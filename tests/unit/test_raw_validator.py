import gzip
import json
from urllib.parse import quote_plus

import pytest

from qafi_platform.raw_validator import check, process

from .conftest import BUCKET


def _upload(s3, key: str, body: bytes) -> dict:
    """Store an object and return the event S3 would send the Lambda for it."""
    version = s3.put_object(Bucket=BUCKET, Key=key, Body=body)["VersionId"]
    obj = {"key": quote_plus(key), "size": len(body), "versionId": version}
    return {
        "Records": [
            {
                "eventTime": "2026-10-07T12:00:00.000Z",
                "userIdentity": {"principalId": "AWS:TESTUSER"},
                "s3": {"bucket": {"name": BUCKET}, "object": obj},
            }
        ]
    }


def _receipt(s3, key: str) -> dict:
    return json.loads(s3.get_object(Bucket=BUCKET, Key=f"_registry/{key}.json")["Body"].read())


def test_good_file_is_accepted_and_registered(s3):
    key = "datasets/proteome/v1.0/data.tsv.gz"
    assert process(_upload(s3, key, gzip.compress(b"a\tb\n1\t2\n")), s3) == {"accepted": 1, "rejected": 0}
    receipt = _receipt(s3, key)
    assert receipt["status"] == "accepted"
    assert receipt["problems"] == []
    assert receipt["checksum"]["value"]
    assert receipt["uploaded_by"] == "AWS:TESTUSER"


def test_content_that_does_not_match_the_name_is_rejected(s3):
    key = "datasets/proteome/v1.0/fake.tsv.gz"
    assert process(_upload(s3, key, b"plain text, not gzip"), s3) == {"accepted": 0, "rejected": 1}
    receipt = _receipt(s3, key)
    assert receipt["status"] == "rejected"
    assert "wrong file signature" in receipt["problems"][0]


def test_receipt_belongs_to_the_version_that_triggered_the_event(s3):
    key = "datasets/proteome/v1.0/data.tsv"
    first = _upload(s3, key, b"a\tb\n")
    _upload(s3, key, b"")  # a newer, empty version arrives before the first event is handled
    process(first, s3)
    receipt = _receipt(s3, key)
    assert receipt["version_id"] == first["Records"][0]["s3"]["object"]["versionId"]
    assert receipt["status"] == "accepted"


def test_url_encoded_key_is_decoded(s3):
    key = "datasets/clinvar/2026-10/variant summary.tsv"
    process(_upload(s3, key, b"a\tb\n"), s3)
    assert _receipt(s3, key)["status"] == "accepted"


def test_files_outside_datasets_are_ignored(s3):
    assert process(_upload(s3, "notes/readme.txt", b"hello"), s3) == {"accepted": 0, "rejected": 0}


@pytest.mark.parametrize(
    ("key", "size", "head", "problem"),
    [
        ("datasets/proteome/data.tsv.gz", 10, b"\x1f\x8b", "key must look like"),
        ("datasets/Proteome/v1/data.tsv", 10, b"a\tb", "key must look like"),
        ("datasets/proteome/v1/data.tsv", 0, b"", "file is empty"),
        ("datasets/proteome/v1/data.xlsx", 10, b"PK\x03\x04", "file type not allowed"),
        ("datasets/proteome/v1/data.tsv", 10, b"a\x00b", "binary data"),
        ("datasets/proteome/v1/data.parquet", 10, b"\x1f\x8b", "wrong file signature"),
    ],
)
def test_check_reports_problems(key, size, head, problem):
    assert any(problem in p for p in check(key, size, head))


def test_receipt_records_where_a_fetched_file_came_from(s3):
    key = "datasets/clinvar/2026-10/variant_summary.txt.gz"
    source = {
        "source-url": "https://ftp.ncbi.nlm.nih.gov/x.txt.gz",
        "source-last-modified": "Mon, 05 Oct 2026",
    }
    version = s3.put_object(Bucket=BUCKET, Key=key, Body=gzip.compress(b"a"), Metadata=source)["VersionId"]
    event = {"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": {"key": key, "versionId": version}}}]}
    process(event, s3)
    assert _receipt(s3, key)["source"] == source


def test_tar_signature_is_checked_at_its_offset():
    tar_head = b"\0" * 257 + b"ustar\x0000" + b"\0" * 760
    assert check("datasets/alphafold-human/v4/UP000005640_9606_HUMAN_v4.tar", 10, tar_head) == []
    problems = check("datasets/alphafold-human/v4/fake.tar", 10, b"\x1f\x8b" + b"\0" * 1000)
    assert "wrong file signature" in problems[0]


def test_check_accepts_the_allowed_types():
    assert check("datasets/proteome/v1.0/all.tar.gz", 10, b"\x1f\x8b\x08") == []
    assert check("datasets/proteome/v1.0/part.parquet", 10, b"PAR1") == []
    assert check("datasets/uniprot/2026_04/human.zip", 10, b"PK\x03\x04") == []
    assert check("datasets/clinvar/2026-10/summary.csv", 10, b"a,b\n") == []
