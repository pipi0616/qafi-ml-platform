import json
import os
from pathlib import Path

import pytest
from boto3.s3.transfer import TransferConfig

from qafi_platform import ingest, raw_fetcher
from qafi_platform.raw_validator import check, process

from .conftest import BUCKET


def test_crc64nvme_matches_the_reference_value(tmp_path):
    path = tmp_path / "check.txt"
    path.write_bytes(b"123456789")
    assert ingest.crc64nvme(path) == "rosUhgp5mIg="  # CRC-64/NVME check value 0xae8b14860a799888


def test_multipart_upload_is_verified_byte_for_byte(s3, tmp_path, monkeypatch):
    small_parts = TransferConfig(multipart_threshold=5 << 20, multipart_chunksize=5 << 20)
    monkeypatch.setattr(ingest, "UPLOAD", small_parts)
    path = tmp_path / "big.bin"
    path.write_bytes(os.urandom(12 << 20))
    stored = ingest.upload(s3, path, BUCKET, "datasets/test/v1/big.bin", progress=False)
    assert stored["crc64nvme"] == ingest.crc64nvme(path)
    assert stored["etag"].endswith('-3"')  # three parts: the ETag is not a checksum of the file


def test_upload_then_lambda_then_receipt(s3, tmp_path):
    path = tmp_path / "data.tsv"
    path.write_bytes(b"gene\tscore\nTP53\t0.9\n")
    key = ingest.raw_key("proteome", "v1.0", path.name)
    stored = ingest.upload(s3, path, BUCKET, key, progress=False, source="QAFI paper pipeline")
    obj = {"key": key, "size": stored["size_bytes"], "versionId": stored["version_id"]}
    process({"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": obj}}]}, s3)  # what S3 + Lambda do
    receipt = ingest.wait_for_receipt(s3, BUCKET, key, stored["version_id"], timeout=1)
    assert receipt["status"] == "accepted"
    assert receipt["checksum"] == {"algorithm": "CRC64NVME", "value": stored["crc64nvme"]}
    assert receipt["source"] == {"source": "QAFI paper pipeline"}


def test_waiting_for_a_receipt_that_never_comes_times_out(s3):
    with pytest.raises(TimeoutError):
        ingest.wait_for_receipt(s3, BUCKET, "datasets/proteome/v1.0/missing.tsv", timeout=0)


def test_every_configured_source_fits_the_raw_rules():
    sources = ingest.load_sources(Path(__file__).parents[2] / "config" / "sources.toml")
    assert len(sources) == 7
    for source in sources:
        assert source["url"].startswith("https://")
        problems = check(source["key"], 10, b"")  # empty content: only naming and type rules apply
        assert not [p for p in problems if "key must" in p or "not allowed" in p], source["key"]


class _LambdaThatRunsNow:
    """Stands in for AWS: runs the fetcher, then the validator its upload would trigger."""

    def __init__(self, s3):
        self.s3 = s3

    def invoke(self, FunctionName, InvocationType, Payload):
        assert (FunctionName, InvocationType) == (ingest.FETCHER, "Event")
        event = json.loads(Payload)
        raw_fetcher.copy_url(self.s3, event["url"], BUCKET, event["key"])
        version = self.s3.head_object(Bucket=BUCKET, Key=event["key"])["VersionId"]
        obj = {"key": event["key"], "versionId": version}
        process({"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": obj}}]}, self.s3)
        return {"StatusCode": 202}


def test_fetch_sources_copies_each_file_and_waits_for_its_receipt(s3, publisher):
    tar = b"\0" * 257 + b"ustar\x0000" + os.urandom(2000)
    sources = [
        {"url": publisher.serve("human_v4.tar", tar), "key": "datasets/alphafold-human/v4/human_v4.tar"},
        {"url": publisher.serve("ref.csv", b"a,b\n1,2\n"), "key": "datasets/proteingym/v1.3/ref.csv"},
    ]
    receipts = ingest.fetch_sources(_LambdaThatRunsNow(s3), s3, BUCKET, sources, timeout=5)
    assert [r["status"] for r in receipts] == ["accepted", "accepted"]
    assert receipts[0]["source"]["source-url"] == sources[0]["url"]


def test_fetching_again_skips_files_already_in_raw(s3, publisher, capsys):
    sources = [{"url": publisher.serve("ref.csv", b"a,b\n1,2\n"), "key": "datasets/proteingym/v1.3/ref.csv"}]
    lam = _LambdaThatRunsNow(s3)
    first = ingest.fetch_sources(lam, s3, BUCKET, sources, timeout=5)
    again = ingest.fetch_sources(lam, s3, BUCKET, sources, timeout=5)
    assert again == first
    assert "skipped" in capsys.readouterr().out
    assert len(s3.list_object_versions(Bucket=BUCKET, Prefix="datasets/")["Versions"]) == 1  # copied once


def test_uploading_the_same_file_again_is_skipped(s3, tmp_path, capsys):
    path = tmp_path / "data.tsv"
    path.write_bytes(b"gene\tscore\nTP53\t0.9\n")
    key = ingest.raw_key("qafi-predictions", "v1.0", path.name)
    first = ingest.upload(s3, path, BUCKET, key)
    obj = {"key": key, "versionId": first["version_id"]}
    process(
        {"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": obj}}]}, s3
    )  # the receipt S3 + Lambda write
    again = ingest.upload(s3, path, BUCKET, key)
    assert again == first
    assert "skipped" in capsys.readouterr().out
    assert len(s3.list_object_versions(Bucket=BUCKET, Prefix=key)["Versions"]) == 1
    forced = ingest.upload(s3, path, BUCKET, key, progress=False, force=True)
    assert forced["version_id"] != first["version_id"]
