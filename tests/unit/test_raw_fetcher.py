import os

import pytest

from qafi_platform import raw_fetcher

from .conftest import BUCKET


@pytest.fixture
def small_parts(monkeypatch):
    monkeypatch.setattr(raw_fetcher, "PART_SIZE", 5 * 1024 * 1024)  # S3's minimum part size


def test_large_file_is_copied_in_parts(s3, publisher, small_parts):
    data = os.urandom(12 * 1024 * 1024)
    url = publisher.serve("structures_v4.tar", data)
    result = raw_fetcher.copy_url(s3, url, BUCKET, "datasets/alphafold-human/v4/structures_v4.tar")
    assert result == {
        "key": "datasets/alphafold-human/v4/structures_v4.tar",
        "size_bytes": len(data),
        "method": "3 parts",
    }
    stored = s3.get_object(Bucket=BUCKET, Key=result["key"], ChecksumMode="ENABLED")
    assert stored["Body"].read() == data
    assert stored["ChecksumCRC32"]
    assert stored["Metadata"] == {"source-url": url, "source-last-modified": "Mon, 05 Oct 2026 13:02:37 GMT"}


def test_server_without_ranges_is_streamed(s3, publisher, small_parts):
    data = os.urandom(6 * 1024 * 1024)
    url = publisher.serve("variant_summary.txt.gz", data, ranges=False)
    result = raw_fetcher.copy_url(s3, url, BUCKET, "datasets/clinvar/2026-10/variant_summary.txt.gz")
    assert result["method"] == "single stream"
    assert s3.get_object(Bucket=BUCKET, Key=result["key"])["Body"].read() == data


def test_small_file_is_copied_whole(s3, publisher):
    url = publisher.serve("DMS_substitutions.csv", b"DMS_id,UniProt_ID\nA4_HUMAN,P05067\n")
    assert (
        raw_fetcher.copy_url(s3, url, BUCKET, "datasets/proteingym/v1.3/DMS_substitutions.csv")["size_bytes"]
        == 34
    )


def test_keys_outside_datasets_are_refused(s3, publisher):
    url = publisher.serve("x.csv", b"a,b\n")
    with pytest.raises(ValueError):
        raw_fetcher.copy_url(s3, url, BUCKET, "_registry/x.csv")


def test_failed_copy_leaves_no_half_upload(s3, publisher, small_parts, monkeypatch):
    url = publisher.serve("big.tar", os.urandom(11 * 1024 * 1024))
    monkeypatch.setattr(
        raw_fetcher, "_get_range", lambda *a, **k: (_ for _ in ()).throw(OSError("network down"))
    )
    with pytest.raises(OSError):
        raw_fetcher.copy_url(s3, url, BUCKET, "datasets/test/v1/big.tar")
    assert s3.list_multipart_uploads(Bucket=BUCKET).get("Uploads", []) == []
    assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET)
