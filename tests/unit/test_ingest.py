import json
import os
from datetime import date
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
    # 13, plus the Swiss-Prot entries for the knowledge base (11), plus 4 for the stability chapter (14a)
    assert len(sources) == 18
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


def test_a_new_version_changes_the_url_and_the_raw_folder():
    config = Path(__file__).parents[2] / "config" / "sources.toml"
    sources = {s["dataset"]: s for s in ingest.load_sources(config, {"clinvar": "2026-11"})}
    assert sources["clinvar"]["url"].endswith("/variant_summary_2026-11.txt.gz")
    assert sources["clinvar"]["key"] == "datasets/clinvar/2026-11/variant_summary_2026-11.txt.gz"
    assert sources["proteingym"]["version"] == "v1.3"  # datasets not named keep their pinned version


def _monthly(publisher, *months):
    """A publisher that has released the given months; returns the URL template of its monthly file."""
    for month in months:
        url = publisher.serve(f"variant_summary_{month}.txt.gz", b"x")
    return url.replace(months[-1], "{version}")


def test_the_newest_published_month_is_found(publisher):
    template = _monthly(publisher, "2026-10", "2026-11")
    assert ingest.newest_monthly_release(template, date(2026, 11, 20), oldest="2026-10") == "2026-11"
    # early in December the December release is not out yet: November is still the newest
    assert ingest.newest_monthly_release(template, date(2026, 12, 2), oldest="2026-10") == "2026-11"


def test_never_older_than_the_pinned_version(publisher):
    template = _monthly(publisher, "2026-09")
    assert ingest.newest_monthly_release(template, date(2026, 10, 3), oldest="2026-10") == "2026-10"


def test_no_release_for_three_months_is_an_error(publisher):
    template = _monthly(publisher, "2026-01")
    with pytest.raises(RuntimeError, match="no monthly release"):
        ingest.newest_monthly_release(template, date(2026, 10, 3), oldest="2026-01")


def test_only_monthly_datasets_move_to_a_newer_version(publisher, tmp_path):
    template = _monthly(publisher, "2026-10", "2026-11")
    config = tmp_path / "sources.toml"
    config.write_text(
        f'[[dataset]]\nname = "clinvar"\nversion = "2026-10"\nrefresh = "monthly"\nurls = ["{template}"]\n'
        '[[dataset]]\nname = "proteingym"\nversion = "v1.3"\nrefresh = "manual"\nurls = ["https://x/{version}.zip"]\n'
    )
    assert ingest.newest_releases(config, date(2026, 11, 20)) == {"clinvar": "2026-11"}


def test_a_dataset_pinned_twice_keeps_both_versions():
    config = Path(__file__).parents[2] / "config" / "sources.toml"
    sources = ingest.load_sources(config, {"clinvar": "2026-11"})
    assert sorted(s["version"] for s in sources if s["dataset"] == "alphafold-human") == ["v4", "v6"]
