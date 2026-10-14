import json
import urllib.request
from urllib.parse import parse_qs, unquote, urlparse

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lakehouse import export
from moto.server import ThreadedMotoServer

from qafi_platform import athena, lake

from .athena_fake import FakeAthena

BUCKET = "test-lake"


@pytest.fixture
def s3(monkeypatch):
    """A local S3 server (moto), so upload links can be used over HTTP like the Databricks job uses them."""
    for var in ("AWS_PROFILE", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    reset = urllib.request.Request(f"http://{host}:{port}/moto-api/reset", method="POST")
    urllib.request.urlopen(reset).close()  # the fake keeps its buckets between servers otherwise
    client = boto3.client("s3", region_name="eu-central-1", endpoint_url=f"http://{host}:{port}")
    client.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
    client.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    yield client
    server.stop()


def write_parts(folder, proteins_per_part):
    """Parquet files like the export job's: variant_effect parts sorted by protein, and the protein list."""
    paths = []
    for i, proteins in enumerate(proteins_per_part):
        table = pa.table(
            {
                "uniprot_id": [p for p in proteins for _ in range(19)],
                "position": pa.array([1] * 19 * len(proteins), pa.int32()),
                "function_score": pa.array([0.5] * 19 * len(proteins), pa.float32()),
            }
        )
        paths.append(folder / f"part-{i:05d}.parquet")
        pq.write_table(table, paths[-1])
    everything = sorted(p for part in proteins_per_part for p in part)
    pq.write_table(pa.table({"uniprot_id": everything}), folder / "proteins.parquet")
    return paths, folder / "proteins.parquet"


def export_release(s3, tmp_path, release="r1"):
    """What notebook 10a and the job do together: sign, upload every file, upload the manifest last."""
    plan = lake.upload_plan(s3, BUCKET, release, files=4)
    parts, proteins = write_parts(tmp_path, [["A00001", "A00002"], ["B00001"]])
    files = []
    for path, link in zip(parts, plan["variant_effect"], strict=False):
        export.put(str(path), link["url"])
        files.append(export.describe(str(path), link["key"]))
    export.put(str(proteins), plan["proteins"]["url"])
    written = export.manifest(
        release, "gold.x", files, export.describe(str(proteins), plan["proteins"]["key"]), [{"v": "1"}]
    )
    export.put(json.dumps(written).encode(), plan["manifest"]["url"], "application/json")
    return plan, written


def test_upload_links_are_temporary_and_name_their_file(s3):
    plan = lake.upload_plan(s3, BUCKET, "2026-10-14", files=2, hours=3)
    link = plan["variant_effect"][1]
    assert link["key"] == "variant_effect/release=2026-10-14/part-01.parquet"
    url = urlparse(link["url"])
    assert unquote(url.path).endswith(link["key"]) and parse_qs(url.query)["X-Amz-Expires"] == ["10800"]
    assert plan["proteins"]["key"] == "proteins/release=2026-10-14/proteins.parquet"
    assert plan["manifest"]["key"] == "manifests/2026-10-14.json"


def test_an_export_lands_complete_and_is_checked_against_its_manifest(s3, tmp_path):
    _, written = export_release(s3, tmp_path)
    assert written["variants"] == 57 and written["proteins"] == 3
    assert written["files"][0]["first_protein"] == "A00001"
    assert written["files"][1]["last_protein"] == "B00001"
    table = lake.check(s3, BUCKET, "r1")
    assert table["ok"].all() and len(table) == 3 and table.attrs["variants"] == 57


def test_a_file_the_job_did_not_write_fails_the_check(s3, tmp_path):
    export_release(s3, tmp_path)
    s3.put_object(Bucket=BUCKET, Key="variant_effect/release=r1/part-09.parquet", Body=b"stale")
    with pytest.raises(RuntimeError, match="part-09"):
        lake.check(s3, BUCKET, "r1")


def test_a_release_is_written_once_unless_overwritten_on_purpose(s3, tmp_path):
    export_release(s3, tmp_path)
    with pytest.raises(FileExistsError, match="r1"):
        lake.upload_plan(s3, BUCKET, "r1")
    lake.upload_plan(s3, BUCKET, "r1", overwrite=True)  # deletes the release first
    assert s3.list_objects_v2(Bucket=BUCKET).get("KeyCount", 0) == 0
    assert len(s3.list_object_versions(Bucket=BUCKET)["Versions"]) == 4  # the bucket keeps the old files


def test_release_names_cannot_smuggle_sql():
    with pytest.raises(ValueError):
        lake.keys("r1'; DROP TABLE proteins; --")


def test_publishing_registers_the_release_and_moves_the_current_views(monkeypatch):
    monkeypatch.setattr(athena.time, "sleep", lambda s: None)
    client = FakeAthena(lambda sql, p: ([], []))
    lake.publish(client, "wg", "db", BUCKET, "r2")
    statements = [r["QueryString"] for r in client.requests]
    assert statements == [
        f"ALTER TABLE variant_effect ADD IF NOT EXISTS PARTITION (release = 'r2') "
        f"LOCATION 's3://{BUCKET}/variant_effect/release=r2/'",
        "CREATE OR REPLACE VIEW variant_effect_current AS SELECT * FROM variant_effect WHERE release = 'r2'",
        f"ALTER TABLE proteins ADD IF NOT EXISTS PARTITION (release = 'r2') LOCATION 's3://{BUCKET}/proteins/release=r2/'",
        "CREATE OR REPLACE VIEW proteins_current AS SELECT * FROM proteins WHERE release = 'r2'",
    ]


def test_a_query_comes_back_as_a_table_with_what_it_scanned(monkeypatch):
    monkeypatch.setattr(athena.time, "sleep", lambda s: None)
    client = FakeAthena(lambda sql, p: ([("n", "bigint")], [(42,)]))
    frame = lake.query(client, "SELECT count(*) AS n FROM proteins_current", "wg", "db")
    assert frame["n"].tolist() == [42] and frame.attrs == {"scanned_mb": 12.3, "seconds": 0.8}


def test_a_broken_upload_is_tried_again_and_labelled_as_a_file(monkeypatch):
    import io

    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise ConnectionResetError("connection reset by peer")
        return io.BytesIO()

    monkeypatch.setattr(export.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(export.time, "sleep", lambda s: None)
    export.put(b"PAR1", "https://link")
    assert len(calls) == 2
    assert calls[1].get_method() == "PUT"
    assert calls[1].get_header("Content-type") == "application/octet-stream"
