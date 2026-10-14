"""The AWS lake behind the API (notebook 10a): the released scores as files in S3, queried with Athena.

Layout in the lake bucket, one folder per release; a release is written once and never changed:

  variant_effect/release=<release>/part-00.parquet ...   every scored variant, sorted by protein
  proteins/release=<release>/proteins.parquet            one row per protein: gene name, length
  manifests/<release>.json                               what the export wrote: files, rows, sizes

Free Edition cannot write to our bucket, so the export travels like the raw files did, the other way round:
``upload_plan`` signs one temporary upload link per file (only those keys, for a few hours) and
``upload_plan_to_volume`` puts the links where the Databricks job ``qafi lake export`` finds them.
``check`` compares S3 with the job's manifest, ``publish`` registers the release with the Glue catalog and
points the views ``*_current`` at it (the data's "champion" label), ``query`` runs SQL for the notebooks.
"""

from __future__ import annotations

import io
import json
import re

import pandas as pd

from qafi_platform import athena

TABLES = ("variant_effect", "proteins")
FILES = 16  # parts of variant_effect: about 200 MB each for the whole proteome
RELEASE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.-]*$")  # also goes into DDL, so nothing else is allowed


def _release(release: str) -> str:
    if not RELEASE.match(release):
        raise ValueError(f"not a release name: {release!r}")
    return release


def keys(release: str, files: int = FILES) -> dict:
    """The S3 keys of one release."""
    _release(release)
    return {
        "variant_effect": [f"variant_effect/release={release}/part-{i:02d}.parquet" for i in range(files)],
        "proteins": f"proteins/release={release}/proteins.parquet",
        "manifest": f"manifests/{release}.json",
    }


def _listed(s3, bucket: str, prefix: str) -> dict[str, int]:
    found = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        found.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    return found


def upload_plan(
    s3, bucket: str, release: str, files: int = FILES, hours: int = 6, overwrite: bool = False
) -> dict:
    """One temporary upload link per file of ``release``; whoever holds a link can write that key only.

    A release is written once: if it is in the bucket already this raises, unless ``overwrite``, which
    first deletes it (the bucket keeps the old versions).
    """
    names = keys(release, files)
    existing = [k for t in TABLES for k in _listed(s3, bucket, f"{t}/release={release}/")]
    existing += list(_listed(s3, bucket, names["manifest"]))
    if existing and not overwrite:
        raise FileExistsError(f"release {release} is already in s3://{bucket} ({len(existing)} files)")
    for key in existing:
        s3.delete_object(Bucket=bucket, Key=key)

    def sign(key: str) -> dict:
        url = s3.generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=hours * 3600
        )
        return {"key": key, "url": url}

    return {
        "release": release,
        "variant_effect": [sign(k) for k in names["variant_effect"]],
        "proteins": sign(names["proteins"]),
        "manifest": sign(names["manifest"]),
    }


def upload_plan_to_volume(w, catalog: str, plan: dict) -> str:
    """Write the links to the volume, where the export job picks them up; returns the file's path."""
    path = f"/Volumes/{catalog}/ml/files/_export/links.json"
    w.files.upload(path, io.BytesIO(json.dumps(plan, indent=1).encode()), overwrite=True)
    return path


def check(s3, bucket: str, release: str) -> pd.DataFrame:
    """Every file the export job wrote, with its rows and size, and whether S3 holds exactly that.

    Raises if a file is missing, has another size, or if the release holds files the job did not write.
    """
    names = keys(release)
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=names["manifest"])["Body"].read())
    stored = {k: v for t in TABLES for k, v in _listed(s3, bucket, f"{t}/release={release}/").items()}
    table = pd.DataFrame(manifest["files"])
    table["in_s3"] = table["key"].map(stored)
    table["ok"] = table["in_s3"] == table["bytes"]
    unexpected = sorted(set(stored) - set(table["key"]))
    if not table["ok"].all() or unexpected:
        wrong = table.loc[~table["ok"], "key"].tolist()
        raise RuntimeError(f"release {release} is incomplete: wrong {wrong}, unexpected {unexpected}")
    table.attrs.update({k: v for k, v in manifest.items() if k != "files"})
    return table


def publish(client, workgroup: str, database: str, bucket: str, release: str) -> None:
    """Register ``release`` with the catalog and point ``variant_effect_current`` and ``proteins_current``
    at it. The API reads only the views, so moving them releases new scores and moving them back rolls back.
    """
    _release(release)
    for table in TABLES:
        athena.run(
            client,
            f"ALTER TABLE {table} ADD IF NOT EXISTS PARTITION (release = '{release}') "
            f"LOCATION 's3://{bucket}/{table}/release={release}/'",
            workgroup,
            database,
        )
        athena.run(
            client,
            f"CREATE OR REPLACE VIEW {table}_current AS SELECT * FROM {table} WHERE release = '{release}'",
            workgroup,
            database,
        )


def query(client, sql: str, workgroup: str, database: str, parameters: list | None = None) -> pd.DataFrame:
    """An Athena query as a DataFrame; ``attrs`` says how much it scanned and how long it took."""
    result = athena.run(client, sql, workgroup, database, parameters)
    frame = pd.DataFrame(result.rows, columns=result.columns)
    frame.attrs.update(scanned_mb=round(result.scanned_bytes / 1e6, 1), seconds=result.milliseconds / 1000)
    return frame
