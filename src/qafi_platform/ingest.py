"""Get files into the raw bucket and confirm they arrived intact (notebooks 03 and 03b).

The raw bucket is the system of record: files are stored exactly as received under
``datasets/<dataset>/<version>/<file>``. Every upload carries a CRC64NVME checksum that S3
verifies on arrival; afterwards the checksum S3 stored is compared with one computed from the
local file, which proves the cloud copy is identical byte for byte. The Lambda in
``raw_validator.py`` then checks the file and writes a receipt to ``_registry/``.

Public datasets listed in ``config/sources.toml`` don't pass through the laptop: the fetcher
Lambda in ``raw_fetcher.py`` copies them from the publisher straight into the raw bucket, and the
same validator checks them.

These helpers print progress because they are run interactively from notebooks.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import threading
import time
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from botocore.httpchecksum import CrtCrc64NvmeChecksum

PART_SIZE = 32 * 1024 * 1024  # the 7.4 GB tarball goes up as 222 parts, 8 at a time
UPLOAD = TransferConfig(multipart_threshold=PART_SIZE, multipart_chunksize=PART_SIZE, max_concurrency=8)
FETCHER = "qafi-dev-raw-fetcher"


def raw_key(dataset: str, version: str, filename: str) -> str:
    """Where a file lives in the raw bucket."""
    return f"datasets/{dataset}/{version}/{filename}"


def crc64nvme(path: Path) -> str:
    """CRC64NVME checksum of a local file, base64-encoded the way S3 reports it."""
    checksum = CrtCrc64NvmeChecksum()
    with open(path, "rb") as f:
        while chunk := f.read(PART_SIZE):
            checksum.update(chunk)
    return checksum.b64digest()


class _Progress:
    """Prints a line every 5% of an upload. boto3 calls it from several threads at once."""

    def __init__(self, total: int):
        self.total, self.done, self.next_mark = total, 0, 5
        self.start = time.monotonic()
        self.lock = threading.Lock()

    def __call__(self, nbytes: int) -> None:
        with self.lock:
            self.done += nbytes
            pct = self.done * 100 // self.total
            if pct >= self.next_mark:
                minutes = (time.monotonic() - self.start) / 60
                print(f"  {pct:3d}%  {self.done / 1e9:5.2f} / {self.total / 1e9:.2f} GB  ({minutes:.1f} min)")
                self.next_mark = pct - pct % 5 + 5


def upload(
    s3,
    path: Path,
    bucket: str,
    key: str,
    progress: bool = True,
    source: str | None = None,
    force: bool = False,
) -> dict:
    """Upload ``path`` with a CRC64NVME checksum and prove the stored copy is identical.

    ``source`` says where a non-public file came from; it is stored with the object and the
    validator copies it into the receipt (ASCII only: S3 metadata travels as HTTP headers).
    Files above ``PART_SIZE`` go up as a multipart upload. Raises if S3's checksum differs
    from the local one; otherwise returns what was stored.

    Idempotent: if the raw bucket already holds an accepted copy with the same checksum, nothing
    is uploaded and the existing copy is returned (``force`` uploads a new version anyway).
    """
    path = Path(path)
    size = path.stat().st_size
    local = crc64nvme(path)
    if progress:
        print(f"local CRC64NVME  {local}  ({size / 1e9:.2f} GB)")
    existing = None if force else _current_receipt(s3, bucket, key)
    if (
        existing
        and existing["status"] == "accepted"
        and existing.get("checksum") == {"algorithm": "CRC64NVME", "value": local}
    ):
        if progress:
            print(f"already in raw with the same checksum (checked {existing['checked_at']}): skipped")
        return {
            "key": key,
            "size_bytes": size,
            "crc64nvme": local,
            "version_id": existing["version_id"],
            "etag": s3.head_object(Bucket=bucket, Key=key, VersionId=existing["version_id"])["ETag"],
        }
    extra = {"ChecksumAlgorithm": "CRC64NVME"} | ({"Metadata": {"source": source}} if source else {})
    s3.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs=extra,
        Config=UPLOAD,
        Callback=_Progress(size) if progress else None,
    )
    head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    remote = head.get("ChecksumCRC64NVME")
    if remote != local:
        raise RuntimeError(f"checksum mismatch for {key}: local {local}, S3 {remote}")
    if progress:
        print(f"S3 CRC64NVME     {remote}  identical")
    return {
        "key": key,
        "size_bytes": size,
        "crc64nvme": local,
        "version_id": head.get("VersionId"),
        "etag": head["ETag"],
    }


def wait_for_receipt(
    s3,
    bucket: str,
    key: str,
    version_id: str | None = None,
    timeout: int = 60,
    newer_than: str | None = None,
) -> dict:
    """Wait until the validator has written the receipt for ``key``.

    ``version_id`` waits for the receipt of that exact version; ``newer_than`` (an ISO time)
    ignores receipts checked before it, such as those left by an earlier run.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            receipt = json.loads(s3.get_object(Bucket=bucket, Key=f"_registry/{key}.json")["Body"].read())
            if (version_id is None or receipt.get("version_id") == version_id) and (
                newer_than is None or receipt.get("checked_at", "") >= newer_than
            ):
                return receipt
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                raise
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no receipt for {key} after {timeout} s. Right after terraform apply the trigger can "
                "take a few minutes to start working: upload again. Otherwise check the Lambdas' logs."
            )
        time.sleep(2)


def smoke_test(s3, bucket: str) -> list[dict]:
    """Upload one good and one broken small file and print what the Lambda decided for each."""
    receipts = []
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp, "smoke_ok.tsv.gz")
        good.write_bytes(gzip.compress(b"gene\tvariant\tscore\nTP53\tR175H\t0.97\n"))
        bad = Path(tmp, "smoke_bad.tsv.gz")  # named .gz but is plain text
        bad.write_bytes(b"gene\tvariant\tscore\nTP53\tR175H\t0.97\n")
        for path in (good, bad):
            stored = upload(
                s3, path, bucket, raw_key("smoke-test", "v1", path.name), progress=False, force=True
            )
            receipt = wait_for_receipt(s3, bucket, stored["key"], stored["version_id"], timeout=120)
            reason = f": {'; '.join(receipt['problems'])}" if receipt["problems"] else ""
            print(f"{receipt['key']:<42} {receipt['status']}{reason}")
            receipts.append(receipt)
    return receipts


def load_sources(path: Path) -> list[dict]:
    """One entry per file in ``config/sources.toml``: dataset, version, official URL and raw key."""
    with open(path, "rb") as f:
        datasets = tomllib.load(f)["dataset"]
    return [
        {
            "dataset": d["name"],
            "version": d["version"],
            "url": url,
            "key": raw_key(d["name"], d["version"], url.rsplit("/", 1)[-1]),
        }
        for d in datasets
        for url in d["urls"]
    ]


def _current_receipt(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"_registry/{key}.json")["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def fetch_sources(
    lambda_client, s3, bucket: str, sources: list[dict], timeout: int = 1200, force: bool = False
) -> list[dict]:
    """Send the fetcher Lambda off for every file at once, then wait for the validator's receipts.

    Nothing is downloaded to this computer: each Lambda copies one file from the publisher into
    the raw bucket, and the upload triggers the validator as usual. A file already accepted from
    the same URL is skipped, so running this again copies only what is new (``force`` re-copies).
    """
    started = (datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds")  # clock-skew margin
    receipts: dict[str, dict] = {}
    for source in sources:
        existing = _current_receipt(s3, bucket, source["key"])
        if (
            not force
            and existing
            and existing["status"] == "accepted"
            and (existing.get("source") or {}).get("source-url") == source["url"]
        ):
            receipts[source["key"]] = existing
            print(f"skipped   {source['key']}  (already in raw)")
            continue
        payload = json.dumps({"url": source["url"], "key": source["key"]})
        lambda_client.invoke(FunctionName=FETCHER, InvocationType="Event", Payload=payload)
        print(f"sent      {source['key']}")
    for source in sources:
        if source["key"] in receipts:
            continue
        receipt = wait_for_receipt(s3, bucket, source["key"], timeout=timeout, newer_than=started)
        reason = f"  ({'; '.join(receipt['problems'])})" if receipt["problems"] else ""
        print(f"{receipt['status']:<9} {receipt['key']:<72} {receipt['size_bytes'] / 1e9:6.2f} GB{reason}")
        receipts[source["key"]] = receipt
    return [receipts[source["key"]] for source in sources]
