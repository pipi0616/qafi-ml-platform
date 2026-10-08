"""Lambda function: copy one public file from its official URL into the raw bucket.

Event: ``{"url": "https://...", "key": "datasets/<dataset>/<version>/<file>"}``.

The file goes straight from the publisher to S3 without touching a disk. When the server
supports HTTP range requests (all our sources do), it is copied in 64 MB parts, 8 at a time:
each part is downloaded with a Range request and uploaded as one part of an S3 multipart
upload carrying a CRC32 checksum, which S3 verifies. A 5 GB file takes a few minutes, well
inside Lambda's 15. The copy must have exactly the size the publisher announced, and the
object records where it came from (URL and the publisher's Last-Modified date). The raw
validator then checks the file and writes its receipt, as for any other upload.

Deployed on its own as a zip, so it only uses the standard library and the runtime's boto3.
"""

from __future__ import annotations

import http.client
import logging
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

log = logging.getLogger(__name__)

PART_SIZE = 64 * 1024 * 1024
WORKERS = 8
USER_AGENT = "qafi-ml-platform raw-fetcher (research data ingestion)"


def _http(url: str, method: str = "GET", byte_range: tuple[int, int] | None = None):
    headers = {"User-Agent": USER_AGENT}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers), timeout=120)


def _get_range(url: str, start: int, end: int, attempts: int = 3) -> bytes:
    """Bytes ``start``..``end`` (inclusive) of ``url``, retried on network errors."""
    for attempt in range(1, attempts + 1):
        try:
            with _http(url, byte_range=(start, end)) as resp:
                data = resp.read()
            if len(data) != end - start + 1:
                raise OSError(f"got {len(data)} bytes for range {start}-{end}")
            return data
        except (OSError, http.client.HTTPException):  # dropped connections, timeouts, short reads
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def copy_url(s3, url: str, bucket: str, key: str) -> dict:
    """Copy ``url`` to ``s3://bucket/key``; returns the size and how it was copied."""
    if not key.startswith("datasets/"):
        raise ValueError(f"refusing to write outside datasets/: {key}")
    with _http(url, method="HEAD") as resp:
        size = int(resp.headers.get("Content-Length") or -1)
        ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
        metadata = {"source-url": url, "source-last-modified": resp.headers.get("Last-Modified", "unknown")}

    if size > PART_SIZE and ranges:
        upload = s3.create_multipart_upload(
            Bucket=bucket, Key=key, Metadata=metadata, ChecksumAlgorithm="CRC32", ChecksumType="FULL_OBJECT"
        )
        upload_id = upload["UploadId"]

        def copy_part(number: int) -> dict:
            start = (number - 1) * PART_SIZE
            data = _get_range(url, start, min(start + PART_SIZE, size) - 1)
            part = s3.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=number,
                Body=data,
                ChecksumAlgorithm="CRC32",
            )
            return {"PartNumber": number, "ETag": part["ETag"], "ChecksumCRC32": part["ChecksumCRC32"]}

        try:
            with ThreadPoolExecutor(WORKERS) as pool:
                parts = list(pool.map(copy_part, range(1, -(-size // PART_SIZE) + 1)))
            s3.complete_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts}
            )
        except Exception:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            raise
        method = f"{len(parts)} parts"
    else:
        with _http(url) as resp:
            s3.upload_fileobj(
                resp, bucket, key, ExtraArgs={"Metadata": metadata, "ChecksumAlgorithm": "CRC32"}
            )
        method = "single stream"

    stored = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    if size >= 0 and stored != size:
        raise RuntimeError(f"{key}: publisher announced {size} bytes but {stored} were stored")
    return {"key": key, "size_bytes": stored, "method": method}


_s3 = None


def lambda_handler(event: dict, context) -> dict:
    """Entry point Lambda calls. The S3 client is created once per container and then reused."""
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3", config=Config(max_pool_connections=2 * WORKERS, retries={"mode": "standard"})
        )
    started = time.monotonic()
    result = copy_url(_s3, event["url"], os.environ["RAW_BUCKET"], event["key"])
    result["seconds"] = round(time.monotonic() - started, 1)
    log.info(
        "copied %s (%d bytes, %s) in %.1f s",
        result["key"],
        result["size_bytes"],
        result["method"],
        result["seconds"],
    )
    return result
