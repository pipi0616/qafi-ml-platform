"""Lambda function: check every new file in the raw bucket and record it in the registry.

S3 invokes this function for each object created under ``datasets/``. Each file is checked:
- the key follows ``datasets/<dataset>/<version>/<file>``;
- the file is not empty;
- its first bytes match its type (a ``.gz`` file must start with the gzip signature, and so on).
The result goes to ``_registry/<key>.json`` as a receipt with the size, the checksum S3 stored,
the object version and the uploader. Rejected files stay where they are; the receipt says why.

Deployed on its own as a zip, so it only uses the standard library and the boto3 that the
Lambda Python runtime provides.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from urllib.parse import unquote_plus

import boto3

log = logging.getLogger(__name__)

DATA_PREFIX = "datasets/"
REGISTRY_PREFIX = os.environ.get("REGISTRY_PREFIX", "_registry/")
KEY_PATTERN = re.compile(r"datasets/[a-z0-9][a-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._-]*/[^/]+")

# A file's signature bytes identify its real type, whatever its name says: (offset, bytes).
SIGNATURES = {
    ".gz": (0, b"\x1f\x8b"),
    ".zip": (0, b"PK\x03\x04"),
    ".parquet": (0, b"PAR1"),
    ".tar": (257, b"ustar"),  # tar archives carry their signature in the first header block
    ".safetensors": (8, b"{"),  # model weights: 8 bytes giving the header's length, then the JSON header
    # A pickle holds instructions, not just data, so loading one runs code. Accepted here because some
    # publishers ship their splits this way; it is read with a restricted unpickler (qafi_core.stability).
    ".pkl": (0, b"\x80\x04"),  # protocol 4: PROTO opcode, then the version
}
TEXT_SUFFIXES = (".tsv", ".csv", ".txt", ".json")
CHECKSUM_FIELDS = ("ChecksumCRC64NVME", "ChecksumCRC32", "ChecksumCRC32C", "ChecksumSHA256", "ChecksumSHA1")


def check(key: str, size: int, head: bytes) -> list[str]:
    """Problems found in one file, given its key, size and first bytes; empty means accepted."""
    problems = []
    if not KEY_PATTERN.fullmatch(key):
        problems.append("key must look like datasets/<dataset>/<version>/<file>")
    if size == 0:
        return [*problems, "file is empty"]
    name = key.rsplit("/", 1)[-1].lower()
    suffix = next((s for s in SIGNATURES if name.endswith(s)), None)
    if suffix:
        offset, signature = SIGNATURES[suffix]
        if head[offset : offset + len(signature)] != signature:
            problems.append(f"content is not a {suffix} file (wrong file signature)")
    elif name.endswith(TEXT_SUFFIXES):
        if b"\x00" in head:
            problems.append("text file contains binary data")
    else:
        problems.append(
            "file type not allowed (use .gz, .zip, .parquet, .tar, .safetensors, .tsv, .csv, .txt or .json)"
        )
    return problems


def process(event: dict, s3) -> dict:
    """Check and register every file in an S3 event; returns how many were accepted and rejected."""
    counts = {"accepted": 0, "rejected": 0}
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])  # S3 URL-encodes keys in events
        version = record["s3"]["object"].get("versionId")
        if not key.startswith(DATA_PREFIX):
            continue  # S3 only sends datasets/ events; this guards against a misconfigured trigger
        target = {"Bucket": bucket, "Key": key} | ({"VersionId": version} if version else {})

        meta = s3.head_object(**target, ChecksumMode="ENABLED")
        size = meta["ContentLength"]
        head = s3.get_object(**target, Range="bytes=0-1023")["Body"].read() if size else b""
        problems = check(key, size, head)
        status = "rejected" if problems else "accepted"
        algorithm = next((f for f in CHECKSUM_FIELDS if meta.get(f)), None)

        receipt = {
            "key": key,
            "status": status,
            "problems": problems,
            "size_bytes": size,
            "checksum": {"algorithm": algorithm.removeprefix("Checksum"), "value": meta[algorithm]}
            if algorithm
            else None,
            "version_id": version,
            "source": meta.get("Metadata") or None,  # set by the fetcher: official URL and release date
            "uploaded_by": record.get("userIdentity", {}).get("principalId"),
            "uploaded_at": record.get("eventTime"),
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        s3.put_object(
            Bucket=bucket,
            Key=f"{REGISTRY_PREFIX}{key}.json",
            Body=json.dumps(receipt, indent=2).encode(),
            ContentType="application/json",
        )
        if problems:
            log.error("rejected %s: %s", key, "; ".join(problems))
        else:
            log.info("accepted %s (%d bytes)", key, size)
        counts[status] += 1
    return counts


_s3 = None


def lambda_handler(event: dict, context) -> dict:
    """Entry point Lambda calls. The S3 client is created once per container and then reused."""
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return process(event, _s3)
