"""Uploading the released scores to the AWS lake (job ``qafi lake export``, notebook 10a).

The job's notebook writes the parquet files with Spark into the volume; ``put`` sends each one to its
temporary upload link and ``describe`` says what it holds, for the manifest the laptop checks S3 against.
"""

from __future__ import annotations

import os
import time
import urllib.request
from datetime import UTC, datetime

import pyarrow.parquet as pq


def describe(path: str, key: str) -> dict:
    """Rows, size and the first and last protein of a parquet file, from its footer (nothing else read)."""
    meta = pq.ParquetFile(path).metadata
    first = last = None
    if meta.num_row_groups:
        column = meta.schema.names.index("uniprot_id")
        stats = [meta.row_group(i).column(column).statistics for i in range(meta.num_row_groups)]
        first, last = min(s.min for s in stats), max(s.max for s in stats)
    return {
        "key": key,
        "rows": meta.num_rows,
        "bytes": os.path.getsize(path),
        "row_groups": meta.num_row_groups,
        "first_protein": first,
        "last_protein": last,
    }


def manifest(
    release: str, source: str, variant_files: list[dict], protein_file: dict, versions: list
) -> dict:
    """What one export wrote: uploaded last, so its presence in S3 means the release is complete."""
    return {
        "release": release,
        "source": source,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "variants": sum(f["rows"] for f in variant_files),
        "proteins": protein_file["rows"],
        "model_versions": versions,
        "files": [*variant_files, protein_file],
    }


def put(body, url: str, content_type: str = "application/octet-stream", attempts: int = 4) -> None:
    """Upload a file (a path) or ``bytes`` to a temporary upload link; a broken transfer is tried again.

    The content type is set on purpose: without it urllib labels the upload as a web form.
    """
    for attempt in range(attempts):
        try:
            if isinstance(body, bytes):
                headers = {"Content-Type": content_type}
                request = urllib.request.Request(url, data=body, method="PUT", headers=headers)
                urllib.request.urlopen(request, timeout=600).close()
            else:
                with open(body, "rb") as data:  # streamed, not read into memory
                    headers = {"Content-Type": content_type, "Content-Length": str(os.path.getsize(body))}
                    request = urllib.request.Request(url, data=data, method="PUT", headers=headers)
                    urllib.request.urlopen(request, timeout=600).close()
            return
        except OSError:  # includes HTTP errors and a lost connection
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)
