"""Laptop-side helpers for the Databricks chapters (notebook 04 onwards).

Free Edition cannot attach our S3 bucket, so raw files travel through temporary download links:
``transfer_links`` signs a read-only link per file (valid a few hours, no AWS key leaves the laptop)
and ``upload_transfer_list`` puts the list into the volume, where the job's copy task reads it.
``sql`` runs a query on the workspace's SQL warehouse and returns a pandas DataFrame.

``w`` is a ``databricks.sdk.WorkspaceClient``; the SDK is only needed where these are called.
"""

from __future__ import annotations

import io
import json
import time

import pandas as pd


def transfer_links(s3, bucket: str, keys: list[str], hours: int = 6) -> dict[str, dict]:
    """For each S3 key, a temporary read-only download link and the size the copy must have.

    With the size in the list, the copy task can skip files it already holds without using the
    link, so a re-run works even after the links have expired.
    """
    return {
        key: {
            "url": s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=hours * 3600
            ),
            "size": s3.head_object(Bucket=bucket, Key=key)["ContentLength"],
        }
        for key in keys
    }


def upload_transfer_list(w, catalog: str, links: dict[str, dict]) -> str:
    """Write the links to the volume, where the copy task picks them up; returns the file's path."""
    path = f"/Volumes/{catalog}/raw/files/_transfer/links.json"
    w.files.upload(path, io.BytesIO(json.dumps(links, indent=1).encode()), overwrite=True)
    return path


def warehouse_id(w) -> str:
    """The workspace's SQL warehouse (Free Edition has exactly one)."""
    return next(iter(w.warehouses.list())).id


def sql(w, statement: str, warehouse: str | None = None, timeout: int = 600) -> pd.DataFrame:
    """Run ``statement`` on the SQL warehouse and return the result as a DataFrame.

    The warehouse starts by itself if it is stopped, which takes a few seconds.
    """
    api = w.statement_execution
    response = api.execute_statement(
        statement=statement, warehouse_id=warehouse or warehouse_id(w), wait_timeout="30s"
    )
    deadline = time.monotonic() + timeout
    while response.status.state.value in ("PENDING", "RUNNING"):
        if time.monotonic() > deadline:
            api.cancel_execution(response.statement_id)
            raise TimeoutError(f"query still running after {timeout} s")
        time.sleep(2)
        response = api.get_statement(response.statement_id)
    if response.status.state.value != "SUCCEEDED":
        error = response.status.error.message if response.status.error else response.status.state.value
        raise RuntimeError(f"query failed: {error}")
    columns = [c.name for c in response.manifest.schema.columns] if response.manifest else []
    rows, chunk = [], response.result
    while chunk is not None:
        rows.extend(chunk.data_array or [])
        if chunk.next_chunk_index is None:
            break
        chunk = api.get_statement_result_chunk_n(response.statement_id, chunk.next_chunk_index)
    return pd.DataFrame(rows, columns=columns)


def latest_run(w, job_name: str) -> pd.DataFrame:
    """State of each task in the job's most recent run, with minutes so far."""
    job = next(iter(w.jobs.list(name=job_name)))
    run = w.jobs.get_run(next(iter(w.jobs.list_runs(job_id=job.job_id, limit=1))).run_id)
    print(run.run_page_url)
    now = time.time() * 1000
    rows = []
    for task in sorted(run.tasks or [], key=lambda t: t.start_time or float("inf")):  # in the order they ran
        state = task.state
        end = task.end_time or now
        rows.append(
            {
                "task": task.task_key,
                "state": state.life_cycle_state.value if state and state.life_cycle_state else None,
                "result": state.result_state.value if state and state.result_state else None,
                "minutes": round((end - task.start_time) / 60000, 1) if task.start_time else None,
            }
        )
    return pd.DataFrame(rows)
