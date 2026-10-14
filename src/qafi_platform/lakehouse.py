"""Laptop-side helpers for the Databricks chapters (notebook 04 onwards).

Free Edition cannot attach our S3 bucket, so raw files travel through temporary download links:
``transfer_links`` signs a read-only link per file (valid a few hours, no AWS key leaves the laptop)
and ``upload_transfer_list`` puts the list into the volume, where the job's copy task reads it.
``sql`` runs a query on the workspace's SQL warehouse and returns a pandas DataFrame; ``download`` saves a
large result as a parquet file.
``create_machine_identity`` gives Airflow its own Databricks identity, with the secret kept in AWS.

``w`` is a ``databricks.sdk.WorkspaceClient``; the SDK is only needed where these are called.
"""

from __future__ import annotations

import io
import json
import subprocess
import time
from pathlib import Path

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


def create_machine_identity(w, secrets, name: str, secret_id: str, days: int = 90) -> str:
    """A service principal (machine identity) for Airflow, with its OAuth secret kept in AWS Secrets Manager.

    The secret goes straight from Databricks into ``secret_id``, in the form of an Airflow connection: it
    is never printed or written to disk. Running this again keeps the identity, stores a new secret and
    revokes the older ones, which is how the secret is rotated. Returns the application ID, the name
    Databricks permissions use for the identity.
    """
    from databricks.sdk.service.iam import ComplexValue

    found = list(w.service_principals.list(filter=f'displayName eq "{name}"'))
    if found:
        sp = found[0]
    else:  # workspace-access: without it the identity cannot call any workspace API
        sp = w.service_principals.create(
            display_name=name, active=True, entitlements=[ComplexValue(value="workspace-access")]
        )
    secret = w.service_principal_secrets_proxy.create(service_principal_id=sp.id, lifetime=f"{days * 86400}s")
    connection = {
        "conn_type": "databricks",
        "host": w.config.host,
        "login": sp.application_id,
        "password": secret.secret,
        "extra": {"service_principal_oauth": True},
    }
    try:
        secrets.put_secret_value(SecretId=secret_id, SecretString=json.dumps(connection))
    except Exception:
        # not stored, so nobody can use it: revoke it rather than leave a valid secret behind
        w.service_principal_secrets_proxy.delete(service_principal_id=sp.id, secret_id=secret.id)
        raise
    for old in w.service_principal_secrets_proxy.list(service_principal_id=sp.id):
        if old.id != secret.id:
            w.service_principal_secrets_proxy.delete(service_principal_id=sp.id, secret_id=old.id)
    return sp.application_id


def warehouse_id(w) -> str:
    """The workspace's SQL warehouse (Free Edition has exactly one)."""
    return next(iter(w.warehouses.list())).id


def _finished(api, response, timeout: int):
    """Wait until the statement has run; returns the final response, or raises with the warehouse's error."""
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
    return response


def sql(w, statement: str, warehouse: str | None = None, timeout: int = 600) -> pd.DataFrame:
    """Run ``statement`` on the SQL warehouse and return the result as a DataFrame.

    The warehouse starts by itself if it is stopped, which takes a few seconds. Results travel as text and
    are limited to 25 MB: for a large result use :func:`download`.
    """
    api = w.statement_execution
    response = _finished(
        api,
        api.execute_statement(
            statement=statement, warehouse_id=warehouse or warehouse_id(w), wait_timeout="30s"
        ),
        timeout,
    )
    columns = [c.name for c in response.manifest.schema.columns] if response.manifest else []
    rows, chunk = [], response.result
    while chunk is not None:
        rows.extend(chunk.data_array or [])
        if chunk.next_chunk_index is None:
            break
        chunk = api.get_statement_result_chunk_n(response.statement_id, chunk.next_chunk_index)
    return pd.DataFrame(rows, columns=columns)


def download(
    w, statement: str, path, warehouse: str | None = None, timeout: int = 1800, workers: int = 8
) -> int:
    """Run ``statement`` and save its result as a parquet file at ``path``; returns the number of rows.

    For results of any size, with their column types kept: the warehouse hands the result out as Arrow
    files behind temporary links. ``workers`` links are fetched at the same time (the transfer, not the
    query, is the slow part of a large result) and written to disk in their order, so memory stays low.
    A link that breaks off is fetched again, for up to half an hour (a patchy connection); the file gets
    its name only when complete, so a stopped download never leaves a file that looks finished.
    """
    import http.client
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq
    from databricks.sdk.service.sql import Disposition, Format

    api = w.statement_execution
    response = _finished(
        api,
        api.execute_statement(
            statement=statement,
            warehouse_id=warehouse or warehouse_id(w),
            disposition=Disposition.EXTERNAL_LINKS,
            format=Format.ARROW_STREAM,
            wait_timeout="30s",
        ),
        timeout,
    )
    chunks = response.manifest.total_chunk_count or 0

    def fetch(index: int, attempts: int = 35):
        """One chunk; a transfer cut off half-way is fetched again, with a fresh link."""
        for attempt in range(attempts):
            try:
                link = api.get_statement_result_chunk_n(response.statement_id, index).external_links[0]
                with urllib.request.urlopen(link.external_link, timeout=120) as stream:  # stalled: retried
                    return pa.ipc.open_stream(stream.read()).read_all()
            except (OSError, http.client.HTTPException):  # OSError includes timeouts and a lost network
                if attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt, 60))

    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    writer, rows = None, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, chunks, workers):  # a window of chunks in flight, written in order
            for table in pool.map(fetch, range(start, min(start + workers, chunks))):
                writer = writer or pq.ParquetWriter(partial, table.schema)
                writer.write_table(table)
                rows += table.num_rows
    if writer:
        writer.close()
        partial.replace(path)
    return rows


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


def dbt_env(w, profile: str = "qafi") -> dict[str, str]:
    """The variables dbt/profiles.yml reads: workspace, SQL warehouse and a token from the CLI login.

    The token comes from ``databricks auth token`` and expires within the hour, so no long-lived
    credential is ever written down; call this again when dbt reports an expired token.
    """
    out = subprocess.run(
        ["databricks", "auth", "token", "--profile", profile, "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "DATABRICKS_HOST": w.config.host.removeprefix("https://"),
        "DATABRICKS_HTTP_PATH": next(iter(w.warehouses.list())).odbc_params.path,
        "DATABRICKS_TOKEN": json.loads(out)["access_token"],
    }
