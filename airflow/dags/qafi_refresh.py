"""Monthly refresh of the QAFI training data (notebook 06).

    fetch_new_releases ──▶ send_transfer_list ──▶ run_lakehouse_job

fetch_new_releases  For datasets with refresh = "monthly" in config/sources.toml, the newest release is copied
                    from the publisher into the S3 raw bucket by the fetcher Lambda. Files already in raw are
                    skipped, so a month without a new release copies nothing.
send_transfer_list  Read-only download links for the files the job loads go into the Databricks volume (Free
                    Edition cannot read the bucket itself).
run_lakehouse_job   The Databricks job "qafi training data" rebuilds bronze, silver and gold (notebooks 04-05)
                    with exactly those versions; Airflow waits for its result.

AWS calls use the session Airflow runs in. Databricks calls use the connection databricks_default, which
Airflow reads from AWS Secrets Manager (see airflow.sh).
"""

from __future__ import annotations

from pathlib import Path

import pendulum
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.sdk import dag, task

SOURCES = Path(__file__).resolve().parents[2] / "config" / "sources.toml"
RAW_BUCKET = "qafi-186972156153-dev-raw"
CATALOG = "qafi_dev"
JOB = "qafi training data"
# dataset in sources.toml -> the job parameter telling the job which version to load
JOB_PARAMETERS = {
    "clinvar": "clinvar_version",
    "uniprot-human": "uniprot_version",
    "proteingym": "proteingym_version",
    "mavedb": "mavedb_version",
}


@dag(
    schedule="0 6 10 * *",  # 06:00 UTC on the 10th of every month: ClinVar's monthly release is out by then
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,  # months missed while Airflow was off are not made up: one run loads the newest data
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    tags=["qafi"],
)
def qafi_refresh():
    @task
    def fetch_new_releases(dag_run=None) -> dict[str, str]:
        """Copy new releases into raw; return the version of each dataset the job is to load."""
        import boto3

        from qafi_platform import ingest

        sources = ingest.load_sources(SOURCES, ingest.newest_releases(SOURCES, dag_run.run_after.date()))
        ingest.fetch_sources(boto3.client("lambda"), boto3.client("s3"), RAW_BUCKET, sources)
        versions = {s["dataset"]: s["version"] for s in sources}
        return {parameter: versions[dataset] for dataset, parameter in JOB_PARAMETERS.items()}

    @task
    def send_transfer_list(job_parameters: dict[str, str]) -> None:
        """Put download links for the job's files into the volume, where the job's copy task reads them."""
        import boto3
        from airflow.sdk import BaseHook
        from databricks.sdk import WorkspaceClient

        from qafi_platform import ingest, lakehouse

        versions = {dataset: job_parameters[parameter] for dataset, parameter in JOB_PARAMETERS.items()}
        keys = [s["key"] for s in ingest.load_sources(SOURCES, versions) if s["dataset"] in JOB_PARAMETERS]
        conn = BaseHook.get_connection("databricks_default")
        if conn.extra_dejson.get("service_principal_oauth"):  # a machine identity: OAuth client ID and secret
            w = WorkspaceClient(host=conn.host, client_id=conn.login, client_secret=conn.password)
        else:  # a token
            w = WorkspaceClient(host=conn.host, token=conn.password)
        links = lakehouse.transfer_links(boto3.client("s3"), RAW_BUCKET, keys, hours=6)
        print(lakehouse.upload_transfer_list(w, CATALOG, links))

    job_parameters = fetch_new_releases()
    run_job = DatabricksRunNowOperator(
        task_id="run_lakehouse_job", job_name=JOB, job_parameters=job_parameters
    )
    send_transfer_list(job_parameters) >> run_job


qafi_refresh()
