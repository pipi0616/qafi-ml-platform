"""The Databricks connection notebook 06 puts into Secrets Manager is exactly what Airflow reads back."""

from types import SimpleNamespace as NS

import boto3
from moto import mock_aws

SECRET = "airflow/connections/databricks_default"


def test_airflow_reads_the_stored_connection(monkeypatch):
    from airflow.providers.amazon.aws.secrets.secrets_manager import SecretsManagerBackend

    from qafi_platform import lakehouse

    for var in ("AWS_PROFILE", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    sp = NS(id="7", application_id="app-1")
    w = NS(
        config=NS(host="https://workspace"),
        service_principals=NS(list=lambda filter: [sp]),
        service_principal_secrets_proxy=NS(
            create=lambda service_principal_id, lifetime: NS(id="s1", secret="oauth-secret"),
            list=lambda service_principal_id: [],
            delete=None,
        ),
    )
    with mock_aws():
        secrets = boto3.client("secretsmanager", region_name="eu-central-1")
        secrets.create_secret(Name=SECRET)
        lakehouse.create_machine_identity(w, secrets, "qafi-airflow", SECRET)
        backend = SecretsManagerBackend(connections_prefix="airflow/connections", region_name="eu-central-1")
        conn = backend.get_connection("databricks_default")
    assert (conn.conn_type, conn.host, conn.login, conn.password) == (
        "databricks",
        "https://workspace",
        "app-1",
        "oauth-secret",
    )
    assert conn.extra_dejson == {"service_principal_oauth": True}  # the provider logs in with OAuth M2M
