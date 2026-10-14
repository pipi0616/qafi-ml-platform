import json
from types import SimpleNamespace as NS
from urllib.parse import parse_qs, urlparse

import pytest
from botocore.exceptions import ClientError

from qafi_platform import lakehouse

from .conftest import BUCKET


def test_transfer_links_are_temporary_read_only_links_with_sizes(s3):
    key = "datasets/clinvar/2026-10/x.txt.gz"
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"12345")
    item = lakehouse.transfer_links(s3, BUCKET, [key], hours=2)[key]
    url = urlparse(item["url"])
    assert BUCKET in url.netloc and url.path.endswith(f"/{key}")
    assert parse_qs(url.query)["X-Amz-Expires"] == ["7200"]
    assert item["size"] == 5


def test_transfer_list_goes_into_the_volume():
    uploads = {}
    w = NS(files=NS(upload=lambda path, data, overwrite: uploads.update({path: json.load(data)})))
    links = {"datasets/a.gz": {"url": "https://signed", "size": 5}}
    path = lakehouse.upload_transfer_list(w, "qafi_dev", links)
    assert path == "/Volumes/qafi_dev/raw/files/_transfer/links.json"
    assert uploads[path] == links


def _response(state, rows=None, next_chunk=None, error=None):
    return NS(
        statement_id="s1",
        status=NS(state=NS(value=state), error=NS(message=error) if error else None),
        manifest=NS(schema=NS(columns=[NS(name="table"), NS(name="rows")])),
        result=NS(data_array=rows, next_chunk_index=next_chunk),
    )


def test_sql_waits_for_the_warehouse_and_reads_every_chunk(monkeypatch):
    monkeypatch.setattr(lakehouse.time, "sleep", lambda s: None)
    api = NS(
        execute_statement=lambda **kw: _response("PENDING"),
        get_statement=lambda sid: _response("SUCCEEDED", [["bronze", "568"]], next_chunk=1),
        get_statement_result_chunk_n=lambda sid, i: NS(data_array=[["silver", "214"]], next_chunk_index=None),
    )
    w = NS(statement_execution=api, warehouses=NS(list=lambda: iter([NS(id="wh1")])))
    df = lakehouse.sql(w, "SELECT ...")
    assert df.to_dict("records") == [{"table": "bronze", "rows": "568"}, {"table": "silver", "rows": "214"}]


def test_sql_reports_the_warehouse_error():
    api = NS(execute_statement=lambda **kw: _response("FAILED", error="TABLE_OR_VIEW_NOT_FOUND"))
    w = NS(statement_execution=api, warehouses=NS(list=lambda: iter([NS(id="wh1")])))
    with pytest.raises(RuntimeError, match="TABLE_OR_VIEW_NOT_FOUND"):
        lakehouse.sql(w, "SELECT * FROM missing")


def test_latest_run_lists_each_task_with_its_state(capsys):
    state = NS(life_cycle_state=NS(value="TERMINATED"), result_state=NS(value="SUCCESS"))
    task = NS(task_key="copy_raw", state=state, start_time=1_000_000, end_time=1_180_000)
    run = NS(run_id=7, run_page_url="https://workspace/jobs/1/runs/7", tasks=[task])
    jobs = NS(
        list=lambda name: iter([NS(job_id=1)]),
        list_runs=lambda job_id, limit: iter([NS(run_id=7)]),
        get_run=lambda run_id: run,
    )
    df = lakehouse.latest_run(NS(jobs=jobs), "qafi raw to silver")
    assert df.to_dict("records") == [
        {"task": "copy_raw", "state": "TERMINATED", "result": "SUCCESS", "minutes": 3.0}
    ]
    assert "runs/7" in capsys.readouterr().out


def test_dbt_env_uses_a_short_lived_token_from_the_cli(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return NS(stdout=json.dumps({"access_token": "one-hour-token", "token_type": "Bearer"}))

    monkeypatch.setattr(lakehouse.subprocess, "run", fake_run)
    w = NS(
        config=NS(host="https://dbc-123.cloud.databricks.com"),
        warehouses=NS(list=lambda: iter([NS(odbc_params=NS(path="/sql/1.0/warehouses/abc"))])),
    )
    assert lakehouse.dbt_env(w) == {
        "DATABRICKS_HOST": "dbc-123.cloud.databricks.com",
        "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc",
        "DATABRICKS_TOKEN": "one-hour-token",
    }
    assert calls == [["databricks", "auth", "token", "--profile", "qafi", "-o", "json"]]


SECRET = "airflow/connections/databricks_default"


class _FakeDatabricks:
    """Records what create_machine_identity asks of Databricks, which hands out the secret 'oauth-secret'."""

    def __init__(self, existing: bool = False):
        sp = NS(id="7", application_id="app-1")
        self.created, self.deleted, self.lifetimes, self.secrets = [], [], [], [NS(id="old")]

        def create_secret(service_principal_id, lifetime):
            self.lifetimes.append(lifetime)
            self.secrets.append(NS(id="new"))
            return NS(id="new", secret="oauth-secret")

        def delete_secret(service_principal_id, secret_id):
            self.deleted.append(secret_id)
            self.secrets = [s for s in self.secrets if s.id != secret_id]

        self.config = NS(host="https://workspace")
        self.service_principals = NS(
            list=lambda filter: [sp] if existing else [],
            create=lambda **kw: self.created.append(kw) or sp,
        )
        self.service_principal_secrets_proxy = NS(
            create=create_secret, list=lambda service_principal_id: list(self.secrets), delete=delete_secret
        )


def test_the_secret_goes_straight_into_secrets_manager_as_an_airflow_connection(secretsmanager, capsys):
    secretsmanager.create_secret(Name=SECRET)
    w = _FakeDatabricks()
    assert lakehouse.create_machine_identity(w, secretsmanager, "qafi-airflow", SECRET) == "app-1"
    assert json.loads(secretsmanager.get_secret_value(SecretId=SECRET)["SecretString"]) == {
        "conn_type": "databricks",
        "host": "https://workspace",
        "login": "app-1",
        "password": "oauth-secret",
        "extra": {"service_principal_oauth": True},
    }
    assert w.created[0]["display_name"] == "qafi-airflow"
    assert w.lifetimes == ["7776000s"]  # valid for 90 days
    assert w.deleted == ["old"]  # rotation: the older secret is revoked
    assert "oauth-secret" not in capsys.readouterr().out


def test_running_again_keeps_the_identity(secretsmanager):
    secretsmanager.create_secret(Name=SECRET)
    w = _FakeDatabricks(existing=True)
    lakehouse.create_machine_identity(w, secretsmanager, "qafi-airflow", SECRET)
    assert w.created == []


def test_a_secret_that_cannot_be_stored_is_revoked(secretsmanager):
    w = _FakeDatabricks()  # the secret does not exist in Secrets Manager yet, so storing fails
    with pytest.raises(ClientError):
        lakehouse.create_machine_identity(w, secretsmanager, "qafi-airflow", SECRET)
    assert w.deleted == ["new"]


def test_download_writes_every_arrow_chunk_into_one_parquet_file(tmp_path, monkeypatch):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    def arrow(rows):  # what the warehouse serves behind a link: one Arrow stream
        table = pa.table({"uniprot_id": [r[0] for r in rows], "embedding": [r[1] for r in rows]})
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as stream:
            stream.write_table(table)
        return sink.getvalue()

    files = {"https://link/0": arrow([("P1", [0.5, 1.5])]), "https://link/1": arrow([("P2", [2.5, 3.5])])}
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: io.BytesIO(files[url]))
    api = NS(
        execute_statement=lambda **kw: NS(
            statement_id="s1",
            status=NS(state=NS(value="SUCCEEDED"), error=None),
            manifest=NS(total_chunk_count=2),
        ),
        get_statement_result_chunk_n=lambda sid, i: NS(
            external_links=[NS(external_link=f"https://link/{i}", next_chunk_index=1 if i == 0 else None)]
        ),
    )
    w = NS(statement_execution=api, warehouses=NS(list=lambda: iter([NS(id="wh1")])))
    path = tmp_path / "result.parquet"
    assert lakehouse.download(w, "SELECT ...", path) == 2
    table = pq.read_table(path)
    assert table["uniprot_id"].to_pylist() == ["P1", "P2"]
    assert table["embedding"].to_pylist() == [[0.5, 1.5], [2.5, 3.5]]  # arrays keep their type


def test_download_fetches_a_broken_transfer_again(tmp_path, monkeypatch):
    import http.client
    import io

    import pyarrow as pa

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, pa.schema([("uniprot_id", pa.string())])) as stream:
        stream.write_table(pa.table({"uniprot_id": ["P1"]}))
    calls = []

    def urlopen(url, timeout=None):  # the first transfer breaks off, the second gets through
        calls.append(url)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"", 10)
        return io.BytesIO(sink.getvalue())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(lakehouse.time, "sleep", lambda s: None)
    api = NS(
        execute_statement=lambda **kw: NS(
            statement_id="s1",
            status=NS(state=NS(value="SUCCEEDED"), error=None),
            manifest=NS(total_chunk_count=1),
        ),
        get_statement_result_chunk_n=lambda sid, i: NS(external_links=[NS(external_link="https://link/0")]),
    )
    w = NS(statement_execution=api, warehouses=NS(list=lambda: iter([NS(id="wh1")])))
    assert lakehouse.download(w, "SELECT ...", tmp_path / "result.parquet") == 1
    assert len(calls) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["result.parquet"]  # no half-written file left


def test_download_of_an_empty_result_writes_nothing(tmp_path):
    api = NS(
        execute_statement=lambda **kw: NS(
            statement_id="s1",
            status=NS(state=NS(value="SUCCEEDED"), error=None),
            manifest=NS(total_chunk_count=0),
        )
    )
    w = NS(statement_execution=api, warehouses=NS(list=lambda: iter([NS(id="wh1")])))
    assert lakehouse.download(w, "SELECT ...", tmp_path / "none.parquet") == 0
    assert not (tmp_path / "none.parquet").exists()
