import pytest

from qafi_platform import athena

from .athena_fake import FakeAthena

COLUMNS = [("uniprot_id", "varchar"), ("position", "integer"), ("score", "float"), ("has_dms", "boolean")]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(athena.time, "sleep", lambda s: None)


def test_rows_come_back_typed_from_every_page():
    rows = [("P04637", i, 0.25 * i, i % 2 == 0) for i in range(5)]
    client = FakeAthena(lambda sql, p: (COLUMNS, rows), page_size=2, polls=3)
    result = athena.run(client, "SELECT * FROM t", "wg", "db")
    assert result.columns == ["uniprot_id", "position", "score", "has_dms"]
    assert [r["position"] for r in result.rows] == [0, 1, 2, 3, 4]  # the header row is not a row
    assert result.rows[2] == {"uniprot_id": "P04637", "position": 2, "score": 0.5, "has_dms": True}
    assert result.scanned_bytes == 12_345_678 and not result.reused


def test_values_travel_as_parameters_not_inside_the_sql():
    client = FakeAthena(lambda sql, p: (COLUMNS, []))
    athena.run(client, "SELECT * FROM t WHERE uniprot_id = ?", "wg", "db", [athena.literal("x' OR 'a'='a")])
    request = client.requests[0]
    assert request["QueryString"] == "SELECT * FROM t WHERE uniprot_id = ?"
    assert request["ExecutionParameters"] == ["'x'' OR ''a''=''a'"]  # one string, quotes doubled
    assert request["WorkGroup"] == "wg" and request["QueryExecutionContext"] == {"Database": "db"}
    assert "ResultReuseConfiguration" not in request


def test_result_reuse_is_asked_for_when_wanted():
    client = FakeAthena(lambda sql, p: (COLUMNS, []))
    athena.run(client, "SELECT 1", "wg", "db", reuse_minutes=60)
    reuse = client.requests[0]["ResultReuseConfiguration"]["ResultReuseByAgeConfiguration"]
    assert reuse == {"Enabled": True, "MaxAgeInMinutes": 60}


def test_null_stays_none_and_ddl_has_no_header_row():
    client = FakeAthena(lambda sql, p: ([("x", "varchar")], [(None,)]))
    assert athena.run(client, "SELECT x FROM t", "wg", "db").rows == [{"x": None}]
    client = FakeAthena(lambda sql, p: ([], []))
    assert athena.run(client, "CREATE OR REPLACE VIEW v AS SELECT 1", "wg", "db").rows == []


def test_a_failed_query_raises_with_athenas_reason():
    reason = "TABLE_NOT_FOUND: line 1:15: Table t does not exist"
    client = FakeAthena(lambda sql, p: (COLUMNS, []), fail=reason)
    with pytest.raises(RuntimeError, match="TABLE_NOT_FOUND"):
        athena.run(client, "SELECT * FROM t", "wg", "db")


def test_a_query_that_takes_too_long_is_stopped():
    client = FakeAthena(lambda sql, p: (COLUMNS, []), polls=10**6)
    with pytest.raises(TimeoutError):
        athena.run(client, "SELECT * FROM t", "wg", "db", timeout=-1)
    assert client.stopped == ["q1"]
