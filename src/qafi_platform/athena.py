"""Running one Athena query and reading its rows (chapter 10), for the API and the notebooks alike.

Athena is serverless SQL over the files in S3: a query is started, runs for a second or more, and its result
is read page by page. No pandas here, so the API's image stays small; the notebooks turn ``Result.rows``
into a DataFrame (``qafi_platform.lake.query``).

Values are passed as execution parameters (``?`` in the SQL), never pasted into the SQL text, so a value
cannot change what the query does. ``literal`` quotes a string for that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

CASTS = {
    "integer": int,
    "bigint": int,
    "smallint": int,
    "tinyint": int,
    "float": float,
    "real": float,
    "double": float,
    "boolean": lambda value: value == "true",
}


@dataclass
class Result:
    columns: list[str]
    rows: list[dict]
    scanned_bytes: int  # what Athena bills: 5 USD per TB, at least 10 MB per query
    milliseconds: int
    reused: bool  # answered from an earlier identical query, nothing scanned


def literal(value: str) -> str:
    """``value`` as an SQL string literal, for an execution parameter."""
    return "'" + value.replace("'", "''") + "'"


def run(
    client,
    sql: str,
    workgroup: str,
    database: str,
    parameters: list | None = None,
    reuse_minutes: int = 0,
    timeout: float = 60,
) -> Result:
    """Run ``sql`` and return its rows, typed; raises if the query fails or takes longer than ``timeout``.

    ``reuse_minutes``: an identical query that succeeded within that many minutes answers again without
    scanning (Athena's result reuse), which is faster and free.
    """
    request = {"QueryString": sql, "WorkGroup": workgroup, "QueryExecutionContext": {"Database": database}}
    if parameters:
        request["ExecutionParameters"] = [str(p) for p in parameters]
    if reuse_minutes:
        request["ResultReuseConfiguration"] = {
            "ResultReuseByAgeConfiguration": {"Enabled": True, "MaxAgeInMinutes": reuse_minutes}
        }
    query_id = client.start_query_execution(**request)["QueryExecutionId"]
    deadline, wait = time.monotonic() + timeout, 0.05
    while True:
        execution = client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            reason = execution["Status"].get("StateChangeReason", "")
            raise RuntimeError(f"Athena query {state.lower()}: {reason}")
        if time.monotonic() > deadline:
            client.stop_query_execution(QueryExecutionId=query_id)
            raise TimeoutError(f"Athena query still running after {timeout} s")
        time.sleep(wait)
        wait = min(wait * 2, 0.5)  # a small query is done in about a second: look often at first

    columns, rows, token = None, [], None
    while True:
        page = client.get_query_results(QueryExecutionId=query_id, **({"NextToken": token} if token else {}))
        if columns is None:
            info = page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
            columns = [c["Name"] for c in info]
            casts = [CASTS.get(c["Type"], str) for c in info]
            data = page["ResultSet"]["Rows"]
            if execution.get("StatementType") == "DML":  # a SELECT's first row repeats the column names
                data = data[1:]
        else:
            data = page["ResultSet"]["Rows"]
        for row in data:
            values = [cell.get("VarCharValue") for cell in row["Data"]]
            rows.append(
                {
                    name: None if value is None else cast(value)
                    for name, cast, value in zip(columns, casts, values, strict=True)
                }
            )
        token = page.get("NextToken")
        if not token:
            break
    statistics = execution.get("Statistics", {})
    return Result(
        columns=columns,
        rows=rows,
        scanned_bytes=statistics.get("DataScannedInBytes", 0),
        milliseconds=statistics.get("TotalExecutionTimeInMillis", 0),
        reused=statistics.get("ResultReuseInformation", {}).get("ReusedPreviousResult", False),
    )
