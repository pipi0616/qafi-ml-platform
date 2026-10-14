"""Athena's API in memory, shaped like the real service, for the tests of chapter 10."""


class FakeAthena:
    """``answer(sql, parameters)`` gives ``(columns, rows)`` for a query, a column being (name, Athena type).

    Like the real service: a query is RUNNING for ``polls`` checks before it succeeds, results come in pages
    of ``page_size`` rows, a SELECT's first row repeats the column names, values are text and NULL is an
    empty cell. Every request is kept in ``requests``.
    """

    def __init__(self, answer, page_size: int = 1000, polls: int = 1, fail: str | None = None):
        self.answer, self.page_size, self.polls, self.fail = answer, page_size, polls, fail
        self.requests, self.results, self.stopped = [], {}, []

    def start_query_execution(self, **request):
        self.requests.append(request)
        sql = request["QueryString"]
        sent = request.get("ExecutionParameters", [])
        parameters = [p[1:-1].replace("''", "'") if p.startswith("'") else p for p in sent]
        columns, rows = self.answer(sql, parameters)
        statement = "DDL" if sql.split()[0].upper() in ("ALTER", "CREATE", "DROP") else "DML"
        query_id = f"q{len(self.requests)}"
        self.results[query_id] = dict(columns=columns, rows=rows, statement=statement, polls=self.polls)
        return {"QueryExecutionId": query_id}

    def get_query_execution(self, QueryExecutionId):
        result = self.results[QueryExecutionId]
        if self.fail:
            state = "FAILED"
        elif result["polls"] > 0:
            result["polls"] -= 1
            state = "RUNNING"
        else:
            state = "SUCCEEDED"
        return {
            "QueryExecution": {
                "Status": {"State": state, "StateChangeReason": self.fail or ""},
                "StatementType": result["statement"],
                "Statistics": {
                    "DataScannedInBytes": 12_345_678,
                    "TotalExecutionTimeInMillis": 800,
                    "ResultReuseInformation": {"ReusedPreviousResult": False},
                },
            }
        }

    def get_query_results(self, QueryExecutionId, NextToken=None):
        result = self.results[QueryExecutionId]
        text = [
            [None if v is None else str(v).lower() if isinstance(v, bool) else str(v) for v in row]
            for row in result["rows"]
        ]
        if result["statement"] == "DML":
            text = [[name for name, _ in result["columns"]], *text]
        start = int(NextToken or 0)
        page = {
            "ResultSet": {
                "ResultSetMetadata": {"ColumnInfo": [{"Name": n, "Type": t} for n, t in result["columns"]]},
                "Rows": [
                    {"Data": [{} if v is None else {"VarCharValue": v} for v in row]}
                    for row in text[start : start + self.page_size]
                ],
            }
        }
        if start + self.page_size < len(text):
            page["NextToken"] = str(start + self.page_size)
        return page

    def stop_query_execution(self, QueryExecutionId):
        self.stopped.append(QueryExecutionId)
