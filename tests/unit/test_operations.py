from types import SimpleNamespace as NS

import pytest

from qafi_platform import operations


def test_a_logs_insights_query_comes_back_as_a_table(monkeypatch):
    monkeypatch.setattr(operations.time, "sleep", lambda s: None)
    states = iter(["Running", "Complete"])
    sent = {}

    def start_query(**request):
        sent.update(request)
        return {"queryId": "q1"}

    def get_query_results(queryId):
        row = [
            {"field": "status", "value": "200"},
            {"field": "requests", "value": "12"},
            {"field": "@ptr", "value": "x"},
        ]
        return {"status": next(states), "results": [row]}

    client = NS(start_query=start_query, get_query_results=get_query_results)
    table = operations.logs(client, "/aws/lambda/qafi-dev-api", "stats count(*) by status", hours=2)
    assert table.to_dict("records") == [{"status": "200", "requests": "12"}]
    assert sent["endTime"] - sent["startTime"] == 7200 and sent["logGroupName"] == "/aws/lambda/qafi-dev-api"


def test_a_failed_logs_query_raises():
    client = NS(
        start_query=lambda **r: {"queryId": "q1"}, get_query_results=lambda queryId: {"status": "Failed"}
    )
    with pytest.raises(RuntimeError, match="failed"):
        operations.logs(client, "g", "bad query")


def test_alarms_are_listed_with_their_state():
    alarm = {"AlarmName": "qafi-dev-api-5xx", "StateValue": "OK", "AlarmDescription": "server errors"}
    client = NS(describe_alarms=lambda AlarmNamePrefix: {"MetricAlarms": [alarm]})
    assert operations.alarms(client, "qafi-dev").to_dict("records") == [
        {"alarm": "qafi-dev-api-5xx", "state": "OK", "what": "server errors"}
    ]


def test_costs_are_summed_per_service_and_tiny_ones_dropped():
    def group(service, amount):
        return {"Keys": [service], "Metrics": {"UnblendedCost": {"Amount": amount, "Unit": "USD"}}}

    response = {"ResultsByTime": [
        {"Groups": [group("Amazon S3", "1.20"), group("AWS Lambda", "0.001")]},
        {"Groups": [group("Amazon S3", "0.30"), group("Amazon Athena", "0.05")]},
    ]}  # fmt: skip
    sent = {}
    client = NS(get_cost_and_usage=lambda **request: sent.update(request) or response)
    table = operations.costs(client, "2026-10-01", "2026-10-16")
    assert sent["Filter"] == {"Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Credit", "Refund"]}}}
    assert table.to_dict("records") == [
        {"service": "Amazon S3", "usd": 1.5},
        {"service": "Amazon Athena", "usd": 0.05},
    ]
