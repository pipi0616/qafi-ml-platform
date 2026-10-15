"""Running the platform (notebook 13): what the logs say, what the alarms say, what it costs.

logs(client, group, query, hours)   a CloudWatch Logs Insights query over a log group, as a table
alarms(client, prefix)              every alarm whose name starts with ``prefix`` and its state
costs(client, start, end)           what each AWS service cost between two dates (Cost Explorer)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pandas as pd


def logs(client, group: str, query: str, hours: float = 24, timeout: float = 60) -> pd.DataFrame:
    """Run a Logs Insights ``query`` over the last ``hours`` of ``group`` and return its rows."""
    end = datetime.now(UTC)
    query_id = client.start_query(
        logGroupName=group,
        startTime=int((end - timedelta(hours=hours)).timestamp()),
        endTime=int(end.timestamp()),
        queryString=query,
    )["queryId"]
    deadline = time.monotonic() + timeout
    while True:
        result = client.get_query_results(queryId=query_id)
        if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        if time.monotonic() > deadline:
            client.stop_query(queryId=query_id)
            raise TimeoutError(f"Logs Insights query still running after {timeout} s")
        time.sleep(1)
    if result["status"] != "Complete":
        raise RuntimeError(f"Logs Insights query {result['status'].lower()}")
    rows = [
        {f["field"]: f["value"] for f in row if not f["field"].startswith("@ptr")}
        for row in result["results"]
    ]
    return pd.DataFrame(rows)


def alarms(client, prefix: str) -> pd.DataFrame:
    """Name, state and description of every alarm whose name starts with ``prefix``."""
    found = client.describe_alarms(AlarmNamePrefix=prefix)["MetricAlarms"]
    return pd.DataFrame(
        [
            {"alarm": a["AlarmName"], "state": a["StateValue"], "what": a.get("AlarmDescription", "")}
            for a in found
        ]
    )


def costs(client, start: str, end: str) -> pd.DataFrame:
    """What each AWS service cost from ``start`` up to (not including) ``end`` (YYYY-MM-DD), before credits:
    the free plan's credits are booked as negative costs and would hide what the platform really uses.

    Each Cost Explorer request itself costs 0.01 USD.
    """
    response = client.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        Filter={"Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Credit", "Refund"]}}},
    )
    rows = [
        {"service": group["Keys"][0], "usd": float(group["Metrics"]["UnblendedCost"]["Amount"])}
        for period in response["ResultsByTime"]
        for group in period["Groups"]
    ]
    frame = pd.DataFrame(rows, columns=["service", "usd"]).groupby("service", as_index=False)["usd"].sum()
    return frame[frame["usd"] >= 0.005].sort_values("usd", ascending=False).round(2).reset_index(drop=True)
