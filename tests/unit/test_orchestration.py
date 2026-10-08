import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from qafi_platform import orchestration


@pytest.fixture
def root(tmp_path):
    """A project folder whose local Airflow has generated the password 's3cret'."""
    home = tmp_path / "airflow" / ".airflow"
    home.mkdir(parents=True)
    (home / "simple_auth_manager_passwords.json.generated").write_text(json.dumps({"admin": "s3cret"}))
    return tmp_path


def test_the_generated_password_is_read_from_airflow_home(root):
    assert orchestration.airflow_password(root) == "s3cret"


def test_the_api_logs_in_first_then_sends_the_token(root, monkeypatch):
    seen = []

    class Airflow(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers.get("Authorization")))
            login = self.path == "/auth/token"
            self._reply({"access_token": "jwt"} if login else {"dag_run_id": "manual__1"})

        def do_GET(self):
            seen.append((self.path, None, self.headers.get("Authorization")))
            self._reply({"dag_runs": []})

        def _reply(self, data):
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Airflow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(orchestration, "UI", f"http://127.0.0.1:{server.server_port}")
    try:
        assert orchestration._runs(root, "qafi_refresh") == []
        run = orchestration._api(root, "POST", "/dags/qafi_refresh/dagRuns", {"logical_date": None})
    finally:
        server.shutdown()
    assert run == {"dag_run_id": "manual__1"}
    assert seen[0] == ("/auth/token", {"username": "admin", "password": "s3cret"}, None)
    assert seen[1] == ("/api/v2/dags/qafi_refresh/dagRuns?order_by=-run_after&limit=100", None, "Bearer jwt")
    assert seen[3] == ("/api/v2/dags/qafi_refresh/dagRuns", {"logical_date": None}, "Bearer jwt")


class _FakeAirflow:
    """Answers the REST calls start_dag_run makes; ``scheduled`` makes switching on start a missed run."""

    def __init__(self, runs, scheduled=False):
        self.runs, self.scheduled, self.calls = list(runs), scheduled, []

    def __call__(self, root, method, path, body=None):
        self.calls.append(method)
        if method == "PATCH" and self.scheduled:
            self.runs.insert(0, {"dag_run_id": "scheduled__2026-10-10T06:00:00+00:00", "state": "queued"})
        if method == "POST":
            return {"dag_run_id": "manual__2026-10-08T16:10:15+00:00"}
        return {"dag_runs": list(self.runs)} if method == "GET" else {}


def test_a_run_already_going_is_used_instead_of_starting_another(root, monkeypatch):
    fake = _FakeAirflow([{"dag_run_id": "manual__ui", "state": "running"}])
    monkeypatch.setattr(orchestration, "_api", fake)
    assert orchestration.start_dag_run(root, "qafi_refresh") == "manual__ui"
    assert fake.calls == ["GET"]


def test_switching_on_starts_the_missed_scheduled_run_without_a_second_one(root, monkeypatch):
    fake = _FakeAirflow([{"dag_run_id": "old", "state": "success"}], scheduled=True)
    monkeypatch.setattr(orchestration, "_api", fake)
    assert orchestration.start_dag_run(root, "qafi_refresh") == "scheduled__2026-10-10T06:00:00+00:00"
    assert "POST" not in fake.calls


def test_without_a_missed_run_a_run_is_triggered_by_hand(root, monkeypatch):
    fake = _FakeAirflow([{"dag_run_id": "old", "state": "success"}])
    monkeypatch.setattr(orchestration, "_api", fake)
    monkeypatch.setattr(orchestration.time, "sleep", lambda s: None)
    assert orchestration.start_dag_run(root, "qafi_refresh", wait=0) == "manual__2026-10-08T16:10:15+00:00"
    assert fake.calls == ["GET", "PATCH", "POST"]


def test_waits_for_the_given_run_and_lists_its_tasks_in_order(root, monkeypatch):
    states = iter(["running", "running", "success"])

    def fake_api(root, method, path, body=None):
        if path.endswith("/taskInstances"):
            ran = [("run_lakehouse_job", "12:01", "12:06"), ("fetch_new_releases", "12:00", "12:01")]
            return {
                "task_instances": [
                    {"task_id": task, "state": "success", "start_date": start, "end_date": end}
                    for task, start, end in ran
                ]
            }
        assert path == "/dags/qafi_refresh/dagRuns/manual__2026-10-08T16%3A10%3A15%2B00%3A00"
        return {"dag_run_id": "new", "state": next(states)}

    monkeypatch.setattr(orchestration, "_api", fake_api)
    monkeypatch.setattr(orchestration.time, "sleep", lambda s: None)
    df = orchestration.wait_for_dag_run(Path("."), "qafi_refresh", "manual__2026-10-08T16:10:15+00:00")
    assert list(df["task_id"]) == ["fetch_new_releases", "run_lakehouse_job"]
