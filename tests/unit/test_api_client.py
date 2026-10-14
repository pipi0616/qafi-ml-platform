import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
import pytest

from qafi_platform import api_client


@pytest.fixture
def gateway():
    """A local server standing in for API Gateway: refuses unsigned requests, like the AWS_IAM route."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(dict(self.headers))
            signed = self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256")
            status, body = (
                (200, {"variant": "R175H"}) if signed else (403, {"message": "Missing Authentication Token"})
            )
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()


def test_a_signed_request_gets_through_and_an_unsigned_one_does_not(gateway):
    url, seen = gateway
    session = boto3.Session(
        aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="secret", region_name="eu-central-1"
    )
    status, body, seconds = api_client.get(session, f"{url}/v1/proteins/TP53/variants/R175H")
    assert status == 200 and body == {"variant": "R175H"} and seconds >= 0
    authorization = seen[0]["Authorization"]
    assert (
        "Credential=AKIDEXAMPLE/" in authorization
        and "/eu-central-1/execute-api/aws4_request" in authorization
    )
    assert "secret" not in json.dumps(seen)  # the secret key itself never travels
    status, body, _ = api_client.get(session, f"{url}/v1/proteins/TP53/variants/R175H", sign=False)
    assert status == 403 and body["message"] == "Missing Authentication Token"
