import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
import pytest
from moto import mock_aws

BUCKET = "test-raw"


@pytest.fixture
def s3(monkeypatch):
    """An S3 client talking to moto's in-memory fake, with a versioned bucket like the real raw bucket."""
    for var in ("AWS_PROFILE", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        client = boto3.client("s3", region_name="eu-central-1")
        client.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        client.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
        yield client


@pytest.fixture
def publisher():
    """A local web server standing in for a data publisher: ``publisher.serve(name, data)`` returns a URL.

    Like NCBI or EBI it answers HEAD and supports Range requests, unless ``ranges=False``.
    """
    files: dict[str, tuple[bytes, bool]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self._reply(send_body=False)

        def do_GET(self):
            self._reply(send_body=True)

        def _reply(self, send_body: bool):
            if self.path not in files:
                self.send_error(404)
                return
            data, ranges = files[self.path]
            body, status = data, 200
            if ranges and self.headers.get("Range"):
                start, end = (int(x) for x in self.headers["Range"].removeprefix("bytes=").split("-"))
                body, status = data[start : end + 1], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", "Mon, 05 Oct 2026 13:02:37 GMT")
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    class Publisher:
        def serve(self, name: str, data: bytes, ranges: bool = True) -> str:
            files[f"/{name}"] = (data, ranges)
            return f"http://127.0.0.1:{server.server_port}/{name}"

    yield Publisher()
    server.shutdown()
