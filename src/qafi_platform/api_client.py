"""Calling the deployed API (notebook 10b). The API lets only signed requests through: each request carries a
signature made with your AWS credentials (Signature Version 4), which API Gateway checks before anything runs.
The credentials themselves never travel; the signature is valid for this one request, for a few minutes.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def get(session, url: str, sign: bool = True, timeout: float = 30) -> tuple[int, dict, float]:
    """GET ``url``: HTTP status, JSON answer and the seconds it took. ``session``: a boto3 Session."""
    headers = {}
    if sign:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        request = AWSRequest(method="GET", url=url)
        credentials = session.get_credentials().get_frozen_credentials()
        SigV4Auth(credentials, "execute-api", session.region_name).add_auth(request)
        headers = dict(request.headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:  # 4xx and 5xx still carry an answer worth reading
        status, body = error.code, error.read()
    return status, json.loads(body or b"{}"), round(time.monotonic() - started, 2)
