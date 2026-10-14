"""Claude on Amazon Bedrock, inside one Region (notebooks 11 and 12).

The free AWS plan cannot use cross-Region inference, and in Ireland (eu-west-1) Claude runs inside the Region
only through the ``bedrock-mantle`` endpoint, which speaks Anthropic's own Messages API. Requests are signed
with your AWS credentials (Signature Version 4) like every other AWS call: no API key to store anywhere.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

REGION = "eu-west-1"  # decision 9 in docs/plan.md
HAIKU = "anthropic.claude-haiku-4-5"
SONNET = "anthropic.claude-sonnet-5"
PRICES = {HAIKU: (1.0, 5.0)}  # USD per million input and output tokens (Bedrock on-demand, Standard tier)


def messages(session, body: dict, region: str = REGION, timeout: float = 120, attempts: int = 5) -> dict:
    """POST ``body`` to the Messages API and return Claude's reply; busy or failing servers are retried."""
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    url = f"https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages"
    data = json.dumps(body)
    for attempt in range(attempts):
        request = AWSRequest(
            method="POST",
            url=url,
            data=data,
            headers={"content-type": "application/json", "anthropic-version": "2023-06-01"},
        )
        SigV4Auth(session.get_credentials().get_frozen_credentials(), "bedrock-mantle", region).add_auth(
            request
        )
        try:
            outgoing = urllib.request.Request(
                url, data=data.encode(), headers=dict(request.headers), method="POST"
            )
            with urllib.request.urlopen(outgoing, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code not in (429, 500, 502, 503, 529) or attempt == attempts - 1:
                raise RuntimeError(f"Claude answered {error.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)  # signed again on the next attempt: a signature is valid for minutes only
    raise AssertionError("unreachable")


def cost(model: str, usage: dict) -> float | None:
    """What a reply cost in USD, from its token counts (``None`` for a model without a price here)."""
    if model not in PRICES:
        return None
    per_input, per_output = PRICES[model]
    return round((usage["input_tokens"] * per_input + usage["output_tokens"] * per_output) / 1e6, 6)
