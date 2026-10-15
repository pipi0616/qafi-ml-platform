"""Claude on Amazon Bedrock (notebooks 11 and 12).

Claude is called through the EU inference profile (model IDs that start with ``eu.``): Bedrock sends each
request to whichever EU Region has capacity, so the data stays in the EU. Tested on this account on
2026-10-15: the single-Region path (the ``bedrock-mantle`` endpoint) is not available to it, the EU profile
is. Requests and replies use Anthropic's Messages format through the standard ``bedrock-runtime`` InvokeModel
API; boto3 signs them with your AWS credentials and retries when Bedrock is busy.
"""

from __future__ import annotations

import json

REGION = "eu-west-1"  # where the knowledge base and the guardrail live (decision 9 in docs/plan.md)
HAIKU = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
PRICES = {HAIKU: (1.1, 5.5)}  # USD per million input and output tokens: EU profile, Standard tier
_clients = {}


def client(session):
    """A bedrock-runtime client for ``session`` (one per session), retrying with backoff when throttled."""
    if id(session) not in _clients:
        from botocore.config import Config

        config = Config(retries={"mode": "adaptive", "max_attempts": 8}, read_timeout=120)
        _clients[id(session)] = session.client("bedrock-runtime", region_name=REGION, config=config)
    return _clients[id(session)]


def messages(session, body: dict) -> dict:
    """Send a Messages API ``body`` (its ``model`` chooses the inference profile); returns Claude's reply."""
    request = {"anthropic_version": "bedrock-2023-05-31", **{k: v for k, v in body.items() if k != "model"}}
    response = client(session).invoke_model(modelId=body["model"], body=json.dumps(request))
    return json.loads(response["body"].read())


def cost(model: str, usage: dict) -> float | None:
    """What a reply cost in USD, from its token counts (``None`` for a model without a price here)."""
    if model not in PRICES:
        return None
    per_input, per_output = PRICES[model]
    return round((usage["input_tokens"] * per_input + usage["output_tokens"] * per_output) / 1e6, 6)
