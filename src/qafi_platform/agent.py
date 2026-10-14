"""The variant interpretation agent (notebook 12): Claude decides which tools to use, this code runs them.

  TOOLS                         what Claude may ask for, each described with a JSON Schema
  Toolbox                       runs the tools: the prediction API (10b) and the knowledge base (11)
  run(session, question, ...)   the loop: ask Claude, run the tools it asks for, give it the results, repeat
                                until it answers (or the turn limit is reached)
  check(client, guardrail, ...) Bedrock Guardrails on the question before the agent sees it, and on the
                                answer before the user does, including whether it rests on the tool results

An agent is a language model, a set of tools and a loop. Frameworks (Strands, LangGraph, Bedrock Agents) wrap
the same loop; written out here, every step can be seen and tested.
"""

from __future__ import annotations

import json
import time
from urllib.parse import quote

from qafi_platform import api_client, claude, knowledge

TOOLS = [
    {
        "name": "get_variant_scores",
        "description": (
            "Released QAFI scores of one missense variant of a human protein: stability_score (low = "
            "likely destabilising) and function_score (low = likely loss of function), both percentiles "
            "between 0 and 1, plus the ESM-2 zero-shot score. Predictions from a model, not clinical "
            "classifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protein": {
                    "type": "string",
                    "description": "Gene symbol (TP53) or UniProt accession (P04637)",
                },
                "variant": {"type": "string", "description": "One-letter substitution, e.g. R175H"},
            },
            "required": ["protein", "variant"],
        },
    },
    {
        "name": "get_position_scores",
        "description": "Released QAFI scores of all 19 substitutions at one position of a human protein.",
        "input_schema": {
            "type": "object",
            "properties": {
                "protein": {"type": "string", "description": "Gene symbol or UniProt accession"},
                "position": {"type": "integer", "description": "Residue number in the UniProt sequence"},
            },
            "required": ["protein", "position"],
        },
    },
    {
        "name": "search_knowledge",
        "description": (
            "Search the knowledge base: UniProt annotations of reviewed human proteins (function, location, "
            "disease, domains, known natural variants, mutagenesis experiments) and the QAFI paper. Returns "
            "passages with their source document."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words"},
                "gene": {"type": "string", "description": "Optional: search only this gene's UniProt entry"},
            },
            "required": ["query"],
        },
    },
]

SYSTEM = (
    "You are a research assistant for interpreting human missense variants. Use the tools: the QAFI scores "
    "for predictions, the knowledge base for what is known. Say where each fact comes from: (QAFI score) or "
    "the knowledge base source document. QAFI scores are model predictions for research, not a clinical "
    "classification. If the tools do not give an answer, say so. Answer in at most 200 words."
)


class Toolbox:
    """Runs the tools Claude asks for. ``api_session`` signs calls to the API (Frankfurt), ``runtime`` is a
    bedrock-agent-runtime client for the knowledge base (Ireland)."""

    def __init__(self, api_session, api_url: str, runtime, knowledge_base: str):
        self.api_session, self.api_url = api_session, api_url.rstrip("/")
        self.runtime, self.knowledge_base = runtime, knowledge_base

    def _api(self, path: str) -> dict:
        status, body, _ = api_client.get(self.api_session, f"{self.api_url}{path}")
        if status != 200:  # the API's own explanation, e.g. "position 175 of TP53 is R, not G"
            raise ValueError(f"{status}: {body.get('detail', body)}")
        return body

    def get_variant_scores(self, protein: str, variant: str) -> dict:
        return self._api(f"/v1/proteins/{quote(protein, safe='')}/variants/{quote(variant, safe='')}")

    def get_position_scores(self, protein: str, position: int) -> dict:
        body = self._api(f"/v1/proteins/{quote(protein, safe='')}/positions/{int(position)}")
        keep = ("variant", "stability_score", "function_score")
        variants = [{k: v[k] for k in keep} for v in body["variants"]]
        return {"position": body["position"], "ref_aa": body["ref_aa"], "variants": variants}

    def search_knowledge(self, query: str, gene: str | None = None) -> list[dict]:
        where = {"equals": {"key": "gene_name", "value": gene.upper()}} if gene else None
        passages = knowledge.retrieve(self.runtime, self.knowledge_base, query, k=5, where=where)
        return [{"source": p["source"], "text": p["text"]} for p in passages]

    def call(self, name: str, arguments: dict):
        if name not in {tool["name"] for tool in TOOLS}:
            raise ValueError(f"no tool named {name}")
        return getattr(self, name)(**arguments)


def evidence(name: str, output) -> str:
    """A tool result as plain sentences, for the guardrail's grounding check: it scores an answer against
    text, and judges JSON poorly (measured in notebook 12: 0.54 for JSON, 0.97 for the same facts as text)."""
    meaning = "Scores are percentiles between 0 and 1; low means likely damaging."
    if name == "get_variant_scores":
        label = f"{output['gene_name'] or output['uniprot_id']} {output['variant']}"
        scores = (
            f"stability score {output['stability_score']:.2f}, function score {output['function_score']:.2f}"
        )
        return f"QAFI scores of {label}: {scores}. {meaning}"
    if name == "get_position_scores":
        listed = "; ".join(
            f"{v['variant']} stability {v['stability_score']:.2f} function {v['function_score']:.2f}"
            for v in output["variants"]
        )
        return f"QAFI scores at position {output['position']} ({output['ref_aa']}): {listed}. {meaning}"
    return "\n".join(f"From {p['source']}: {p['text']}" for p in output)


def run(session, question: str, toolbox: Toolbox, model: str = claude.HAIKU, max_turns: int = 6) -> dict:
    """Answer ``question`` with the tools: the answer, every tool call (name, input, seconds, error), the tool
    results as sentences (``sources``, for the grounding check), the number of turns, tokens and cost."""
    messages = [{"role": "user", "content": question}]
    steps, sources, usage = [], [], {"input_tokens": 0, "output_tokens": 0}
    for turn in range(1, max_turns + 1):
        reply = claude.messages(
            session,
            {"model": model, "max_tokens": 1024, "system": SYSTEM, "tools": TOOLS, "messages": messages},
        )
        for key in usage:
            usage[key] += reply["usage"][key]
        messages.append({"role": "assistant", "content": reply["content"]})
        if reply["stop_reason"] != "tool_use":
            return {
                "answer": "".join(b.get("text", "") for b in reply["content"] if b["type"] == "text"),
                "steps": steps,
                "sources": sources,
                "turns": turn,
                **usage,
                "usd": claude.cost(model, usage),
            }
        results = []
        for block in reply["content"]:
            if block["type"] != "tool_use":
                continue
            started = time.monotonic()
            try:
                output, error = toolbox.call(block["name"], block["input"]), False
                sources.append(evidence(block["name"], output))
            except Exception as problem:  # the agent is told what went wrong and can try differently
                output, error = {"error": str(problem)}, True
            steps.append(
                {
                    "tool": block["name"],
                    "input": block["input"],
                    "seconds": round(time.monotonic() - started, 2),
                    "error": error,
                }
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(output)[:20000],
                    "is_error": error,
                }
            )
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"no answer after {max_turns} turns: {[s['tool'] for s in steps]}")


def check(
    client,
    guardrail: str,
    version: str,
    text: str,
    source: str = "INPUT",
    grounding: list | None = None,
    query: str | None = None,
) -> dict:
    """Apply the guardrail to ``text``: was it blocked, the text to go on with (masked, or the refusal), why.

    ``source`` is INPUT for the user's question and OUTPUT for the agent's answer. For an answer, pass the
    tool results as ``grounding`` and the question as ``query``: the guardrail then also scores whether
    the answer is supported by those results (grounding) and answers the question (relevance).
    """
    content = [{"text": {"text": text, "qualifiers": ["guard_content"]}}]
    for passage in grounding or []:
        content.append({"text": {"text": passage[:5000], "qualifiers": ["grounding_source"]}})
    if query:
        content.append({"text": {"text": query, "qualifiers": ["query"]}})
    response = client.apply_guardrail(
        guardrailIdentifier=guardrail, guardrailVersion=version, source=source, content=content
    )
    reasons, actions = [], []
    for assessment in response.get("assessments", []):
        for topic in assessment.get("topicPolicy", {}).get("topics", []):
            reasons.append(f"topic: {topic['name']}")
            actions.append(topic.get("action"))
        for entity in assessment.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
            reasons.append(f"personal data: {entity['type']} ({entity.get('action', '').lower()})")
            actions.append(entity.get("action"))
        for item in assessment.get("contentPolicy", {}).get("filters", []):
            reasons.append(f"content: {item['type']}")
            actions.append(item.get("action"))
        for item in assessment.get("contextualGroundingPolicy", {}).get("filters", []):
            reasons.append(f"{item['type'].lower()} {item['score']:.2f} (needs {item['threshold']})")
            actions.append(item.get("action"))
    # Intervening is either blocking (show the refusal) or masking personal data (go on with the masked text).
    intervened = response["action"] == "GUARDRAIL_INTERVENED"
    blocked = intervened and "BLOCKED" in actions
    shown = response["outputs"][0]["text"] if intervened and response.get("outputs") else text
    return {"blocked": blocked, "text": shown, "reasons": reasons}


def ask(session, guard, guardrail: str, version: str, question: str, toolbox: Toolbox, **options) -> dict:
    """The whole path of one question: guardrail on the question, the agent, guardrail on the answer.

    ``guard`` is a bedrock-runtime client. Returns what the user sees (``answer``), whether and where it was
    stopped (``stopped``: None, "question" or "answer"), and the agent's run (tools, cost), if it ran.
    """
    entry = check(guard, guardrail, version, question, "INPUT")
    if entry["blocked"]:
        return {"answer": entry["text"], "stopped": "question", "reasons": entry["reasons"], "run": None}
    result = run(session, entry["text"], toolbox, **options)  # entry["text"]: personal data already masked
    exit_ = check(guard, guardrail, version, result["answer"], "OUTPUT", result["sources"] or None, question)
    return {
        "answer": exit_["text"],
        "stopped": "answer" if exit_["blocked"] else None,
        "reasons": entry["reasons"] + exit_["reasons"],
        "run": result,
    }
