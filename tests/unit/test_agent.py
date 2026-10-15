import json
from types import SimpleNamespace as NS

import pytest

from qafi_platform import agent, api_client, claude


def tool_use(id_, name, arguments):
    return {"type": "tool_use", "id": id_, "name": name, "input": arguments}


def reply(content, stop="end_turn"):
    return {"content": content, "stop_reason": stop, "usage": {"input_tokens": 1000, "output_tokens": 100}}


@pytest.fixture
def toolbox(monkeypatch):
    def get(session, url, sign=True, timeout=30):
        if url.endswith("/v1/proteins/TP53/variants/R175H"):
            body = {"uniprot_id": "P04637", "gene_name": "TP53", "variant": "R175H"}
            return 200, {**body, "stability_score": 0.2214, "function_score": 0.3011, "zero_shot": -9.13}, 1.2
        return 422, {"detail": "position 175 of TP53 is R, not G"}, 0.1

    monkeypatch.setattr(api_client, "get", get)
    runtime = NS(retrieve=lambda **kw: {"retrievalResults": [{
        "content": {"text": "R175H: in LFS"}, "location": {"s3Location": {"uri": "s3://kb/uniprot/P04637.md"}},
        "score": 0.8, "metadata": {}}]})  # fmt: skip
    box = agent.Toolbox(None, "https://api.example/", runtime, "KB1")
    box.runtime_requests = runtime
    return box


def test_the_agent_runs_the_tools_claude_asks_for_and_returns_its_answer(monkeypatch, toolbox):
    scores = tool_use("t1", "get_variant_scores", {"protein": "TP53", "variant": "R175H"})
    search = tool_use("t2", "search_knowledge", {"query": "R175H", "gene": "tp53"})
    script = iter([
        reply([{"type": "text", "text": "Let me look."}, scores, search], stop="tool_use"),
        reply([{"type": "text", "text": "Likely damaging (QAFI score); seen in LFS (P04637.md)."}]),
    ])  # fmt: skip
    sent = []  # what was sent at each call (a copy: the conversation keeps growing afterwards)
    monkeypatch.setattr(
        claude, "messages", lambda session, body: sent.append(json.loads(json.dumps(body))) or next(script)
    )
    result = agent.run(None, "Is TP53 R175H damaging?", toolbox)
    assert result["answer"] == "Likely damaging (QAFI score); seen in LFS (P04637.md)."
    assert [s["tool"] for s in result["steps"]] == ["get_variant_scores", "search_knowledge"]
    assert (
        result["turns"] == 2
        and result["input_tokens"] == 2000
        and result["usd"] == round((2000 * 1.1 + 200 * 5.5) / 1e6, 6)
    )
    second = sent[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    results = second[2]["content"]
    assert results[0]["tool_use_id"] == "t1" and '"stability_score": 0.2214' in results[0]["content"]
    assert "P04637.md" in results[1]["content"]
    assert result["sources"] == [  # sentences for the grounding check
        "QAFI scores of TP53 R175H: stability score 0.22, function score 0.30, ESM-2 zero-shot score -9.13 "
        "(log-probability of the new amino acid minus the old one; negative means less expected by the "
        "language model). Stability and function scores are percentiles between 0 and 1: low means likely "
        "damaging (destabilising, loss of function), high means likely tolerated. They are model predictions "
        "for research, not clinical classifications.",
        "From P04637.md: R175H: in LFS",
    ]
    assert sent[0]["tools"] == agent.TOOLS and "not a clinical" in sent[0]["system"]


def test_a_failing_tool_is_reported_to_claude_not_raised(monkeypatch, toolbox):
    wrong = tool_use("t1", "get_variant_scores", {"protein": "TP53", "variant": "G175H"})
    script = iter([
        reply([wrong], stop="tool_use"),
        reply([{"type": "text", "text": "Position 175 of TP53 is R, so G175H does not exist."}]),
    ])  # fmt: skip
    sent = []
    monkeypatch.setattr(
        claude, "messages", lambda session, body: sent.append(json.loads(json.dumps(body))) or next(script)
    )
    result = agent.run(None, "Score of TP53 G175H?", toolbox)
    assert result["steps"][0]["error"] is True
    assert result["sources"] == ["The tool answered with an error: 422: position 175 of TP53 is R, not G"]
    tool_result = sent[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] and "is R, not G" in tool_result["content"]


def test_only_the_declared_tools_can_run(toolbox):
    with pytest.raises(ValueError, match="no tool named"):
        toolbox.call("delete_everything", {})


def test_an_agent_that_never_answers_is_stopped(monkeypatch, toolbox):
    endless = reply([tool_use("t", "search_knowledge", {"query": "x"})], stop="tool_use")
    monkeypatch.setattr(claude, "messages", lambda session, body: endless)
    with pytest.raises(RuntimeError, match="no answer after 3 turns"):
        agent.run(None, "?", toolbox, max_turns=3)


def test_the_guardrail_verdict_is_explained():
    sent = {}

    def apply_guardrail(**request):
        sent.update(request)
        return {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "I can't give medical advice about a specific person."}],
            "assessments": [
                {
                    "topicPolicy": {"topics": [{"name": "Personal medical advice", "action": "BLOCKED"}]},
                    "contextualGroundingPolicy": {
                        "filters": [{"type": "GROUNDING", "score": 0.31, "threshold": 0.7}]
                    },
                }
            ],
        }

    verdict = agent.check(
        NS(apply_guardrail=apply_guardrail), "G1", "1", "answer", "OUTPUT", ["source"], "question"
    )
    assert verdict == {
        "blocked": True,
        "text": "I can't give medical advice about a specific person.",
        "reasons": ["topic: Personal medical advice", "grounding 0.31 (needs 0.7)"],
    }
    assert [c["text"]["qualifiers"] for c in sent["content"]] == [
        ["guard_content"],
        ["grounding_source"],
        ["query"],
    ]


def test_masking_personal_data_is_not_a_refusal():
    def apply_guardrail(**request):
        return {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "Send the TP53 summary to {EMAIL}"}],
            "assessments": [
                {"sensitiveInformationPolicy": {"piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]}}
            ],
        }

    verdict = agent.check(NS(apply_guardrail=apply_guardrail), "G1", "1", "Send the TP53 summary to a@b.org")
    assert verdict == {
        "blocked": False,
        "text": "Send the TP53 summary to {EMAIL}",
        "reasons": ["personal data: EMAIL (anonymized)"],
    }


def test_a_question_passes_the_guardrail_twice(monkeypatch, toolbox):
    checks = []

    def apply_guardrail(**request):
        checks.append(request)
        if request["source"] == "INPUT":
            return {"action": "NONE", "outputs": [], "assessments": []}
        grounding = {"type": "GROUNDING", "score": 0.95, "threshold": 0.7, "action": "NONE"}
        return {
            "action": "NONE",
            "outputs": [],
            "assessments": [{"contextualGroundingPolicy": {"filters": [grounding]}}],
        }

    scores = tool_use("t1", "get_variant_scores", {"protein": "TP53", "variant": "R175H"})
    script = iter([
        reply([scores], stop="tool_use"),
        reply([{"type": "text", "text": "Predicted damaging (QAFI score)."}]),
    ])  # fmt: skip
    monkeypatch.setattr(claude, "messages", lambda session, body: next(script))
    result = agent.ask(
        None, NS(apply_guardrail=apply_guardrail), "G1", "2", "Is TP53 R175H damaging?", toolbox
    )
    assert result["answer"] == "Predicted damaging (QAFI score)." and result["stopped"] is None
    assert result["reasons"] == ["grounding 0.95 (needs 0.7)"]
    assert [c["source"] for c in checks] == ["INPUT", "OUTPUT"]
    qualifiers = [c["text"]["qualifiers"] for c in checks[1]["content"]]
    assert qualifiers == [["guard_content"], ["grounding_source"], ["query"]]


def test_a_refused_question_never_reaches_the_agent(monkeypatch, toolbox):
    def apply_guardrail(**request):
        topic = {"name": "Personal medical advice", "action": "BLOCKED"}
        return {"action": "GUARDRAIL_INTERVENED", "outputs": [{"text": "Please ask a clinical geneticist."}],
                "assessments": [{"topicPolicy": {"topics": [topic]}}]}  # fmt: skip

    monkeypatch.setattr(claude, "messages", lambda session, body: pytest.fail("the agent must not run"))
    question = "My father has TP53 R175H, should he get chemotherapy?"
    result = agent.ask(None, NS(apply_guardrail=apply_guardrail), "G1", "2", question, toolbox)
    assert result == {
        "answer": "Please ask a clinical geneticist.",
        "stopped": "question",
        "reasons": ["topic: Personal medical advice"],
        "run": None,
    }


def test_an_answer_without_sources_is_checked_without_the_grounding_policy():
    sent = {}

    def apply_guardrail(**request):
        sent.update(request)
        return {"action": "NONE", "outputs": [], "assessments": []}

    agent.check(NS(apply_guardrail=apply_guardrail), "G1", "2", "An answer.", "OUTPUT", None, "A question?")
    assert sent["content"] == [
        {"text": {"text": "An answer."}}
    ]  # with no sources there is nothing to ground on
