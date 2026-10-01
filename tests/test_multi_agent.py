"""Scripted tests for the multi-agent workflow. No API key required."""

from unittest.mock import patch

import pytest

from src.agent.loop import AgentUnavailableError
from src.agent.state import (
    AnalystOutput,
    ClaimCheck,
    ComparisonOutput,
    OrchestratorPlan,
    PolicyClaim,
    SearchSubtask,
    VerificationOutput,
)
from src.agent.tools import PolicyTools, passthrough_rerank
from src.agent.workflow import MultiAgentWorkflow
from src.api.main import AgentStepResponse, QueryRequest
from src.eval.metrics import evaluate_multi_result
from src.generation.prompts import REFUSAL_MESSAGE


CHAS = {
    "chunk_id": "chas-1",
    "text": "CHAS provides subsidised medical and dental care at participating GP clinics for Singapore Citizens.",
    "title": "CHAS",
    "url": "https://www.healthhub.sg/chas",
    "domain": "healthhub.sg",
    "source": "healthhub",
    "category": "subsidies",
    "doc_id": "d1",
    "score": 0.9,
}

MSL = {
    "chunk_id": "msl-1",
    "text": "MediShield Life is basic health insurance covering all Singapore Citizens and Permanent Residents.",
    "title": "MediShield Life",
    "url": "https://www.moh.gov.sg/medishield-life",
    "domain": "moh.gov.sg",
    "source": "moh",
    "category": "subsidies",
    "doc_id": "d2",
    "score": 0.85,
}


class FakeRetriever:
    def __init__(self, chunks=None):
        self._chunks = {c["chunk_id"]: c for c in (chunks or [CHAS, MSL])}

    def retrieve(self, query, mode="hybrid", top_k=None):
        q = query.lower()
        hits = []
        if "chas" in q:
            hits.append(dict(CHAS))
        if "medishield" in q or "shield" in q:
            hits.append(dict(MSL))
        return hits

    def get_chunk(self, chunk_id):
        return self._chunks.get(chunk_id)


class ScriptedStructuredModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.invoke_count = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        parent = self

        class _Structured:
            def invoke(self, messages, **kwargs):
                parent.invoke_count += 1
                if not parent.outputs:
                    raise AssertionError("no more scripted structured outputs")
                return parent.outputs.pop(0)

        return _Structured()

    def invoke(self, messages, **kwargs):
        raise AssertionError("multi-agent specialists should use structured output")


def _workflow(outputs):
    retriever = FakeRetriever()
    return MultiAgentWorkflow(
        retriever=retriever,
        chat_model=ScriptedStructuredModel(outputs),
        tools=PolicyTools(retriever, rerank_fn=passthrough_rerank),
        rerank_fn=passthrough_rerank,
    )


def test_lookup_skips_comparison():
    result = _workflow(
        [
            OrchestratorPlan(
                intent="lookup",
                subtasks=[SearchSubtask(query="CHAS", domain="healthhub.sg")],
            ),
            AnalystOutput(
                claims=[
                    PolicyClaim(
                        statement="CHAS subsidises GP care for Singapore Citizens",
                        citation_index=1,
                        eligibility="Singapore Citizens",
                    )
                ]
            ),
            VerificationOutput(
                checks=[ClaimCheck(claim_index=0, status="supported", reason="stated in chunk")]
            ),
        ]
    ).query("What is CHAS?")

    agents = [s.get("agent") for s in result.steps]
    assert result.intent == "lookup"
    assert result.refused is False
    assert result.retrieval_mode == "multi"
    assert "comparison" not in agents
    assert "policy_analyst" in agents
    assert "verification" in agents
    assert [s["tool"] for s in result.steps if s.get("tool")][0] == "search_policy"
    assert "[1]" in result.answer
    assert "orchestrator" in result.trace_text


def test_comparison_issues_two_searches():
    result = _workflow(
        [
            OrchestratorPlan(
                intent="comparison",
                subtasks=[
                    SearchSubtask(query="CHAS", domain="healthhub.sg", label="CHAS"),
                    SearchSubtask(query="MediShield Life", domain="moh.gov.sg", label="MediShield"),
                ],
            ),
            ComparisonOutput(
                side_a_label="CHAS",
                side_b_label="MediShield Life",
                claims=[
                    PolicyClaim(statement="CHAS subsidises GP care for citizens", citation_index=1, side="A"),
                    PolicyClaim(statement="MediShield Life is basic insurance for citizens and PRs", citation_index=2, side="B"),
                ],
            ),
            VerificationOutput(
                checks=[
                    ClaimCheck(claim_index=0, status="supported", reason="ok"),
                    ClaimCheck(claim_index=1, status="supported", reason="ok"),
                ]
            ),
        ]
    ).query("Compare CHAS and MediShield Life.")

    tools = [s["tool"] for s in result.steps if s.get("tool")]
    agents = [s.get("agent") for s in result.steps]
    assert result.intent == "comparison"
    assert result.refused is False
    assert tools.count("search_policy") == 2
    assert "comparison" in agents
    assert "policy_analyst" not in agents
    assert {c["domain"] for c in result.citations} == {"healthhub.sg", "moh.gov.sg"}


def test_verifier_rejection_retries_once_then_refuses():
    result = _workflow(
        [
            OrchestratorPlan(
                intent="lookup",
                subtasks=[SearchSubtask(query="CHAS", domain="healthhub.sg")],
            ),
            AnalystOutput(
                claims=[PolicyClaim(statement="Invented subsidy amount of $9999", citation_index=1)]
            ),
            VerificationOutput(
                checks=[ClaimCheck(claim_index=0, status="unsupported", reason="not in chunk")]
            ),
            AnalystOutput(
                claims=[PolicyClaim(statement="Still invented", citation_index=1)]
            ),
            VerificationOutput(
                checks=[ClaimCheck(claim_index=0, status="unsupported", reason="still unsupported")]
            ),
        ]
    ).query("What is CHAS?")

    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.retry_count == 1
    assert any(s.get("retry") for s in result.steps)
    assert "retry" in result.trace_text
    agents = [s.get("agent") for s in result.steps]
    assert agents.count("verification") == 2
    assert agents.count("retrieval") >= 2


def test_trace_includes_request_id_and_agents():
    result = _workflow(
        [
            OrchestratorPlan(intent="lookup", subtasks=[SearchSubtask(query="CHAS")]),
            AnalystOutput(
                claims=[PolicyClaim(statement="CHAS is a subsidy scheme", citation_index=1)]
            ),
            VerificationOutput(checks=[ClaimCheck(claim_index=0, status="supported")]),
        ]
    ).query("What is CHAS?")

    first = result.steps[0]
    assert first["request_id"]
    assert first["step_number"] == 1
    assert first["agent"] == "orchestrator"
    assert "latency_ms" in first
    assert result.trace_text.startswith("User")
    parsed = AgentStepResponse(**first)
    assert parsed.agent == "orchestrator"


def test_out_of_scope_still_searches_then_can_refuse():
    result = _workflow(
        [
            OrchestratorPlan(
                intent="out_of_scope",
                subtasks=[SearchSubtask(query="Apple stock price")],
            ),
        ]
    ).query("What is the stock price of Apple Inc today?")

    assert result.intent == "out_of_scope"
    assert result.refused is True
    assert result.retry_count == 1
    assert any(s.get("tool") == "search_policy" for s in result.steps)


def test_multi_unavailable_without_chat_model():
    with patch("src.agent.workflow.get_chat_model", return_value=None):
        with pytest.raises(AgentUnavailableError, match="chat model"):
            MultiAgentWorkflow(retriever=FakeRetriever(), rerank_fn=passthrough_rerank)


def test_query_request_accepts_multi_mode():
    req = QueryRequest(question="What is CHAS?", mode="multi")
    assert req.mode == "multi"
    assert QueryRequest(question="What is CHAS?").mode == "pipeline"


def test_multi_trajectory_metrics_include_route():
    result = _workflow(
        [
            OrchestratorPlan(
                intent="comparison",
                subtasks=[
                    SearchSubtask(query="CHAS"),
                    SearchSubtask(query="MediShield Life"),
                ],
            ),
            ComparisonOutput(
                claims=[
                    PolicyClaim(statement="CHAS is a subsidy", citation_index=1, side="A"),
                    PolicyClaim(statement="MediShield is insurance", citation_index=2, side="B"),
                ]
            ),
            VerificationOutput(
                checks=[
                    ClaimCheck(claim_index=0, status="supported"),
                    ClaimCheck(claim_index=1, status="supported"),
                ]
            ),
        ]
    ).query("Compare CHAS and MediShield Life.")
    row = evaluate_multi_result(
        {
            "id": "t1",
            "question": "Compare CHAS and MediShield Life.",
            "expect_refusal": False,
            "expected_source_domains": ["healthhub.sg"],
            "reference_answer": "",
            "expected_tools": ["search_policy", "search_policy"],
            "expected_route": "comparison",
        },
        result,
    )
    assert row["route_accuracy"] is True
    assert row["expected_tools_used"] is True
    assert row["retry_count"] == 0
    assert row["success"] is True
