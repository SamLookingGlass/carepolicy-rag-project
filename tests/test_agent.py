"""Unit tests for the tool-calling agent. Uses a scripted model — no API key required."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from src.agent.loop import AgentUnavailableError, FinalAnswer, PolicyAgent
from src.agent.tools import PolicyTools, passthrough_rerank
from src.api.main import QueryRequest
from src.eval.metrics import evaluate_agent_result
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
        if not hits:
            hits = [dict(c) for c in self._chunks.values()]
        return hits

    def get_chunk(self, chunk_id):
        return self._chunks.get(chunk_id)


class ScriptedChatModel:
    def __init__(self, tool_turns, final):
        self.tool_turns = list(tool_turns)
        self.final = final
        self.invoke_count = 0
        self._i = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.invoke_count += 1
        if self._i < len(self.tool_turns):
            msg = self.tool_turns[self._i]
            self._i += 1
            return msg
        return AIMessage(content="")

    def with_structured_output(self, schema, **kwargs):
        parent = self

        class _Structured:
            def invoke(self, messages, **kwargs):
                parent.invoke_count += 1
                return parent.final

        return _Structured()


class AlwaysSearchModel:
    def __init__(self):
        self.invoke_count = 0
        self.n = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.invoke_count += 1
        self.n += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_policy",
                    "args": {"query": f"noise {self.n}"},
                    "id": f"call_{self.n}",
                    "type": "tool_call",
                }
            ],
        )

    def with_structured_output(self, schema, **kwargs):
        raise AssertionError("structured output should not run when the budget is exhausted")


def _tool_msg(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _agent(model, tools=None):
    retriever = FakeRetriever()
    return PolicyAgent(
        retriever=retriever,
        chat_model=model,
        tools=tools or PolicyTools(retriever, rerank_fn=passthrough_rerank),
        rerank_fn=passthrough_rerank,
    )


def test_search_policy_assigns_citation_indices_and_filters_domain():
    tools = PolicyTools(FakeRetriever(), rerank_fn=passthrough_rerank)
    filtered = tools.search_policy("healthcare schemes", domain="healthhub.sg")
    assert "error" not in filtered
    assert filtered["results"]
    assert all(r["domain"] == "healthhub.sg" for r in filtered["results"])
    assert filtered["results"][0]["citation_index"] == 1

    bad = tools.search_policy("chas", domain="example.com")
    assert "error" in bad


def test_read_chunk_unknown_id():
    tools = PolicyTools(FakeRetriever(), rerank_fn=passthrough_rerank)
    result = tools.read_chunk("does-not-exist")
    assert "error" in result
    assert "does-not-exist" in result["error"]


def test_read_chunk_returns_full_text_after_search():
    tools = PolicyTools(FakeRetriever(), rerank_fn=passthrough_rerank)
    tools.search_policy("CHAS")
    result = tools.read_chunk("chas-1")
    assert result["text"] == CHAS["text"]
    assert result["citation_index"] == 1


def test_two_search_comparison():
    model = ScriptedChatModel(
        [
            _tool_msg("search_policy", {"query": "CHAS eligibility", "domain": "healthhub.sg"}, "c1"),
            _tool_msg("search_policy", {"query": "MediShield Life coverage", "domain": "moh.gov.sg"}, "c2"),
            AIMessage(content=""),
        ],
        FinalAnswer(
            answer=(
                "CHAS subsidises GP care for Singapore Citizens [1]. "
                "MediShield Life is basic insurance for citizens and PRs [2]."
            ),
            citation_indices=[1, 2],
            refused=False,
        ),
    )
    result = _agent(model).query("Compare CHAS and MediShield Life.")

    assert result.refused is False
    assert result.retrieval_mode == "agent"
    assert [s["tool"] for s in result.steps] == ["search_policy", "search_policy"]
    assert result.steps[0]["arguments"]["query"] == "CHAS eligibility"
    assert result.steps[1]["arguments"]["domain"] == "moh.gov.sg"
    assert result.tool_call_count == 2
    assert result.model_turns == 3
    assert result.citation_verification["valid"] is True
    assert {c["domain"] for c in result.citations} == {"healthhub.sg", "moh.gov.sg"}


def test_budget_stops_infinite_tool_calls():
    model = AlwaysSearchModel()
    result = _agent(model).query("What is CHAS?")

    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.model_turns == 4
    assert result.tool_call_count <= 6
    assert model.invoke_count == 4
    assert result.steps


def test_bad_chunk_id_is_recorded_and_can_refuse():
    model = ScriptedChatModel(
        [
            _tool_msg("read_chunk", {"chunk_id": "does-not-exist"}, "c1"),
            AIMessage(content=""),
        ],
        FinalAnswer(answer="I cannot answer from the sources.", citation_indices=[], refused=True),
    )
    result = _agent(model).query("What is CHAS?")

    assert result.refused is True
    assert result.steps[0]["tool"] == "read_chunk"
    assert "error" in result.steps[0]["observation"]
    assert result.citation_verification["valid"] is True


def test_citation_verification_after_agent_answer():
    model = ScriptedChatModel(
        [
            _tool_msg("search_policy", {"query": "CHAS"}, "c1"),
            AIMessage(content=""),
        ],
        FinalAnswer(answer="Invented claim with a fake marker [9].", citation_indices=[9], refused=False),
    )
    result = _agent(model).query("What is CHAS?")

    assert result.refused is False
    assert result.citation_verification["valid"] is False
    assert result.citations == []


def test_agent_unavailable_without_chat_model():
    with patch("src.agent.loop.get_chat_model", return_value=None):
        with pytest.raises(AgentUnavailableError, match="chat model"):
            PolicyAgent(retriever=FakeRetriever(), rerank_fn=passthrough_rerank)


def test_query_request_defaults_to_pipeline():
    req = QueryRequest(question="What is CHAS and who is eligible?")
    assert req.mode == "pipeline"


def test_agent_trajectory_metrics():
    model = ScriptedChatModel(
        [
            _tool_msg("search_policy", {"query": "CHAS"}, "c1"),
            _tool_msg("search_policy", {"query": "MediShield Life"}, "c2"),
            AIMessage(content=""),
        ],
        FinalAnswer(
            answer="CHAS is a subsidy scheme [1]. MediShield Life is insurance [2].",
            citation_indices=[1, 2],
            refused=False,
        ),
    )
    result = _agent(model).query("Compare CHAS and MediShield Life.")
    row = evaluate_agent_result(
        {
            "id": "t1",
            "question": "Compare CHAS and MediShield Life.",
            "expect_refusal": False,
            "expected_source_domains": ["healthhub.sg"],
            "reference_answer": "",
            "expected_tools": ["search_policy", "search_policy"],
        },
        result,
    )
    assert row["expected_tools_used"] is True
    assert row["within_budget"] is True
    assert row["tool_call_count"] == 2
    assert row["success"] is True
