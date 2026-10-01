"""Owned multi-agent loop: orchestrator routes, specialists write shared state, one retry."""

from __future__ import annotations

import time
from typing import Any

from src.agent.loop import AgentUnavailableError
from src.agent.specialists import (
    broaden_subtasks,
    plan_route,
    run_analyst,
    run_comparison,
    run_retrieval,
    run_verification,
)
from src.agent.state import WorkflowState
from src.agent.tools import PolicyTools
from src.agent.trace import format_trace, tool_names
from src.config import get_settings
from src.generation.citations import build_citation_objects, verify_citations
from src.generation.prompts import REFUSAL_MESSAGE
from src.llm_provider import get_chat_model
from src.pipeline import QueryResult
from src.retrieval.hybrid import HybridRetriever

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def _needs_retry(state: WorkflowState) -> bool:
    if not state.retrieved_chunks:
        return True
    if not state.claims:
        return True
    if not state.verification:
        return True
    return not any(check.status == "supported" for check in state.verification)


def _supported_claims(state: WorkflowState):
    supported_idx = {
        check.claim_index
        for check in state.verification
        if check.status == "supported"
    }
    return [claim for i, claim in enumerate(state.claims) if i in supported_idx]


def _compose_answer(state: WorkflowState) -> str:
    claims = _supported_claims(state)
    if not claims:
        return REFUSAL_MESSAGE
    parts: list[str] = []
    for claim in claims:
        sentence = claim.statement.rstrip(".")
        extras = []
        if claim.eligibility:
            extras.append(f"eligibility: {claim.eligibility}")
        if claim.conditions:
            extras.append(f"conditions: {claim.conditions}")
        if claim.exceptions:
            extras.append(f"exceptions: {claim.exceptions}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        parts.append(f"{sentence}{suffix} [{claim.citation_index}].")
    return " ".join(parts)


def _supported_rate(state: WorkflowState) -> float | None:
    if not state.verification:
        return None
    return sum(1 for c in state.verification if c.status == "supported") / len(state.verification)


class MultiAgentWorkflow:
    """Orchestrator + retrieval + analyst or comparison + verification, with one retry."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        chat_model: Any | None = None,
        tools: PolicyTools | None = None,
        rerank_fn=None,
    ):
        self.settings = get_settings()
        self.retriever = retriever or HybridRetriever()
        self.chat_model = chat_model if chat_model is not None else get_chat_model()
        if self.chat_model is None:
            raise AgentUnavailableError(
                "Multi-agent mode requires a chat model. Set OPENAI_API_KEY or Azure credentials."
            )
        self.rerank_fn = rerank_fn
        self._bound_tools = tools

    def _new_tools(self) -> PolicyTools:
        if self._bound_tools is not None:
            return self._bound_tools
        return PolicyTools(self.retriever, rerank_fn=self.rerank_fn)

    def _result(
        self,
        start: float,
        state: WorkflowState,
        *,
        refused: bool,
        answer: str | None = None,
    ) -> QueryResult:
        chunks = list(state.retrieved_chunks)
        text = REFUSAL_MESSAGE if refused else (answer or _compose_answer(state))
        if refused:
            citations = []
            verification = {"valid": True, "citation_accuracy": 1.0, "cited_count": 0}
        else:
            citations = build_citation_objects(text, chunks)
            verification = verify_citations(text, chunks)
        tools = state.tools
        confidence = float(getattr(tools, "best_score", 0.0) or 0.0)
        if refused:
            confidence = min(confidence, 0.3) if confidence else 0.0
        cost = state.tokens_used / 1000.0 * self.settings.llm_usd_per_1k_tokens
        names = tool_names(state.trace)
        llm_steps = [s for s in state.trace if not s.get("tool")]
        return QueryResult(
            answer=text,
            citations=citations,
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            retrieval_mode="multi",
            chunks_used=0 if refused else len(chunks),
            refused=refused,
            citation_verification=verification,
            retrieved_chunks=chunks,
            steps=list(state.trace),
            model_turns=len(llm_steps),
            tool_call_count=len(names),
            intent=state.intent,
            trace_text=format_trace(state),
            tokens=state.tokens_used,
            estimated_cost_usd=round(cost, 6),
            retry_count=state.retry_count,
            supported_claim_rate=_supported_rate(state),
        )

    def _analyze(self, state: WorkflowState) -> None:
        if state.intent == "comparison":
            run_comparison(state, self.chat_model)
        else:
            run_analyst(state, self.chat_model)

    def _over_budget(self, state: WorkflowState) -> bool:
        return state.remaining_budget.get("steps", 0) <= 0

    @traceable(name="carepolicy_multi_agent_query", run_type="chain")
    def query(self, question: str) -> QueryResult:
        start = time.perf_counter()
        tools = self._new_tools()
        tools.chunks = []
        tools._index_by_id = {}
        tools.best_score = 0.0

        state = WorkflowState(
            user_query=question,
            remaining_budget={
                "retries": self.settings.multi_max_retries,
                "steps": self.settings.multi_max_steps,
            },
            tools=tools,
        )

        try:
            plan_route(state, self.chat_model)
        except Exception as exc:  # noqa: BLE001
            state.agent_outputs["orchestrator_error"] = str(exc)
            return self._result(start, state, refused=True)

        if self._over_budget(state):
            return self._result(start, state, refused=True)

        while True:
            run_retrieval(state, retry=state.retry_count > 0)
            if state.retrieved_chunks and not self._over_budget(state):
                try:
                    self._analyze(state)
                    run_verification(state, self.chat_model)
                except Exception as exc:  # noqa: BLE001
                    state.agent_outputs["specialist_error"] = str(exc)
                    return self._result(start, state, refused=True)

            if not _needs_retry(state):
                return self._result(start, state, refused=False)

            if state.remaining_budget.get("retries", 0) <= 0 or self._over_budget(state):
                return self._result(start, state, refused=True)

            state.remaining_budget["retries"] -= 1
            state.retry_count += 1
            broaden_subtasks(state)
