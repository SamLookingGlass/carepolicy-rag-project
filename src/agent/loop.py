"""Bounded tool-calling loop: search/read until the model stops, then a structured answer."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agent.prompts import AGENT_SYSTEM_PROMPT, FINAL_ANSWER_USER_TEMPLATE
from src.agent.tools import PolicyTools, SearchPolicyArgs, ReadChunkArgs
from src.config import get_settings
from src.generation.citations import build_citation_objects, verify_citations
from src.generation.prompts import REFUSAL_MESSAGE
from src.llm_provider import get_chat_model
from src.pipeline import QueryResult
from src.retrieval.hybrid import HybridRetriever

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    from langchain.tools import StructuredTool

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


class AgentUnavailableError(RuntimeError):
    """Raised when agent mode is requested but no chat model is configured."""


class FinalAnswer(BaseModel):
    answer: str = Field(..., description="Grounded answer with [N] citations, or a short refusal")
    citation_indices: list[int] = Field(default_factory=list)
    refused: bool = Field(..., description="True if the sources do not support an answer")


def _tool_calls(response: Any) -> list[dict]:
    raw = getattr(response, "tool_calls", None) or []
    parsed: list[dict] = []
    for i, tc in enumerate(raw):
        if isinstance(tc, dict):
            parsed.append(
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args") or {},
                    "id": tc.get("id") or f"call_{i}",
                }
            )
        else:
            parsed.append(
                {
                    "name": getattr(tc, "name", "") or "",
                    "args": getattr(tc, "args", None) or {},
                    "id": getattr(tc, "id", None) or f"call_{i}",
                }
            )
    return parsed


def _coerce_final(raw: Any) -> FinalAnswer:
    if isinstance(raw, FinalAnswer):
        return raw
    if hasattr(raw, "model_dump"):
        return FinalAnswer(**raw.model_dump())
    if isinstance(raw, dict):
        return FinalAnswer(**raw)
    raise TypeError(f"Unexpected structured output type: {type(raw)!r}")


class PolicyAgent:
    """Model-in-the-loop over search_policy and read_chunk, with a hard step budget."""

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
                "Agent mode requires a chat model. Set OPENAI_API_KEY or Azure credentials."
            )
        self.rerank_fn = rerank_fn
        self._bound_tools = tools
        self._langchain_tools = [
            StructuredTool.from_function(
                func=lambda query, domain=None: None,
                name="search_policy",
                description=(
                    "Search Singapore healthcare policy documents (HealthHub, MOH, HPB, PDPC). "
                    "Use domain to restrict one search when comparing schemes or mixing "
                    "clinical policy with PDPA."
                ),
                args_schema=SearchPolicyArgs,
            ),
            StructuredTool.from_function(
                func=lambda chunk_id: None,
                name="read_chunk",
                description="Read the full text of a chunk returned by search_policy.",
                args_schema=ReadChunkArgs,
            ),
        ]

    def _new_tools(self) -> PolicyTools:
        if self._bound_tools is not None:
            return self._bound_tools
        return PolicyTools(self.retriever, rerank_fn=self.rerank_fn)

    def _refusal(
        self,
        start: float,
        tools: PolicyTools,
        steps: list[dict],
        model_turns: int,
        tool_call_count: int,
        answer: str | None = None,
    ) -> QueryResult:
        return QueryResult(
            answer=answer or REFUSAL_MESSAGE,
            citations=[],
            confidence=min(tools.best_score, 0.3) if tools.best_score else 0.0,
            latency_ms=(time.perf_counter() - start) * 1000,
            retrieval_mode="agent",
            chunks_used=0,
            refused=True,
            citation_verification={"valid": True, "citation_accuracy": 1.0, "cited_count": 0},
            retrieved_chunks=list(tools.chunks),
            steps=steps,
            model_turns=model_turns,
            tool_call_count=tool_call_count,
        )

    def _finish(
        self,
        start: float,
        tools: PolicyTools,
        steps: list[dict],
        model_turns: int,
        tool_call_count: int,
        final: FinalAnswer,
    ) -> QueryResult:
        chunks = list(tools.chunks)
        if final.refused:
            return self._refusal(
                start,
                tools,
                steps,
                model_turns,
                tool_call_count,
                answer=final.answer or REFUSAL_MESSAGE,
            )

        answer = final.answer.strip()
        citations = build_citation_objects(answer, chunks)
        verification = verify_citations(answer, chunks)
        confidence = float(tools.best_score)
        return QueryResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            retrieval_mode="agent",
            chunks_used=len(chunks),
            refused=False,
            citation_verification=verification,
            retrieved_chunks=chunks,
            steps=steps,
            model_turns=model_turns,
            tool_call_count=tool_call_count,
        )

    @traceable(name="carepolicy_agent_query", run_type="chain")
    def query(self, question: str) -> QueryResult:
        start = time.perf_counter()
        tools = self._new_tools()
        # Fresh citation state when PolicyTools is reused across queries (tests)
        tools.chunks = []
        tools._index_by_id = {}
        tools.best_score = 0.0

        max_turns = self.settings.agent_max_model_turns
        max_tools = self.settings.agent_max_tool_calls
        steps: list[dict] = []
        model_turns = 0
        tool_call_count = 0

        bound = self.chat_model.bind_tools(self._langchain_tools)
        messages: list = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        while True:
            if model_turns >= max_turns:
                return self._refusal(start, tools, steps, model_turns, tool_call_count)

            response = bound.invoke(messages)
            model_turns += 1
            messages.append(response)
            calls = _tool_calls(response)

            if not calls:
                break

            for call in calls:
                if tool_call_count >= max_tools:
                    return self._refusal(start, tools, steps, model_turns, tool_call_count)

                try:
                    observation = tools.execute(call["name"], call["args"])
                except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
                    observation = {"error": str(exc)}

                tool_call_count += 1
                steps.append(
                    {
                        "tool": call["name"],
                        "arguments": dict(call["args"] or {}),
                        "observation": observation,
                    }
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(observation, ensure_ascii=False, default=str),
                        tool_call_id=str(call["id"]),
                    )
                )

        prompt = FINAL_ANSWER_USER_TEMPLATE.format(citation_map=tools.citation_map())
        structured = self.chat_model.with_structured_output(FinalAnswer)
        raw = structured.invoke([*messages, HumanMessage(content=prompt)])
        final = _coerce_final(raw)
        return self._finish(start, tools, steps, model_turns, tool_call_count, final)
