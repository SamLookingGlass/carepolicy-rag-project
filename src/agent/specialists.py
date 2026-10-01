"""Specialists that only read and write WorkflowState. They do not call each other."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import (
    ANALYST_PROMPT,
    COMPARISON_PROMPT,
    ORCHESTRATOR_PROMPT,
    VERIFIER_PROMPT,
)
from src.agent.state import (
    AnalystOutput,
    ComparisonOutput,
    OrchestratorPlan,
    SearchSubtask,
    VerificationOutput,
    WorkflowState,
)
from src.agent.trace import elapsed_ms, invoke_structured, record_step, timed


def plan_route(state: WorkflowState, chat_model: Any) -> OrchestratorPlan:
    start = timed()
    plan, tokens = invoke_structured(
        chat_model,
        OrchestratorPlan,
        [
            SystemMessage(content=ORCHESTRATOR_PROMPT),
            HumanMessage(content=state.user_query),
        ],
    )
    if not plan.subtasks:
        plan.subtasks = [SearchSubtask(query=state.user_query)]
    state.intent = plan.intent
    state.subtasks = list(plan.subtasks)
    state.agent_outputs["orchestrator"] = plan.model_dump()
    record_step(
        state,
        agent="orchestrator",
        input_data={"question": state.user_query},
        output=plan.model_dump(),
        latency_ms=elapsed_ms(start),
        tokens=tokens,
    )
    return plan


def run_retrieval(state: WorkflowState, retry: bool = False) -> None:
    tools = state.tools
    hits: list[dict] = []
    for subtask in state.subtasks:
        start = timed()
        observation = tools.search_policy(subtask.query, domain=subtask.domain)
        record_step(
            state,
            agent="retrieval",
            tool="search_policy",
            input_data={"query": subtask.query, "domain": subtask.domain},
            output=observation,
            latency_ms=elapsed_ms(start),
            status="ok" if "error" not in observation else "fail",
            retry=retry,
        )
        results = observation.get("results") or []
        if results:
            top_id = results[0].get("chunk_id")
            if top_id:
                read_start = timed()
                full = tools.read_chunk(top_id)
                record_step(
                    state,
                    agent="retrieval",
                    tool="read_chunk",
                    input_data={"chunk_id": top_id},
                    output=full,
                    latency_ms=elapsed_ms(read_start),
                    retry=retry,
                )
        hits = list(tools.chunks)
    state.retrieved_chunks = hits
    state.agent_outputs["retrieval"] = {
        "chunk_ids": [c.get("chunk_id") for c in hits],
        "retry": retry,
    }


def _evidence_block(state: WorkflowState) -> str:
    if not state.retrieved_chunks:
        return "(no chunks retrieved)"
    parts = []
    for i, chunk in enumerate(state.retrieved_chunks, start=1):
        parts.append(
            f"[{i}] {chunk.get('title', 'Untitled')} ({chunk.get('domain', '')})\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(parts)


def run_analyst(state: WorkflowState, chat_model: Any) -> None:
    start = timed()
    output, tokens = invoke_structured(
        chat_model,
        AnalystOutput,
        [
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(
                content=f"Question: {state.user_query}\n\nEvidence:\n{_evidence_block(state)}"
            ),
        ],
    )
    state.claims = list(output.claims)
    state.agent_outputs["policy_analyst"] = output.model_dump()
    record_step(
        state,
        agent="policy_analyst",
        input_data={"question": state.user_query, "chunks": len(state.retrieved_chunks)},
        output=output.model_dump(),
        latency_ms=elapsed_ms(start),
        tokens=tokens,
    )


def run_comparison(state: WorkflowState, chat_model: Any) -> None:
    start = timed()
    labels = [t.label or t.query for t in state.subtasks]
    output, tokens = invoke_structured(
        chat_model,
        ComparisonOutput,
        [
            SystemMessage(content=COMPARISON_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state.user_query}\n"
                    f"Sides: {labels}\n\nEvidence:\n{_evidence_block(state)}"
                )
            ),
        ],
    )
    state.claims = list(output.claims)
    state.agent_outputs["comparison"] = output.model_dump()
    record_step(
        state,
        agent="comparison",
        input_data={"question": state.user_query, "sides": labels},
        output=output.model_dump(),
        latency_ms=elapsed_ms(start),
        tokens=tokens,
    )


def run_verification(state: WorkflowState, chat_model: Any) -> None:
    if not state.claims:
        state.verification = []
        record_step(
            state,
            agent="verification",
            input_data={"claims": 0},
            output={"checks": []},
            status="fail",
        )
        return

    claim_lines = []
    for i, claim in enumerate(state.claims):
        chunk = None
        idx = claim.citation_index
        if 1 <= idx <= len(state.retrieved_chunks):
            chunk = state.retrieved_chunks[idx - 1]
        chunk_text = (chunk or {}).get("text", "")
        claim_lines.append(
            f"claim_index={i} citation=[{idx}] statement={claim.statement}\n"
            f"cited_chunk: {chunk_text or '(missing)'}"
        )

    start = timed()
    output, tokens = invoke_structured(
        chat_model,
        VerificationOutput,
        [
            SystemMessage(content=VERIFIER_PROMPT),
            HumanMessage(content="\n\n".join(claim_lines)),
        ],
    )
    state.verification = list(output.checks)
    state.agent_outputs["verification"] = output.model_dump()
    supported = sum(1 for c in output.checks if c.status == "supported")
    record_step(
        state,
        agent="verification",
        input_data={"claims": len(state.claims)},
        output=output.model_dump(),
        latency_ms=elapsed_ms(start),
        tokens=tokens,
        status="ok" if supported else "fail",
    )


def broaden_subtasks(state: WorkflowState) -> None:
    """Drop domain filters and add the raw user question for one wider pass."""
    broadened = []
    seen = set()
    for subtask in state.subtasks:
        key = (subtask.query.strip().lower(), None)
        if key in seen:
            continue
        seen.add(key)
        broadened.append(SearchSubtask(query=subtask.query, domain=None, label=subtask.label))
    raw = SearchSubtask(query=state.user_query, domain=None, label="broader")
    if (raw.query.strip().lower(), None) not in seen:
        broadened.append(raw)
    state.subtasks = broadened
