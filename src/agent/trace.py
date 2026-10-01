"""Record and format multi-agent traces (agent, tool, latency, tokens, retry)."""

from __future__ import annotations

import time
from typing import Any

from src.agent.state import WorkflowState


def extract_tokens(raw: Any) -> int:
    meta = getattr(raw, "usage_metadata", None)
    if isinstance(meta, dict):
        return int(meta.get("total_tokens") or 0)
    if hasattr(raw, "response_metadata"):
        usage = (raw.response_metadata or {}).get("token_usage") or {}
        total = usage.get("total_tokens")
        if total:
            return int(total)
    return 0


def coerce_model(raw: Any, schema: type):
    if isinstance(raw, schema):
        return raw
    if hasattr(raw, "model_dump"):
        return schema(**raw.model_dump())
    if isinstance(raw, dict):
        if raw.get("parsed") is not None and not any(
            k in raw for k in schema.model_fields
        ):
            return coerce_model(raw["parsed"], schema)
        return schema(**{k: v for k, v in raw.items() if k in schema.model_fields})
    raise TypeError(f"Unexpected structured output type: {type(raw)!r}")


def invoke_structured(model: Any, schema: type, messages: list) -> tuple[Any, int]:
    try:
        structured = model.with_structured_output(schema, include_raw=True)
    except TypeError:
        structured = model.with_structured_output(schema)
    raw = structured.invoke(messages)
    tokens = 0
    parsed = raw
    if isinstance(raw, dict) and "parsed" in raw:
        tokens = extract_tokens(raw.get("raw"))
        parsed = raw.get("parsed")
        if parsed is None:
            raise TypeError("structured output parse failed")
    else:
        tokens = extract_tokens(raw)
    return coerce_model(parsed, schema), tokens


def record_step(
    state: WorkflowState,
    *,
    agent: str,
    tool: str = "",
    input_data: dict | None = None,
    output: Any = None,
    latency_ms: float = 0.0,
    tokens: int = 0,
    status: str = "ok",
    retry: bool = False,
) -> dict:
    payload = {
        "request_id": state.request_id,
        "step_number": len(state.trace) + 1,
        "agent": agent,
        "tool": tool,
        "arguments": dict(input_data or {}),
        "observation": output,
        "input": dict(input_data or {}),
        "output": output,
        "latency_ms": round(latency_ms, 2),
        "tokens": int(tokens),
        "status": status,
        "retry": retry,
    }
    state.trace.append(payload)
    state.tokens_used += int(tokens)
    if state.remaining_budget.get("steps", 0) > 0:
        state.remaining_budget["steps"] -= 1
    return payload


def timed() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def format_trace(state: WorkflowState) -> str:
    lines = ["User"]
    last_agent = ""
    for step in state.trace:
        agent = step.get("agent") or ""
        tool = step.get("tool") or ""
        ms = float(step.get("latency_ms") or 0)
        retry = " retry" if step.get("retry") else ""
        status = step.get("status") or "ok"
        mark = "✓" if status == "ok" else "✗"
        if tool:
            lines.append(f" ├─ {tool} {ms:.0f}ms {mark}{retry}")
            continue
        if agent != last_agent:
            lines.append(" ↓")
            extra = f" ({state.intent})" if agent == "orchestrator" and state.intent else ""
            lines.append(f"{agent}{extra} {ms:.0f}ms{retry}")
            last_agent = agent
        else:
            lines.append(f" ├─ {agent} {ms:.0f}ms {mark}{retry}")
    lines.append(" ↓")
    lines.append("Final Answer")
    return "\n".join(lines)


def tool_names(trace: list[dict]) -> list[str]:
    return [s["tool"] for s in trace if s.get("tool")]
