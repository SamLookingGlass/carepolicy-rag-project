"""Shared workflow state and structured I/O for the multi-agent path."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["lookup", "comparison", "conditions", "out_of_scope"]


class SearchSubtask(BaseModel):
    query: str
    domain: str | None = None
    label: str = ""


class OrchestratorPlan(BaseModel):
    intent: Intent
    subtasks: list[SearchSubtask] = Field(default_factory=list)
    rationale: str = ""


class PolicyClaim(BaseModel):
    statement: str
    citation_index: int = Field(..., ge=1)
    eligibility: str = ""
    conditions: str = ""
    exceptions: str = ""
    side: str = ""


class AnalystOutput(BaseModel):
    claims: list[PolicyClaim] = Field(default_factory=list)


class ComparisonOutput(BaseModel):
    side_a_label: str = "A"
    side_b_label: str = "B"
    claims: list[PolicyClaim] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim_index: int = Field(..., ge=0)
    status: Literal["supported", "unsupported"]
    reason: str = ""


class VerificationOutput(BaseModel):
    checks: list[ClaimCheck] = Field(default_factory=list)


@dataclass
class WorkflowState:
    user_query: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    intent: str = ""
    subtasks: list[SearchSubtask] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    claims: list[PolicyClaim] = field(default_factory=list)
    verification: list[ClaimCheck] = field(default_factory=list)
    remaining_budget: dict[str, int] = field(
        default_factory=lambda: {"retries": 1, "steps": 12}
    )
    trace: list[dict] = field(default_factory=list)
    tokens_used: int = 0
    retry_count: int = 0
    tools: Any = None
