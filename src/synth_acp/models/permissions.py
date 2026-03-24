"""Permission decision types and rule model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class PermissionDecision(StrEnum):
    """Outcome for a persisted permission rule."""

    allow = "allow"
    reject = "reject"


class PermissionRule(BaseModel, frozen=True):
    """A persisted per-agent permission rule keyed on (agent_id, tool_kind)."""

    agent_id: str
    tool_kind: str
    decision: PermissionDecision
