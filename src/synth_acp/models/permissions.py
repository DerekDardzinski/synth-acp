"""Permission decision types and rule model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class PermissionDecision(StrEnum):
    """Outcome for a persisted permission rule.

    Values match ACP ``PermissionOptionKind`` literals.
    """

    allow_once = "allow_once"
    allow_always = "allow_always"
    reject_once = "reject_once"
    reject_always = "reject_always"


class PermissionRule(BaseModel, frozen=True):
    """A persisted per-agent permission rule keyed on (agent_id, tool_kind, session_id)."""

    agent_id: str
    tool_kind: str
    session_id: str
    decision: PermissionDecision
