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


def permission_rule_key(tool_kind: str, tool_identity: str | None) -> str:
    """The key a permission rule is stored and looked up under.

    The single place rule keys are formed. An MCP tool's kind is usually ``"other"``,
    so keying it on kind would let one "Always allow" cover every other kind-``"other"``
    tool; a known identity keys the rule to that one tool instead.

    Args:
        tool_kind: The tool call's ACP kind.
        tool_identity: ``"<server>/<tool>"`` when known, else None.

    Returns:
        ``"tool:<tool_identity>"`` when tool_identity is set, else tool_kind.
    """
    return f"tool:{tool_identity}" if tool_identity else tool_kind


class PermissionRule(BaseModel, frozen=True):
    """A persisted per-agent permission rule keyed on (agent_id, tool_kind, session_id).

    ``tool_kind`` holds the rule key from ``permission_rule_key``: the tool identity
    when one is known, else the tool kind. The name is kept to match the
    ``rules.tool_kind`` column.
    """

    agent_id: str
    tool_kind: str
    session_id: str
    decision: PermissionDecision
