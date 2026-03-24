"""Broker events emitted to the frontend."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from acp.schema import PermissionOption
from pydantic import BaseModel, Field

from synth_acp.models.agent import AgentState
from synth_acp.models.permissions import PermissionDecision


def _now() -> datetime:
    return datetime.now(UTC)


class BrokerEvent(BaseModel, frozen=True):
    """Base for all events the broker emits."""

    timestamp: datetime = Field(default_factory=_now)
    agent_id: str


class AgentStateChanged(BrokerEvent):
    """Agent transitioned between lifecycle states."""

    old_state: AgentState
    new_state: AgentState


class MessageChunkReceived(BrokerEvent):
    """A streaming text chunk from an agent response."""

    chunk: str


class ToolCallUpdated(BrokerEvent):
    """A tool call started, progressed, or completed."""

    tool_call_id: str
    title: str
    kind: str
    status: str


class BrokerError(BrokerEvent):
    """Non-fatal error surfaced to the UI."""

    message: str
    severity: Literal["warning", "error"] = "error"


class PermissionRequested(BrokerEvent):
    """Agent is blocked waiting for a permission decision."""

    request_id: str
    title: str
    kind: str
    options: list[PermissionOption]


class PermissionAutoResolved(BrokerEvent):
    """A permission was auto-resolved by a persisted rule."""

    request_id: str
    decision: PermissionDecision


class TurnComplete(BrokerEvent):
    """Agent finished processing a prompt."""

    stop_reason: str


class McpMessageDelivered(BrokerEvent):
    """An inter-agent message was delivered via the poller."""

    from_agent: str
    to_agent: str
