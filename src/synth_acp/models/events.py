"""Broker events emitted to the frontend."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from synth_acp.models.agent import AgentState


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
