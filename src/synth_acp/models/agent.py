"""Agent state machine and configuration."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, field_validator


class AgentState(StrEnum):
    """ACP session lifecycle states."""

    UNSTARTED = "unstarted"
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    AWAITING_PERMISSION = "awaiting_permission"
    TERMINATED = "terminated"


TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.UNSTARTED: {AgentState.INITIALIZING, AgentState.TERMINATED},
    AgentState.INITIALIZING: {AgentState.IDLE, AgentState.TERMINATED},
    AgentState.IDLE: {AgentState.BUSY, AgentState.TERMINATED},
    AgentState.BUSY: {
        AgentState.IDLE,
        AgentState.AWAITING_PERMISSION,
        AgentState.TERMINATED,
    },
    AgentState.AWAITING_PERMISSION: {AgentState.BUSY, AgentState.TERMINATED},
    AgentState.TERMINATED: set(),
}


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


class AgentConfig(BaseModel, frozen=True):
    """Configuration for a single agent from .synth.json."""

    id: str
    binary: str
    args: list[str] = []
    cwd: str = "."
    autostart: bool = False

    @field_validator("id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                f"Agent ID '{v}' must start with alphanumeric and contain only "
                f"letters, digits, hyphens, underscores"
            )
        return v
