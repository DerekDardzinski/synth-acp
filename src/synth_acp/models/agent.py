"""Agent state machine and configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentState(StrEnum):
    """ACP session lifecycle states."""

    UNSTARTED = "unstarted"
    INITIALIZING = "initializing"
    IDLE = "idle"
    CONFIGURING = "configuring"
    BUSY = "busy"
    AWAITING_PERMISSION = "awaiting_permission"
    TERMINATED = "terminated"


TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.UNSTARTED: {AgentState.INITIALIZING, AgentState.TERMINATED},
    AgentState.INITIALIZING: {AgentState.IDLE, AgentState.TERMINATED},
    AgentState.IDLE: {AgentState.BUSY, AgentState.CONFIGURING, AgentState.TERMINATED},
    AgentState.CONFIGURING: {AgentState.IDLE, AgentState.TERMINATED},
    AgentState.BUSY: {
        AgentState.IDLE,
        AgentState.AWAITING_PERMISSION,
        AgentState.TERMINATED,
    },
    AgentState.AWAITING_PERMISSION: {AgentState.BUSY, AgentState.TERMINATED},
    AgentState.TERMINATED: {AgentState.INITIALIZING},
}


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


@dataclass(frozen=True)
class AgentMode:
    """A single mode advertised by an ACP agent after session creation."""

    id: str
    name: str
    description: str | None = None


@dataclass(frozen=True)
class AgentModel:
    """A model advertised by an ACP agent after session creation.

    Populated from the UNSTABLE ``models`` field in NewSessionResponse.
    May be absent even on agents that support modes.
    """

    id: str
    name: str
    description: str | None = None


class AgentConfig(BaseModel, frozen=True):
    """Configuration for a single agent.

    Aligned with the ``launch_agent`` MCP tool parameters:

    - ``agent_id``: unique identifier and display name shown to other agents.
    - ``harness``: short name resolved via the harness registry (e.g. ``kiro``,
      ``claude``, ``opencode``).
    - ``agent_mode``: ACP mode id applied via ``set_session_mode()`` after the
      session is created. Optional — when omitted the agent starts in its default
      mode.
    """

    agent_id: str
    harness: str
    agent_mode: str | None = None
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, data: Any) -> Any:
        """Coerce legacy fields to the new schema.

        - ``id`` → ``agent_id``
        - ``profile`` → ``agent_mode`` (only when ``agent_mode`` absent)
        - ``cmd``, ``binary``, ``args``, ``label``, ``autostart`` are dropped.
        """
        if isinstance(data, dict):
            data = dict(data)
            if "id" in data and "agent_id" not in data:
                data["agent_id"] = data.pop("id")
            else:
                data.pop("id", None)
            if "profile" in data and "agent_mode" not in data:
                data["agent_mode"] = data.pop("profile")
            else:
                data.pop("profile", None)
            for key in ("cmd", "binary", "args", "label", "autostart"):
                data.pop(key, None)
        return data

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent_id matches the allowed identifier pattern."""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                f"Agent ID '{v}' must start with alphanumeric and contain only "
                f"letters, digits, hyphens, underscores"
            )
        return v

    @field_validator("harness")
    @classmethod
    def validate_harness(cls, v: str) -> str:
        """Validate harness is a non-empty string."""
        if not v.strip():
            raise ValueError("harness must not be empty")
        return v

    @property
    def display_name(self) -> str:
        """Human-readable display name shown to other agents."""
        return self.agent_id
