"""Agent state machine and configuration."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Configuration for a single agent.

    Supports both the new ``cmd`` format and legacy ``binary``/``args``.
    Legacy fields are coerced to ``cmd`` via a model validator.
    """

    id: str
    cmd: list[str]
    label: str | None = None
    profile: str | None = None
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, data: Any) -> Any:
        """Coerce legacy ``binary``/``args`` to ``cmd`` and drop ``autostart``."""
        if isinstance(data, dict):
            data = dict(data)
            if "binary" in data and "cmd" not in data:
                binary = data.pop("binary")
                args = data.pop("args", [])
                data["cmd"] = [binary, *args]
            else:
                data.pop("binary", None)
                data.pop("args", None)
            data.pop("autostart", None)
        return data

    @field_validator("id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent ID matches allowed pattern."""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                f"Agent ID '{v}' must start with alphanumeric and contain only "
                f"letters, digits, hyphens, underscores"
            )
        return v

    @field_validator("cmd")
    @classmethod
    def validate_cmd(cls, v: list[str]) -> list[str]:
        """Validate cmd is a non-empty list."""
        if len(v) == 0:
            raise ValueError("cmd must not be empty")
        return v

    @property
    def binary(self) -> str:
        """Return the executable name (first element of ``cmd``)."""
        return self.cmd[0]

    @property
    def args(self) -> list[str]:
        """Return command arguments (all elements after the first)."""
        return self.cmd[1:]

    @property
    def display_name(self) -> str:
        """Return the human-readable display name."""
        return self.label or self.id
