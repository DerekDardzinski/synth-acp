"""Normalized values that per-harness ACP extensions hand back to synth.

An extension (``synth_acp.acp.extensions``) translates one harness's proprietary
protocol into these types.  They live in ``models`` so the broker and UI can read them
without importing any extension module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExtensionCapabilities(BaseModel, frozen=True):
    """What synth may rely on for one harness engine. Read by ACPSession, broker and UI.

    Attributes:
        steer: In-turn steering is supported.
        profile_switch: How agent_mode is changed: ``"mode"`` in place via
            set_session_mode (Kiro), or ``"handoff"`` by an agent-written handoff to a
            fresh session started as the new profile (Claude, whose main-thread agent is
            fixed at session creation), or ``"config_option"`` through the native
            ``mode``-category config option, which ACPSession also uses to apply
            agent_mode after session creation and restore (OpenCode, which advertises
            its agents only as that option's values, with no legacy ``modes``).
        reload_after_mode_switch: A mode switch drops MCP servers, so reload them via
            load_session (Kiro v2).
        no_replay_load: ``session_meta("load")`` suppresses the agent's history replay.
    """

    steer: bool = False
    profile_switch: Literal["mode", "handoff", "config_option"] = "mode"
    reload_after_mode_switch: bool = False
    no_replay_load: bool = False


class ContextUsage(BaseModel, frozen=True):
    """Context-window fullness reported through a proprietary channel."""

    used: int
    size: int
    cost_amount: float | None = None
    cost_currency: str | None = None


class ProfileUnavailable(BaseModel, frozen=True):
    """The requested agent profile cannot be used; the harness falls back."""

    requested: str
    fallback: str | None
    reason: str


class TurnActivity(BaseModel, frozen=True):
    """The harness reports a turn it is running started (active=True) or ended (active=False), whether or not synth started it."""

    active: bool


Signal = ContextUsage | ProfileUnavailable | TurnActivity
"""Closed union: a new kind needs a new model plus its mapping in ACPSession."""


class UpdateHints(BaseModel, frozen=True):
    """Proprietary data an extension read from one session/update payload."""

    parent_tool_call_id: str | None = None
    is_replay: bool = False
    signals: tuple[Signal, ...] = ()


class PermissionHints(BaseModel, frozen=True):
    """Proprietary data an extension read from one session/request_permission.

    Attributes:
        tool_identity: ``"<server>/<tool>"`` for an MCP tool; None when the extension
            cannot tell.
    """

    tool_identity: str | None = None


class SteerOutcome(BaseModel, frozen=True):
    """Result of one attempt to deliver text into a running turn."""

    status: Literal[
        "delivered", "queued", "dropped", "prompt_required", "unsupported", "failed", "started_turn"
    ]
    message_id: str | None = None

    @property
    def accepted(self) -> bool:
        """True for 'delivered', 'queued' and 'started_turn': the harness owns delivery, so synth must NOT also enqueue the text.

        ``started_turn`` means the harness found no running turn and started one of its
        own with the text; ACPSession keeps the agent BUSY until that turn ends.
        """
        return self.status in ("delivered", "queued", "started_turn")
