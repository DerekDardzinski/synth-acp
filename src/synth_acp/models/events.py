"""Broker events emitted to the frontend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from acp.schema import (
    PermissionOption,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
)
from pydantic import BaseModel, Field

from synth_acp.models.agent import AgentMode, AgentModel, AgentState
from synth_acp.models.permissions import PermissionDecision


@dataclass(frozen=True)
class ToolCallDiff:
    """A file edit diff extracted from an ACP tool call update.

    Attributes:
        path: File path the diff applies to.
        old_text: Original text, or None for new files.
        new_text: Replacement text.
    """

    path: str
    old_text: str | None
    new_text: str


@dataclass(frozen=True)
class ToolCallLocation:
    """A file location referenced by a tool call.

    Attributes:
        path: File path.
        line: Line number, or None if unspecified.
    """

    path: str
    line: int | None = None


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
    locations: list[ToolCallLocation] = Field(default_factory=list)
    raw_input: Any = None
    raw_output: Any = None
    diffs: list[ToolCallDiff] = Field(default_factory=list)
    text_content: str | None = None
    terminal_id: str | None = None
    parent_tool_call_id: str | None = None


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


class AgentThoughtReceived(BrokerEvent):
    """A streaming thought/reasoning chunk from an agent."""

    chunk: str


class UsageUpdated(BrokerEvent):
    """Context window and cost snapshot from an agent turn.

    ``cost_amount`` and ``cost_currency`` reflect per-turn values as
    reported by the ACP SDK.  The broker may accumulate these into
    cumulative totals before surfacing to the UI.
    """

    size: int
    used: int
    cost_amount: float | None = None
    cost_currency: str | None = None


class HookFired(BrokerEvent):
    """A lifecycle hook was executed. UI renders as a dim system line."""

    hook_name: str


class MessageSteered(BrokerEvent):
    """An inter-agent message was injected into the agent's running turn.

    A steered message sends no prompt, so no UserPromptSubmitted is emitted for
    it. This event is the only record that it was delivered, and the UI renders
    it inside the turn it arrived in.
    """

    from_agent: str | None
    text: str


class QueueItemSnapshot(BaseModel, frozen=True):
    """Serializable snapshot of a single queued prompt item."""

    id: str
    text: str
    source: Literal["user", "mcp"]
    from_agent: str | None = None
    editing: bool = False


class QueueUpdated(BrokerEvent):
    """Full queue state snapshot for UI reconciliation.

    Emitted after every queue mutation. The UI reconciles its widget
    state against this snapshot.
    """

    items: list[QueueItemSnapshot]


class AgentModesReceived(BrokerEvent):
    """Agent advertised available modes after session creation.

    Emitted once per session, immediately before AgentStateChanged(IDLE).
    Only emitted if the agent's NewSessionResponse includes a modes payload.
    """

    available_modes: list[AgentMode]
    current_mode_id: str


class AgentModeChanged(BrokerEvent):
    """Agent confirmed a mode switch via current_mode_update stream event."""

    mode_id: str


class AgentModelsReceived(BrokerEvent):
    """Agent advertised available models after session creation.

    Only emitted if the agent's NewSessionResponse includes a models payload.
    This capability is marked UNSTABLE in the ACP SDK — many agents will not
    include it. The UI must handle the case where this event never arrives.
    """

    available_models: list[AgentModel]
    current_model_id: str


class AgentModelChanged(BrokerEvent):
    """The agent's active model changed.

    Fired by the session after a successful set_session_model() call.
    There is no ACP push notification for model changes, so this event is
    emitted optimistically by the client immediately after the call returns.
    """

    model_id: str


class ConfigOptionsReceived(BrokerEvent):
    """Emitted once after session creation with native or synthesized config options."""

    config_options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean]


class ConfigOptionChanged(BrokerEvent):
    """Emitted after a successful config option change or from a ConfigOptionUpdate push notification."""

    config_id: str
    value: str | bool


class PlanReceived(BrokerEvent):
    """Agent sent a full plan update."""

    entries: list[Any]


class AvailableCommandsReceived(BrokerEvent):
    """Agent advertised its available slash commands."""

    commands: list[Any]


class TerminalCreated(BrokerEvent):
    """A terminal was created for an agent.

    Attributes:
        terminal_id: Unique identifier for the terminal.
        command: The command that was executed.
        terminal_process: Reference to the TerminalProcess (Any to avoid layer import).
    """

    terminal_id: str
    command: str
    terminal_process: Any


class SessionRestoreComplete(BrokerEvent):
    """Emitted after the broker finishes replaying the event journal for an agent.

    The UI can use this to dismiss loading indicators. The broker has already
    pushed all journaled events into the queue before this event.
    """


class UserPromptSubmitted(BrokerEvent):
    """A user prompt was submitted to an agent.

    Emitted by the broker when it handles SendPrompt, so user messages
    flow through the same event pipeline as agent responses and get
    journaled for session restore.
    """

    text: str


class AgentHandedOff(BrokerEvent):
    """An agent was retired and a successor took over its id.

    Emitted AFTER the database transaction has committed and after all broker-layer
    in-memory state has been re-keyed, and BEFORE the successor's session is started.
    A consumer can therefore rely on the successor's AgentStateChanged(INITIALIZING)
    arriving strictly after this event.  Both are non-chunk events delivered through the
    awaited FIFO put on broker._event_queue, so the relative order is guaranteed; the
    put_nowait drop path applies only to MessageChunkReceived and cannot reorder these.

    A consumer must NOT assume the predecessor's UI state still exists when this
    arrives.  The predecessor's AgentStateChanged(TERMINATED) is emitted EARLIER, during
    task cancellation, and still carries the ORIGINAL id, so a consumer that tears down
    per-agent state on TERMINATED has already done so.  Rebuild from the journal
    instead: the predecessor's ui_events rows have moved, so load_journal(retired_agent_id)
    returns its full transcript and load_journal(agent_id) correctly returns nothing.

    NOT journaled.  Do not add it to broker._JOURNALABLE: on a session restore the
    predecessor is 'inactive' and is not restored at all, while the successor is 'active'
    and restored by the normal path, so no consumer needs to replay this.
    """

    agent_id: str
    """The ORIGINAL id.  After this event it denotes the SUCCESSOR."""
    retired_agent_id: str
    """The predecessor's new suffixed id.  It owns all history authored before the
    handoff, is status 'inactive', and is resurrectable."""
    parent: str | None
    """Unchanged by the handoff.  Predecessor and successor share this parent."""
    task: str
    """Copied from the predecessor to the successor."""


type AgentEvent = (
    AgentStateChanged | MessageChunkReceived | ToolCallUpdated | TurnComplete
    | AgentThoughtReceived | PlanReceived | AvailableCommandsReceived | TerminalCreated
)

type ConfigEvent = (
    AgentModesReceived | AgentModeChanged | AgentModelsReceived | AgentModelChanged
    | ConfigOptionsReceived | ConfigOptionChanged
)

type SystemEvent = (
    BrokerError | PermissionRequested | PermissionAutoResolved
    | UsageUpdated | HookFired | QueueUpdated
    | SessionRestoreComplete | UserPromptSubmitted | MessageSteered
    | AgentHandedOff
)

type BrokerEventUnion = AgentEvent | ConfigEvent | SystemEvent