"""SynthApp — Textual TUI bridging the ACPBroker to the terminal."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sqlite3
from typing import ClassVar, NamedTuple

from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer, TextArea
from textual.worker import Worker, WorkerState

from synth_acp.broker.broker import ACPBroker
from synth_acp.db import (
    configure_connection,
    get_unembedded_agents_sync,
    store_embedding_sync,
)
from synth_acp.diagnostics import DiagnosticsHandle, start_diagnostics
from synth_acp.embeddings import EmbeddingEngine, embedding_available
from synth_acp.models.agent import AgentConfig, AgentState, css_id
from synth_acp.models.commands import (
    CommitQueueEdit,
    DeleteQueueItem,
    EditQueueItem,
    LaunchAgent,
    RespondPermission,
    TerminateAgent,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentHandedOff,
    AgentStateChanged,
    AgentThoughtReceived,
    AvailableCommandsReceived,
    BrokerError,
    BrokerEvent,
    ConfigOptionChanged,
    ConfigOptionsReceived,
    HookFired,
    MessageChunkReceived,
    MessageSteered,
    PermissionRequested,
    PlanReceived,
    QueueUpdated,
    SessionRestoreComplete,
    TerminalCreated,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
    UserPromptSubmitted,
)
from synth_acp.runtime_gc import GCFreezeManager
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.screens.help import HelpScreen
from synth_acp.ui.screens.launch import LaunchAgentScreen
from synth_acp.ui.screens.permission import PermissionBar
from synth_acp.ui.screens.session_picker import SessionPickerScreen
from synth_acp.ui.widgets.agent_list import AgentList, AgentTile
from synth_acp.ui.widgets.conversation import (
    ConversationFeed,
    FirstPaintWindow,
    first_paint_window,
)
from synth_acp.ui.widgets.diff_view import prewarm_highlighting
from synth_acp.ui.widgets.gradient_bar import ActivityBar
from synth_acp.ui.widgets.input_bar import InputBar

_DISABLED_STATES = {AgentState.AWAITING_PERMISSION}

_BUSY_INPUT_STATES = frozenset(
    {
        AgentState.INITIALIZING,
        AgentState.BUSY,
        AgentState.CONFIGURING,
        AgentState.AWAITING_PERMISSION,
    }
)
"""States for which the input bar shows the cancel button and its activity bar.

Shared by the live ``AgentStateChanged`` route and the feed-creation tail so the two
cannot disagree.  They previously did, in opposite directions, and the disagreement
was reachable: the live route listed neither IDLE/TERMINATED nor busy for
AWAITING_PERMISSION, so it called nothing and the bar kept whatever it had, while the
tail used if/else and so treated AWAITING_PERMISSION as idle.  A feed created while a
permission was pending therefore showed no cancel button and no activity bar while the
agent was genuinely not IDLE -- so a prompt typed into it was correctly queued by the
broker, which reads like the queue misbehaving.
"""

_RENDERABLE_EVENTS = (
    MessageChunkReceived,
    AgentThoughtReceived,
    ToolCallUpdated,
    TurnComplete,
    PlanReceived,
    HookFired,
    UserPromptSubmitted,
    MessageSteered,
)

log = logging.getLogger(__name__)


def _coalesce_events(events: list[BrokerEvent]) -> list[BrokerEvent]:
    """Merge consecutive MessageChunkReceived/AgentThoughtReceived events.

    Consecutive events of the same type and agent_id are collapsed into a
    single event with concatenated chunks.  All other event types pass
    through unchanged.

    Args:
        events: Raw event buffer to coalesce.

    Returns:
        New list with consecutive streamable events merged.
    """
    if not events:
        return []
    result: list[BrokerEvent] = []
    for event in events:
        if (
            isinstance(event, (MessageChunkReceived, AgentThoughtReceived))
            and result
            and type(result[-1]) is type(event)
            and result[-1].agent_id == event.agent_id
        ):
            prev = result[-1]
            assert isinstance(prev, (MessageChunkReceived, AgentThoughtReceived))
            result[-1] = prev.model_copy(update={"chunk": prev.chunk + event.chunk})
        else:
            result.append(event)
    return result


class DynamicAgentInfo(NamedTuple):
    """Metadata for a dynamically launched agent.

    Attributes:
        parent: Agent ID of the parent, or None.
        task: Task description.
        harness: Harness short name.
        cwd: Working directory.
    """

    parent: str | None
    task: str
    harness: str
    cwd: str = ""


class SynthApp(App):
    """Textual TUI for SYNTH multi-agent orchestration."""

    TITLE = "SYNTH"
    THEME = "catppuccin-mocha"
    CSS_PATH = "css/app.tcss"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "next_agent", "Next agent", show=False),
        Binding("l", "launch", "Launch agent"),
        Binding("ctrl+r", "restore", "Restore session"),
        Binding("f1", "help", "Help"),
    ]

    selected_agent: reactive[str] = reactive("")
    selected_thread: reactive[str] = reactive("")

    def __init__(
        self,
        broker: ACPBroker,
        config: SessionConfig,
        initial_agent: AgentConfig | None = None,
        css_path: str | None = None,
        restore: bool = False,
    ) -> None:
        if css_path:
            self.CSS_PATH = css_path
        super().__init__()
        self.broker = broker
        self.config = config
        self._initial_agent = initial_agent or broker._initial_agent
        self._restore_mode = restore
        self._event_buffers: dict[str, list[BrokerEvent]] = {}
        self._panels: dict[str, ConversationFeed] = {}
        self._agent_states: dict[str, AgentState] = {}
        self._dynamic_agents: dict[str, DynamicAgentInfo] = {}
        self._agent_config_options: dict[str, list] = {}
        self._tiles: dict[str, AgentTile] = {}
        self._selecting: dict[str, asyncio.Task[None]] = {}
        self._draining: dict[str, asyncio.Event] = {}
        self._indexing_complete: bool = False
        self._embedding_engine: EmbeddingEngine | None = None
        self._auto_selected_from: str | None = None
        """Which id an AUTOMATIC selection move moved away from, if any.

        A handoff cannot ask "is the original id still selected?": the predecessor's
        TERMINATED event has already moved the selection to the first other live agent,
        or to "", so that check fails whenever any other agent is live. This records the
        move so the successor can take the selection back -- and is cleared on any
        EXPLICIT user selection, so focus is never stolen from a deliberate choice.
        """
        self._diagnostics: DiagnosticsHandle | None = None
        self._gc_freeze: GCFreezeManager | None = None

    def _handle_exception(self, error: Exception) -> None:
        """Log unhandled exceptions to file before Textual's default handling."""
        import logging

        logging.getLogger("synth_acp.ui.app").error("Textual unhandled exception", exc_info=error)
        super()._handle_exception(error)

    def compose(self) -> ComposeResult:
        """Build the main layout with sidebar and footer."""
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield AgentList([])
            yield ContentSwitcher(id="right")
        yield Footer()

    def _is_composing(self, agent_id: str) -> bool:
        """Check if user is composing in the text area for an agent.

        Called by the broker at delivery decision time — reads actual UI state.
        """
        feed = self._panels.get(agent_id)
        if not feed or not feed.input_bar:
            return False
        return feed.input_bar.is_composing

    async def on_mount(self) -> None:
        """Launch all agents, select the first, and start the broker event consumer."""
        self.theme = "catppuccin-mocha"
        self._start_runtime_subsystems()
        self.broker.set_composing_check(self._is_composing)
        initial = self._initial_agent
        self._event_buffers[initial.agent_id] = []
        if self._restore_mode:
            self.run_worker(
                self._consume_broker_events(),
                exit_on_error=False,
                name="broker-consumer",
                group="broker",
            )
            self._do_restore(from_startup=True)
        else:
            await self.broker.handle(LaunchAgent(agent_id=initial.agent_id, config=initial))
            await self.select_agent(initial.agent_id)
            self.run_worker(
                self._consume_broker_events(),
                exit_on_error=False,
                name="broker-consumer",
                group="broker",
            )
        self._index_sessions()

    def _start_runtime_subsystems(self) -> None:
        """Start the GC freeze worker and, when enabled, the diagnostics workers.

        The freeze worker is UNCONDITIONAL: it is a production fix for the
        stop-the-world gen-2 pause, not instrumentation, so it must not be gated on
        SYNTH_DIAG. Diagnostics gate themselves inside ``start_diagnostics``.
        """
        self._gc_freeze = GCFreezeManager()
        # Callable, not a coroutine: a worker cancelled before its task starts then
        # leaves nothing un-awaited.
        self.run_worker(self._gc_freeze.run, name="gc-freeze", group="gc", exit_on_error=False)
        self._diagnostics = start_diagnostics(self)
        self._prewarm_highlighting()

    @work(thread=True, group="prewarm")
    def _prewarm_highlighting(self) -> None:
        """Pre-warm the syntax highlighter OFF the loop thread.

        Its own worker group, and never awaited, so startup does not wait on it. Measured in
        a fresh interpreter, the first highlight costs 458.9 ms against 4.5 ms once warmed —
        running this on the loop thread would simply move that stall into startup.
        """
        prewarm_highlighting()

    def _stop_runtime_subsystems(self) -> None:
        """Stop diagnostics and the GC freeze worker.

        The diagnostics ``gc.callbacks`` hook is process-global, so a recorder left
        installed survives repeated app instances in one process, retains state,
        duplicates measurements, and changes the callback baseline later runs see.
        """
        if self._diagnostics is not None:
            self._diagnostics.stop()
            self._diagnostics = None
        with contextlib.suppress(Exception):
            self.workers.cancel_group(self, "gc")
        self._gc_freeze = None

    async def _consume_broker_events(self) -> None:
        """Consume broker events and post them as Textual messages."""
        async for event in self.broker.events():
            self.post_message(BrokerEventMessage(event))

    @work(thread=True, group="indexer")
    def _index_sessions(self) -> None:
        """Background worker: embed all un-embedded sessions.

        Runs on mount. Fire-and-forget — if app exits, no harm.
        Sets self._indexing_complete = True when done.
        """
        self._do_index_sessions()

    def _do_index_sessions(self) -> None:
        """Index session logic, separated for testability."""
        try:
            from textual.worker import NoActiveWorker, get_current_worker

            if not embedding_available():
                return
            try:
                worker = get_current_worker()
            except NoActiveWorker:
                worker = None
            engine = EmbeddingEngine()
            self._embedding_engine = engine
            engine.ensure_model()
            db_path = str(self.broker._db_path)
            with contextlib.closing(sqlite3.connect(db_path)) as conn:
                configure_connection(conn)
                conn.row_factory = sqlite3.Row
                pairs = get_unembedded_agents_sync(conn)
                for sid, agent_id in pairs:
                    if worker is not None and worker.is_cancelled:
                        return
                    text = self._query_agent_text(conn, sid, agent_id)
                    if text is None:
                        store_embedding_sync(conn, sid, agent_id, "", b"")
                    else:
                        text_h = hashlib.sha256(text.encode()).hexdigest()
                        embedding = engine.embed(text)
                        store_embedding_sync(conn, sid, agent_id, text_h, embedding.tobytes())
            self._indexing_complete = True
        except Exception:
            log.debug("Background indexing failed", exc_info=True)

    @staticmethod
    def _query_agent_text(conn: sqlite3.Connection, session_id: str, agent_id: str) -> str | None:
        """Get the first inbound message text for an agent.

        Returns None if no text found or text < 20 chars.
        """
        row = conn.execute(
            "SELECT payload FROM ui_events "
            "WHERE session_id = ? AND agent_id = ? "
            "AND event_type IN ('UserPromptSubmitted', 'InitialPromptDelivered') "
            "ORDER BY seq LIMIT 1",
            (session_id, agent_id),
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["payload"])
            text = data.get("text")
        except (json.JSONDecodeError, TypeError):
            return None
        if not text or len(text) < 20:
            return None
        return text

    async def on_broker_event_message(self, message: BrokerEventMessage) -> None:
        """Route broker events to the appropriate widgets.

        Args:
            message: Wrapped broker event.
        """
        event = message.event

        # Handle queue state updates — forward to prompt queue widget and drain button
        if isinstance(event, QueueUpdated):
            if event.agent_id in self._panels:
                feed = self._panels[event.agent_id]
                if feed.input_bar:
                    from synth_acp.ui.widgets.prompt_queue import PromptQueue as PQWidget

                    try:
                        pq = feed.input_bar.query_one(PQWidget)
                        pq.reconcile(event.items)
                    except Exception:
                        log.debug(
                            "PromptQueue widget not found for %s", event.agent_id, exc_info=True
                        )
                    # Show drain button when queue has items and agent is idle
                    has_items = bool(event.items)
                    is_idle = self._agent_states.get(event.agent_id) == AgentState.IDLE
                    feed.input_bar.set_drain_visible(
                        has_items and is_idle and not feed.input_bar._busy
                    )
            return

        # Handle session history restore — render static snapshot into the feed.
        if isinstance(event, SessionRestoreComplete):
            if event.agent_id in self._panels:
                feed = self._panels[event.agent_id]
                if feed.input_bar is not None:
                    feed.input_bar.set_busy(False)
            return

        if isinstance(event, AgentHandedOff):
            await self._handle_handed_off(event)
            return

        # Buffer events for agents without panels or during drain
        if event.agent_id not in self._panels or event.agent_id in self._draining:
            if event.agent_id not in self._event_buffers:
                self._event_buffers[event.agent_id] = []
            self._event_buffers[event.agent_id].append(event)

        if isinstance(event, AgentStateChanged):
            prev_state = self._agent_states.get(event.agent_id)
            self._agent_states[event.agent_id] = event.new_state
            # Resurrection: agent was TERMINATED in the UI but a new session
            # started (UNSTARTED → INITIALIZING). Reload journal so the
            # previous conversation is visible, and re-create the tile.
            if prev_state == AgentState.TERMINATED and event.new_state == AgentState.INITIALIZING:
                aid = event.agent_id
                self._event_buffers.setdefault(aid, [])
                try:
                    journal = await self.broker.load_journal(aid, self.broker.session_id)
                    self._event_buffers[aid] = journal + self._event_buffers[aid]
                except Exception:
                    log.debug("Failed to load journal for resurrected agent %s", aid, exc_info=True)
                if aid not in self._tiles:
                    try:
                        agent_list = self.query_one(AgentList)
                        parent = self.broker.get_agent_parent(aid)
                        tile = agent_list.add_agent_tile(aid, parent=parent)
                        self._tiles[aid] = tile
                    except Exception:
                        log.debug("Failed to re-create tile for %s", aid, exc_info=True)
                # Re-register dynamic agent info so select_agent can resolve harness/cwd
                if aid not in self._dynamic_agents:
                    parent = self.broker.get_agent_parent(aid)
                    harness = self.broker.get_agent_harness(aid)
                    self._dynamic_agents[aid] = DynamicAgentInfo(
                        parent=parent, task="", harness=harness, cwd=self.broker.get_agent_cwd(aid)
                    )
            elif event.agent_id not in self._dynamic_agents:
                parent = self.broker.get_agent_parent(event.agent_id)
                harness = self.broker.get_agent_harness(event.agent_id)
                self._dynamic_agents[event.agent_id] = DynamicAgentInfo(
                    parent=parent,
                    task="",
                    harness=harness,
                    cwd=self.broker.get_agent_cwd(event.agent_id),
                )
                self._event_buffers.setdefault(event.agent_id, [])
                try:
                    agent_list = self.query_one(AgentList)
                    tile = agent_list.add_agent_tile(event.agent_id, parent=parent)
                    self._tiles[event.agent_id] = tile
                except Exception:
                    self.log.error(f"Failed to add tile for {event.agent_id}", exc_info=True)
            tile = self._tiles.get(event.agent_id)
            if tile is not None:
                if event.new_state == AgentState.TERMINATED:
                    self._agent_config_options.pop(event.agent_id, None)
                    self._tiles.pop(event.agent_id, None)
                    tile.remove()
                else:
                    tile.update_state(event.new_state)
            # A successor claims the selection its predecessor lost, once it actually
            # exists. Deferred to here rather than done when AgentHandedOff arrived,
            # because selecting an id creates its panel and the original id must be absent
            # from every per-agent dict until the successor mounts. The marker is cleared
            # by any explicit user selection, so this never steals focus.
            if (
                event.new_state == AgentState.INITIALIZING
                and self._auto_selected_from == event.agent_id
            ):
                self._auto_selected_from = None
                await self.select_agent(event.agent_id)
            if event.new_state == AgentState.TERMINATED:
                feed = self._panels.pop(event.agent_id, None)
                if feed is not None:
                    await feed.remove()
                self._event_buffers.pop(event.agent_id, None)
                self._dynamic_agents.pop(event.agent_id, None)
                if self.selected_agent == event.agent_id:
                    live = [
                        aid for aid, st in self._agent_states.items() if st != AgentState.TERMINATED
                    ]
                    # Record what we are moving AWAY from before moving, so a handoff can
                    # put the user back on the id that is about to continue the work.
                    #
                    # ONLY during a handoff.  Armed on every termination of the selected
                    # agent it has no expiry, so a later INITIALIZING for the same id --
                    # an ordinary resurrect_agent, or a launch reusing a terminated id --
                    # would yank the selection out from under the user from inside an event
                    # handler.  The reservation is NOT usable as the predicate here: it is
                    # installed after the predecessor is killed, so it does not exist yet
                    # when this terminal event arrives.
                    if self.broker.handoff_in_flight(event.agent_id):
                        self._auto_selected_from = event.agent_id
                    if live:
                        await self.select_agent(live[0], explicit=False)
                    else:
                        self.selected_agent = ""

        if isinstance(event, ConfigOptionsReceived):
            self._agent_config_options[event.agent_id] = list(event.config_options)
            agent_opt = self._synthesize_agent_option(event.agent_id)
            if agent_opt is not None:
                self._agent_config_options[event.agent_id].insert(0, agent_opt)
            self._update_tile_mode_from_config(event.agent_id)
            self._update_input_bar_config_options(event.agent_id)

        if isinstance(event, ConfigOptionChanged):
            options = self._agent_config_options.get(event.agent_id, [])
            for i, opt in enumerate(options):
                if opt.id == event.config_id and hasattr(opt, "current_value"):
                    options[i] = opt.model_copy(update={"current_value": event.value})
                    break
            if isinstance(event.value, str):
                bar = self._get_input_bar(event.agent_id)
                if bar:
                    bar.update_config_option_value(event.config_id, event.value)
            # Update tile mode if the changed option is the mode category
            opt_match = next((o for o in options if o.id == event.config_id), None)
            if opt_match and getattr(opt_match, "category", None) == "mode":
                self._update_tile_mode_from_config(event.agent_id)

        if isinstance(event, UsageUpdated):
            self._update_usage_display(event)

        # Route to the agent's panel if it exists and not draining
        if event.agent_id in self._panels and event.agent_id not in self._draining:
            feed = self._panels[event.agent_id]
            await self._route_event_to_feed(feed, event)

            # Update InputBar disable state on state changes
            if isinstance(event, AgentStateChanged):
                self._update_input_bar_state(event.agent_id, event.new_state)
        elif isinstance(event, BrokerError):
            self.notify(event.message, severity=event.severity)

    async def _route_event_to_feed(self, feed: ConversationFeed, event: BrokerEvent) -> None:
        """Route a single LIVE event to a conversation feed, atomically.

        This is the steady-state broker route, the one that runs while an agent is streaming.
        The lock covers RECORDING AND RENDERING TOGETHER for the same reason it does in
        `_replay_event`: recording is synchronous and happens first, so with the lock taken
        only inside the feed's render methods a live event recorded against a replay's
        suppressed `_late_window_turn` and then rendered into the restored tail — the recorded
        batch and the DOM diverged silently, which AC3 exists to forbid.

        Every branch below is inside the lock, not just the chunk ones. `ConversationFeed`
        also asserts the invariant at the recording step, so a future branch added outside the
        lock fails at the point of the bug instead of diverging quietly.

        Args:
            feed: Target conversation feed.
            event: The broker event to route.
        """
        async with feed.exclusive_render():
            await self._route_event_to_feed_locked(feed, event)

    async def _route_event_to_feed_locked(
        self, feed: ConversationFeed, event: BrokerEvent
    ) -> None:
        """Body of _route_event_to_feed; runs holding the feed's render lock."""
        if isinstance(event, _RENDERABLE_EVENTS):
            feed.record_event(event)
        if isinstance(event, PermissionRequested):
            self._mount_permission_bar(feed, event)
            return
        if isinstance(event, MessageChunkReceived):
            await feed.add_chunk(event.chunk)
        elif isinstance(event, AgentThoughtReceived):
            await feed.add_thought_chunk(event.chunk)
        elif isinstance(event, ToolCallUpdated):
            await feed.add_tool_call(
                event.tool_call_id,
                event.title,
                event.kind,
                event.status,
                locations=event.locations,
                raw_input=event.raw_input,
                raw_output=event.raw_output,
                diffs=event.diffs,
                text_content=event.text_content,
                terminal_id=event.terminal_id,
                parent_tool_call_id=event.parent_tool_call_id,
            )
        elif isinstance(event, TerminalCreated):
            await feed.mount_terminal(event.terminal_id, event.terminal_process)
        elif isinstance(event, TurnComplete):
            await feed.finalize_current_message()
            if feed.input_bar is not None:
                feed.input_bar.set_busy(False)
        elif isinstance(event, PlanReceived):
            await feed.update_plan(event.entries)
        elif isinstance(event, AvailableCommandsReceived):
            if feed.input_bar is not None:
                feed.input_bar.update_slash_commands(event.commands)
        elif isinstance(event, HookFired):
            await feed.add_hook_notification(event.hook_name)
        elif isinstance(event, MessageSteered):
            await feed.add_steered_message(event.from_agent, event.text)
        elif isinstance(event, UserPromptSubmitted):
            await feed.add_prompt(event.text)
        elif isinstance(event, BrokerError):
            self.notify(event.message, severity=event.severity)
        elif isinstance(event, AgentStateChanged):
            if feed.input_bar is None:
                return
            feed.input_bar.set_busy(event.new_state in _BUSY_INPUT_STATES)
            if event.new_state == AgentState.IDLE and event.agent_id == self.selected_agent:
                feed.input_bar.query_one("#prompt-input").focus()

    def _mount_permission_bar(self, feed: ConversationFeed, event: PermissionRequested) -> None:
        """Mount a PermissionBar at the top of the InputBar.

        Args:
            feed: Target conversation feed.
            event: The permission request event.
        """
        position = self.broker.permission_position(event.agent_id)
        bar = PermissionBar(
            event.agent_id, event.request_id, event.title, event.options, position=position
        )
        if feed.input_bar is not None:
            feed.input_bar.mount(bar, before=0)
        else:
            feed.mount(bar)

    async def on_permission_bar_resolved(self, message: PermissionBar.Resolved) -> None:
        """Handle the resolved permission from the PermissionBar.

        Args:
            message: The resolved permission message.
        """
        if message.option_id:
            await self.broker.handle(
                RespondPermission(
                    agent_id=message.agent_id,
                    request_id=message.request_id,
                    option_id=message.option_id,
                )
            )

    def on_prompt_queue_edit_requested(self, message: object) -> None:
        """Forward edit request from PromptQueue widget to broker."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        if not isinstance(message, PromptQueue.EditRequested):
            return
        if self.selected_agent:
            self.run_worker(
                self.broker.handle(
                    EditQueueItem(agent_id=self.selected_agent, item_id=message.item_id)
                )
            )

    def on_prompt_queue_edit_committed(self, message: object) -> None:
        """Forward edit commit from PromptQueue widget to broker."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        if not isinstance(message, PromptQueue.EditCommitted):
            return
        if self.selected_agent:
            self.run_worker(
                self.broker.handle(
                    CommitQueueEdit(
                        agent_id=self.selected_agent, item_id=message.item_id, text=message.text
                    )
                )
            )

    def on_prompt_queue_delete_requested(self, message: object) -> None:
        """Forward delete request from PromptQueue widget to broker."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        if not isinstance(message, PromptQueue.DeleteRequested):
            return
        if self.selected_agent:
            self.run_worker(
                self.broker.handle(
                    DeleteQueueItem(agent_id=self.selected_agent, item_id=message.item_id)
                )
            )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Trigger queue drain when text area becomes empty (user stopped composing)."""
        if event.text_area.id != "prompt-input":
            return
        if event.text_area.text.strip():
            return  # Still composing — nothing to do
        # Text is empty — user may have deleted content or submitted.
        # Try draining any queued MCP messages for this agent.
        agent_id: str | None = None
        for ancestor in event.text_area.ancestors_with_self:
            if isinstance(ancestor, InputBar):
                agent_id = ancestor._agent_id
                break
        if agent_id:
            from synth_acp.models.commands import ReleaseQueue

            self.run_worker(self.broker.handle(ReleaseQueue(agent_id=agent_id)))

    async def on_agent_tile_terminate_clicked(self, message: AgentTile.TerminateClicked) -> None:
        """Handle the close button on an agent tile."""
        await self.broker.handle(TerminateAgent(agent_id=message.agent_id))

    def _record_only(self, feed: ConversationFeed, event: BrokerEvent) -> None:
        """Record a buffered event into its turn batch WITHOUT constructing widgets.

        The skipped-turn counterpart of ``_replay_event``, used by the first-selection
        drain for turns older than the mounted tail. Doing the recording but not the
        rendering is what keeps full scrollback intact while first paint stays cheap:
        ``record_event`` is a single list append, and ``close_turn_batch`` is another.

        THE RENDERABLE FILTER IS DELIBERATE AND NORMATIVE. It is the same
        ``_RENDERABLE_EVENTS`` check ``_replay_event`` applies, so the recorded stream is
        byte-identical to an unwindowed drain. Recording non-renderables instead would be a
        defect: ``_turn_events`` is consumed by ``_restore_turns``, which replays batches
        through ``replay_event``, and that silently ignores non-renderables — so they would
        sit in the batches unrenderable by any scroll-up, and the recorded stream would
        change for the MOUNTED region too.

        Two non-renderable side effects ARE still applied, because both are cheap, mount no
        widgets, and are emitted EARLY enough to land in the skipped region on almost every
        restore — dropping them would silently lose slash commands and leave the input bar
        stuck busy.

        Args:
            feed: Target conversation feed.
            event: The buffered broker event to record.
        """
        if isinstance(event, _RENDERABLE_EVENTS):
            feed.record_event(event)
        if isinstance(event, TurnComplete):
            feed.close_turn_batch()
        elif isinstance(event, AvailableCommandsReceived):
            if feed.input_bar is not None:
                feed.input_bar.update_slash_commands(event.commands)
        elif isinstance(event, SessionRestoreComplete):
            if feed.input_bar is not None:
                feed.input_bar.set_busy(False)

    async def _replay_event(self, feed: ConversationFeed, event: BrokerEvent) -> None:
        """Record and render one event ATOMICALLY with respect to a historical replay.

        The lock is taken HERE, around recording and rendering together, not inside the feed's
        render methods alone. `record_event` is synchronous and runs before the render is
        awaited, so with the lock held only further down, a live event's recording landed in
        `_current_turn_events` while a restore had `_late_window_turn` suppressed — the
        recorded batch and the rendered DOM then disagreed, silently, with no error. The lock
        is reentrant per task, so the feed's own wrappers underneath skip it.
        """
        async with feed.exclusive_render():
            await self._replay_event_locked(feed, event)

    async def _replay_event_locked(
        self, feed: ConversationFeed, event: BrokerEvent
    ) -> None:
        """Replay a buffered event to a conversation feed during drain.

        Skips BrokerError and PermissionAutoResolved events. AgentStateChanged
        updates are already tracked in _agent_states.

        Args:
            feed: Target conversation feed.
            event: The buffered broker event to replay.
        """
        if isinstance(event, _RENDERABLE_EVENTS):
            feed.record_event(event)
        if isinstance(event, MessageChunkReceived):
            await feed.add_chunk(event.chunk)
        elif isinstance(event, AgentThoughtReceived):
            await feed.add_thought_chunk(event.chunk)
        elif isinstance(event, ToolCallUpdated):
            await feed.add_tool_call(
                event.tool_call_id,
                event.title,
                event.kind,
                event.status,
                locations=event.locations,
                raw_input=event.raw_input,
                raw_output=event.raw_output,
                diffs=event.diffs,
                text_content=event.text_content,
                terminal_id=event.terminal_id,
                parent_tool_call_id=event.parent_tool_call_id,
            )
        elif isinstance(event, TerminalCreated):
            await feed.mount_terminal(event.terminal_id, event.terminal_process)
        elif isinstance(event, PermissionRequested):
            if self.broker.is_permission_pending(event.agent_id):
                self._mount_permission_bar(feed, event)
        elif isinstance(event, TurnComplete):
            await feed.finalize_current_message()
        elif isinstance(event, PlanReceived):
            await feed.update_plan(event.entries)
        elif isinstance(event, AvailableCommandsReceived):
            if feed.input_bar is not None:
                feed.input_bar.update_slash_commands(event.commands)
        elif isinstance(event, HookFired):
            await feed.add_hook_notification(event.hook_name)
        elif isinstance(event, MessageSteered):
            await feed.add_steered_message(event.from_agent, event.text)
        elif isinstance(event, UserPromptSubmitted):
            await feed.add_prompt(event.text)
        elif isinstance(event, SessionRestoreComplete):
            if feed.input_bar is not None:
                feed.input_bar.set_busy(False)

    def _sync_input_bar(self, agent_id: str) -> None:
        """Re-derive BOTH input-bar properties for an agent from ``_agent_states``.

        ``_agent_states`` is written on every ``AgentStateChanged``, including the ones
        the live route drops because the agent had no panel or was draining -- and
        ``_replay_event_locked`` deliberately ignores that event type, so the widgets are
        the only thing that goes stale.  Call this wherever the input bar may have been
        created or reattached after an event was dropped.

        Does nothing until the bar exists.  ``_ensure_feed``'s tail runs before the feed
        has mounted, so the bar is frequently None there; the selection watcher is the
        seam where it reliably exists.

        Args:
            agent_id: The agent whose input bar to re-derive.
        """
        state = self._agent_states.get(agent_id, AgentState.IDLE)
        feed = self._panels.get(agent_id)
        if feed is None or feed.input_bar is None:
            return
        feed.input_bar.set_busy(state in _BUSY_INPUT_STATES)
        self._update_input_bar_state(agent_id, state)

    def _update_input_bar_state(self, agent_id: str, state: AgentState) -> None:
        """Update the InputBar disabled state for an agent.

        Args:
            agent_id: The agent whose input bar to update.
            state: The agent's current state.
        """
        if agent_id not in self._panels:
            return
        feed = self._panels[agent_id]
        bar = feed.input_bar
        if bar is None:
            return
        if state in _DISABLED_STATES:
            hint = f"{agent_id} is {state.value.replace('_', ' ')}…"
            bar.set_disabled(disabled=True, hint=hint)
        else:
            bar.set_disabled(disabled=False, hint=f"Message {agent_id}…")

    def _synthesize_agent_option(self, agent_id: str) -> SessionConfigOptionSelect | None:
        """Build a synthesized agent picker option from discovery results."""
        if self.broker._registry.get_agent_mode_target(agent_id) != "meta_agent":
            return None
        agents = self.broker.get_discovered_agents(agent_id)
        if not agents:
            return None
        current_value = self.broker._registry.get_agent_mode(agent_id) or ""
        return SessionConfigOptionSelect(
            id="agent",
            name="Agent",
            category="agent",
            type="select",
            current_value=current_value,
            options=[
                SessionConfigSelectOption(name="Default", value=""),
                *[SessionConfigSelectOption(name=a.name, value=a.qualified_name) for a in agents],
            ],
        )

    def _update_tile_mode_from_config(self, agent_id: str) -> None:
        """Resolve the current mode name from config options and push to tile."""
        # For harnesses where mode config option != agent name (e.g. Claude Code),
        # use the broker's display name derived from the configured agent_mode.
        display_name = self.broker.get_agent_display_name(agent_id)
        if display_name:
            tile = self._tiles.get(agent_id)
            if tile is not None:
                tile.update_mode(display_name)
            return

        options = self._agent_config_options.get(agent_id, [])
        mode_opt = next(
            (
                o
                for o in options
                if getattr(o, "category", None) == "mode" and hasattr(o, "current_value")
            ),
            None,
        )
        mode_name: str | None = None
        if mode_opt is not None:
            for entry in mode_opt.options:
                if hasattr(entry, "value") and entry.value == mode_opt.current_value:
                    mode_name = entry.name
                    break
                if hasattr(entry, "options"):
                    for sub in entry.options:
                        if sub.value == mode_opt.current_value:
                            mode_name = sub.name
                            break
                    if mode_name:
                        break
        tile = self._tiles.get(agent_id)
        if tile is not None:
            tile.update_mode(mode_name)

    def _get_input_bar(self, agent_id: str) -> InputBar | None:
        """Return the InputBar for an agent, or None."""
        feed = self._panels.get(agent_id)
        return feed.input_bar if feed else None

    def _update_input_bar_config_options(self, agent_id: str) -> None:
        """Push stored config options to the agent's input bar."""
        bar = self._get_input_bar(agent_id)
        if bar:
            bar.update_config_options(self._agent_config_options.get(agent_id, []))

    def _update_usage_display(self, event: UsageUpdated) -> None:
        """Update usage display for the agent's tile and (if selected) input bar.

        Args:
            event: Usage snapshot from the broker.
        """
        cost_text = self._format_cost(event.cost_amount, event.cost_currency)

        # Always update the tile
        tile = self._tiles.get(event.agent_id)
        if tile is not None:
            tile.update_usage(event.used, event.size, cost_text)

        # Also update the input bar if this is the selected agent
        if event.agent_id == self.selected_agent:
            bar = self._get_input_bar(event.agent_id)
            if bar is not None:
                bar.query_one(ActivityBar).update_usage(event.used, event.size, cost_text)

    def _format_cost(self, amount: float | None, currency: str | None) -> str:
        """Format cost amount and currency into display string.

        Args:
            amount: Cost amount (None = no cost data).
            currency: Currency code (e.g. "USD").

        Returns:
            Formatted string like "$1.23" or "" if no data.
        """
        if amount is None:
            return ""
        if currency and currency.upper() == "USD":
            return f"${amount:.2f}"
        if currency:
            return f"{amount:.2f} {currency}"
        return f"{amount:.2f}"

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes — notify and restart on error.

        Args:
            event: Textual worker state change event.
        """
        if event.worker.name != "broker-consumer" or event.state != WorkerState.ERROR:
            return
        error = event.worker.error
        self.notify(
            f"Broker consumer crashed: {error}",
            severity="error",
            timeout=0,
        )
        self.run_worker(
            self._consume_broker_events(),
            exit_on_error=False,
            name="broker-consumer",
            group="broker",
        )

    async def select_agent(self, agent_id: str, *, explicit: bool = True) -> None:
        """Switch the right panel to the given agent.

        Creates the panel and drains buffered events on first visit,
        then sets the reactive to trigger the watcher. Concurrent calls
        for the same agent_id are deduplicated — the second caller awaits
        the in-flight task and returns.

        Args:
            agent_id: The agent to display.
            explicit: True when the user asked for this agent — a tile click, Tab, a
                launch, a restore. Clearing the auto-move marker here covers every such
                caller without editing each one. The automatic move made when an agent
                terminates passes False, since that is the move a handoff needs to undo.
        """
        if explicit:
            self._auto_selected_from = None
        if agent_id in self._selecting:
            await asyncio.shield(self._selecting[agent_id])
            return
        task = asyncio.ensure_future(self._do_select_agent(agent_id))
        self._selecting[agent_id] = task
        try:
            await task
        finally:
            self._selecting.pop(agent_id, None)

    async def _handle_handed_off(self, event: AgentHandedOff) -> None:
        """Rebuild the retired predecessor's view and leave the original id free.

        REBUILD, do not move. The predecessor's AgentStateChanged(TERMINATED) arrives
        BEFORE this event and still carries the ORIGINAL id, so the existing handler has
        already removed that id's tile, feed, buffer, dynamic metadata and config options.
        Missing keys are the normal case here, not an error. Textual widget ids are
        write-once anyway, so renaming the predecessor's widgets was never possible.

        Returns before the generic event-buffering block below on purpose: this event
        carries the ORIGINAL id, so falling through would create _event_buffers[original]
        and re-occupy the id the successor is about to mount under.
        """
        original, retired = event.agent_id, event.retired_agent_id

        # The one per-agent dict the TERMINATED handler does NOT clear. Left in place, the
        # successor's INITIALIZING would read prev_state == TERMINATED and take the
        # RESURRECTION branch instead of the new-agent branch.
        self._agent_states.pop(original, None)

        # Operational state, not just widgets: the resurrection branch is gated on
        # prev_state == TERMINATED, so without this a later resurrect of the retired agent
        # falls into the new-agent branch, calls add_agent_tile for a tile mounted just
        # below, and Textual raises DuplicateIds.
        self._agent_states[retired] = AgentState.TERMINATED
        self._dynamic_agents[retired] = DynamicAgentInfo(
            parent=event.parent,
            task=event.task,
            harness=self.broker.get_agent_harness(retired),
            cwd=self.broker.get_agent_cwd(retired),
        )

        # The rename moved the predecessor's ui_events rows, so this returns its full
        # transcript while load_journal(original) correctly returns nothing.
        try:
            self._event_buffers[retired] = await self.broker.load_journal(
                retired, self.broker.session_id
            )
        except Exception:
            # SURFACED, not swallowed. The rows still exist under the retired id, so an
            # empty feed here would tell the user their predecessor's transcript was lost
            # when it was not. Mount the feed anyway so the agent stays reachable.
            log.error("Failed to load journal for retired agent %s", retired, exc_info=True)
            self._event_buffers.setdefault(retired, [])
            self.notify(
                f"Could not load the transcript for retired agent {retired}. "
                "Its history is still stored under that id; see the log for the "
                "read error.",
                severity="error",
            )
        # Feed BEFORE tile, deliberately. _ensure_feed subscribes an existing tile to the
        # feed's streaming signal, which only exists once the feed has mounted; and a
        # retired agent produces no further output, so there is nothing to subscribe to.
        await self._ensure_feed(retired)

        try:
            tile = self.query_one(AgentList).add_agent_tile(
                retired, parent=event.parent, retired=True
            )
            self._tiles[retired] = tile
        except Exception:
            # Also surfaced: without a tile the retired agent is unreachable in the
            # sidebar, which looks exactly like the handoff having discarded it.
            log.error("Failed to add retired tile for %s", retired, exc_info=True)
            self.notify(
                f"Could not add a sidebar entry for retired agent {retired}.",
                severity="error",
            )

        # Selection is NOT restored here. Doing so would create _panels[original] before
        # the successor exists, leaving the original id present in a per-agent dict that
        # must be empty for the successor to mount through the ordinary new-agent branch.
        # The marker survives instead, and the successor's INITIALIZING claims it.

    async def _ensure_feed(self, agent_id: str) -> None:
        """Mount this agent's conversation feed and drain its buffered events, once.

        Extracted from ``_do_select_agent`` unchanged so a handoff can mount the retired
        agent's transcript without selecting it, and without becoming a second copy of the
        tail-first windowing loop. A no-op if the feed already exists.
        """
        if agent_id in self._panels:
            return
        initial = self._initial_agent
        agent_cfg = initial if initial.agent_id == agent_id else None
        agent_name = agent_cfg.display_name if agent_cfg else agent_id
        harness = agent_cfg.harness if agent_cfg else ""
        cwd = agent_cfg.cwd if agent_cfg else ""
        if not harness:
            dyn = self._dynamic_agents.get(agent_id)
            if dyn:
                harness = dyn.harness
                cwd = cwd or dyn.cwd
        feed = ConversationFeed(
            agent_id,
            agent_name,
            self.config.project,
            harness=harness,
            cwd=cwd,
            id=f"feed-{css_id(agent_id)}",
        )
        await self.query_one("#right", ContentSwitcher).add_content(feed, set_current=False)
        self._panels[agent_id] = feed
        tile = self._tiles.get(agent_id)
        if tile is not None:
            tile.subscribe_feed(feed)
        evt = asyncio.Event()
        self._draining[agent_id] = evt
        try:
            first_pass = True
            while self._event_buffers.get(agent_id):
                batch = self._event_buffers[agent_id]
                self._event_buffers[agent_id] = []
                events = _coalesce_events(batch)
                # TAIL-FIRST WINDOWING, first pass ONLY. The first pass is the buffered
                # HISTORY, where mounting everything costs ~8.8s on a large feed and 96%
                # of that is widget construction. Later passes carry events that arrived
                # DURING the drain — those are live and must be replayed in full, or
                # agent output would vanish whenever it raced a selection.
                # The window is computed on the COALESCED list because coalescing
                # changes event counts, so the budget must see what the replay sees.
                window = first_paint_window(events) if first_pass else FirstPaintWindow(0, 0)
                first_pass = False
                for i, event in enumerate(events):
                    if i < window.start_index:
                        self._record_only(feed, event)
                    else:
                        if i == window.start_index and window.skipped_turns:
                            # Assigned HERE, not before the loop: the periodic yield
                            # below lets a NearTop restore worker run mid-drain, and an
                            # index set before its batches were closed would make
                            # _restore_turns slice a short or empty range.
                            feed._mounted_start_idx = window.skipped_turns
                        await self._replay_event(feed, event)
                    if i % 20 == 19:
                        await asyncio.sleep(0)
        finally:
            evt.set()
            self._draining.pop(agent_id, None)
        # Scroll to bottom after replay so restored sessions show latest messages.
        if feed._scroll:
            feed._scroll.anchor()
        # Re-derive the input bar from _agent_states, which is written on EVERY
        # AgentStateChanged (including ones dropped by the live route while this agent had
        # no panel or was draining -- _replay_event_locked deliberately ignores them).
        # Both properties are re-derived: set_busy alone left a disabled flag from an
        # AWAITING_PERMISSION whose exit was never live-routed, and re-selecting the agent
        # was the only code in the app that repaired it.
        # Deferred: ConversationFeed.on_mount is what assigns ``input_bar``, and it runs on
        # the message pump, so the bar is still None immediately after ``add_content``
        # returns.  Syncing here directly would be a silent no-op -- which is exactly how
        # the stale-busy-state bug survived.
        self.call_after_refresh(self._sync_input_bar, agent_id)
        # Push any config option data that arrived before the panel existed
        self._update_input_bar_config_options(agent_id)

    async def _do_select_agent(self, agent_id: str) -> None:
        """Inner body of select_agent — creates panel, drains buffer, switches view."""
        await self._ensure_feed(agent_id)

        # Sync cached usage into the InputBar on every selection
        usage = self.broker.get_usage(agent_id)
        if usage is not None:
            cost_text = self._format_cost(usage.cost_amount, usage.cost_currency)
            bar = self._get_input_bar(agent_id)
            if bar is not None:
                bar.query_one(ActivityBar).update_usage(usage.used, usage.size, cost_text)

        switcher = self.query_one("#right", ContentSwitcher)
        if self.selected_agent == agent_id and switcher.current != f"feed-{css_id(agent_id)}":
            self.watch_selected_agent(agent_id)
        else:
            self.selected_agent = agent_id

    def watch_selected_agent(self, old_agent: str, agent_id: str) -> None:
        """React to selected_agent changes — switch panel, update tiles and topbar.

        Args:
            old_agent: The previously selected agent ID.
            agent_id: The newly selected agent ID.
        """
        # Try draining queued MCP messages for previous agent if not composing
        if old_agent:
            old_feed = self._panels.get(old_agent)
            if not old_feed or not old_feed.input_bar or not old_feed.input_bar.is_composing:
                from synth_acp.models.commands import ReleaseQueue

                self.run_worker(self.broker.handle(ReleaseQueue(agent_id=old_agent)))
        if not agent_id:
            return
        feed_id = f"feed-{css_id(agent_id)}"
        switcher = self.query_one(ContentSwitcher)
        try:
            switcher.get_child_by_id(feed_id)
        except Exception:
            return
        switcher.current = feed_id
        for tile in self._tiles.values():
            tile.set_class(tile._agent_id == agent_id, "tile-active")
        self._sync_input_bar(agent_id)

    async def action_next_agent(self) -> None:
        """Cycle to the next live agent, skipping terminated ones."""
        ids = [aid for aid, state in self._agent_states.items() if state != AgentState.TERMINATED]
        if not ids:
            return
        idx = ids.index(self.selected_agent) if self.selected_agent in ids else -1
        await self.select_agent(ids[(idx + 1) % len(ids)])

    @work(exclusive=True, group="modal")
    async def action_launch(self) -> None:
        """Open the launch agent modal and launch the selected agent."""
        await self._do_launch()

    async def _do_launch(self) -> bool:
        """Launch modal logic, separated for testability.

        Returns:
            True if an agent was launched, False if the modal was cancelled.
        """
        result = await self.push_screen_wait(LaunchAgentScreen())
        if result is not None:
            self._dynamic_agents[result.agent_id] = DynamicAgentInfo(
                parent=None, task="", harness=result.harness, cwd=result.cwd
            )
            self._event_buffers[result.agent_id] = []
            try:
                tile = self.query_one(AgentList).add_agent_tile(result.agent_id)
                self._tiles[result.agent_id] = tile
            except Exception:
                log.debug("Failed to add tile for %s", result.agent_id, exc_info=True)
            await self.select_agent(result.agent_id)
            await self.broker.handle(LaunchAgent(agent_id=result.agent_id, config=result))
            return True
        return False

    @work(exclusive=True, group="modal")
    async def action_restore(self) -> None:
        """Open the session picker modal (ctrl+r)."""
        active = [
            aid
            for aid, state in self._agent_states.items()
            if state not in (AgentState.TERMINATED,)
        ]
        if active:
            self.notify("Cannot restore while agents are running.", severity="warning")
            return
        await self._show_session_picker(from_startup=False)

    @work(exclusive=True, group="modal")
    async def _do_restore(self, *, from_startup: bool) -> None:
        """Worker wrapper for the session picker flow."""
        await self._show_session_picker(from_startup=from_startup)

    async def _show_session_picker(self, *, from_startup: bool) -> None:
        """Show the session picker and handle the result."""
        from synth_acp.models.commands import RestoreSession

        sessions = await ACPBroker.list_restorable_sessions(self.broker._db_path)
        result = await self.push_screen_wait(
            SessionPickerScreen(
                sessions,
                db_path=self.broker._db_path,
                engine=self._embedding_engine,
                indexing_complete=self._indexing_complete,
            )
        )
        if result is not None:
            # Pre-initialise event buffers for all agents in the restored
            # session before the broker starts launching them.
            session_info = next((s for s in sessions if s["session_id"] == result), None)
            if session_info:
                for aid in session_info["agents"]:
                    self._event_buffers.setdefault(aid, [])

            await self.broker.handle(RestoreSession(broker_session_id=result))

            # Remove tiles for config agents not in the restored session.
            if session_info:
                restored_ids = set(session_info["agents"])
                initial_id = self._initial_agent.agent_id
                if initial_id not in restored_ids:
                    tile = self._tiles.pop(initial_id, None)
                    if tile is not None:
                        tile.remove()

            # Load journal events into buffers BEFORE creating panels.
            # select_agent will drain them through _replay_event after
            # the feed is mounted and its widget tree is ready.
            if session_info:
                for aid in session_info["agents"]:
                    journal = await self.broker.load_journal(aid, result)
                    # The broker-consumer worker runs concurrently and pops the
                    # buffer for any agent that transitions to TERMINATED during
                    # RestoreSession handling (see AgentStateChanged handler).
                    # Re-seed with setdefault so a concurrently-removed buffer
                    # doesn't raise KeyError here.
                    self._event_buffers.setdefault(aid, []).extend(journal)

            # Select the first restored agent to create its panel and drain
            # the buffer (which will include journal-replayed events).
            if session_info and session_info["agents"]:
                # Pick the first non-terminated agent — the agents list may
                # include terminated agents that weren't restored.
                first = next(
                    (
                        aid
                        for aid in session_info["agents"]
                        if self._agent_states.get(aid) != AgentState.TERMINATED
                    ),
                    session_info["agents"][0],
                )
                await self.select_agent(first)
        elif from_startup:
            # Cancelled at startup — fall through to normal launch
            initial = self._initial_agent
            await self.broker.handle(LaunchAgent(agent_id=initial.agent_id, config=initial))
            await self.select_agent(initial.agent_id)

    async def action_help(self) -> None:
        """Open the help modal showing key bindings and usage."""
        await self.push_screen_wait(HelpScreen())

    async def action_quit(self) -> None:
        """Quit the app. Cleanup happens in on_unmount."""
        self.exit()

    async def on_unmount(self) -> None:
        """Terminate all agent subprocesses during Textual shutdown."""
        import threading

        # Runs before the watchdog and before broker shutdown, so the process-global
        # gc hook comes off even if shutdown hangs.
        self._stop_runtime_subsystems()
        watchdog = threading.Timer(5.0, os._exit, args=(0,))
        watchdog.daemon = True
        watchdog.start()
        try:
            await self.broker.shutdown()
        except Exception:
            log.debug("Broker shutdown error", exc_info=True)
        finally:
            try:
                loop = asyncio.get_running_loop()
                await loop.shutdown_default_executor(1)
                loop._default_executor = None  # type: ignore[attr-defined]
            except Exception:
                pass
            watchdog.cancel()
