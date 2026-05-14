"""SynthApp — Textual TUI bridging the ACPBroker to the terminal."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
from typing import ClassVar, NamedTuple

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer, TextArea
from textual.worker import Worker, WorkerState

from synth_acp.broker.broker import ACPBroker
from synth_acp.db import (
    _build_embedding_text,
    _text_hash,
    get_unembedded_sessions_sync,
    store_embedding_sync,
)
from synth_acp.embeddings import EmbeddingEngine, embedding_available
from synth_acp.models.agent import AgentConfig, AgentState, css_id
from synth_acp.models.commands import (
    HoldDelivery,
    LaunchAgent,
    ReleaseDelivery,
    RespondPermission,
    SendPrompt,
    TerminateAgent,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    AvailableCommandsReceived,
    BrokerError,
    BrokerEvent,
    ConfigOptionChanged,
    ConfigOptionsReceived,
    HookFired,
    InitialPromptDelivered,
    McpMessageDelivered,
    McpMessageHeld,
    MessageChunkReceived,
    PermissionRequested,
    PlanReceived,
    SessionRestoreComplete,
    TerminalCreated,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
    UserPromptSubmitted,
)
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.screens.help import HelpScreen
from synth_acp.ui.screens.launch import LaunchAgentScreen
from synth_acp.ui.screens.permission import PermissionBar
from synth_acp.ui.screens.session_picker import SessionPickerScreen
from synth_acp.ui.widgets.agent_list import AgentList, AgentTile, MCPButton
from synth_acp.ui.widgets.conversation import ConversationFeed
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.message_queue import MessageQueue

_DISABLED_STATES = {AgentState.AWAITING_PERMISSION}

_RENDERABLE_EVENTS = (
    MessageChunkReceived,
    AgentThoughtReceived,
    ToolCallUpdated,
    TurnComplete,
    PlanReceived,
    McpMessageDelivered,
    HookFired,
    InitialPromptDelivered,
    UserPromptSubmitted,
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
        Binding("m", "messages", "MCP messages"),
        Binding("l", "launch", "Launch agent"),
        Binding("ctrl+r", "restore", "Restore session"),
        Binding("f1", "help", "Help"),
    ]

    selected_agent: reactive[str] = reactive("")
    selected_thread: reactive[str] = reactive("")

    def __init__(self, broker: ACPBroker, config: SessionConfig, initial_agent: AgentConfig | None = None, css_path: str | None = None, restore: bool = False) -> None:
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
        self._mcp_threads: dict[tuple[str, str], list[McpMessageDelivered]] = {}
        self._mcp_count: int = 0
        self._mcp_panel: MessageQueue | None = None
        self._delivery_holding: set[str] = set()
        self._drain_suppressed: set[str] = set()
        self._dynamic_agents: dict[str, DynamicAgentInfo] = {}
        self._agent_config_options: dict[str, list] = {}
        self._tiles: dict[str, AgentTile] = {}
        self._selecting: dict[str, asyncio.Task[None]] = {}
        self._draining: dict[str, asyncio.Event] = {}
        self._indexing_complete: bool = False
        self._embedding_engine: EmbeddingEngine | None = None

    def _handle_exception(self, error: Exception) -> None:
        """Log unhandled exceptions to file before Textual's default handling."""
        import logging

        logging.getLogger("synth_acp.ui.app").error(
            "Textual unhandled exception", exc_info=error
        )
        super()._handle_exception(error)

    def compose(self) -> ComposeResult:
        """Build the main layout with sidebar and footer."""
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield AgentList([])
            yield ContentSwitcher(id="right")
        yield Footer()

    async def on_mount(self) -> None:
        """Launch all agents, select the first, and start the broker event consumer."""
        self.theme = "catppuccin-mocha"
        initial = self._initial_agent
        self._event_buffers[initial.agent_id] = []
        if self._restore_mode:
            self.run_worker(self._consume_broker_events(), exit_on_error=False, name="broker-consumer", group="broker")
            self._do_restore(from_startup=True)
        else:
            await self.broker.handle(LaunchAgent(agent_id=initial.agent_id, config=initial))
            await self.select_agent(initial.agent_id)
            self.run_worker(self._consume_broker_events(), exit_on_error=False, name="broker-consumer", group="broker")
        self._index_sessions()

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
            if not embedding_available():
                return
            engine = EmbeddingEngine()
            self._embedding_engine = engine
            engine.ensure_model()
            db_path = str(self.broker._db_path)
            with contextlib.closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.row_factory = sqlite3.Row
                session_ids = get_unembedded_sessions_sync(conn)
                for sid in session_ids:
                    session = self._query_session_metadata(conn, sid)
                    text = _build_embedding_text(session)
                    text_h = _text_hash(text)
                    embedding = engine.embed(text)
                    store_embedding_sync(conn, sid, text_h, embedding.tobytes())
            self._indexing_complete = True
        except Exception:
            log.debug("Background indexing failed", exc_info=True)

    @staticmethod
    def _query_session_metadata(conn: sqlite3.Connection, session_id: str) -> dict:
        """Query session metadata for embedding text composition."""
        agents = [
            r[0] for r in conn.execute(
                "SELECT agent_id FROM agents WHERE session_id = ?", (session_id,)
            ).fetchall()
        ]
        cwd_row = conn.execute(
            "SELECT cwd FROM agents WHERE session_id = ? ORDER BY registered ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        cwd = cwd_row["cwd"] if cwd_row else None
        tasks = [
            r[0] for r in conn.execute(
                "SELECT task FROM agents WHERE session_id = ? AND task IS NOT NULL",
                (session_id,),
            ).fetchall()
        ]
        first_messages: list[str] = []
        msg_rows = conn.execute(
            "SELECT payload FROM ui_events "
            "WHERE session_id = ? AND event_type = 'UserPromptSubmitted' "
            "ORDER BY seq LIMIT 3",
            (session_id,),
        ).fetchall()
        for row in msg_rows:
            try:
                data = json.loads(row["payload"])
                text = data.get("text", "")
                if text:
                    first_messages.append(text)
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "session_id": session_id,
            "agents": agents,
            "cwd": cwd,
            "tasks": tasks,
            "first_messages": first_messages,
        }

    async def on_broker_event_message(self, message: BrokerEventMessage) -> None:
        """Route broker events to the appropriate widgets.

        Args:
            message: Wrapped broker event.
        """
        event = message.event

        # Handle held MCP messages — show in prompt queue
        if isinstance(event, McpMessageHeld):
            recipient = event.agent_id
            if recipient in self._panels and recipient not in self._draining:
                feed = self._panels[recipient]
                if feed.input_bar:
                    feed.input_bar.enqueue(event.preview, "mcp", event.from_agent)
                    self._attempt_drain(recipient)
            return

        # Handle MCP messages — update threads and show in conversation
        if isinstance(event, McpMessageDelivered):
            key = tuple(sorted([event.from_agent, event.to_agent]))
            self._mcp_threads.setdefault(key, []).append(event)  # type: ignore[arg-type]
            self._mcp_count += 1
            try:
                self.query_one("#mcp-btn", MCPButton).update_count(self._mcp_count)
            except Exception:
                log.debug("MCP button not found", exc_info=True)
            # Update MCP panel if visible
            if (
                self._mcp_panel is not None
                and self.query_one(ContentSwitcher).current == "messages"
            ):
                await self._mcp_panel.update_threads(self._mcp_threads)
            # Show in recipient's conversation feed
            recipient = event.to_agent
            if recipient in self._panels and recipient not in self._draining:
                feed = self._panels[recipient]
                feed.record_event(event)
                await feed.add_mcp_message(event.from_agent, event.to_agent, event.preview)
            else:
                self._event_buffers.setdefault(recipient, []).append(event)
            return

        # Handle session history restore — render static snapshot into the feed.
        if isinstance(event, SessionRestoreComplete):
            if event.agent_id in self._panels:
                feed = self._panels[event.agent_id]
                if feed.input_bar is not None:
                    feed.input_bar.set_busy(False)
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
                    self._dynamic_agents[aid] = DynamicAgentInfo(parent=parent, task="", harness=harness, cwd=self.broker.get_agent_cwd(aid))
            elif event.agent_id not in self._dynamic_agents:
                parent = self.broker.get_agent_parent(event.agent_id)
                harness = self.broker.get_agent_harness(event.agent_id)
                self._dynamic_agents[event.agent_id] = DynamicAgentInfo(parent=parent, task="", harness=harness, cwd=self.broker.get_agent_cwd(event.agent_id))
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
            if event.new_state == AgentState.TERMINATED:
                feed = self._panels.pop(event.agent_id, None)
                if feed is not None:
                    await feed.remove()
                self._event_buffers.pop(event.agent_id, None)
                self._dynamic_agents.pop(event.agent_id, None)
                if self.selected_agent == event.agent_id:
                    live = [aid for aid, st in self._agent_states.items() if st != AgentState.TERMINATED]
                    if live:
                        await self.select_agent(live[0])
                    else:
                        self.selected_agent = ""

        if isinstance(event, ConfigOptionsReceived):
            self._agent_config_options[event.agent_id] = list(event.config_options)
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
        """Route a single event to a conversation feed.

        Args:
            feed: Target conversation feed.
            event: The broker event to route.
        """
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
        elif isinstance(event, UsageUpdated):
            self._update_usage_display(event)
        elif isinstance(event, HookFired):
            await feed.add_hook_notification(event.hook_name)
        elif isinstance(event, InitialPromptDelivered):
            await feed.add_prompt(event.text)
        elif isinstance(event, UserPromptSubmitted):
            self._drain_suppressed.discard(event.agent_id)
            await feed.add_prompt(event.text)
        elif isinstance(event, BrokerError):
            self.notify(event.message, severity=event.severity)
        elif isinstance(event, AgentStateChanged):
            if feed.input_bar is None:
                return
            if event.new_state in {AgentState.IDLE, AgentState.TERMINATED}:
                feed.input_bar.set_busy(False)
                if event.new_state == AgentState.IDLE and event.agent_id == self.selected_agent:
                    feed.input_bar.query_one("#prompt-input").focus()
                if event.new_state == AgentState.IDLE:
                    self._attempt_drain(event.agent_id)
            elif event.new_state in {AgentState.INITIALIZING, AgentState.BUSY, AgentState.CONFIGURING}:
                feed.input_bar.set_busy(True)

    def _mount_permission_bar(self, feed: ConversationFeed, event: PermissionRequested) -> None:
        """Mount a PermissionBar at the top of the InputBar.

        Args:
            feed: Target conversation feed.
            event: The permission request event.
        """
        position = self.broker.permission_position(event.agent_id)
        bar = PermissionBar(event.agent_id, event.request_id, event.title, event.options, position=position)
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
                RespondPermission(agent_id=message.agent_id, request_id=message.request_id, option_id=message.option_id)
            )

    def on_input_bar_drain_ready(self, event: InputBar.DrainReady) -> None:
        """Handle DrainReady — attempt drain if agent is idle.

        Args:
            event: The DrainReady message from InputBar.
        """
        self._drain_suppressed.discard(event.agent_id)
        if self._agent_states.get(event.agent_id) == AgentState.IDLE:
            self._attempt_drain(event.agent_id)

    def on_input_bar_cancel_clicked(self, event: InputBar.CancelClicked) -> None:
        """Suppress auto-drain after cancel — user wants to correct course."""
        self._drain_suppressed.add(event.agent_id)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Hold/release MCP delivery based on whether user is composing."""
        if event.text_area.id != "prompt-input":
            return
        # Derive agent_id from the InputBar ancestor, not selected_agent
        agent_id: str | None = None
        for ancestor in event.text_area.ancestors_with_self:
            if isinstance(ancestor, InputBar):
                agent_id = ancestor._agent_id
                break
        if not agent_id:
            return
        if event.text_area.text.strip():
            if agent_id not in self._delivery_holding:
                self._delivery_holding.add(agent_id)
                self.run_worker(self.broker.handle(HoldDelivery(agent_id=agent_id)))
        elif agent_id in self._delivery_holding:
            self._delivery_holding.discard(agent_id)
            self.run_worker(self.broker.handle(ReleaseDelivery(agent_id=agent_id)))

    async def on_agent_tile_terminate_clicked(self, message: AgentTile.TerminateClicked) -> None:
        """Handle the close button on an agent tile."""
        await self.broker.handle(TerminateAgent(agent_id=message.agent_id))

    async def _replay_event(self, feed: ConversationFeed, event: BrokerEvent) -> None:
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
        elif isinstance(event, McpMessageDelivered):
            await feed.add_mcp_message(event.from_agent, event.to_agent, event.preview)
        elif isinstance(event, HookFired):
            await feed.add_hook_notification(event.hook_name)
        elif isinstance(event, InitialPromptDelivered):
            await feed.add_prompt(event.text)
        elif isinstance(event, UserPromptSubmitted):
            await feed.add_prompt(event.text)
        elif isinstance(event, SessionRestoreComplete):
            if feed.input_bar is not None:
                feed.input_bar.set_busy(False)

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

    def _attempt_drain(self, agent_id: str) -> None:
        """Drain the next queued prompt and dispatch it to the broker.

        Single drain owner — only app.py calls this.
        If the user is actively composing, shows the drain button instead.

        Args:
            agent_id: The agent whose queue to drain.
        """
        feed = self._panels.get(agent_id)
        if not feed or not feed.input_bar:
            return
        if not feed.input_bar.has_queue_items:
            feed.input_bar.set_drain_pending(False)
            return
        if feed.input_bar.is_composing or agent_id in self._drain_suppressed:
            feed.input_bar.set_drain_pending(True)
            return
        feed.input_bar.set_drain_pending(False)
        queued = feed.input_bar.drain_next()
        if queued:
            self.run_worker(self.broker.handle(SendPrompt(agent_id=agent_id, text=queued.text)))

    def _update_tile_mode_from_config(self, agent_id: str) -> None:
        """Resolve the current mode name from config options and push to tile."""
        options = self._agent_config_options.get(agent_id, [])
        mode_opt = next((o for o in options if getattr(o, "category", None) == "mode" and hasattr(o, "current_value")), None)
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
        """Update usage display for the selected agent.

        Args:
            event: Usage snapshot from the broker.
        """
        if event.agent_id != self.selected_agent:
            return

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
        self.run_worker(self._consume_broker_events(), exit_on_error=False, name="broker-consumer", group="broker")

    async def select_agent(self, agent_id: str) -> None:
        """Switch the right panel to the given agent.

        Creates the panel and drains buffered events on first visit,
        then sets the reactive to trigger the watcher. Concurrent calls
        for the same agent_id are deduplicated — the second caller awaits
        the in-flight task and returns.

        Args:
            agent_id: The agent to display.
        """
        if agent_id in self._selecting:
            await asyncio.shield(self._selecting[agent_id])
            return
        task = asyncio.ensure_future(self._do_select_agent(agent_id))
        self._selecting[agent_id] = task
        try:
            await task
        finally:
            self._selecting.pop(agent_id, None)

    async def _do_select_agent(self, agent_id: str) -> None:
        """Inner body of select_agent — creates panel, drains buffer, switches view."""
        if agent_id not in self._panels:
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
            feed = ConversationFeed(agent_id, agent_name, self.config.project, harness=harness, cwd=cwd, id=f"feed-{css_id(agent_id)}")
            await self.query_one("#right", ContentSwitcher).add_content(feed, set_current=False)
            self._panels[agent_id] = feed
            tile = self._tiles.get(agent_id)
            if tile is not None:
                tile.subscribe_feed(feed)
            evt = asyncio.Event()
            self._draining[agent_id] = evt
            try:
                while self._event_buffers.get(agent_id):
                    batch = self._event_buffers[agent_id]
                    self._event_buffers[agent_id] = []
                    for i, event in enumerate(_coalesce_events(batch)):
                        await self._replay_event(feed, event)
                        if i % 20 == 19:
                            await asyncio.sleep(0)
            finally:
                evt.set()
                self._draining.pop(agent_id, None)
            # Scroll to bottom after replay so restored sessions show latest messages.
            if feed._scroll:
                feed._scroll.anchor()
            # Set busy state based on current agent state after replay
            state = self._agent_states.get(agent_id)
            if feed.input_bar is not None:
                if state in {AgentState.INITIALIZING, AgentState.BUSY, AgentState.CONFIGURING}:
                    feed.input_bar.set_busy(True)
                else:
                    feed.input_bar.set_busy(False)
            # Push any config option data that arrived before the panel existed
            self._update_input_bar_config_options(agent_id)
            # Refresh MCP badge in case replayed events updated the count
            if self._mcp_count:
                try:
                    self.query_one("#mcp-btn", MCPButton).update_count(self._mcp_count)
                except Exception:
                    pass

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
        # Release hold on previous agent if it's not composing
        if old_agent and old_agent in self._delivery_holding:
            old_feed = self._panels.get(old_agent)
            if not old_feed or not old_feed.input_bar or not old_feed.input_bar.is_composing:
                self._delivery_holding.discard(old_agent)
                self.run_worker(self.broker.handle(ReleaseDelivery(agent_id=old_agent)))
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
        try:
            self.query_one("#mcp-btn", MCPButton).remove_class("btn-active")
        except Exception:
            log.debug("MCP button deselect failed", exc_info=True)
        self._update_input_bar_state(agent_id, self._agent_states.get(agent_id, AgentState.IDLE))

    async def show_messages(self) -> None:
        """Toggle the MCP messages view. Close it if already active."""
        switcher = self.query_one("#right", ContentSwitcher)

        if switcher.current == "messages":
            # Close messages panel — return to selected agent
            if self.selected_agent:
                await self.select_agent(self.selected_agent)
            return

        for tile in self._tiles.values():
            tile.remove_class("tile-active")
        try:
            self.query_one("#mcp-btn", MCPButton).add_class("btn-active")
        except Exception:
            log.debug("MCP button select failed", exc_info=True)

        if self._mcp_panel is None:
            panel = MessageQueue(self._mcp_threads, id="messages")
            self._mcp_panel = panel
            await switcher.add_content(panel, set_current=False)
        else:
            await self._mcp_panel.update_threads(self._mcp_threads)

        switcher.current = "messages"

    async def action_next_agent(self) -> None:
        """Cycle to the next live agent, skipping terminated ones."""
        ids = [aid for aid, state in self._agent_states.items() if state != AgentState.TERMINATED]
        if not ids:
            return
        idx = ids.index(self.selected_agent) if self.selected_agent in ids else -1
        await self.select_agent(ids[(idx + 1) % len(ids)])

    async def action_messages(self) -> None:
        """Show the MCP messages panel."""
        await self.show_messages()

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
            self._dynamic_agents[result.agent_id] = DynamicAgentInfo(parent=None, task="", harness=result.harness, cwd=result.cwd)
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
            aid for aid, state in self._agent_states.items()
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
                    self._event_buffers[aid].extend(journal)

            # Select the first restored agent to create its panel and drain
            # the buffer (which will include journal-replayed events).
            if session_info and session_info["agents"]:
                # Pick the first non-terminated agent — the agents list may
                # include terminated agents that weren't restored.
                first = next(
                    (aid for aid in session_info["agents"]
                     if self._agent_states.get(aid) != AgentState.TERMINATED),
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
