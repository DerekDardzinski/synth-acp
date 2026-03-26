"""SynthApp — Textual TUI bridging the ACPBroker to the terminal."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer, LoadingIndicator, Static
from textual.worker import WorkerState

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentState
from synth_acp.models.commands import LaunchAgent
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerError,
    BrokerEvent,
    McpMessageDelivered,
    MessageChunkReceived,
    PermissionAutoResolved,
    PermissionRequested,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
)
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.screens.help import HelpScreen
from synth_acp.ui.screens.launch import LaunchAgentScreen
from synth_acp.ui.widgets.agent_list import AgentList, AgentTile, MCPButton
from synth_acp.ui.widgets.conversation import ConversationFeed
from synth_acp.ui.widgets.message_queue import MessageQueue

PALETTE = [
    "#3b82f6",
    "#a78bfa",
    "#f97316",
    "#2dd4bf",
    "#f472b6",
    "#06b6d4",
    "#84cc16",
    "#e879f9",
    "#fb923c",
    "#4ade80",
]

_DISABLED_STATES = {AgentState.BUSY, AgentState.AWAITING_PERMISSION}


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
        Binding("f1", "help", "Help"),
    ]

    selected_agent: reactive[str] = reactive("")
    selected_thread: reactive[str] = reactive("")

    def __init__(self, broker: ACPBroker, config: SessionConfig) -> None:
        super().__init__()
        self.broker = broker
        self.config = config
        self._agent_colors: dict[str, str] = {
            agent.id: PALETTE[i % len(PALETTE)] for i, agent in enumerate(config.agents)
        }
        self._event_buffers: dict[str, list[BrokerEvent]] = {}
        self._panels: dict[str, ConversationFeed] = {}
        self._agent_states: dict[str, AgentState] = {}
        self._mcp_threads: dict[tuple[str, str], list[McpMessageDelivered]] = {}
        self._mcp_count: int = 0
        self._mcp_panel: MessageQueue | None = None

    def compose(self) -> ComposeResult:
        """Build the top bar, main layout with sidebar, and footer."""
        with Horizontal(id="topbar"):
            yield Static("SYNTH", id="tb-title")
            yield Static("│", id="tb-sep")
            yield Static(f"project: {self.config.project}", id="tb-session")
            yield Static("", id="tb-right")
        with Horizontal(id="main"):
            agents = [(a.id, self._agent_colors[a.id]) for a in self.config.agents]
            with Vertical(id="sidebar"):
                yield AgentList(agents)
            yield ContentSwitcher(id="right")
        yield Footer()

    async def on_mount(self) -> None:
        """Launch all agents and start the broker event consumer."""
        for agent in self.config.agents:
            self._event_buffers[agent.id] = []
        for agent in self.config.agents:
            await self.broker.handle(LaunchAgent(agent_id=agent.id))
        self.run_worker(self._consume_broker_events(), exit_on_error=False, name="broker-consumer")

    async def _consume_broker_events(self) -> None:
        """Consume broker events and post them as Textual messages."""
        async for event in self.broker.events():
            self.post_message(BrokerEventMessage(event))

    async def on_broker_event_message(self, message: BrokerEventMessage) -> None:
        """Route broker events to the appropriate widgets.

        Args:
            message: Wrapped broker event.
        """
        event = message.event

        # Handle MCP messages — update threads and show in conversation
        if isinstance(event, McpMessageDelivered):
            key = tuple(sorted([event.from_agent, event.to_agent]))
            self._mcp_threads.setdefault(key, []).append(event)  # type: ignore[arg-type]
            self._mcp_count += 1
            try:
                self.query_one("#mcp-btn", MCPButton).update_count(self._mcp_count)
            except Exception:
                pass
            # Update MCP panel if visible
            if (
                self._mcp_panel is not None
                and self.query_one(ContentSwitcher).current == "messages"
            ):
                self._mcp_panel.update_threads(self._mcp_threads)
            # Show in recipient's conversation feed
            recipient = event.to_agent
            if recipient in self._panels:
                feed = self._panels[recipient]
                feed.add_mcp_message(event.from_agent, event.to_agent, event.preview)
            elif recipient in self._event_buffers:
                self._event_buffers[recipient].append(event)
            return

        # Buffer events for agents without panels
        if event.agent_id in self._event_buffers and event.agent_id not in self._panels:
            self._event_buffers[event.agent_id].append(event)

        if isinstance(event, AgentStateChanged):
            self._agent_states[event.agent_id] = event.new_state
            try:
                tile = self.query_one(f"#tile-{event.agent_id}", AgentTile)
                tile.update_state(event.new_state)
            except Exception:
                pass

        # Route to the agent's panel if it exists (regardless of selection)
        if event.agent_id in self._panels:
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
        if isinstance(event, MessageChunkReceived):
            await feed.add_chunk(event.chunk)
        elif isinstance(event, AgentThoughtReceived):
            await feed.add_thought_chunk(event.chunk)
        elif isinstance(event, ToolCallUpdated):
            feed.add_tool_call(event.tool_call_id, event.title, event.kind, event.status)
        elif isinstance(event, PermissionRequested):
            feed.add_permission(
                event.agent_id, event.request_id, event.title, event.kind, event.options
            )
        elif isinstance(event, PermissionAutoResolved):
            feed.remove_permission(event.request_id)
        elif isinstance(event, TurnComplete):
            await feed.finalize_current_message()
        elif isinstance(event, UsageUpdated):
            self._update_usage_display(event)
        elif isinstance(event, BrokerError):
            self.notify(event.message, severity=event.severity)
        elif isinstance(event, AgentStateChanged):
            if event.new_state == AgentState.IDLE:
                try:
                    await feed.query_one("#loading-spinner", LoadingIndicator).remove()
                except NoMatches:
                    pass
            elif event.new_state == AgentState.INITIALIZING:
                await feed.query_one(".conv-scroll").mount(LoadingIndicator(id="loading-spinner"))

    async def _replay_event(self, feed: ConversationFeed, event: BrokerEvent) -> None:
        """Replay a buffered event to a conversation feed during drain.

        Skips BrokerError and PermissionAutoResolved events. AgentStateChanged
        updates are already tracked in _agent_states.

        Args:
            feed: Target conversation feed.
            event: The buffered broker event to replay.
        """
        if isinstance(event, MessageChunkReceived):
            await feed.add_chunk(event.chunk)
        elif isinstance(event, AgentThoughtReceived):
            await feed.add_thought_chunk(event.chunk)
        elif isinstance(event, ToolCallUpdated):
            feed.add_tool_call(event.tool_call_id, event.title, event.kind, event.status)
        elif isinstance(event, PermissionRequested):
            feed.add_permission(
                event.agent_id, event.request_id, event.title, event.kind, event.options
            )
        elif isinstance(event, TurnComplete):
            await feed.finalize_current_message()
        elif isinstance(event, McpMessageDelivered):
            feed.add_mcp_message(event.from_agent, event.to_agent, event.preview)

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

    def _update_usage_display(self, event: UsageUpdated) -> None:
        """Update the topbar usage display for the selected agent.

        Args:
            event: Usage snapshot from the broker.
        """
        if event.agent_id != self.selected_agent:
            return
        parts: list[str] = []
        used = event.used
        parts.append(f"{used // 1000}k ctx" if used >= 1000 else f"{used} ctx")
        if event.cost_amount is not None:
            parts.append(f"${event.cost_amount:.2f}")
        try:
            self.query_one("#tb-right", Static).update(
                f"[dim]{'  '.join(parts)}[/dim]" if parts else ""
            )
        except Exception:
            pass

    def on_worker_state_changed(self, event: SynthApp.WorkerStateChanged) -> None:
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
        self.run_worker(self._consume_broker_events(), exit_on_error=False, name="broker-consumer")

    async def select_agent(self, agent_id: str) -> None:
        """Switch the right panel to the given agent.

        Creates the panel and drains buffered events on first visit,
        then sets the reactive to trigger the watcher.

        Args:
            agent_id: The agent to display.
        """
        if agent_id not in self._panels:
            color = self._agent_colors.get(agent_id, "#94a3b8")
            feed = ConversationFeed(agent_id, color, id=f"feed-{agent_id}")
            self._panels[agent_id] = feed
            await self.query_one("#right").mount(feed)
            for event in self._event_buffers.get(agent_id, []):
                await self._replay_event(feed, event)
            self._event_buffers[agent_id] = []
            # Remove spinner if agent already passed INITIALIZING while buffered
            state = self._agent_states.get(agent_id)
            if state != AgentState.INITIALIZING:
                try:
                    feed.query_one("#loading-spinner", LoadingIndicator).remove()
                except NoMatches:
                    pass

        self.selected_agent = agent_id

    def watch_selected_agent(self, agent_id: str) -> None:
        """React to selected_agent changes — switch panel, update tiles and topbar.

        Args:
            agent_id: The newly selected agent ID.
        """
        if not agent_id:
            return
        self.query_one(ContentSwitcher).current = f"feed-{agent_id}"
        for tile in self.query(AgentTile):
            tile.set_class(tile._agent_id == agent_id, "tile-active")
        try:
            self.query_one("#mcp-btn", MCPButton).remove_class("btn-active")
        except Exception:
            pass
        # Update topbar with agent display name
        display_name = agent_id
        for a in self.config.agents:
            if a.id == agent_id:
                display_name = getattr(a, "display_name", a.id)
                break
        try:
            self.query_one("#tb-session", Static).update(display_name)
        except Exception:
            pass
        self._update_input_bar_state(agent_id, self._agent_states.get(agent_id, AgentState.IDLE))

    async def show_messages(self) -> None:
        """Switch the right panel to the MCP messages view."""
        for tile in self.query(AgentTile):
            tile.remove_class("tile-active")
        try:
            self.query_one("#mcp-btn", MCPButton).add_class("btn-active")
        except Exception:
            pass

        switcher = self.query_one("#right", ContentSwitcher)

        if self._mcp_panel is None:
            panel = MessageQueue(self._mcp_threads, self._agent_colors, id="messages")
            self._mcp_panel = panel
            await switcher.mount(panel)
        else:
            self._mcp_panel.update_threads(self._mcp_threads)

        switcher.current = "messages"

    async def action_next_agent(self) -> None:
        """Cycle to the next agent in config order."""
        ids = [a.id for a in self.config.agents]
        if not ids:
            return
        idx = ids.index(self.selected_agent) if self.selected_agent in ids else -1
        await self.select_agent(ids[(idx + 1) % len(ids)])

    async def action_messages(self) -> None:
        """Show the MCP messages panel."""
        await self.show_messages()

    async def action_launch(self) -> None:
        """Open the launch agent modal and launch the selected agent."""
        agents = [(a.id, a.display_name, self._agent_states.get(a.id)) for a in self.config.agents]
        result = await self.push_screen_wait(LaunchAgentScreen(agents))
        if result is not None:
            await self.broker.handle(LaunchAgent(agent_id=result))

    async def action_help(self) -> None:
        """Open the help modal showing key bindings and usage."""
        await self.push_screen_wait(HelpScreen())

    async def action_quit(self) -> None:
        """Shut down the broker and exit.

        Exits the TUI immediately and runs broker shutdown in background
        to avoid blocking on agent process termination.
        """
        self.exit()
        try:
            await self.broker.shutdown()
        except Exception:
            pass
