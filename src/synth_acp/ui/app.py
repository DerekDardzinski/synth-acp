"""SynthApp — Textual TUI bridging the ACPBroker to the terminal."""

from __future__ import annotations

import logging
from typing import ClassVar, NamedTuple

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer
from textual.worker import Worker, WorkerState

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentMode, AgentModel, AgentState
from synth_acp.models.commands import LaunchAgent, RespondPermission, TerminateAgent
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentModeChanged,
    AgentModelChanged,
    AgentModelsReceived,
    AgentModesReceived,
    AgentStateChanged,
    AgentThoughtReceived,
    AvailableCommandsReceived,
    BrokerError,
    BrokerEvent,
    HookFired,
    InitialPromptDelivered,
    McpMessageDelivered,
    MessageChunkReceived,
    PermissionRequested,
    PlanReceived,
    TerminalCreated,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
)
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.screens.help import HelpScreen
from synth_acp.ui.screens.launch import LaunchAgentScreen
from synth_acp.ui.screens.permission import PermissionBar
from synth_acp.ui.widgets.agent_list import AgentList, AgentTile, MCPButton
from synth_acp.ui.widgets.conversation import ConversationFeed
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.message_queue import MessageQueue

_DISABLED_STATES = {AgentState.BUSY, AgentState.CONFIGURING, AgentState.AWAITING_PERMISSION}

log = logging.getLogger(__name__)


class DynamicAgentInfo(NamedTuple):
    """Metadata for a dynamically launched agent.

    Attributes:
        parent: Agent ID of the parent, or None.
        task: Task description.
    """

    parent: str | None
    task: str
    harness: str


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

    def __init__(self, broker: ACPBroker, config: SessionConfig, css_path: str | None = None) -> None:
        if css_path:
            self.CSS_PATH = css_path
        super().__init__()
        self.broker = broker
        self.config = config
        self._event_buffers: dict[str, list[BrokerEvent]] = {}
        self._panels: dict[str, ConversationFeed] = {}
        self._agent_states: dict[str, AgentState] = {}
        self._mcp_threads: dict[tuple[str, str], list[McpMessageDelivered]] = {}
        self._mcp_count: int = 0
        self._mcp_panel: MessageQueue | None = None
        self._dynamic_agents: dict[str, DynamicAgentInfo] = {}
        self._agent_modes: dict[str, list[AgentMode]] = {}
        self._agent_current_mode: dict[str, str] = {}
        self._agent_models: dict[str, list[AgentModel]] = {}
        self._agent_current_model: dict[str, str] = {}

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
            agents = [a.agent_id for a in self.config.agents]
            with Vertical(id="sidebar"):
                yield AgentList(agents)
            yield ContentSwitcher(id="right")
        yield Footer()

    async def on_mount(self) -> None:
        """Launch all agents, select the first, and start the broker event consumer."""
        self.theme = "catppuccin-mocha"
        for agent in self.config.agents:
            self._event_buffers[agent.agent_id] = []
        for agent in self.config.agents:
            await self.broker.handle(LaunchAgent(agent_id=agent.agent_id))
        if self.config.agents:
            await self.select_agent(self.config.agents[0].agent_id)
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
                log.debug("MCP button not found", exc_info=True)
            # Update MCP panel if visible
            if (
                self._mcp_panel is not None
                and self.query_one(ContentSwitcher).current == "messages"
            ):
                await self._mcp_panel.update_threads(self._mcp_threads)
            # Show in recipient's conversation feed
            recipient = event.to_agent
            if recipient in self._panels:
                feed = self._panels[recipient]
                feed.add_mcp_message(event.from_agent, event.to_agent, event.preview)
            elif recipient in self._event_buffers:
                self._event_buffers[recipient].append(event)
            return

        # Buffer events for agents without panels
        if event.agent_id not in self._panels:
            if event.agent_id not in self._event_buffers:
                self._event_buffers[event.agent_id] = []
            self._event_buffers[event.agent_id].append(event)

        if isinstance(event, AgentStateChanged):
            self._agent_states[event.agent_id] = event.new_state
            if event.agent_id not in self._dynamic_agents and event.agent_id not in {a.agent_id for a in self.config.agents}:
                parent = self.broker.get_agent_parent(event.agent_id)
                harness = self.broker.get_agent_harness(event.agent_id)
                self._dynamic_agents[event.agent_id] = DynamicAgentInfo(parent=parent, task="", harness=harness)
                self._event_buffers.setdefault(event.agent_id, [])
                try:
                    agent_list = self.query_one(AgentList)
                    agent_list.add_agent_tile(event.agent_id, parent=parent)
                except Exception:
                    self.log.error(f"Failed to add tile for {event.agent_id}", exc_info=True)
            try:
                tile = self.query_one(f"#tile-{event.agent_id}", AgentTile)
                if event.new_state == AgentState.TERMINATED:
                    self._agent_modes.pop(event.agent_id, None)
                    self._agent_current_mode.pop(event.agent_id, None)
                    self._agent_models.pop(event.agent_id, None)
                    self._agent_current_model.pop(event.agent_id, None)
                    tile.remove()
                else:
                    tile.update_state(event.new_state)
            except Exception:
                log.debug("Agent tile not found for %s", event.agent_id, exc_info=True)

        if isinstance(event, AgentModesReceived):
            self._agent_modes[event.agent_id] = event.available_modes
            self._agent_current_mode[event.agent_id] = event.current_mode_id
            self._update_tile_mode(event.agent_id)
            self._update_input_bar_modes(event.agent_id)

        if isinstance(event, AgentModeChanged):
            self._agent_current_mode[event.agent_id] = event.mode_id
            self._update_tile_mode(event.agent_id)
            self._update_input_bar_current_mode(event.agent_id)

        if isinstance(event, AgentModelsReceived):
            self._agent_models[event.agent_id] = event.available_models
            self._agent_current_model[event.agent_id] = event.current_model_id
            self._update_input_bar_models(event.agent_id)

        if isinstance(event, AgentModelChanged):
            self._agent_current_model[event.agent_id] = event.model_id
            self._update_input_bar_current_model(event.agent_id)

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
            )
        elif isinstance(event, TerminalCreated):
            await feed.mount_terminal(event.terminal_id, event.terminal_process)
        elif isinstance(event, TurnComplete):
            await feed.finalize_current_message()
            feed.input_bar.set_busy(False)
        elif isinstance(event, PlanReceived):
            await feed.update_plan(event.entries)
        elif isinstance(event, AvailableCommandsReceived):
            if feed.input_bar is not None:
                feed.input_bar.update_slash_commands(event.commands)
        elif isinstance(event, UsageUpdated):
            self._update_usage_display(event)
        elif isinstance(event, HookFired):
            feed.add_hook_notification(event.hook_name)
        elif isinstance(event, InitialPromptDelivered):
            feed.add_prompt(event.text)
        elif isinstance(event, BrokerError):
            self.notify(event.message, severity=event.severity)
        elif isinstance(event, AgentStateChanged):
            if feed.input_bar is None:
                return
            if event.new_state in {AgentState.IDLE, AgentState.TERMINATED}:
                feed.input_bar.set_busy(False)
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
            feed.add_mcp_message(event.from_agent, event.to_agent, event.preview)
        elif isinstance(event, HookFired):
            feed.add_hook_notification(event.hook_name)
        elif isinstance(event, InitialPromptDelivered):
            feed.add_prompt(event.text)

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

    def _update_tile_mode(self, agent_id: str) -> None:
        """Resolve the current mode name and push it to the agent's tile."""
        mode_id = self._agent_current_mode.get(agent_id)
        modes = self._agent_modes.get(agent_id, [])
        mode_name = next((m.name for m in modes if m.id == mode_id), None)
        try:
            tile = self.query_one(f"#tile-{agent_id}", AgentTile)
            tile.update_mode(mode_name)
        except Exception:
            log.debug("Agent tile not found for mode update: %s", agent_id, exc_info=True)

    def _get_input_bar(self, agent_id: str) -> InputBar | None:
        """Return the InputBar for an agent, or None."""
        feed = self._panels.get(agent_id)
        return feed.input_bar if feed else None

    def _update_input_bar_modes(self, agent_id: str) -> None:
        """Push full mode list to the agent's input bar."""
        bar = self._get_input_bar(agent_id)
        if bar:
            bar.update_mode_info(
                self._agent_modes.get(agent_id, []),
                self._agent_current_mode.get(agent_id),
            )

    def _update_input_bar_current_mode(self, agent_id: str) -> None:
        """Push just the current mode id to the agent's input bar."""
        bar = self._get_input_bar(agent_id)
        mid = self._agent_current_mode.get(agent_id)
        if bar and mid:
            bar.update_current_mode(mid)

    def _update_input_bar_models(self, agent_id: str) -> None:
        """Push full model list to the agent's input bar."""
        bar = self._get_input_bar(agent_id)
        if bar:
            bar.update_model_info(
                self._agent_models.get(agent_id, []),
                self._agent_current_model.get(agent_id),
            )

    def _update_input_bar_current_model(self, agent_id: str) -> None:
        """Push just the current model id to the agent's input bar."""
        bar = self._get_input_bar(agent_id)
        mid = self._agent_current_model.get(agent_id)
        if bar and mid:
            bar.update_current_model(mid)

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
        self.run_worker(self._consume_broker_events(), exit_on_error=False, name="broker-consumer")

    async def select_agent(self, agent_id: str) -> None:
        """Switch the right panel to the given agent.

        Creates the panel and drains buffered events on first visit,
        then sets the reactive to trigger the watcher.

        Args:
            agent_id: The agent to display.
        """
        if agent_id not in self._panels:
            agent_cfg = next((a for a in self.config.agents if a.agent_id == agent_id), None)
            agent_name = agent_cfg.display_name if agent_cfg else agent_id
            harness = agent_cfg.harness if agent_cfg else ""
            if not harness:
                dyn = self._dynamic_agents.get(agent_id)
                if dyn:
                    harness = dyn.harness
            feed = ConversationFeed(agent_id, agent_name, self.config.project, harness=harness, cwd=agent_cfg.cwd if agent_cfg else "", id=f"feed-{agent_id}")
            self._panels[agent_id] = feed
            await self.query_one("#right").mount(feed)
            for event in self._event_buffers.get(agent_id, []):
                await self._replay_event(feed, event)
            self._event_buffers[agent_id] = []
            # Hide spinner if agent already passed INITIALIZING while buffered
            state = self._agent_states.get(agent_id)
            if state not in {AgentState.INITIALIZING, AgentState.BUSY, AgentState.CONFIGURING}:
                if feed.input_bar is not None:
                    feed.input_bar.set_busy(False)
            # Push any mode/model data that arrived before the panel existed
            self._update_input_bar_modes(agent_id)
            self._update_input_bar_models(agent_id)

        switcher = self.query_one("#right", ContentSwitcher)
        if self.selected_agent == agent_id and switcher.current != f"feed-{agent_id}":
            self.watch_selected_agent(agent_id)
        else:
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
            log.debug("MCP button deselect failed", exc_info=True)
        self._update_input_bar_state(agent_id, self._agent_states.get(agent_id, AgentState.IDLE))

    async def show_messages(self) -> None:
        """Toggle the MCP messages view. Close it if already active."""
        switcher = self.query_one("#right", ContentSwitcher)

        if switcher.current == "messages":
            # Close messages panel — return to selected agent
            if self.selected_agent:
                switcher.current = f"feed-{self.selected_agent}"
                self.watch_selected_agent(self.selected_agent)
            return

        for tile in self.query(AgentTile):
            tile.remove_class("tile-active")
        try:
            self.query_one("#mcp-btn", MCPButton).add_class("btn-active")
        except Exception:
            log.debug("MCP button select failed", exc_info=True)

        if self._mcp_panel is None:
            panel = MessageQueue(self._mcp_threads, id="messages")
            self._mcp_panel = panel
            await switcher.mount(panel)
        else:
            await self._mcp_panel.update_threads(self._mcp_threads)

        switcher.current = "messages"

    async def action_next_agent(self) -> None:
        """Cycle to the next agent in config order."""
        ids = [a.agent_id for a in self.config.agents]
        if not ids:
            return
        idx = ids.index(self.selected_agent) if self.selected_agent in ids else -1
        await self.select_agent(ids[(idx + 1) % len(ids)])

    async def action_messages(self) -> None:
        """Show the MCP messages panel."""
        await self.show_messages()

    @work
    async def action_launch(self) -> None:
        """Open the launch agent modal and launch the selected agent."""
        await self._do_launch()

    async def _do_launch(self) -> None:
        """Launch modal logic, separated for testability."""
        result = await self.push_screen_wait(LaunchAgentScreen())
        if result is not None:
            self._dynamic_agents[result.agent_id] = DynamicAgentInfo(parent=None, task="", harness=result.harness)
            self._event_buffers[result.agent_id] = []
            try:
                self.query_one(AgentList).add_agent_tile(result.agent_id)
            except Exception:
                log.debug("Failed to add tile for %s", result.agent_id, exc_info=True)
            await self.select_agent(result.agent_id)
            await self.broker.handle(LaunchAgent(agent_id=result.agent_id, config=result))

    async def action_help(self) -> None:
        """Open the help modal showing key bindings and usage."""
        await self.push_screen_wait(HelpScreen())

    async def action_quit(self) -> None:
        """Quit the app. Cleanup happens in on_unmount."""
        self.exit()

    async def on_unmount(self) -> None:
        """Terminate all agent subprocesses during Textual shutdown."""
        try:
            await self.broker.shutdown()
        except Exception:
            log.debug("Broker shutdown error", exc_info=True)
