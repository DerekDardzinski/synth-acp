"""Sidebar widgets: AgentTile, AgentList, LaunchButton, MCPButton."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.markup import escape
from textual.message import Message
from textual.widgets import Static

from synth_acp.models.agent import AgentState
from synth_acp.ui.widgets.gradient_bar import ActivityBar

STATUS_DOT: dict[AgentState, str] = {
    AgentState.INITIALIZING: "[$accent]●[/]",
    AgentState.IDLE: "[$success]●[/]",
    AgentState.BUSY: "[$warning]●[/]",
    AgentState.AWAITING_PERMISSION: "[$warning bold]●[/]",
    AgentState.TERMINATED: "[$text-muted]○[/]",
}

PREVIEW_TEXT: dict[AgentState, str] = {
    AgentState.INITIALIZING: "[$text-muted italic]initializing…[/]",
    AgentState.IDLE: "[$text-muted italic]idle[/]",
    AgentState.BUSY: "[$warning italic]working…[/]",
    AgentState.TERMINATED: "[$text-muted italic]terminated[/]",
    AgentState.AWAITING_PERMISSION: "[$warning bold italic]awaiting permission…[/]",
}

DEFAULT_PREVIEW = "[$text-muted italic]idle[/]"


_BUSY_STATES = {AgentState.INITIALIZING, AgentState.BUSY, AgentState.AWAITING_PERMISSION}


class _TerminateBtn(Static):
    """Inline close button for terminating an agent."""

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        super().__init__("\u2297", classes="tile-close")

    def on_click(self, event: object) -> None:
        """Post TerminateClicked and stop propagation."""
        from textual.events import Click

        if isinstance(event, Click):
            event.stop()
        if self.parent is not None:
            self.parent.post_message(AgentTile.TerminateClicked(self._agent_id))


class AgentTile(Vertical):
    """Clickable agent tile showing status dot, name, activity preview, and activity bar.

    Args:
        agent_id: Unique agent identifier.
        state: Initial agent state.
    """

    class TerminateClicked(Message):
        """Posted when the close button is clicked on a tile."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    def __init__(
        self,
        agent_id: str,
        state: AgentState = AgentState.IDLE,
        *,
        task: str = "",
        parent: str | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._state = state
        self._agent_task = task
        self._parent_agent = parent
        self._current_mode: str | None = None
        super().__init__(id=f"tile-{agent_id}")
        if state == AgentState.AWAITING_PERMISSION:
            self.add_class("tile-permission")

    def compose(self) -> ComposeResult:
        yield Static(self._build_markup(), classes="tile-label")
        yield _TerminateBtn(self._agent_id)
        yield ActivityBar(classes="tile-activity")

    def on_mount(self) -> None:
        self.query_one(ActivityBar).active = self._state in _BUSY_STATES

    def _build_markup(self) -> str:
        """Build the tile markup from current state."""
        dot = STATUS_DOT.get(self._state, "[$text-muted]○[/]")
        warn = (
            "  [$warning bold]⚠[/]"
            if self._state == AgentState.AWAITING_PERMISSION
            else ""
        )
        name = f"[$primary bold]{escape(self._agent_id)}[/]"
        if self._parent_agent:
            name += f" [$text-muted](via {escape(self._parent_agent)})[/]"
        mode_badge = (
            f"  [$accent dim]{escape(self._current_mode)}[/]"
            if self._current_mode
            else ""
        )
        preview = (
            f"[dim italic]{escape(self._agent_task)}[/dim italic]"
            if self._agent_task
            else PREVIEW_TEXT.get(self._state, DEFAULT_PREVIEW)
        )
        return f"{dot} {name}{warn}{mode_badge}\n  {preview}"

    def update_state(self, new_state: AgentState) -> None:
        """Update the tile to reflect a new agent state.

        Args:
            new_state: The new agent state.
        """
        self._state = new_state
        self.query_one(".tile-label", Static).update(self._build_markup())
        self.query_one(ActivityBar).active = new_state in _BUSY_STATES
        if new_state == AgentState.AWAITING_PERMISSION:
            self.add_class("tile-permission")
        else:
            self.remove_class("tile-permission")

    def update_mode(self, mode_name: str | None) -> None:
        """Update the mode badge shown in the tile.

        Pass None to clear the badge (e.g. on agent termination).

        Args:
            mode_name: Human-readable mode name, or None to clear.
        """
        self._current_mode = mode_name
        self.query_one(".tile-label", Static).update(self._build_markup())

    def on_click(self) -> None:
        """Select this agent in the app."""
        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
                    return
        app.run_worker(app.select_agent(self._agent_id))


class LaunchButton(Static):
    """'+ launch agent' button with dashed border."""

    def __init__(self) -> None:
        super().__init__("+ launch agent", id="launch-btn")

    def on_click(self) -> None:
        """Open the launch agent modal."""
        self.app.action_launch()


class MCPButton(Static):
    """MCP Messages button with pending count badge."""

    def __init__(self) -> None:
        super().__init__(self._build_markup(0), id="mcp-btn")
        self._count = 0

    def _build_markup(self, count: int) -> str:
        """Build button markup with optional badge."""
        badge = f"  [$warning bold]{count}[/]" if count else ""
        return f"◈  MCP Messages{badge}"

    def update_count(self, count: int) -> None:
        """Update the pending message count badge.

        Args:
            count: Number of pending messages.
        """
        self._count = count
        self.update(self._build_markup(count))

    def on_click(self) -> None:
        """Switch to MCP messages panel."""
        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
                    return
        app.run_worker(app.show_messages())


class AgentList(Vertical):
    """Sidebar container with agent tiles, launch button, and MCP button.

    Args:
        agents: List of agent IDs.
    """

    def __init__(self, agents: list[str]) -> None:
        super().__init__()
        self._agents = agents

    def compose(self) -> ComposeResult:
        """Yield sidebar label, scrollable agent tiles, and buttons."""
        with ScrollableContainer(id="agent-list"):
            yield Vertical(
                Static("AGENTS", id="sidebar-label"),
                id="sidebar-label-dock",
            )
            for agent_id in self._agents:
                yield AgentTile(agent_id)
        yield Vertical(
            LaunchButton(),
            MCPButton(),
            id="sidebar-buttons-dock",
        )

    def add_agent_tile(
        self, agent_id: str, *, task: str = "", parent: str | None = None
    ) -> None:
        """Mount a new agent tile into the scrollable container.

        Args:
            agent_id: Unique agent identifier.
            task: Optional task description.
            parent: Optional parent agent ID.
        """
        tile = AgentTile(agent_id, task=task, parent=parent)
        self.query_one("#agent-list", ScrollableContainer).mount(tile)
