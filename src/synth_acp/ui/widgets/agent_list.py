"""Sidebar widgets: AgentTile, AgentList, LaunchButton, MCPButton."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static

from synth_acp.models.agent import AgentState

STATUS_DOT: dict[AgentState, str] = {
    AgentState.INITIALIZING: "[cyan]●[/cyan]",
    AgentState.IDLE: "[green]●[/green]",
    AgentState.BUSY: "[yellow]●[/yellow]",
    AgentState.AWAITING_PERMISSION: "[bold yellow]●[/bold yellow]",
    AgentState.TERMINATED: "[dim]○[/dim]",
}

PREVIEW_TEXT: dict[AgentState, str] = {
    AgentState.INITIALIZING: "[dim italic]initializing…[/dim italic]",
    AgentState.IDLE: "[dim italic]idle[/dim italic]",
    AgentState.BUSY: "[yellow italic]working…[/yellow italic]",
    AgentState.TERMINATED: "[dim italic]terminated[/dim italic]",
    AgentState.AWAITING_PERMISSION: "[bold yellow italic]awaiting permission…[/bold yellow italic]",
}

DEFAULT_PREVIEW = "[dim italic]idle[/dim italic]"


class AgentTile(Static):
    """Clickable agent tile showing status dot, colored name, and activity preview.

    Args:
        agent_id: Unique agent identifier.
        color: Hex color for the agent name.
        state: Initial agent state.
    """

    def __init__(
        self,
        agent_id: str,
        color: str,
        state: AgentState = AgentState.IDLE,
        *,
        task: str = "",
        parent: str | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._color = color
        self._state = state
        self._task = task
        self._parent = parent
        super().__init__(self._build_markup(), id=f"tile-{agent_id}")
        if state == AgentState.AWAITING_PERMISSION:
            self.add_class("tile-permission")

    def _build_markup(self) -> str:
        """Build the tile markup from current state."""
        dot = STATUS_DOT.get(self._state, "[dim]○[/dim]")
        warn = (
            "  [bold yellow]⚠[/bold yellow]"
            if self._state == AgentState.AWAITING_PERMISSION
            else ""
        )
        if self._task:
            preview = f"[dim italic]{self._task}[/dim italic]"
        elif self._parent:
            preview = f"[dim]via {self._parent}[/dim]"
        else:
            preview = PREVIEW_TEXT.get(self._state, DEFAULT_PREVIEW)
        return f"{dot} [bold {self._color}]{self._agent_id}[/bold {self._color}]{warn}\n  {preview}"

    def update_state(self, new_state: AgentState) -> None:
        """Update the tile to reflect a new agent state.

        Args:
            new_state: The new agent state.
        """
        self._state = new_state
        self.update(self._build_markup())
        if new_state == AgentState.AWAITING_PERMISSION:
            self.add_class("tile-permission")
        else:
            self.remove_class("tile-permission")

    def on_click(self) -> None:
        """Select this agent in the app."""
        from synth_acp.ui.app import SynthApp

        app = self.app
        assert isinstance(app, SynthApp)
        app.run_worker(app.select_agent(self._agent_id))


class LaunchButton(Static):
    """'+ launch agent' button with dashed border."""

    def __init__(self) -> None:
        super().__init__("[dim]+ launch agent[/dim]", id="launch-btn")

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
        badge = f"  [bold #f97316]{count}[/bold #f97316]" if count else ""
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
        assert isinstance(app, SynthApp)
        app.run_worker(app.show_messages())


class AgentList(Vertical):
    """Sidebar container with agent tiles, launch button, and MCP button.

    Args:
        agents: List of (agent_id, color) pairs.
    """

    def __init__(self, agents: list[tuple[str, str]]) -> None:
        super().__init__()
        self._agents = agents

    def compose(self) -> ComposeResult:
        """Yield sidebar label, scrollable agent tiles, and buttons."""
        yield Static("AGENTS", id="sidebar-label")
        with ScrollableContainer(id="agent-list"):
            for agent_id, color in self._agents:
                yield AgentTile(agent_id, color)
            yield LaunchButton()
        yield MCPButton()

    def add_agent_tile(
        self, agent_id: str, color: str, *, task: str = "", parent: str | None = None
    ) -> None:
        """Mount a new agent tile into the scrollable container.

        Args:
            agent_id: Unique agent identifier.
            color: Hex color for the agent name.
            task: Optional task description.
            parent: Optional parent agent ID.
        """
        tile = AgentTile(agent_id, color, task=task, parent=parent)
        launch_btn = self.query_one("#launch-btn", LaunchButton)
        self.query_one("#agent-list", ScrollableContainer).mount(tile, before=launch_btn)
