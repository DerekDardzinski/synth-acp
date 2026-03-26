"""LaunchAgentScreen — modal for selecting an agent to launch."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button

from synth_acp.models.agent import AgentState

_RUNNING_STATES = {
    AgentState.INITIALIZING,
    AgentState.IDLE,
    AgentState.BUSY,
    AgentState.AWAITING_PERMISSION,
}


class LaunchAgentScreen(ModalScreen[str | None]):
    """Modal listing agents for launch selection.

    Args:
        agents: List of ``(agent_id, display_name, state)`` tuples.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Close")]

    def __init__(self, agents: list[tuple[str, str, AgentState | None]]) -> None:
        super().__init__()
        self._agents = agents

    def compose(self) -> ComposeResult:
        """Yield a centered container with one button per agent."""
        with Vertical(id="launch-container"):
            for agent_id, display_name, state in self._agents:
                label = f"{display_name} ({state.value if state else 'unstarted'})"
                disabled = state in _RUNNING_STATES if state else False
                yield Button(label, id=f"launch-{agent_id}", disabled=disabled)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the selected agent ID.

        Args:
            event: Button press event.
        """
        agent_id = event.button.id
        if agent_id:
            self.dismiss(agent_id.removeprefix("launch-"))

    def action_dismiss_none(self) -> None:
        """Dismiss the modal without selecting an agent."""
        self.dismiss(None)
