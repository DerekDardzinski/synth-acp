"""LaunchAgentScreen — modal form for configuring and launching a new agent."""

from __future__ import annotations

import shutil
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import HarnessEntry


def _detect_harnesses() -> list[HarnessEntry]:
    """Return harness entries whose binary is found in PATH."""
    available: list[HarnessEntry] = []
    for entry in load_harness_registry():
        for binary in entry.binary_names:
            if shutil.which(binary):
                available.append(entry)
                break
    return available


class LaunchAgentScreen(ModalScreen[AgentConfig | None]):
    """Modal form for configuring a new agent to launch.

    Returns an AgentConfig on submit, or None on cancel.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Close")]

    def __init__(self) -> None:
        super().__init__()
        self._harnesses = _detect_harnesses()

    def compose(self) -> ComposeResult:
        """Yield the launch form."""
        options = [(entry.name, entry.short_name) for entry in self._harnesses]
        with Vertical(id="launch-container"):
            yield Label("Launch Agent", id="launch-title")
            yield Label("Harness", classes="field-label")
            yield Select(options, id="harness-select", prompt="Select harness")
            yield Label("Agent ID", classes="field-label")
            yield Input(placeholder="e.g. my-agent", id="agent-id-input")
            yield Label("Agent Mode [dim](optional)[/dim]", classes="field-label")
            yield Input(placeholder="e.g. code, plan, chat", id="agent-mode-input")
            yield Button("Launch", id="launch-submit", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Validate and dismiss with an AgentConfig."""
        if event.button.id != "launch-submit":
            return

        harness = self.query_one("#harness-select", Select).value
        agent_id = self.query_one("#agent-id-input", Input).value.strip()
        agent_mode = self.query_one("#agent-mode-input", Input).value.strip() or None

        if harness is Select.BLANK:
            self.notify("Select a harness", severity="warning")
            return
        if not agent_id:
            self.notify("Agent ID is required", severity="warning")
            return

        try:
            config = AgentConfig(
                agent_id=agent_id,
                harness=str(harness),
                agent_mode=agent_mode,
            )
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return

        self.dismiss(config)

    def action_dismiss_none(self) -> None:
        """Dismiss the modal without launching."""
        self.dismiss(None)
