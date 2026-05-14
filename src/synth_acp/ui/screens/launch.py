"""LaunchAgentScreen — modal form for configuring and launching a new agent."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select
from textual.widgets.option_list import Option

from synth_acp.discovery import DiscoveredAgent, discover_agents
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import HarnessEntry
from synth_acp.ui.file_discovery import fuzzy_score


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
        self._agents: list[DiscoveredAgent] = []
        self._selected_agent: str | None = None

    def compose(self) -> ComposeResult:
        """Yield the launch form."""
        options = [(entry.name, entry.short_name) for entry in self._harnesses]
        with Vertical(id="launch-container"):
            yield Label("Launch Agent", id="launch-title")
            yield Label("Harness", classes="field-label")
            yield Select(options, id="harness-select", prompt="Select harness")
            yield Label("Agent ID", classes="field-label")
            yield Input(placeholder="e.g. my-agent", id="agent-id-input")
            yield Label("Agent [dim](optional)[/dim]", classes="field-label")
            yield Input(
                placeholder="Type to filter agents...", id="agent-filter-input"
            )
            yield OptionList(id="agent-option-list")
            yield Label("Working Directory", classes="field-label")
            yield Input(value=str(Path.cwd().resolve()), id="cwd-input")
            yield Button("Launch", id="launch-submit", variant="primary")

    def on_mount(self) -> None:
        """Hide agent widgets until a harness is selected."""
        self.query_one("#agent-filter-input", Input).display = False
        self.query_one("#agent-option-list", OptionList).display = False

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle harness Select change."""
        if event.select.id == "harness-select":
            self._on_harness_changed(event)

    def _on_harness_changed(self, event: Select.Changed) -> None:
        """Populate agent list from discovery when harness changes."""
        filter_input = self.query_one("#agent-filter-input", Input)
        option_list = self.query_one("#agent-option-list", OptionList)

        if event.value is Select.BLANK:
            filter_input.display = False
            option_list.display = False
            self._agents = []
            self._selected_agent = None
            return

        harness = next(
            (h for h in self._harnesses if h.short_name == event.value), None
        )
        if harness is None:
            filter_input.display = True
            option_list.display = False
            self._agents = []
            return

        self._agents = discover_agents(harness, Path.cwd())
        self._selected_agent = None
        filter_input.value = ""
        if self._agents:
            filter_input.display = True
            option_list.display = True
            self._refresh_agent_list("")
        else:
            filter_input.display = True
            filter_input.placeholder = "Enter agent mode..."
            option_list.display = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter agent list on input change."""
        if event.input.id == "agent-filter-input":
            self._selected_agent = None
            self._refresh_agent_list(event.value)

    def _refresh_agent_list(self, query: str) -> None:
        """Update the OptionList with fuzzy-filtered agents."""
        option_list = self.query_one("#agent-option-list", OptionList)
        option_list.clear_options()

        if not self._agents:
            option_list.display = False
            return

        if query:
            scored = []
            for agent in self._agents:
                # Score against both name and qualified_name
                s1 = fuzzy_score(query, agent.name)
                s2 = fuzzy_score(query, agent.qualified_name)
                best = max(s for s in (s1, s2) if s is not None) if (s1 is not None or s2 is not None) else None
                if best is not None:
                    scored.append((best, agent))
            scored.sort(key=lambda x: (-x[0], len(x[1].qualified_name)))
            agents = [a for _, a in scored[:15]]
        else:
            agents = self._agents[:15]

        for agent in agents:
            option_list.add_option(
                Option(f"{agent.name} [dim]({agent.source})[/]", id=agent.qualified_name)
            )
        option_list.display = bool(agents)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle agent selection from the filtered list."""
        if event.option_list.id != "agent-option-list":
            return
        event.stop()
        if event.option_id:
            self._selected_agent = event.option_id
            # Show selection in the filter input
            agent = next((a for a in self._agents if a.qualified_name == event.option_id), None)
            if agent:
                filter_input = self.query_one("#agent-filter-input", Input)
                filter_input.value = agent.name
                self.query_one("#agent-option-list", OptionList).display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Validate and dismiss with an AgentConfig."""
        if event.button.id != "launch-submit":
            return

        harness = self.query_one("#harness-select", Select).value
        agent_id = self.query_one("#agent-id-input", Input).value.strip()
        filter_value = self.query_one("#agent-filter-input", Input).value.strip()
        cwd = self.query_one("#cwd-input", Input).value.strip()

        if harness is Select.BLANK:
            self.notify("Select a harness", severity="warning")
            return
        if not agent_id:
            self.notify("Agent ID is required", severity="warning")
            return

        # Use selected agent from list, or raw input as agent_mode
        agent_mode: str | None = self._selected_agent or filter_value or None

        resolved_cwd = str(Path(cwd).resolve()) if cwd else str(Path.cwd().resolve())

        try:
            config = AgentConfig(
                agent_id=agent_id,
                harness=str(harness),
                agent_mode=agent_mode,
                cwd=resolved_cwd,
            )
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return

        self.dismiss(config)

    def action_dismiss_none(self) -> None:
        """Dismiss the modal without launching."""
        self.dismiss(None)
