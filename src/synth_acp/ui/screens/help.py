"""HelpScreen — modal showing key bindings and usage info."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class HelpScreen(ModalScreen[None]):
    """Modal displaying key bindings and routing syntax."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Close")]

    def compose(self) -> ComposeResult:
        """Yield a centered container with bindings table and usage info."""
        with Vertical(id="help-container"):
            yield Static(self._build_bindings_text(), id="help-bindings")
            yield Static(
                "[bold]Routing[/bold]\n  @agent-id  — Route a message to a specific agent",
                id="help-routing",
            )
            yield Static(
                "[bold]Layout[/bold]\n"
                "  Left sidebar: agent list and MCP button\n"
                "  Right panel: conversation feed or MCP messages",
                id="help-layout",
            )

    def _build_bindings_text(self) -> str:
        """Build a formatted string of key bindings from the app.

        Returns:
            Formatted bindings text.
        """
        lines = ["[bold]Key Bindings[/bold]"]
        for binding in self.app.BINDINGS:
            if isinstance(binding, Binding):
                lines.append(f"  {binding.key:<12} {binding.description}")
            else:
                key, _action, desc = binding
                lines.append(f"  {key:<12} {desc}")
        return "\n".join(lines)

    def action_dismiss_none(self) -> None:
        """Dismiss the help modal."""
        self.dismiss(None)
