"""Plan widget showing agent task entries with status indicators."""

from __future__ import annotations

from acp.schema import PlanEntry
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.markup import escape
from textual.widgets import Static

_STATUS_ICON: dict[str, str] = {
    "completed": "[green]✓[/green]",
    "in_progress": "[yellow]⟳[/yellow]",
    "pending": "[dim]·[/dim]",
}

_PRIORITY_BADGE: dict[str, str] = {
    "high": " [red dim]high[/red dim]",
    "medium": "",
    "low": "",
}


class PlanBlock(Vertical):
    """Displays an agent plan as a list of status-tracked entries.

    Args:
        entries: List of PlanEntry objects from the ACP SDK.
    """

    DEFAULT_CSS = """
    PlanBlock {
        height: auto;
        border-left: wide $success;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    PlanBlock .plan-header {
        color: $text-muted;
        text-style: italic;
    }
    PlanBlock .plan-entry {
        height: auto;
        padding-left: 1;
        color: $text;
    }
    PlanBlock .plan-entry.completed {
        color: $text-muted;
        text-style: strike;
    }
    """

    def __init__(self, entries: list[PlanEntry]) -> None:
        super().__init__()
        self._entries = entries

    def compose(self) -> ComposeResult:
        yield Static("[dim]Plan[/dim]", classes="plan-header")
        for entry in self._entries:
            icon = _STATUS_ICON.get(entry.status, "[dim]·[/dim]")
            priority = _PRIORITY_BADGE.get(entry.priority, "")
            text = f"{icon} {escape(entry.content)}{priority}"
            classes = f"plan-entry {entry.status}"
            yield Static(text, classes=classes)
