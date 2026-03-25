"""Tool call block showing kind icon, title, and status badge."""

from __future__ import annotations

from textual.widgets import Static

TOOL_KIND_STYLE: dict[str, tuple[str, str]] = {
    "read": ("◎", "#3b82f6"),
    "edit": ("✎", "#a78bfa"),
    "execute": ("⚡", "#f97316"),
    "delete": ("✕", "#f87171"),
}

_FALLBACK_STYLE = ("◈", "#64748b")

_STATUS_BADGE: dict[str, str] = {
    "completed": "[green]✓[/green]",
    "in_progress": "[yellow]⟳[/yellow]",
    "pending": "[dim]·[/dim]",
    "failed": "[red]✕[/red]",
}


class ToolCallBlock(Static):
    """Displays a tool call with kind icon, title, and status badge.

    Args:
        tool_call_id: Unique tool call identifier.
        title: Human-readable tool call description.
        kind: Tool kind (read, edit, execute, delete, other).
        status: Current status (completed, in_progress, pending, failed).
    """

    def __init__(self, tool_call_id: str, title: str, kind: str, status: str) -> None:
        self._title = title
        self._kind = kind
        self._status = status
        super().__init__(self._build_markup(), id=f"tool-{tool_call_id}")

    def _build_markup(self) -> str:
        """Build the tool call markup."""
        icon, color = TOOL_KIND_STYLE.get(self._kind, _FALLBACK_STYLE)
        badge = _STATUS_BADGE.get(self._status, "[dim]·[/dim]")
        return f"[{color}]{icon}[/{color}] {self._title}  {badge}"

    def update_status(self, status: str) -> None:
        """Update the status badge.

        Args:
            status: New status string.
        """
        self._status = status
        self.update(self._build_markup())
