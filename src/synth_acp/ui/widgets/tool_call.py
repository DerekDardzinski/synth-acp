"""Tool call block showing kind icon, title, status badge, and optional content."""

from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.widgets import Markdown, Static

from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.widgets.diff_view import DiffView

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


class ToolCallBlock(Vertical):
    """Displays a tool call with kind icon, title, status badge, and content.

    Args:
        tool_call_id: Unique tool call identifier.
        title: Human-readable tool call description.
        kind: Tool kind (read, edit, execute, delete, other).
        status: Current status (completed, in_progress, pending, failed).
        locations: File locations referenced by the tool call.
        raw_input: Raw input payload from the ACP SDK.
        diffs: File edit diffs extracted from the tool call.
        text_content: Extracted text content from the tool call.
    """

    def __init__(
        self,
        tool_call_id: str,
        title: str,
        kind: str,
        status: str,
        *,
        locations: list[ToolCallLocation] | None = None,
        raw_input: Any = None,
        diffs: list[ToolCallDiff] | None = None,
        text_content: str | None = None,
    ) -> None:
        super().__init__(id=f"tool-{tool_call_id}")
        self._title = title
        self._kind = kind
        self._status = status
        self._initial_locations = locations
        self._initial_raw_input = raw_input
        self._initial_diffs = diffs
        self._initial_text_content = text_content
        self._locations_rendered = False
        self._raw_input_rendered = False
        self._text_rendered = False

    def _build_markup(self) -> str:
        """Build the header markup."""
        icon, color = TOOL_KIND_STYLE.get(self._kind, _FALLBACK_STYLE)
        badge = _STATUS_BADGE.get(self._status, "[dim]·[/dim]")
        return f"[{color}]{icon}[/{color}] {self._title}  {badge}"

    def compose(self):
        """Compose header and initial content widgets."""
        yield Static(self._build_markup(), id="tc-header")
        yield from self._location_widgets(self._initial_locations)
        yield from self._raw_input_widgets(self._initial_raw_input)
        yield from self._text_widgets(self._initial_text_content)
        yield from self._diff_widgets(self._initial_diffs)

    def _location_widgets(self, locations: list[ToolCallLocation] | None) -> list[Static]:
        """Build location widget if applicable."""
        if not locations or self._locations_rendered:
            return []
        self._locations_rendered = True
        loc = locations[0]
        label = f"{loc.path}:{loc.line}" if loc.line is not None else loc.path
        return [Static(label, id="tc-location")]

    def _raw_input_widgets(self, raw_input: Any) -> list[Static]:
        """Build raw input widget if applicable."""
        if raw_input is None or self._raw_input_rendered:
            return []
        cmd = None
        if isinstance(raw_input, dict):
            cmd = raw_input.get("command") or raw_input.get("cmd")
        elif isinstance(raw_input, str):
            cmd = raw_input
        if cmd is None:
            return []
        self._raw_input_rendered = True
        return [Static(f"$ {cmd}", id="tc-raw-input")]

    def _text_widgets(self, text_content: str | None) -> list[Markdown]:
        """Build text content widget if applicable."""
        if not text_content or self._text_rendered:
            return []
        self._text_rendered = True
        return [Markdown(text_content, id="tc-text")]

    def _diff_widgets(self, diffs: list[ToolCallDiff] | None) -> list[DiffView]:
        """Build DiffView widgets for diffs."""
        if not diffs:
            return []
        collapsed = self._status == "completed"
        return [
            DiffView(d.path, d.old_text, d.new_text, collapsed=collapsed)
            for d in diffs
        ]

    def update_status(self, status: str) -> None:
        """Update the status badge.

        Args:
            status: New status string.
        """
        self._status = status
        self.query_one("#tc-header", Static).update(self._build_markup())

    async def update_content(
        self,
        locations: list[ToolCallLocation] | None = None,
        raw_input: Any = None,
        diffs: list[ToolCallDiff] | None = None,
        text_content: str | None = None,
    ) -> None:
        """Append new content widgets. Diffs always append; others are no-ops if rendered.

        Args:
            locations: File locations referenced by the tool call.
            raw_input: Raw input payload from the ACP SDK.
            diffs: File edit diffs extracted from the tool call.
            text_content: Extracted text content from the tool call.
        """
        widgets: list[Static | Markdown | DiffView] = []
        widgets.extend(self._location_widgets(locations))
        widgets.extend(self._raw_input_widgets(raw_input))
        widgets.extend(self._text_widgets(text_content))
        widgets.extend(self._diff_widgets(diffs))
        for w in widgets:
            await self.mount(w)
