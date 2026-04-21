"""Tool call block showing kind icon, title, status badge, and optional content."""

from __future__ import annotations

import re
from typing import Any

from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.highlight import highlight
from textual.widgets import Label, Markdown, Rule, Static

from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.widgets.copy_button import CopyButton
from synth_acp.ui.widgets.diff_view import DiffView

_ANSI_RE = re.compile(r"\x1b\[[\d;]*[A-Za-z]")


def _render_output(text: str, language: str | None) -> Content:
    """Render output text, interpreting ANSI escapes when present."""
    if _ANSI_RE.search(text):
        from rich.ansi import AnsiDecoder

        decoder = AnsiDecoder()
        lines = list(decoder.decode(text))
        parts = [Content.from_rich_text(line) for line in lines]
        return Content("\n").join(parts)
    return highlight(text, language=language)


TOOL_KIND_STYLE: dict[str, tuple[str, str]] = {
    "read": ("◎", "#3b82f6"),
    "edit": ("✎", "#a78bfa"),
    "execute": ("⚡", "#f97316"),
    "delete": ("✕", "#f87171"),
    "move": ("⇄", "#94a3b8"),
    "search": ("⌕", "#34d399"),
    "think": ("◌", "#c4b5fd"),
    "fetch": ("↓", "#38bdf8"),
    "switch_mode": ("⊞", "#64748b"),
}

_FALLBACK_STYLE = ("◈", "#64748b")

_STATUS_BADGE: dict[str, str] = {
    "completed": "[green]✓[/green]",
    "in_progress": "[yellow]⟳[/yellow]",
    "pending": "[dim]·[/dim]",
    "failed": "[red]✕[/red]",
}


def _extract_raw_output_text(raw_output: Any) -> str | None:
    """Extract display text from raw_output, handling nested formats.

    Supports:
    - Direct string
    - Dict with top-level keys: output, stdout, result, content
    - Kiro format: {"items": [{"Json": {"stdout": "...", "stderr": "..."}}]}
    """
    if isinstance(raw_output, str):
        return raw_output
    if not isinstance(raw_output, dict):
        return None
    # Top-level keys
    for key in ("output", "stdout", "result", "content"):
        if key in raw_output:
            return str(raw_output[key])
    # Kiro nested format: items[].Json.{stdout,stderr}
    items = raw_output.get("items")
    if isinstance(items, list):
        parts: list[str] = []
        for item in items:
            if isinstance(item, dict):
                json_val = item.get("Json") or item.get("json")
                if isinstance(json_val, dict):
                    for key in ("stdout", "output", "result", "content"):
                        if json_val.get(key):
                            parts.append(str(json_val[key]))
                            break
        if parts:
            return "".join(parts)
    return None


def _extract_exit_status(raw_output: Any) -> int | None:
    """Extract exit code from raw_output if available."""
    if not isinstance(raw_output, dict):
        return None
    items = raw_output.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                json_val = item.get("Json") or item.get("json")
                if isinstance(json_val, dict):
                    es = json_val.get("exit_status", "")
                    if isinstance(es, str) and "exit status:" in es:
                        try:
                            return int(es.split(":")[-1].strip())
                        except ValueError:
                            pass
    return None


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
        raw_output: Any = None,
        diffs: list[ToolCallDiff] | None = None,
        text_content: str | None = None,
        terminal_id: str | None = None,
    ) -> None:
        super().__init__(id=f"tool-{tool_call_id}")
        self._title = title
        self._kind = kind
        self._status = status
        self._terminal_id = terminal_id
        self._initial_locations = locations
        self._initial_raw_input = raw_input
        self._initial_raw_output = raw_output
        self._initial_diffs = diffs
        self._initial_text_content = text_content
        self._locations_rendered = False
        self._raw_input_rendered = False
        self._raw_output_rendered = False
        self._text_rendered = False
        self._copyable_parts: list[str] = []

    def _build_markup(self) -> Content:
        """Build the header as a Content object."""
        icon, color = TOOL_KIND_STYLE.get(self._kind, _FALLBACK_STYLE)
        badge = _STATUS_BADGE.get(self._status, "[dim]·[/dim]")
        return Content.from_markup(
            f"[{color}]{icon}[/{color}] $title  {badge}",
            title=self._title,
        )

    def compose(self):
        """Compose header and initial content widgets."""
        yield CopyButton(lambda: "\n".join(self._copyable_parts))
        yield Static(self._build_markup(), id="tc-header")
        yield from self._location_widgets(self._initial_locations)
        yield from self._raw_input_widgets(self._initial_raw_input)
        yield from self._text_widgets(self._initial_text_content)
        yield from self._diff_widgets(self._initial_diffs)
        yield from self._raw_output_widgets(self._initial_raw_output)

    def _location_widgets(self, locations: list[ToolCallLocation] | None) -> list[Static]:
        """Build location widget if applicable."""
        if not locations or self._locations_rendered:
            return []
        self._locations_rendered = True
        loc = locations[0]
        label = f"{loc.path}:{loc.line}" if loc.line is not None else loc.path
        return [Static(label, id="tc-location", markup=False)]

    def _raw_input_widgets(self, raw_input: Any) -> list[Label]:
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
        self._copyable_parts.append(f"$ {cmd}")
        content = highlight(f"$ {cmd}", language="bash")
        return [Label(content, id="tc-raw-input")]

    def _text_widgets(self, text_content: str | None) -> list[Markdown]:
        """Build text content widget if applicable."""
        if not text_content or self._text_rendered:
            return []
        self._text_rendered = True
        self._copyable_parts.append(text_content)
        return [Markdown(text_content, id="tc-text", open_links=False)]

    def _diff_widgets(self, diffs: list[ToolCallDiff] | None) -> list[DiffView]:
        """Build DiffView widgets for diffs."""
        if not diffs:
            return []
        return [
            DiffView(d.path, d.path, d.old_text or "", d.new_text)
            for d in diffs
        ]

    def _raw_output_widgets(self, raw_output: Any) -> list[Static | Label | VerticalScroll | Rule]:
        """Build raw output widget for execute/search/fetch kinds."""
        if self._kind not in {"execute", "search", "fetch"}:
            return []
        if raw_output is None or self._raw_output_rendered:
            return []
        text = _extract_raw_output_text(raw_output)
        if not text:
            return []
        self._raw_output_rendered = True
        self._copyable_parts.append(text)
        widgets: list[Static | Label | VerticalScroll | Rule] = []
        widgets.append(Rule(line_style="dashed", id="tc-output-sep"))
        lang = "bash" if self._kind == "execute" else None
        content = _render_output(text, language=lang)
        label = Label(content, id="tc-raw-output-label")
        scroll = VerticalScroll(label, id="tc-raw-output")
        widgets.append(scroll)
        exit_status = _extract_exit_status(raw_output)
        if exit_status is not None:
            exit_style = "success" if exit_status == 0 else "error"
            widgets.append(Rule(line_style="dashed", classes=f"shell-exit-{exit_style}"))
            self._exit_style = exit_style
        return widgets

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
        raw_output: Any = None,
        diffs: list[ToolCallDiff] | None = None,
        text_content: str | None = None,
    ) -> None:
        """Append new content widgets. Diffs always append; others are no-ops if rendered.

        Args:
            locations: File locations referenced by the tool call.
            raw_input: Raw input payload from the ACP SDK.
            raw_output: Raw output payload from the ACP SDK.
            diffs: File edit diffs extracted from the tool call.
            text_content: Extracted text content from the tool call.
        """
        widgets: list[Static | Label | Markdown | DiffView | VerticalScroll | Rule] = []
        widgets.extend(self._location_widgets(locations))
        widgets.extend(self._raw_input_widgets(raw_input))
        widgets.extend(self._text_widgets(text_content))
        diff_views = self._diff_widgets(diffs)
        for dv in diff_views:
            await dv.prepare()
        widgets.extend(diff_views)
        widgets.extend(self._raw_output_widgets(raw_output))
        for w in widgets:
            await self.mount(w)
        if hasattr(self, "_exit_style"):
            try:
                self.query_one("#tc-output-sep").add_class(f"shell-exit-{self._exit_style}")
            except Exception:
                pass
