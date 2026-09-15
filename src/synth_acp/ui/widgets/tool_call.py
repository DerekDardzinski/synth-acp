"""Tool call block showing kind icon, title, status badge, and optional content."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rich.highlighter import ReprHighlighter
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.highlight import highlight
from textual.lazy import Lazy
from textual.widget import Widget
from textual.widgets import Label, Markdown, Rule, Static

from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.widgets.copy_button import CopyButton
from synth_acp.ui.widgets.diff_view import DiffView
from synth_acp.ui.widgets.expandable_section import ExpandableSection

_HIGHLIGHTER = ReprHighlighter()

MAX_DIFF_ATTEMPTS = 2
"""Render attempts allowed per diff before it is abandoned permanently."""


@dataclass(frozen=True)
class DiffKey:
    """Immutable identity of one diff within a tool call."""

    path: str
    old_hash: str
    new_hash: str


class DiffState(StrEnum):
    """Lifecycle of one diff's render work."""

    QUEUED = "queued"
    RENDERING = "rendering"
    RENDERED = "rendered"
    FAILED = "failed"


@dataclass
class DiffRecord:
    """Per-key work state: payload, claim, retry count and fallback reference."""

    diff: ToolCallDiff
    state: DiffState
    attempts: int = 0
    fallback: Widget | None = None

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


class ToolCallBlock(Vertical, can_focus=False):
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
        diffs: list[ToolCallDiff] | None = None,  # noqa: ARG002 — scheduled by the feed
        text_content: str | None = None,
        terminal_id: str | None = None,
    ) -> None:
        super().__init__(id=f"tool-{tool_call_id}")
        self._tool_call_id = tool_call_id
        self._title = title
        self._kind = kind
        self._status = status
        self._terminal_id = terminal_id
        self._initial_locations = locations
        self._initial_raw_input = raw_input
        self._initial_raw_output = raw_output
        self._initial_text_content = text_content
        self._locations_rendered = False
        self._raw_input_rendered = False
        self._raw_output_rendered = False
        self._text_rendered = False
        self._copyable_parts: list[str] = []
        self._nested_section: ExpandableSection | None = None
        self._nested_count: int = 0
        self._diff_states: dict[DiffKey, DiffRecord] = {}

    def _build_markup(self) -> Content:
        """Build the header as a Content object."""
        icon, color = TOOL_KIND_STYLE.get(self._kind, _FALLBACK_STYLE)
        badge = _STATUS_BADGE.get(self._status, "[dim]·[/dim]")
        return Content.from_markup(
            f"[{color}]{icon}[/{color}] $title  {badge}",
            title=self._title,
        )

    def compose(self):
        """Compose header and initial content widgets.

        Diffs are NOT composed here. Highlighting a diff costs ~186ms of synchronous
        tree-sitter work, so DiffViews are prepared off-thread by the feed's diff
        executor and mounted when ready — see ``schedule_diffs``.
        """
        yield CopyButton(lambda: "\n".join(self._copyable_parts))
        yield Static(self._build_markup(), id="tc-header")
        for w in self._location_widgets(self._initial_locations):
            yield w
        for w in self._raw_input_widgets(self._initial_raw_input):
            yield Lazy(w)
        for w in self._text_widgets(self._initial_text_content):
            yield w
        for w in self._raw_output_widgets(self._initial_raw_output):
            yield w

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
        if self._kind not in {"think", "other"}:
            return []
        self._text_rendered = True
        self._copyable_parts.append(text_content)
        return [Markdown(text_content, id="tc-text", open_links=False)]

    @staticmethod
    def _diff_key(diff: ToolCallDiff) -> DiffKey:
        """Identity of a diff: its path plus content hashes."""
        return DiffKey(
            path=diff.path,
            old_hash=hashlib.sha256((diff.old_text or "").encode()).hexdigest(),
            new_hash=hashlib.sha256(diff.new_text.encode()).hexdigest(),
        )

    def schedule_diffs(self, diffs: Sequence[ToolCallDiff]) -> None:
        """Register diffs for rendering. SYNCHRONOUS, non-blocking, does not wake.

        Single-flight per DiffKey: an identical diff already queued, being rendered, or
        already rendered is a no-op, while a distinct diff always enqueues. A failed diff
        gets one retry and is then abandoned permanently.

        Waking the executor is the feed's job — it is already the caller on both entry
        paths, and holding a feed reference here would create an import cycle.

        Args:
            diffs: Diffs carried by this tool call update.
        """
        for diff in diffs:
            key = self._diff_key(diff)
            record = self._diff_states.get(key)
            if record is None:
                self._diff_states[key] = DiffRecord(diff=diff, state=DiffState.QUEUED)
            elif record.state is DiffState.FAILED and record.attempts < MAX_DIFF_ATTEMPTS:
                record.state = DiffState.QUEUED

    def claim_next_diff(self) -> tuple[DiffKey, DiffRecord] | None:
        """Atomically claim one QUEUED diff, transitioning it to RENDERING.

        Synchronous, so the read-and-mutate cannot yield and two workers can never claim
        the same key.

        Returns:
            The claimed key and record, or None when nothing is claimable.
        """
        for key, record in self._diff_states.items():
            if record.state is DiffState.QUEUED:
                record.state = DiffState.RENDERING
                return key, record
        return None

    def release_claim(self, key: DiffKey) -> None:
        """Return a RENDERING claim to QUEUED.

        Called from the executor's finally block: a worker cancelled between claim and
        completion would otherwise leave the key RENDERING forever, and redelivery of a
        RENDERING key is a no-op.

        Args:
            key: The key to release.
        """
        record = self._diff_states.get(key)
        if record is not None and record.state is DiffState.RENDERING:
            record.state = DiffState.QUEUED

    def has_pending_diffs(self) -> bool:
        """True while any diff is queued or being rendered."""
        return any(
            record.state in {DiffState.QUEUED, DiffState.RENDERING}
            for record in self._diff_states.values()
        )

    @staticmethod
    def make_diff_view(record: DiffRecord) -> DiffView:
        """Build an unmounted DiffView for a record's payload."""
        diff = record.diff
        return DiffView(diff.path, diff.path, diff.old_text or "", diff.new_text)

    @staticmethod
    def make_diff_fallback(record: DiffRecord) -> Static:
        """Build a plain unhighlighted rendering, used when highlighting fails."""
        return Static(record.diff.new_text, classes="diff-fallback", markup=False)

    def _raw_output_widgets(self, raw_output: Any) -> list[Rule | VerticalScroll]:
        """Build raw output widget for execute/search/fetch kinds."""
        if self._kind not in {"execute", "search", "fetch"}:
            return []
        if raw_output is None or self._raw_output_rendered or self._terminal_id is not None:
            return []
        text = _extract_raw_output_text(raw_output)
        if not text:
            return []
        self._raw_output_rendered = True
        self._copyable_parts.append(text)
        widgets: list[Rule | VerticalScroll] = []
        widgets.append(Rule(line_style="dashed", id="tc-output-sep"))
        rich_text = Text.from_ansi(text)
        _HIGHLIGHTER.highlight(rich_text)
        content = Content.from_rich_text(rich_text)
        label = Label(content, id="tc-raw-output-label")
        widgets.append(VerticalScroll(label, id="tc-raw-output"))
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
        try:
            self.query_one("#tc-header", Static).update(self._build_markup())
        except NoMatches:
            pass

    async def update_content(
        self,
        locations: list[ToolCallLocation] | None = None,
        raw_input: Any = None,
        raw_output: Any = None,
        diffs: list[ToolCallDiff] | None = None,  # noqa: ARG002
        text_content: str | None = None,
    ) -> None:
        """Append new content widgets. Non-diff content is a no-op once rendered.

        Diffs are NOT rendered here: they are registered by ``schedule_diffs`` and mounted
        off-thread by the feed's diff executor, so this method never awaits ``prepare()``
        and never blocks the App message pump on tree-sitter highlighting.

        Args:
            locations: File locations referenced by the tool call.
            raw_input: Raw input payload from the ACP SDK.
            raw_output: Raw output payload from the ACP SDK.
            diffs: Accepted and ignored — the feed schedules these instead.
            text_content: Extracted text content from the tool call.
        """
        widgets: list[Static | Label | Markdown | VerticalScroll | Rule] = []
        widgets.extend(self._location_widgets(locations))
        widgets.extend(self._raw_input_widgets(raw_input))
        widgets.extend(self._text_widgets(text_content))
        widgets.extend(self._raw_output_widgets(raw_output))
        if widgets:
            await self.mount_compose(iter(widgets))
        if hasattr(self, "_exit_style"):
            try:
                self.query_one("#tc-output-sep").add_class(f"shell-exit-{self._exit_style}")
            except Exception:
                pass

    async def mount_nested_child(self, block: ToolCallBlock) -> None:
        """Mount a nested tool call into the expandable section.

        Creates the ExpandableSection on first call (lazy init, starts collapsed).
        Updates preview to the new child's title. Increments nested count.
        """
        if self._nested_section is None:
            section = ExpandableSection()
            self._nested_section = section
            await self.mount(section)
            section.set_activity(True)
        await self._nested_section.content.mount(block)
        self._nested_section.set_preview(block._title)
        self._nested_count += 1

    def finalize_nested(self) -> None:
        """Called when parent tool call completes.

        Sets activity=False, preview to summary.
        """
        if self._nested_section is None:
            return
        self._nested_section.set_activity(False)
        self._nested_section.set_preview(f"✓ {self._nested_count} tool calls")
