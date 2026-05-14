"""Thought/reasoning block using ExpandableSection with streaming markdown."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets.markdown import Markdown, MarkdownStream

from synth_acp.ui.widgets.copy_button import CopyButton
from synth_acp.ui.widgets.expandable_section import ExpandableSection


class ThoughtBlock(Vertical, can_focus=False):
    """Thought/reasoning block using ExpandableSection.

    Starts expanded while streaming. Auto-collapses on finalize.
    Preview shows last ~60 chars of thought content (debounced).
    """

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._stream: MarkdownStream | None = None
        self._preview_timer: asyncio.TimerHandle | None = None

    def compose(self) -> ComposeResult:
        """Yield CopyButton + ExpandableSection(Markdown, start_expanded=True)."""
        yield CopyButton(lambda: "".join(self._chunks))
        yield ExpandableSection(
            Markdown("", open_links=False),
            start_expanded=True,
        )

    @property
    def _section(self) -> ExpandableSection:
        return self.query_one(ExpandableSection)

    @property
    def _markdown(self) -> Markdown:
        return self.query_one(Markdown)

    async def append_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk. Updates preview (debounced ~200ms)."""
        if not self._chunks:
            self._section.set_activity(True)
        self._chunks.append(chunk)
        if self._stream is None:
            self._stream = Markdown.get_stream(self._markdown)
        await self._stream.write(chunk)
        # Debounce preview update
        if self._preview_timer is not None:
            self._preview_timer.cancel()
        loop = asyncio.get_event_loop()
        self._preview_timer = loop.call_later(0.2, self._update_preview)

    def _update_preview(self) -> None:
        """Extract last ~60 chars from chunks and update section preview."""
        text = "".join(self._chunks).rstrip()
        if len(text) > 60:
            text = "…" + text[-60:]
        self._section.set_preview(text)

    async def finalize(self) -> None:
        """Stop stream, collapse section, set activity=False, freeze preview."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
            # Work around Textual Markdown.append() cursor bug that can
            # leave fenced code blocks empty after incremental re-parse.
            full_content = "".join(self._chunks)
            if full_content:
                await self._markdown.update(full_content)
        if self._preview_timer is not None:
            self._preview_timer.cancel()
            self._preview_timer = None
        self._update_preview()
        self._section.collapsed = True
        self._section.set_activity(False)
