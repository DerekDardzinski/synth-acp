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
        self._streaming_collapsed: bool = False
        # Held directly rather than queried, matching AgentMessage. `AwaitMount` resolves
        # when THIS widget has composed, but its ExpandableSection composes its own body a
        # step later, so a DOM query for the Markdown can race and raise NoMatches on the
        # first chunk under a heavy event backlog.
        self._md = Markdown("", open_links=False)
        self._expandable = ExpandableSection(self._md, start_expanded=True)

    def compose(self) -> ComposeResult:
        """Yield CopyButton + ExpandableSection(Markdown, start_expanded=True)."""
        yield CopyButton(lambda: "".join(self._chunks))
        yield self._expandable

    @property
    def _section(self) -> ExpandableSection:
        return self._expandable

    @property
    def _markdown(self) -> Markdown:
        return self._md

    async def append_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk. Updates preview (debounced ~200ms)."""
        if not self._chunks:
            self._section.set_activity(True)
        self._chunks.append(chunk)
        if self._stream is None:
            self._stream = Markdown.get_stream(self._markdown)
        await self._stream.write(chunk)
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        """(Re)arm the debounced preview update, replacing any pending one."""
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
        # Remembered so reopen() restores the state actually held while streaming rather
        # than forcing a thought the user collapsed back open.
        self._streaming_collapsed = self._section.collapsed
        self._section.collapsed = True
        self._section.set_activity(False)

    async def reopen(self) -> None:
        """Undo finalize() so reasoning streaming can resume.

        Restores everything finalize stopped: a FRESH MarkdownStream over the existing
        Markdown widget, the expanded/collapsed state held while streaming, the activity
        indicator, and preview timing. ``Markdown.update`` is NOT called — the stream
        appends to the existing document.

        Idempotent.
        """
        if self._stream is not None:
            return
        self._stream = Markdown.get_stream(self._markdown)
        self._section.collapsed = self._streaming_collapsed
        self._section.set_activity(True)
        self._schedule_preview()
