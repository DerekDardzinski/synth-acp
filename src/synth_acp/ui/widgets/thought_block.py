"""Collapsible thought/reasoning block with streaming markdown."""

from __future__ import annotations

from textual.widgets import Collapsible
from textual.widgets.markdown import Markdown, MarkdownStream


class ThoughtBlock(Collapsible):
    """Collapsible block for agent reasoning/thought chunks.

    Displays "Thinking…" while streaming, collapses with title "Thought"
    when finalized.
    """

    def __init__(self) -> None:
        super().__init__(Markdown("", open_links=False), title="Thinking…", collapsed=False)
        self._stream: MarkdownStream | None = None

    @property
    def _markdown(self) -> Markdown:
        """Return the inner Markdown widget."""
        return self.query_one(Markdown)

    async def append_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk.

        Args:
            chunk: Markdown fragment to append.
        """
        if self._stream is None:
            self._stream = Markdown.get_stream(self._markdown)
        await self._stream.write(chunk)

    async def finalize(self) -> None:
        """Stop streaming, set title to 'Thought', and collapse."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        self.title = "Thought"
        self.collapsed = True
