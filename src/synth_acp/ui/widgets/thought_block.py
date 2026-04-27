"""Collapsible thought/reasoning block with streaming markdown."""

from __future__ import annotations

from textual.widgets import Collapsible
from textual.widgets.markdown import Markdown, MarkdownStream

from synth_acp.ui.widgets.copy_button import CopyButton


class ThoughtBlock(Collapsible, can_focus=False):
    """Collapsible block for agent reasoning/thought chunks.

    Displays "Thinking…" while streaming, collapses with title "Thought"
    when finalized.
    """

    def __init__(self) -> None:
        super().__init__(
            CopyButton(lambda: "".join(self._chunks)),
            Markdown("", open_links=False),
            title="Thinking…",
            collapsed=False,
        )
        self._stream: MarkdownStream | None = None
        self._chunks: list[str] = []

    @property
    def _markdown(self) -> Markdown:
        """Return the inner Markdown widget."""
        return self.query_one(Markdown)

    async def append_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk.

        Args:
            chunk: Markdown fragment to append.
        """
        self._chunks.append(chunk)
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
