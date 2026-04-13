"""Streaming agent message using Textual's Markdown + MarkdownStream."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets.markdown import Markdown, MarkdownStream

from synth_acp.ui.widgets.copy_button import CopyButton


class AgentMessage(Vertical):
    """Agent response rendered as streaming markdown.

    Args:
        agent_id: The agent that produced this message.
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__()
        self._agent_id = agent_id
        self._stream: MarkdownStream | None = None
        self._chunks: list[str] = []
        self._md = Markdown("", open_links=False)

    def compose(self) -> ComposeResult:
        yield CopyButton(lambda: "".join(self._chunks))
        yield self._md

    async def append_chunk(self, chunk: str) -> None:
        """Append a streaming markdown chunk.

        Args:
            chunk: Markdown fragment to append.
        """
        self._chunks.append(chunk)
        if self._stream is None:
            self._stream = Markdown.get_stream(self._md)
        await self._stream.write(chunk)

    async def finalize(self) -> None:
        """Stop the markdown stream if active."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
