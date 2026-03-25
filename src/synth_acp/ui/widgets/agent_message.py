"""Streaming agent message using Textual's Markdown + MarkdownStream."""

from __future__ import annotations

from textual.widgets.markdown import Markdown, MarkdownStream


class AgentMessage(Markdown):
    """Agent response rendered as streaming markdown.

    Args:
        agent_id: The agent that produced this message.
        color: Hex color for the agent's border.
    """

    def __init__(self, agent_id: str, color: str) -> None:
        super().__init__("")
        self._agent_id = agent_id
        self._stream: MarkdownStream | None = None
        self.styles.border = ("round", color)

    async def append_chunk(self, chunk: str) -> None:
        """Append a streaming markdown chunk.

        Args:
            chunk: Markdown fragment to append.
        """
        if self._stream is None:
            self._stream = Markdown.get_stream(self)
        await self._stream.write(chunk)

    async def finalize(self) -> None:
        """Stop the markdown stream if active."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
