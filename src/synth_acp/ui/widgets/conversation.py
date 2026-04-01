"""Conversation feed container for agent messages, prompts, tools, and permissions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static

from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.prompt_bubble import PromptBubble
from synth_acp.ui.widgets.thought_block import ThoughtBlock
from synth_acp.ui.widgets.tool_call import ToolCallBlock

log = logging.getLogger(__name__)


class ConversationFeed(Vertical):
    """Container holding conversation widgets for a single agent.

    Args:
        agent_id: The agent this feed belongs to.
        agent_name: Display name for the agent.
    """

    def __init__(self, agent_id: str, agent_name: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._current_message: AgentMessage | None = None
        self._current_thought: ThoughtBlock | None = None
        self._scroll: ScrollableContainer | None = None
        self.input_bar: InputBar | None = None

    def compose(self) -> ComposeResult:
        """Yield the scrollable container and input bar."""
        with ScrollableContainer(classes="conv-scroll"):
            pass
        yield InputBar(self._agent_id, self._agent_name)

    def on_mount(self) -> None:
        """Cache the scroll container and input bar references."""
        self._scroll = self.query_one(".conv-scroll", ScrollableContainer)
        self.input_bar = self.query_one(InputBar)

    def add_prompt(self, text: str) -> None:
        """Mount a user prompt bubble and scroll to end.

        Args:
            text: The user's message text.
        """
        ts = datetime.now(UTC).strftime("%H:%M")
        bubble = PromptBubble(text, ts)
        if self._scroll is None:
            return
        self._scroll.mount(bubble)
        self._scroll.scroll_end(animate=False)

    async def add_chunk(self, chunk: str) -> None:
        """Append a streaming chunk, creating an AgentMessage if needed.

        Args:
            chunk: Markdown fragment from the agent.
        """
        if self._current_message is None:
            self._current_message = AgentMessage(self._agent_id)
            if self._scroll is None:
                return
            self._scroll.mount(self._current_message)
        await self._current_message.append_chunk(chunk)
        if self._scroll is None:
            return
        self._scroll.scroll_end(animate=False)

    async def add_thought_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk, creating a ThoughtBlock if needed.

        Args:
            chunk: Markdown fragment from agent reasoning.
        """
        if self._current_thought is None:
            self._current_thought = ThoughtBlock()
            if self._scroll is None:
                return
            self._scroll.mount(self._current_thought)
        await self._current_thought.append_chunk(chunk)
        if self._scroll is None:
            return
        self._scroll.scroll_end(animate=False)

    async def add_tool_call(
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
        """Mount a new ToolCallBlock or update an existing one.

        Finalizes any in-progress AgentMessage so the tool call visually
        splits the response stream.

        Args:
            tool_call_id: Unique tool call identifier.
            title: Human-readable tool call description.
            kind: Tool kind string.
            status: Current status string.
            locations: File locations referenced by the tool call.
            raw_input: Raw input payload from the ACP SDK.
            diffs: File edit diffs extracted from the tool call.
            text_content: Extracted text content from the tool call.
        """
        try:
            existing = self.query_one(f"#tool-{tool_call_id}", ToolCallBlock)
            existing.update_status(status)
            await existing.update_content(
                locations=locations,
                raw_input=raw_input,
                diffs=diffs,
                text_content=text_content,
            )
        except Exception:
            log.debug("Tool call query failed", exc_info=True)
            if self._current_message is not None:
                await self._current_message.finalize()
                self._current_message = None
            block = ToolCallBlock(
                tool_call_id,
                title,
                kind,
                status,
                locations=locations,
                raw_input=raw_input,
                diffs=diffs,
                text_content=text_content,
            )
            if self._scroll is None:
                return
            await self._scroll.mount(block)
            self._scroll.scroll_end(animate=False)

    async def finalize_current_message(self) -> None:
        """Finalize the active streaming message and thought block."""
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        if self._current_message is not None:
            await self._current_message.finalize()
            self._current_message = None

    def add_mcp_message(self, from_agent: str, to_agent: str, preview: str) -> None:
        """Mount an MCP message delivery notification.

        Args:
            from_agent: Sender agent ID.
            to_agent: Recipient agent ID.
            preview: Message preview text.
        """
        ts = datetime.now(UTC).strftime("%H:%M")
        escaped = preview.replace("[", r"\[")
        widget = Static(
            f"{escaped}\n[dim]◈ {from_agent} → {to_agent}  {ts}[/dim]",
            classes="mcp-msg",
        )
        if self._scroll is None:
            return
        self._scroll.mount(widget, before=self._current_message)
        self._scroll.scroll_end(animate=False)
