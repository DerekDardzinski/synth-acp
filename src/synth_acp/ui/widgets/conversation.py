"""Conversation feed container for agent messages, prompts, tools, and permissions."""

from __future__ import annotations

from datetime import UTC, datetime

from acp.schema import PermissionOption
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static

from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.permission import PermissionRequest
from synth_acp.ui.widgets.prompt_bubble import PromptBubble
from synth_acp.ui.widgets.tool_call import ToolCallBlock


class ConversationFeed(Vertical):
    """Container holding conversation widgets for a single agent.

    Args:
        agent_id: The agent this feed belongs to.
        color: Hex color for the agent.
    """

    def __init__(self, agent_id: str, color: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._agent_id = agent_id
        self._color = color
        self._current_message: AgentMessage | None = None
        self._scroll: ScrollableContainer | None = None
        self.input_bar: InputBar | None = None

    def compose(self):
        """Yield the scrollable container and input bar."""
        yield ScrollableContainer(classes="conv-scroll")
        yield InputBar(self._agent_id, self._color)

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
        assert self._scroll is not None
        self._scroll.mount(bubble)
        self._scroll.scroll_end(animate=False)

    async def add_chunk(self, chunk: str) -> None:
        """Append a streaming chunk, creating an AgentMessage if needed.

        Args:
            chunk: Markdown fragment from the agent.
        """
        if self._current_message is None:
            self._current_message = AgentMessage(self._agent_id, self._color)
            assert self._scroll is not None
            self._scroll.mount(self._current_message)
        await self._current_message.append_chunk(chunk)
        assert self._scroll is not None
        self._scroll.scroll_end(animate=False)

    def add_tool_call(self, tool_call_id: str, title: str, kind: str, status: str) -> None:
        """Mount a new ToolCallBlock or update an existing one.

        Args:
            tool_call_id: Unique tool call identifier.
            title: Human-readable tool call description.
            kind: Tool kind string.
            status: Current status string.
        """
        try:
            existing = self.query_one(f"#tool-{tool_call_id}", ToolCallBlock)
            existing.update_status(status)
        except Exception:
            block = ToolCallBlock(tool_call_id, title, kind, status)
            assert self._scroll is not None
            self._scroll.mount(block)
            self._scroll.scroll_end(animate=False)

    def add_permission(
        self,
        agent_id: str,
        request_id: str,
        title: str,
        kind: str,
        options: list[PermissionOption],
    ) -> None:
        """Mount a permission request widget.

        Args:
            agent_id: Agent requesting permission.
            request_id: Unique request identifier.
            title: Permission title.
            kind: Permission kind.
            options: List of permission options.
        """
        widget = PermissionRequest(agent_id, request_id, title, kind, options)
        assert self._scroll is not None
        self._scroll.mount(widget)
        self._scroll.scroll_end(animate=False)

    def remove_permission(self, request_id: str) -> None:
        """Remove a permission request widget by request_id.

        Args:
            request_id: The request to remove.
        """
        try:
            self.query_one(f"#perm-{request_id}", PermissionRequest).remove()
        except Exception:
            pass

    async def finalize_current_message(self) -> None:
        """Finalize the active streaming message."""
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
        snippet = preview[:80] + "…" if len(preview) > 80 else preview
        ts = datetime.now(UTC).strftime("%H:%M")
        widget = Static(
            f"[dim]◈ {from_agent} → {to_agent}  {ts}[/dim]\n  [dim]{snippet}[/dim]",
            classes="mcp-msg",
        )
        assert self._scroll is not None
        self._scroll.mount(widget)
        self._scroll.scroll_end(animate=False)
