"""Conversation feed container for agent messages, prompts, tools, and permissions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static

from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.plan_block import PlanBlock
from synth_acp.ui.widgets.prompt_bubble import PromptBubble
from synth_acp.ui.widgets.shell_result import ShellResultBlock
from synth_acp.ui.widgets.thought_block import ThoughtBlock
from synth_acp.ui.widgets.tool_call import ToolCallBlock

if TYPE_CHECKING:
    from synth_acp.terminal.manager import TerminalProcess

log = logging.getLogger(__name__)


class TurnContainer(Vertical):
    """Groups all widgets belonging to a single conversational turn."""

    DEFAULT_CSS = ""


class ConversationFeed(Vertical):
    """Container holding conversation widgets for a single agent.

    Args:
        agent_id: The agent this feed belongs to.
        agent_name: Display name for the agent.
    """

    def __init__(self, agent_id: str, agent_name: str, project: str = "", harness: str = "", cwd: str = "", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._project = project
        self._harness = harness
        self._cwd = cwd
        self._current_message: AgentMessage | None = None
        self._current_thought: ThoughtBlock | None = None
        self._current_turn: TurnContainer | None = None
        self._plan_block: PlanBlock | None = None
        self._scroll: ScrollableContainer | None = None
        self.input_bar: InputBar | None = None
        self._pending_terminals: dict[str, TerminalProcess] = {}

    def compose(self) -> ComposeResult:
        """Yield the scrollable container and input bar."""
        with ScrollableContainer(classes="conv-scroll"):
            pass
        yield InputBar(self._agent_id, self._agent_name, self._harness, cwd=self._cwd)

    def on_mount(self) -> None:
        """Cache the scroll container and input bar references."""
        self._scroll = self.query_one(".conv-scroll", ScrollableContainer)
        self.input_bar = self.query_one(InputBar)

    @property
    def _mount_target(self) -> TurnContainer | ScrollableContainer | None:
        """Return the current turn container, creating one lazily if needed."""
        if self._current_turn is None:
            self._start_turn()
        return self._current_turn or self._scroll

    def _start_turn(self) -> TurnContainer | None:
        """Create and mount a new turn container, returning it."""
        if self._scroll is None:
            return None
        turn = TurnContainer(classes="turn-container")
        self._current_turn = turn
        self._scroll.mount(turn)
        return turn

    def add_prompt(self, text: str) -> None:
        """Mount a user prompt bubble inside a new turn container.

        Args:
            text: The user's message text.
        """
        if self._scroll is None:
            return
        turn = self._start_turn()
        if turn is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        turn.mount(PromptBubble(text, ts))
        self._scroll.scroll_end(animate=False)

    async def add_chunk(self, chunk: str) -> None:
        """Append a streaming chunk, creating an AgentMessage if needed.

        Args:
            chunk: Markdown fragment from the agent.
        """
        if self._current_message is None:
            self._current_message = AgentMessage(self._agent_id)
            target = self._mount_target
            if target is None:
                return
            target.mount(self._current_message)
        await self._current_message.append_chunk(chunk)
        if self._scroll is not None:
            self._scroll.scroll_end(animate=False)

    async def add_thought_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk, creating a ThoughtBlock if needed.

        Args:
            chunk: Markdown fragment from agent reasoning.
        """
        if self._current_thought is None:
            self._current_thought = ThoughtBlock()
            target = self._mount_target
            if target is None:
                return
            target.mount(self._current_thought)
        await self._current_thought.append_chunk(chunk)
        if self._scroll is not None:
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
        raw_output: Any = None,
        diffs: list[ToolCallDiff] | None = None,
        text_content: str | None = None,
        terminal_id: str | None = None,
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
            raw_output: Raw output payload from the ACP SDK.
            diffs: File edit diffs extracted from the tool call.
            text_content: Extracted text content from the tool call.
            terminal_id: Terminal ID to associate with this tool call.
        """
        try:
            existing = self.query_one(f"#tool-{tool_call_id}", ToolCallBlock)
            existing.update_status(status)
            await existing.update_content(
                locations=locations,
                raw_input=raw_input,
                raw_output=raw_output,
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
                raw_output=raw_output,
                diffs=diffs,
                text_content=text_content,
                terminal_id=terminal_id,
            )
            if self._scroll is None:
                return
            target = self._mount_target or self._scroll
            await target.mount(block)
            if terminal_id and terminal_id in self._pending_terminals:
                from synth_acp.ui.widgets.terminal import Terminal

                process = self._pending_terminals.pop(terminal_id)
                await block.mount(Terminal(process))
            self._scroll.scroll_end(animate=False)

    async def finalize_current_message(self) -> None:
        """Finalize the active streaming message, thought block, and turn."""
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        if self._current_message is not None:
            await self._current_message.finalize()
            self._current_message = None
        self._current_turn = None

    async def update_plan(self, entries: list[object]) -> None:
        """Replace the plan block with updated entries.

        Args:
            entries: Full replacement list of plan entries from the agent.
        """
        if self._scroll is None:
            return
        if self._plan_block is not None:
            await self._plan_block.remove()
            self._plan_block = None
        block = PlanBlock(entries)  # type: ignore[arg-type]
        self._plan_block = block
        target = self._mount_target or self._scroll
        await target.mount(block)
        self._scroll.scroll_end(animate=False)

    def add_mcp_message(self, from_agent: str, to_agent: str, preview: str) -> None:
        """Mount an MCP message delivery notification inside a new turn.

        MCP messages trigger a full agent turn (the broker calls
        ``session.prompt`` after delivery), so they start a new
        turn container just like user prompts.

        Args:
            from_agent: Sender agent ID.
            to_agent: Recipient agent ID.
            preview: Message preview text.
        """
        if self._scroll is None:
            return
        turn = self._start_turn()
        if turn is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        from textual.containers import Vertical
        from textual.widgets.markdown import Markdown

        container = Vertical(classes="mcp-msg")
        turn.mount(container)
        container.mount(Markdown(preview, open_links=False))
        container.mount(Static(f"[dim]◈ {from_agent} → {to_agent}  {ts}[/dim]", classes="bubble-ts"))
        self._scroll.scroll_end(animate=False)

    def add_hook_notification(self, hook_name: str) -> None:
        """Mount a dim system line indicating a lifecycle hook fired."""
        target = self._mount_target
        if target is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        widget = Static(
            f"[dim]synth: {hook_name} hook fired  {ts}[/dim]",
            classes="hook-notification",
        )
        target.mount(widget)
        if self._scroll is not None:
            self._scroll.scroll_end(animate=False)

    async def mount_terminal(self, terminal_id: str, terminal_process: TerminalProcess) -> None:
        """Mount a Terminal widget inside the matching ToolCallBlock.

        If no matching block exists yet, stash in _pending_terminals for
        later mounting when add_tool_call creates the block.

        Args:
            terminal_id: Terminal identifier to match against ToolCallBlock.
            terminal_process: The TerminalProcess to display.
        """
        from synth_acp.ui.widgets.terminal import Terminal

        for block in self.query(ToolCallBlock):
            if block._terminal_id == terminal_id:
                await block.mount(Terminal(terminal_process))
                return
        self._pending_terminals[terminal_id] = terminal_process

    async def run_shell_command(self, command: str) -> None:
        """Run a shell command and display the output in the feed.

        Args:
            command: Shell command string to execute.
        """
        import asyncio

        if self._scroll is None:
            return
        turn = self._start_turn()
        if turn is None:
            return
        block = ShellResultBlock(command)
        turn.mount(block)
        self._scroll.scroll_end(animate=False)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace") if stdout else ""
        block.set_output(output, proc.returncode or 0)
        self._current_turn = None
