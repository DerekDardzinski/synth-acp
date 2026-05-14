"""Conversation feed container for agent messages, prompts, tools, and permissions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.markup import escape
from textual.message import Message
from textual.signal import Signal
from textual.widgets import Static

from synth_acp.models.events import (
    AgentThoughtReceived,
    BrokerEvent,
    HookFired,
    InitialPromptDelivered,
    McpMessageDelivered,
    MessageChunkReceived,
    PlanReceived,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UserPromptSubmitted,
)
from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.copy_button import CopyButton
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.plan_block import PlanBlock
from synth_acp.ui.widgets.prompt_bubble import PromptBubble
from synth_acp.ui.widgets.shell_result import ShellResultBlock
from synth_acp.ui.widgets.thought_block import ThoughtBlock
from synth_acp.ui.widgets.tool_call import ToolCallBlock

if TYPE_CHECKING:
    from synth_acp.terminal.manager import TerminalProcess

log = logging.getLogger(__name__)


class TurnContainer(Vertical, can_focus=False):
    """Groups all widgets belonging to a single conversational turn."""

    DEFAULT_CSS = ""


class PruningScrollContainer(ScrollableContainer):
    """ScrollableContainer that posts NearTop when scroll_y <= threshold."""

    LOAD_THRESHOLD: ClassVar[int] = 20

    class NearTop(Message):
        """Posted when user scrolls near the top of the container."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Detect scroll-near-top and post NearTop message."""
        super().watch_scroll_y(old_value, new_value)
        if new_value <= self.LOAD_THRESHOLD and new_value < old_value:
            self.post_message(self.NearTop())


RESTORE_BATCH: int = 10


class ConversationFeed(Vertical):
    """Container holding conversation widgets for a single agent.

    Args:
        agent_id: The agent this feed belongs to.
        agent_name: Display name for the agent.
    """

    HIGH_MARK: ClassVar[int] = 40
    LOW_MARK: ClassVar[int] = 30

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        project: str = "",
        harness: str = "",
        cwd: str = "",
        **kwargs: object,
    ) -> None:
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
        self._scroll: PruningScrollContainer | None = None
        self.input_bar: InputBar | None = None
        self._turn_events: list[list[BrokerEvent]] = []
        self._current_turn_events: list[BrokerEvent] = []
        self._mounted_start_idx: int = 0
        self._pending_terminals: dict[str, TerminalProcess] = {}
        self._pending_children: dict[str, list[tuple[ToolCallBlock, str | None, str]]] = {}
        self._tool_call_blocks: dict[str, ToolCallBlock] = {}
        self._loading_more: bool = False

    def compose(self) -> ComposeResult:
        """Yield the scrollable container and input bar."""
        with PruningScrollContainer(classes="conv-scroll"):
            pass
        yield InputBar(self._agent_id, self._agent_name, self._harness, cwd=self._cwd)

    def on_mount(self) -> None:
        """Cache the scroll container and input bar references."""
        self._scroll = self.query_one(".conv-scroll", PruningScrollContainer)
        self._scroll.anchor()
        self.input_bar = self.query_one(InputBar)
        self.streaming_signal: Signal[bool] = Signal(self, "streaming")

    def record_event(self, event: BrokerEvent) -> None:
        """Record a renderable event for the current turn.

        Args:
            event: The broker event to track.
        """
        self._current_turn_events.append(event)

    async def replay_event(self, event: BrokerEvent) -> None:
        """Replay a single renderable event into the current turn.

        Dispatches renderable events to the appropriate feed method.
        Non-renderable events are silently skipped.

        Args:
            event: The broker event to replay.
        """
        if isinstance(event, MessageChunkReceived):
            await self.add_chunk(event.chunk)
        elif isinstance(event, AgentThoughtReceived):
            await self.add_thought_chunk(event.chunk)
        elif isinstance(event, ToolCallUpdated):
            await self.add_tool_call(
                event.tool_call_id,
                event.title,
                event.kind,
                event.status,
                locations=event.locations,
                raw_input=event.raw_input,
                raw_output=event.raw_output,
                diffs=event.diffs,
                text_content=event.text_content,
                terminal_id=event.terminal_id,
                parent_tool_call_id=event.parent_tool_call_id,
            )
        elif isinstance(event, TurnComplete):
            await self.finalize_current_message()
        elif isinstance(event, PlanReceived):
            await self.update_plan(event.entries)
        elif isinstance(event, McpMessageDelivered):
            await self.add_mcp_message(event.from_agent, event.to_agent, event.preview)
        elif isinstance(event, HookFired):
            await self.add_hook_notification(event.hook_name)
        elif isinstance(event, (InitialPromptDelivered, UserPromptSubmitted)):
            await self.add_prompt(event.text)

    async def _mount_target(self) -> TurnContainer | PruningScrollContainer | None:
        """Return the current turn container, creating one lazily if needed."""
        if self._current_turn is None:
            await self._start_turn()
        return self._current_turn or self._scroll

    async def _start_turn(self) -> TurnContainer | None:
        """Create and mount a new turn container, returning it."""
        if self._scroll is None:
            return None
        turn = TurnContainer(classes="turn-container")
        self._current_turn = turn
        await self._scroll.mount(turn)
        return turn

    async def add_prompt(self, text: str) -> None:
        """Mount a user prompt bubble inside a new turn container.

        Args:
            text: The user's message text.
        """
        if self._scroll is None:
            return
        turn = await self._start_turn()
        if turn is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        await turn.mount(PromptBubble(text, ts))
        self._scroll.scroll_end(animate=False)

    async def add_chunk(self, chunk: str) -> None:
        """Append a streaming chunk, creating an AgentMessage if needed.

        Args:
            chunk: Markdown fragment from the agent.
        """
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        if self._current_message is None:
            self._current_message = AgentMessage(self._agent_id)
            target = await self._mount_target()
            if target is None:
                return
            await target.mount(self._current_message)
            self.streaming_signal.publish(True)
        await self._current_message.append_chunk(chunk)

    async def add_thought_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk, creating a ThoughtBlock if needed.

        Args:
            chunk: Markdown fragment from agent reasoning.
        """
        if self._current_thought is None:
            self._current_thought = ThoughtBlock()
            target = await self._mount_target()
            if target is None:
                return
            await target.mount(self._current_thought)
        await self._current_thought.append_chunk(chunk)

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
        parent_tool_call_id: str | None = None,
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
            parent_tool_call_id: If set, nest this block inside the parent.
        """
        existing = self._tool_call_blocks.get(tool_call_id)
        if existing is not None:
            existing.update_status(status)
            if status == "completed" and existing._nested_section is not None:
                existing.finalize_nested()
            await existing.update_content(
                locations=locations,
                raw_input=raw_input,
                raw_output=raw_output,
                diffs=diffs,
                text_content=text_content,
            )
        else:
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
            self._tool_call_blocks[tool_call_id] = block
            if parent_tool_call_id:
                block.add_class("nested-tool-call")
                parent_block = self._tool_call_blocks.get(parent_tool_call_id)
                if parent_block is not None:
                    await parent_block.mount_nested_child(block)
                    await self._mount_pending_terminal(block, terminal_id)
                    await self._flush_pending_children(block, tool_call_id)
                else:
                    self._pending_children.setdefault(parent_tool_call_id, []).append(
                        (block, terminal_id, tool_call_id)
                    )
            else:
                if self._current_thought is not None:
                    await self._current_thought.finalize()
                    self._current_thought = None
                if self._current_message is not None:
                    await self._current_message.finalize()
                    self._current_message = None
                if self._scroll is None:
                    return
                target = await self._mount_target() or self._scroll
                async with target.batch():
                    await target.mount(block)
                    await self._mount_pending_terminal(block, terminal_id)
                await self._flush_pending_children(block, tool_call_id)

    async def _mount_pending_terminal(self, block: ToolCallBlock, terminal_id: str | None) -> None:
        """Mount a pending terminal inside a block if one is buffered."""
        if terminal_id and terminal_id in self._pending_terminals:
            from synth_acp.ui.widgets.terminal import Terminal

            process = self._pending_terminals.pop(terminal_id)
            await block.mount(Terminal(process))

    async def _flush_pending_children(self, block: ToolCallBlock, tool_call_id: str) -> None:
        """Recursively mount buffered children inside a newly-mounted block."""
        for child_block, child_terminal_id, child_tool_call_id in self._pending_children.pop(tool_call_id, []):
            await block.mount_nested_child(child_block)
            await self._mount_pending_terminal(child_block, child_terminal_id)
            await self._flush_pending_children(child_block, child_tool_call_id)

    async def finalize_current_message(self) -> None:
        """Finalize the active streaming message, thought block, and turn."""
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        if self._current_message is not None:
            await self._current_message.finalize()
            self._current_message = None
            self.streaming_signal.publish(False)
        if self._current_turn_events:
            self._turn_events.append(self._current_turn_events)
            self._current_turn_events = []
        self._current_turn = None
        await self._check_prune()

    async def _check_prune(self) -> None:
        """Remove oldest turns from DOM if count exceeds HIGH_MARK."""
        if self._scroll is None:
            return
        turns = [c for c in self._scroll.children if isinstance(c, TurnContainer)]
        if len(turns) <= self.HIGH_MARK:
            return
        if self._scroll.scroll_y < self._scroll.max_scroll_y:
            return
        to_remove = turns[: len(turns) - self.LOW_MARK]
        # Clean up _tool_call_blocks for pruned turns
        pruned_blocks = set()
        for turn in to_remove:
            for block in turn.query(ToolCallBlock):
                pruned_blocks.add(block)
        self._tool_call_blocks = {
            tid: blk for tid, blk in self._tool_call_blocks.items() if blk not in pruned_blocks
        }
        self._mounted_start_idx += len(to_remove)
        await self._scroll.remove_children(to_remove)

    def on_pruning_scroll_container_near_top(self) -> None:
        """Trigger restore when user scrolls near the top."""
        self.run_worker(self._restore_turns(), exclusive=True, group="restore")

    async def _restore_turns(self) -> None:
        """Restore a batch of pruned turns at the top of the scroll container.

        Debounced by _loading_more flag. Adjusts scroll_y to prevent visual jump.
        Always resets _loading_more in finally block.
        """
        import asyncio

        if self._loading_more or self._mounted_start_idx == 0 or self._scroll is None:
            return
        self._loading_more = True
        try:
            batch_start = max(0, self._mounted_start_idx - RESTORE_BATCH)
            batch = self._turn_events[batch_start : self._mounted_start_idx]

            # Save current state
            saved_turn = self._current_turn
            saved_message = self._current_message
            saved_thought = self._current_thought

            # Replay each turn
            restored_turns: list[TurnContainer] = []
            for turn_events in batch:
                await self._start_turn()
                for event in turn_events:
                    if isinstance(event, TurnComplete):
                        continue
                    await self.replay_event(event)
                if self._current_thought is not None:
                    await self._current_thought.finalize()
                    self._current_thought = None
                if self._current_message is not None:
                    await self._current_message.finalize()
                    self._current_message = None
                if self._current_turn is not None:
                    restored_turns.append(self._current_turn)
                self._current_turn = None

            # Restore saved state
            self._current_turn = saved_turn
            self._current_message = saved_message
            self._current_thought = saved_thought

            if not restored_turns:
                return

            # Record height before move
            old_vh = self._scroll.virtual_size.height

            # Move restored turns from bottom to top
            first_child = self._scroll.children[0] if self._scroll.children else None
            for turn in restored_turns:
                await turn.remove()
            if first_child is not None:
                await self._scroll.mount_all(restored_turns, before=first_child)
            else:
                await self._scroll.mount_all(restored_turns)

            # Update index BEFORE yield
            self._mounted_start_idx = batch_start

            # Yield for layout
            await asyncio.sleep(0)

            # Adjust scroll to prevent visual jump
            added_height = self._scroll.virtual_size.height - old_vh
            self._scroll.scroll_to(
                y=self._scroll.scroll_y + added_height,
                animate=False,
                immediate=True,
                release_anchor=False,
            )
        except Exception:
            log.exception("Error restoring turns")
        finally:
            self._loading_more = False

    async def update_plan(self, entries: list[object]) -> None:
        """Replace the plan block with updated entries.

        Args:
            entries: Full replacement list of plan entries from the agent.
        """
        if self._scroll is None:
            return
        target = await self._mount_target() or self._scroll
        async with target.batch():
            if self._plan_block is not None:
                await self._plan_block.remove()
                self._plan_block = None
            block = PlanBlock(entries)  # type: ignore[arg-type]
            self._plan_block = block
            await target.mount(block)

    async def add_mcp_message(self, from_agent: str, to_agent: str, preview: str) -> None:
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
        turn = await self._start_turn()
        if turn is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        from textual.containers import Vertical
        from textual.widgets.markdown import Markdown

        container = Vertical(classes="mcp-msg")
        async with turn.batch():
            await turn.mount(container)
            await container.mount(CopyButton(lambda p=preview: p))
            await container.mount(Markdown(preview, open_links=False))
            await container.mount(
                Static(f"[dim]◈ {escape(from_agent)} → {escape(to_agent)}  {ts}[/dim]", classes="bubble-ts")
            )
        self._scroll.scroll_end(animate=False)

    async def add_hook_notification(self, hook_name: str) -> None:
        """Mount a dim system line indicating a lifecycle hook fired."""
        target = await self._mount_target()
        if target is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        widget = Static(
            f"[dim]synth: {escape(hook_name)} hook fired  {ts}[/dim]",
            classes="hook-notification",
        )
        await target.mount(widget)

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
        turn = await self._start_turn()
        if turn is None:
            return
        block = ShellResultBlock(command)
        await turn.mount(block)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace") if stdout else ""
        block.set_output(output, proc.returncode or 0)
        self._current_turn = None
