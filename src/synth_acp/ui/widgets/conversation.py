"""Conversation feed container for agent messages, prompts, tools, and permissions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.markup import escape
from textual.message import Message
from textual.signal import Signal
from textual.widgets import Static
from textual.widgets.markdown import Markdown
from textual.worker import Worker

from synth_acp.models.events import (
    AgentThoughtReceived,
    BrokerEvent,
    HookFired,
    MessageChunkReceived,
    MessageSteered,
    PlanReceived,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UserPromptSubmitted,
)
from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.copy_button import CopyButton
from synth_acp.ui.widgets.diff_view import DiffView
from synth_acp.ui.widgets.input_bar import InputBar
from synth_acp.ui.widgets.plan_block import PlanBlock
from synth_acp.ui.widgets.prompt_bubble import PromptBubble
from synth_acp.ui.widgets.shell_result import ShellResultBlock
from synth_acp.ui.widgets.thought_block import ThoughtBlock
from synth_acp.ui.widgets.tool_call import DiffKey, DiffRecord, DiffState, ToolCallBlock

if TYPE_CHECKING:
    from synth_acp.terminal.manager import TerminalProcess

log = logging.getLogger(__name__)

DIFF_CONCURRENCY = 2
"""Maximum diffs highlighted concurrently across the whole app.

DO NOT RAISE THIS. Measured on the first-selection replay path, wall time to visible:
2 (this value) 8,846 ms; 4 21,903 ms; 8 31,073 ms; unbounded 22,605 ms; the old inline
behaviour 11,334 ms. Raising the cap is 2.5-3.5x WORSE, because the highlight is
GIL-holding while the replay loop the user is waiting on is a coroutine on the loop
thread, so every extra highlight thread steals interpreter time from the thing being
awaited. The cap protects the replay, not keystrokes.
"""

_LIMITER_ATTR = "_synth_diff_limiter"


def diff_limiter(app: App) -> asyncio.Semaphore:
    """Return the app-scoped diff concurrency limiter, creating it on first use.

    Cached on the app instance so it binds to that app's event loop. A module-level
    Semaphore would bind to the first contending loop and then raise "bound to a
    different event loop" in a later app instance.

    The highlight path builds Content/Span objects and is substantially GIL-holding, so
    threading converts one long stall into interpreter slices but does not remove the CPU
    cost — unbounded submission to the shared executor would starve the loop.

    Args:
        app: The running app.

    Returns:
        The semaphore guarding concurrent diff preparation.
    """
    limiter = getattr(app, _LIMITER_ATTR, None)
    if limiter is None:
        limiter = asyncio.Semaphore(DIFF_CONCURRENCY)
        setattr(app, _LIMITER_ATTR, limiter)
    return limiter


class TurnContainer(Vertical, can_focus=False):
    """Groups all widgets belonging to a single conversational turn."""

    DEFAULT_CSS = ""


class PruningScrollContainer(ScrollableContainer):
    """ScrollableContainer that posts NearTop when scroll_y <= threshold."""

    LOAD_THRESHOLD: ClassVar[int] = 20

    class NearTop(Message):
        """Posted when user scrolls near the top of the container."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Post NearTop when the USER scrolls down through the threshold.

        Two guards, both load-bearing for the tail-first window.

        ANCHOR GATE. While the bottom anchor is engaged, every ``scroll_y`` change is
        PROGRAMMATIC — layout, mounting, or the anchor itself — and must not be mistaken for
        the user asking for older history. Measured on a windowed feed: switching the panel
        into view produced a single 32 -> 0 jump, which restored the ENTIRE history the
        window had just avoided mounting, and the repeated triggers cancelled each other's
        exclusive restore worker and left empty TurnContainers behind. Textual releases the
        anchor when the user genuinely scrolls (``release_anchor``), so requiring a released
        anchor is exactly the "user initiated" test that Textual does not otherwise expose.

        CROSSING RULE. The trigger requires crossing the threshold from ABOVE rather than
        any decrease while already inside the band, so one upward gesture requests one
        batch. A window too short to scroll at all is handled by the background backfill,
        not here.
        """
        super().watch_scroll_y(old_value, new_value)
        anchor_engaged = self._anchored and not self._anchor_released
        if anchor_engaged:
            return
        if old_value > self.LOAD_THRESHOLD >= new_value:
            self.post_message(self.NearTop())


_RENDER_LOCK_HELD: ContextVar[bool] = ContextVar("synth_render_lock_held", default=False)
"""True while the CURRENT TASK already holds a feed's render lock.

A ContextVar, not an instance attribute. The distinction is the whole point: an attribute is
feed-global, so a CONCURRENT live task reads the flag another task set and skips the very lock
it is supposed to wait on — measured, that let live text render into a restored message as
"LIVEH1". A ContextVar is scoped to the logical call stack, which is the reentrancy question,
and it lets the lock be taken once at the routing boundary and skipped by every nested render
call underneath.

EXACT SEMANTICS, because the obvious summary is wrong and someone will rely on it: a
ContextVar is NOT invisible to other tasks. `asyncio.Task` COPIES the current context at
creation, so a task created while this is True observes True and would skip the lock. No
current path creates a task inside the locked region that then calls a wrapped render method,
so this is not a live defect — but it is why `_assert_record_lock` below enforces the
invariant at the recording step rather than trusting this flag alone.
"""

_STRICT_RENDER_LOCK: bool = False
"""When True, an out-of-lock recording RAISES instead of logging. Enabled in the test suite.

Production logs rather than raises: this guards against a silent data divergence, and turning
that into an exception on the UI message pump would trade a wrong batch for a dead app.
"""


RESTORE_BATCH: int = 10

FIRST_PAINT_TURNS: int = 3
"""Maximum LOGICAL turns mounted on a first-selection drain."""

FIRST_PAINT_EVENT_BUDGET: int = 40
"""Maximum events mounted on a first-selection drain, whichever cap is reached first."""


@dataclass(frozen=True)
class FirstPaintWindow:
    """The mounted tail of a first-selection drain.

    Attributes:
        start_index: Index of the first event to MOUNT. Earlier events are recorded only.
        skipped_turns: Number of complete turn batches left unmounted, which is what
            ``ConversationFeed._mounted_start_idx`` is initialised to.
    """

    start_index: int
    skipped_turns: int


def first_paint_window(events: Sequence[BrokerEvent]) -> FirstPaintWindow:
    """Choose the tail to mount on a first-selection drain.

    Walks backwards taking whole turns, stopping when EITHER cap is reached, and always
    mounting at least one COMPLETE turn. Both caps are load-bearing: the turn count is
    what delivers the measured first paint, and the event budget exists because turn
    sizes vary enormously — one measured turn held ~1,700 widgets — so a pathological
    single turn must not reconstruct most of the feed under the guise of "3 turns".

    A trailing OPEN segment (events after the last TurnComplete) is a turn the user sees,
    so it counts toward ``FIRST_PAINT_TURNS``. The at-least-one-complete-turn floor
    overrides BOTH caps, which is why a single final turn larger than the budget still
    mounts whole, and why an over-budget open segment still pulls one complete turn in
    behind it.

    Args:
        events: The coalesced buffered events, oldest first.

    Returns:
        The window to mount. ``FirstPaintWindow(0, 0)`` means mount everything, which is
        returned when no TurnComplete is present — there is then no closed batch for
        ``_restore_turns`` to index, so nothing may be skipped.
    """
    turn_completes = [i for i, event in enumerate(events) if isinstance(event, TurnComplete)]
    if not turn_completes:
        return FirstPaintWindow(0, 0)

    # Start index of complete turn k. Turn k spans starts[k] .. turn_completes[k].
    starts = [0 if k == 0 else turn_completes[k - 1] + 1 for k in range(len(turn_completes))]

    # Seed with the trailing OPEN segment: it is the newest content and always mounts.
    start = turn_completes[-1] + 1
    count = len(events) - start
    turns = 1 if start < len(events) else 0

    # index doubles as the complete-turn floor: while it still equals len(turn_completes)
    # no complete turn has been taken yet, so neither cap may stop the walk.
    index = len(turn_completes)
    for k in range(len(turn_completes) - 1, -1, -1):
        turn_length = turn_completes[k] + 1 - starts[k]
        took_a_complete_turn = index < len(turn_completes)
        if took_a_complete_turn and (
            turns >= FIRST_PAINT_TURNS or count + turn_length > FIRST_PAINT_EVENT_BUDGET
        ):
            break
        start = starts[k]
        count += turn_length
        turns += 1
        index = k
    return FirstPaintWindow(start, index)


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
        # INVARIANT, shared by all three writers: the number of LEADING batches in
        # _turn_events that are recorded but NOT mounted. The first-selection drain
        # initialises it to the number of turns it skipped, _check_prune INCREMENTS it by
        # the number of turns it unmounts, and _restore_turns DECREMENTS it by the number
        # of batches it re-mounts. It is therefore always a valid index into _turn_events,
        # and _turn_events[:_mounted_start_idx] is exactly the history reachable only by
        # scrolling up.
        self._mounted_start_idx: int = 0
        self._pending_terminals: dict[str, TerminalProcess] = {}
        self._pending_children: dict[str, list[tuple[ToolCallBlock, str | None, str]]] = {}
        self._tool_call_blocks: dict[str, ToolCallBlock] = {}
        self._loading_more: bool = False
        self._diff_wake: asyncio.Event = asyncio.Event()
        self._diff_blocks: dict[str, ToolCallBlock] = {}
        self._executor_handle: Worker | None = None
        self._late_window_turn: TurnContainer | None = None
        # Serializes LIVE rendering against an in-flight historical replay. Both go through
        # the same add_* entry points and the same _current_* pointers, so without this a
        # live chunk arriving during a restore's await appends into a HISTORICAL message —
        # measured, live text "LIVE" landed inside a restored message as "LIVEH1", and the
        # pointer was then reset to None, silently losing the live output. The depth counter
        # is what keeps the replay's own nested calls from deadlocking on the lock it holds.
        self._render_lock: asyncio.Lock = asyncio.Lock()

    @asynccontextmanager
    async def exclusive_render(self) -> AsyncIterator[None]:
        """Hold this feed's render lock, unless THIS TASK already holds it.

        Reentrant by task, so the routing boundary can take it once around a whole live event
        — RECORDING INCLUDED — and every nested render call underneath skips it. That matters
        because ``SynthApp._replay_event`` records synchronously before awaiting the render:
        with the lock taken only inside the render methods, a live event's recording landed in
        ``_current_turn_events`` while a restore had ``_late_window_turn`` suppressed, so the
        recorded batch and the rendered DOM diverged silently.
        """
        if _RENDER_LOCK_HELD.get():
            yield
            return
        async with self._render_lock:
            token = _RENDER_LOCK_HELD.set(True)
            try:
                yield
            finally:
                _RENDER_LOCK_HELD.reset(token)

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

    def _late_window_is_tail(self) -> bool:
        """True when ``_late_window_turn`` is set AND is still the LAST TurnContainer.

        The single eligibility predicate shared by ``record_event`` (batching) and the
        chunk/thought rendering paths, so batch placement and DOM placement can never
        disagree. It is also what covers the lazy path: an intervening HookFired,
        PlanReceived or ToolCallUpdated can create a later turn through ``_mount_target``
        without passing any named boundary, and requiring the eligible turn to still be
        last makes a late chunk start a new message there instead of mounting above it.
        """
        turn = self._late_window_turn
        if turn is None or self._scroll is None:
            return False
        turns = [c for c in self._scroll.children if isinstance(c, TurnContainer)]
        return bool(turns) and turns[-1] is turn

    def _late_tail_widget[W: (AgentMessage, ThoughtBlock)](self, widget_type: type[W]) -> W | None:
        """Return the eligible turn's last child when it is a reopenable widget.

        The immediate-last-child half of eligibility, applied by rendering only: if a tool
        call was mounted after the message, reopening it would interleave content out of
        order.
        """
        if not self._late_window_is_tail():
            return None
        turn = self._late_window_turn
        if turn is None or not turn.children:
            return None
        last = turn.children[-1]
        return last if isinstance(last, widget_type) else None

    async def _late_mount_target(self) -> TurnContainer | PruningScrollContainer | None:
        """Mount target for late content: the eligible turn, else the normal target.

        Non-reopenable same-boundary late content mounts INSIDE the eligible old turn so
        live rendering and replay agree structurally.
        """
        if self._current_turn is None and self._late_window_is_tail():
            self._current_turn = self._late_window_turn
            return self._current_turn
        return await self._mount_target()

    async def _close_late_window(self) -> None:
        """Close any reopened widget, then clear every current pointer.

        MUST be awaited by every new-turn boundary BEFORE the new turn is mounted,
        otherwise a reopened stream stays open and later content appends into the previous
        turn's widget. Idempotent when no reopen is active.
        """
        if self._current_message is not None:
            await self._current_message.finalize()
            self._current_message = None
            self.streaming_signal.publish(False)
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        self._current_turn = None
        self._late_window_turn = None

    def _assert_record_lock(self) -> None:
        """Fail LOUDLY if this event is being recorded outside the render lock.

        THE INVARIANT: recording and rendering one event must be atomic with respect to a
        historical replay. `_restore_turns` suppresses `_late_window_turn` for its replay, so a
        live event that RECORDS while the replay holds the lock and RENDERS after it releases
        lands in a different batch than the turn it renders into — the recorded history and the
        DOM then disagree with no error anywhere, which is exactly what AC3 forbids.

        Enforced HERE, at the recording step, rather than by auditing call sites. Four separate
        review rounds found a path that recorded outside the lock — the record/render split in
        `_replay_event`, then the steady-state `_route_event_to_feed` route — because "did we
        remember every path?" is a question a reviewer has to re-ask on every change. As a
        runtime invariant it answers itself: any new path that records without the lock fails at
        the point of the bug.

        Deliberately conditional on the lock being HELD. Recording with no replay in flight is
        legitimate and common, so requiring the lock unconditionally would flag ordinary use
        while catching nothing extra: the divergence only exists when a replay is running.
        """
        if not self._render_lock.locked() or _RENDER_LOCK_HELD.get():
            return
        message = (
            f"record_event called outside the render lock while a replay holds it "
            f"(agent {self._agent_id!r}); the recorded batch and the rendered DOM will "
            f"diverge. Wrap the routing path in ConversationFeed.exclusive_render()."
        )
        if _STRICT_RENDER_LOCK:
            raise AssertionError(message)
        log.error(message)

    def record_event(self, event: BrokerEvent) -> None:
        """Record a renderable event into the correct turn batch. Type-aware.

        A late MessageChunkReceived or AgentThoughtReceived joins the ELIGIBLE finalized
        turn's batch, decided by ``_late_window_is_tail()`` — the same predicate the
        rendering path uses, so batch placement and DOM placement cannot diverge.
        UserPromptSubmitted and all post-boundary events go to the next batch. This is the
        only place batching happens.

        Args:
            event: The broker event to track.
        """
        self._assert_record_lock()
        if isinstance(event, (MessageChunkReceived, AgentThoughtReceived)) and (
            self._late_window_is_tail()
        ):
            if not self._turn_events:
                # Unreachable under current invariants: TurnComplete is itself a recorded
                # renderable event, so any turn that reached finalize_current_message
                # committed a batch. Defensive only, so an impossible state cannot raise.
                self._turn_events.append([])
            self._turn_events[-1].append(event)
            return
        self._current_turn_events.append(event)

    async def replay_event(self, event: BrokerEvent) -> None:
        """Replay a single renderable event into the current turn.

        Dispatches renderable events to the appropriate feed method.
        Non-renderable events are silently skipped.

        Args:
            event: The broker event to replay.
        """
        if isinstance(event, MessageChunkReceived):
            await self._add_chunk_locked(event.chunk)
        elif isinstance(event, AgentThoughtReceived):
            await self._add_thought_chunk_locked(event.chunk)
        elif isinstance(event, ToolCallUpdated):
            await self._add_tool_call_impl(
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
            await self._finalize_current_message_impl()
        elif isinstance(event, PlanReceived):
            await self._update_plan_impl(event.entries)
        elif isinstance(event, HookFired):
            await self._add_hook_notification_impl(event.hook_name)
        elif isinstance(event, MessageSteered):
            await self._add_steered_message_impl(event.from_agent, event.text)
        elif isinstance(event, UserPromptSubmitted):
            await self._add_prompt_impl(event.text)

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

    async def _add_prompt_impl(self, text: str) -> None:
        """Mount a user prompt bubble inside a new turn container.

        Args:
            text: The user's message text.
        """
        if self._scroll is None:
            return
        await self._close_late_window()
        turn = await self._start_turn()
        if turn is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        await turn.mount(PromptBubble(text, ts))
        self._scroll.scroll_end(animate=False)

    async def add_chunk(self, chunk: str) -> None:
        """Append a streaming chunk, serialized against any in-flight historical replay.

        Args:
            chunk: Markdown fragment from the agent.
        """
        async with self.exclusive_render():
            await self._add_chunk_locked(chunk)

    async def add_thought_chunk(self, chunk: str) -> None:
        """Append a streaming thought chunk, serialized against historical replay.

        Args:
            chunk: Markdown fragment from agent reasoning.
        """
        async with self.exclusive_render():
            await self._add_thought_chunk_locked(chunk)

    async def add_prompt(self, text: str) -> None:
        """Mount a user prompt bubble, serialized against any historical replay.

        Args:
            text: The user's message text.
        """
        async with self.exclusive_render():
            await self._add_prompt_impl(text)

    async def add_hook_notification(self, hook_name: str) -> None:
        """Mount a lifecycle-hook line, serialized against any historical replay.

        Args:
            hook_name: Name of the hook that fired.
        """
        async with self.exclusive_render():
            await self._add_hook_notification_impl(hook_name)

    async def update_plan(self, entries: list[object]) -> None:
        """Replace the plan block, serialized against any historical replay.

        Args:
            entries: Full replacement list of plan entries from the agent.
        """
        async with self.exclusive_render():
            await self._update_plan_impl(entries)

    async def finalize_current_message(self) -> None:
        """Finalize the streaming message and turn, serialized against historical replay."""
        async with self.exclusive_render():
            await self._finalize_current_message_impl()

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
        """Mount or update a ToolCallBlock, serialized against any historical replay.

        The full signature is repeated rather than forwarded through ``*args, **kwargs``: this
        is the public interface every caller and the type checker see, and collapsing it to
        untyped varargs to save a few lines would silently discard parameter names and static
        checking for the sake of a locking wrapper.

        Serialization matters here as much as for chunks: a live tool call arriving during a
        replay was measured landing in RESTORED turn position 0 instead of at the tail, because
        it mounts through the same ``_current_turn`` pointer the replay is driving.

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
        async with self.exclusive_render():
            await self._add_tool_call_impl(
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
                parent_tool_call_id=parent_tool_call_id,
            )

    async def run_shell_command(self, command: str) -> None:
        """Run a shell command, serialized against any historical replay.

        Serialized for the same reason as the render paths: it opens a turn through
        `_close_late_window`/`_start_turn`, and interleaving that with a replay was measured
        losing ten prompts from restored history.

        Args:
            command: Shell command string to execute.
        """
        async with self.exclusive_render():
            await self._run_shell_command_impl(command)

    async def _add_chunk_locked(self, chunk: str) -> None:
        """Append a streaming chunk, creating an AgentMessage if needed.

        When no message is streaming, a chunk that arrived after TurnComplete reopens the
        eligible turn's last AgentMessage and continues into it, rather than creating a
        spurious second bubble. Owns WIDGETS ONLY — batching is record_event's job.

        Args:
            chunk: Markdown fragment from the agent.
        """
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        if self._current_message is None:
            reopened = self._late_tail_widget(AgentMessage)
            if reopened is not None:
                await reopened.reopen()
                # BOTH pointers: leaving _current_turn unset would let the next renderable
                # mount into a different container, so one logical turn would straddle two
                # and live rendering would diverge from replay.
                self._current_turn = self._late_window_turn
                self._current_message = reopened
                self.streaming_signal.publish(True)
            else:
                self._current_message = AgentMessage(self._agent_id)
                target = await self._late_mount_target()
                if target is None:
                    self._current_message = None
                    return
                await target.mount(self._current_message)
                self.streaming_signal.publish(True)
        # Captured into a LOCAL before the await. _restore_turns suppresses all four
        # pointers for its isolated replay, so a scroll-up or backfill restore landing
        # between here and the append would otherwise null _current_message under us and
        # raise AttributeError on live agent output.
        message = self._current_message
        if message is None:
            return
        await message.append_chunk(chunk)

    async def _add_thought_chunk_locked(self, chunk: str) -> None:
        """Append a streaming thought chunk, with the same turn-level late tolerance.

        Args:
            chunk: Markdown fragment from agent reasoning.
        """
        if self._current_thought is None:
            reopened = self._late_tail_widget(ThoughtBlock)
            if reopened is not None:
                await reopened.reopen()
                self._current_turn = self._late_window_turn
                self._current_thought = reopened
            else:
                self._current_thought = ThoughtBlock()
                target = await self._late_mount_target()
                if target is None:
                    self._current_thought = None
                    return
                await target.mount(self._current_thought)
        # Local capture for the same reason as add_chunk: a concurrent restore suppresses
        # this pointer for the duration of its replay.
        thought = self._current_thought
        if thought is None:
            return
        await thought.append_chunk(chunk)

    async def _add_tool_call_impl(
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
            if diffs:
                existing.schedule_diffs(diffs)
                self.wake_diff_executor(existing)
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
            block.schedule_diffs(diffs or [])
            if parent_tool_call_id:
                block.add_class("nested-tool-call")
                parent_block = self._tool_call_blocks.get(parent_tool_call_id)
                if parent_block is not None:
                    await parent_block.mount_nested_child(block)
                    await self._mount_pending_terminal(block, terminal_id)
                    await self._flush_pending_children(block, tool_call_id)
                    self._wake_for_diffs(block)
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
                self._wake_for_diffs(block)

    def _wake_for_diffs(self, block: ToolCallBlock) -> None:
        """Wake the diff executor if a freshly mounted block has diff work."""
        if block.has_pending_diffs():
            self.wake_diff_executor(block)

    def wake_diff_executor(self, block: ToolCallBlock) -> None:
        """Register queued diff work and wake the executor. SYNCHRONOUS.

        Ensures liveness: a worker that is None, finished or cancelled while the feed is
        still mounted is replaced BEFORE the Event is set, because ``Event.set()`` cannot
        revive a terminated task and ``release_claim`` alone would not recover its work.
        Feed unmount cancels permanently and must NOT restart.

        Args:
            block: The block holding scheduled diffs.
        """
        self._diff_blocks[block._tool_call_id] = block
        if not self._feed_is_live():
            return
        handle = self._executor_handle
        if handle is None or handle.is_cancelled or handle.is_finished:
            self._executor_handle = self.run_worker(
                self._diff_executor(),
                group="diffs",
                exclusive=False,
                name=f"diff-executor-{self._agent_id}",
            )
        self._diff_wake.set()

    def _feed_is_live(self) -> bool:
        """False once this feed is being pruned from the DOM."""
        return self.is_running and not self._pruning

    def _claim_any_diff(self) -> tuple[ToolCallBlock, DiffKey, DiffRecord] | None:
        """Claim one diff from any registered block, dropping dead registrations."""
        for tool_call_id, block in list(self._diff_blocks.items()):
            if block._pruning or not block.is_running:
                del self._diff_blocks[tool_call_id]
                continue
            claim = block.claim_next_diff()
            if claim is not None:
                return block, claim[0], claim[1]
        return None

    async def _diff_executor(self) -> None:
        """Standing per-feed worker that renders queued diffs off the message pump.

        Waits on ``_diff_wake``, drains every registered block, then waits again. Never
        raises; returns cleanly on cancellation leaving no unprepared widget mounted.
        """
        try:
            while True:
                await self._diff_wake.wait()
                self._diff_wake.clear()
                while (claim := self._claim_any_diff()) is not None:
                    await self._render_diff(*claim)
        except asyncio.CancelledError:
            # Contract: return cleanly. _render_diff re-raises so its finally can release
            # the claim first; swallowing it here is what makes the worker's own exit clean.
            return

    async def _render_diff(self, block: ToolCallBlock, key: DiffKey, record: DiffRecord) -> None:
        """Prepare one diff off-thread and mount it, or mount a plain fallback.

        The DiffView is prepared while UNMOUNTED: ``_check_auto_split`` and ``compose``
        both read the highlighted lines, so mounting an unprepared view would run the
        ~186ms highlight synchronously and defeat the whole mechanism.

        Args:
            block: The block owning the diff.
            key: The claimed key.
            record: The claimed record.
        """
        view: DiffView | None = None
        try:
            async with diff_limiter(self.app):
                view = block.make_diff_view(record)
                await view.prepare()
                view.resolve_split(block.content_size.width)
                if block._pruning or not block.is_running or not self._feed_is_live():
                    return
                if self._diff_blocks.get(block._tool_call_id) is not block:
                    return
                if record.fallback is not None:
                    await record.fallback.remove()
                    record.fallback = None
                await block.mount(view)
                record.state = DiffState.RENDERED
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Failed to render diff for %s", key.path)
            record.attempts += 1
            record.state = DiffState.FAILED
            if view is not None and view.is_running:
                await view.remove()
            await self._mount_diff_fallback(block, record)
        finally:
            # Covers cancellation, the discard paths, and any unexpected exit. Without it
            # a key can stay RENDERING forever, and redelivery of a RENDERING key is a
            # no-op, so the diff would silently never render.
            if record.state is DiffState.RENDERING:
                block.release_claim(key)

    async def _mount_diff_fallback(self, block: ToolCallBlock, record: DiffRecord) -> None:
        """Mount a plain unhighlighted rendering so failed content is never lost."""
        try:
            if block._pruning or not block.is_running:
                return
            fallback = block.make_diff_fallback(record)
            await block.mount(fallback)
            record.fallback = fallback
        except Exception:
            log.exception("Failed to mount diff fallback for %s", record.diff.path)

    async def _mount_pending_terminal(self, block: ToolCallBlock, terminal_id: str | None) -> None:
        """Mount a pending terminal inside a block if one is buffered."""
        if terminal_id and terminal_id in self._pending_terminals:
            from synth_acp.ui.widgets.terminal import Terminal

            process = self._pending_terminals.pop(terminal_id)
            await block.mount(Terminal(process))

    async def _flush_pending_children(self, block: ToolCallBlock, tool_call_id: str) -> None:
        """Recursively mount buffered children inside a newly-mounted block."""
        for child_block, child_terminal_id, child_tool_call_id in self._pending_children.pop(
            tool_call_id, []
        ):
            await block.mount_nested_child(child_block)
            await self._mount_pending_terminal(child_block, child_terminal_id)
            await self._flush_pending_children(child_block, child_tool_call_id)
            self._wake_for_diffs(child_block)

    def close_turn_batch(self) -> None:
        """Flush the accumulating turn batch into ``_turn_events``. NO widget work.

        The BATCH-FLUSH half of ``finalize_current_message``. It exists so a turn that is
        recorded but never mounted can still CLOSE its batch: batches were previously
        flushed only inside ``finalize_current_message``, which is reachable only through
        ``replay_event``'s TurnComplete branch, so skipping replay left ``_turn_events`` as
        one unflushed list and ``_restore_turns`` with nothing to index.

        Constructs and finalizes nothing, so a skipped turn costs one list append.
        """
        if self._current_turn_events:
            self._turn_events.append(self._current_turn_events)
            self._current_turn_events = []

    async def _finalize_current_message_impl(self) -> None:
        """Finalize the active streaming message, thought block, and turn.

        The WIDGET-FINALIZE half, which also closes the batch via ``close_turn_batch``.
        Behaviour and ordering are unchanged from before the split.
        """
        if self._current_thought is not None:
            await self._current_thought.finalize()
            self._current_thought = None
        if self._current_message is not None:
            await self._current_message.finalize()
            self._current_message = None
            self.streaming_signal.publish(False)
        self.close_turn_batch()
        # The only site that still knows the finalized container. TurnComplete reaches this
        # method on live routing, app-level replay AND feed replay, so one assignment
        # covers all three; without it no reopen can ever happen. After a windowed drain
        # this lands on the last MOUNTED turn, because skipped turns run close_turn_batch
        # only and never reach here.
        self._late_window_turn = self._current_turn
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
        # Second writer of _mounted_start_idx; see the invariant on its declaration. The
        # turns being unmounted here are the LEADING mounted ones, so incrementing by their
        # count keeps the index pointing at the first still-mounted batch — the same
        # meaning the first-selection drain establishes when it initialises it to the
        # number of turns it skipped.
        self._mounted_start_idx += len(to_remove)
        await self._scroll.remove_children(to_remove)

    def on_pruning_scroll_container_near_top(self) -> None:
        """Trigger restore when user scrolls near the top.

        NOT exclusive. `exclusive=True` would CANCEL an in-flight restore when a second
        NearTop arrives — two quick scroll-up gestures are enough — and interrupting a
        restore part-way corrupts scrollback. `_restore_turns` already no-ops via
        `_loading_more` when one is running, so a redundant worker is harmless and a
        cancelled one is not.
        """
        self.run_worker(self._restore_turns(), group="restore")

    def on_show(self) -> None:
        """Start the eager backfill once the feed is actually visible.

        NOT at the end of the drain. Two reasons, both measured: scroll headroom is
        meaningless while the panel is not current, because the container has size (0, 0);
        and mounting into a TurnContainer that is not in the compositor does not complete,
        so replaying a batch there raises MountError. Textual posts Show when the panel is
        switched into view, which is exactly the moment scrolling becomes possible.
        """
        self.start_window_backfill()

    BACKFILL_MAX_STALLS: ClassVar[int] = 20
    """Consecutive FRUITLESS restore attempts tolerated before the backfill gives up.

    Only attempts that actually ran a restore and failed to advance count against this.
    Waiting for another restore to finish does NOT, because that is legitimate work in
    progress rather than a stuck state, and charging it here would abandon the backfill
    whenever a user's scroll-up restore simply took a while — reintroducing the
    unreachable-history bug this ceiling exists alongside.
    """

    BACKFILL_WAIT_SECONDS: ClassVar[float] = 0.01
    """Pause between polls while another restore holds the lock. Short enough to resume
    promptly, long enough not to spin the loop thread."""

    BACKFILL_MAX_WAITS: ClassVar[int] = 500
    """Poll budget for an in-flight restore, ~5s at BACKFILL_WAIT_SECONDS.

    Separate from the stall ceiling so a slow restore cannot be mistaken for a stuck one,
    but still bounded: a restore that never clears `_loading_more` must not loop forever.
    """

    def start_window_backfill(self) -> None:
        """Start the background eager-restore worker. SYNCHRONOUS, fire-and-forget.

        Returns immediately in the common case: a tail that already overflows the viewport
        needs no backfill, and the measured window overfills a 32-row viewport many times
        over.

        IDEMPOTENT BY DECLINING, NOT BY CANCELLING. `on_show` fires again whenever the panel
        is switched back into view, and `exclusive=True` would cancel a running backfill
        mid-restore — measured, that duplicated a turn and corrupted scrollback order, because
        an interrupted restore leaves partially rendered turns the replacement worker then
        replays again. A second call while one is live is simply dropped.
        """
        if self._mounted_start_idx == 0:
            return
        if any(
            worker.group == "backfill" and not (worker.is_finished or worker.is_cancelled)
            for worker in self.workers
        ):
            return
        self.run_worker(self._ensure_scrollable(), group="backfill")

    async def _ensure_scrollable(self) -> None:
        """Restore older batches until the container has real scroll headroom.

        WHY THIS EXISTS. ``NearTop`` can only fire if the user can actually scroll up. When
        the mounted window is shorter than the viewport, ``max_scroll_y`` is 0, no scroll is
        possible, and every older batch would be permanently unreachable — a silent
        full-scrollback violation with no error anywhere.

        Stops as soon as there is headroom, so it does NOT undo the windowing: it restores
        the minimum needed to make scrolling possible, not the history.
        """
        stalls = 0
        waits = 0
        while self._mounted_start_idx > 0 and self._scroll is not None:
            # Layout-derived: max_scroll_y comes from virtual_size, which is not current
            # synchronously. Reading it without yielding first can see a stale 0 and restore
            # batches a tail that already overflows never needed.
            await asyncio.sleep(0)
            if self._scroll.max_scroll_y > PruningScrollContainer.LOAD_THRESHOLD:
                return
            if self._loading_more:
                # A user scroll-up restore is already in flight. WAIT for it rather than
                # restoring concurrently or giving up: exiting here would abandon the
                # backfill permanently, and if that restore's batch still does not overflow
                # the viewport, the remaining history becomes unreachable forever.
                # Counted against its OWN budget, not the stall ceiling — see
                # BACKFILL_MAX_STALLS.
                await asyncio.sleep(self.BACKFILL_WAIT_SECONDS)
                waits += 1
                if waits > self.BACKFILL_MAX_WAITS:
                    return
                continue
            waits = 0
            before = self._mounted_start_idx
            await self._restore_turns()
            if self._mounted_start_idx < before:
                stalls = 0
            else:
                stalls += 1
                if stalls > self.BACKFILL_MAX_STALLS:
                    return

    async def _restore_turns(self) -> None:
        """Restore a batch of pruned turns at the top of the scroll container.

        Debounced by _loading_more flag. Adjusts scroll_y to prevent visual jump.
        Always resets _loading_more in finally block.
        """

        if self._loading_more or self._mounted_start_idx == 0 or self._scroll is None:
            return
        # Held for the WHOLE replay, so live rendering cannot interleave with it and land in
        # a historical message. _HISTORY_REPLAY is set inside so this method's own nested
        # add_* calls skip the lock instead of deadlocking on it.
        async with self.exclusive_render():
            await self._restore_turns_locked()

    async def _restore_turns_locked(self) -> None:
        """Body of _restore_turns; runs holding the render lock. See that method."""
        if self._scroll is None:
            return
        self._loading_more = True
        # Saved here and restored in the FINALLY: restoring mid-try would leave every
        # pointer aimed at replayed widgets whenever the replay raises and the exception is
        # swallowed below. For _late_window_turn that corruption is permanent — reopen
        # would target a removed widget for the rest of the feed's life.
        saved_turn = self._current_turn
        saved_message = self._current_message
        saved_thought = self._current_thought
        saved_late_window = self._late_window_turn
        saved_start_idx = self._mounted_start_idx
        # Identity snapshot for the all-or-nothing rollback in the finally.
        turns_before = {
            child for child in self._scroll.children if isinstance(child, TurnContainer)
        }
        committed = False
        # ALL FOUR pointers are suppressed for the duration, not just eligibility. This is an
        # isolated replay path: if a live message were still streaming, replay_event would
        # find it through _current_message and append HISTORICAL text into the live widget,
        # then finalize it — corrupting both the live output and the restored history.
        self._current_turn = None
        self._current_message = None
        self._current_thought = None
        self._late_window_turn = None
        try:
            batch_start = max(0, self._mounted_start_idx - RESTORE_BATCH)
            batch = self._turn_events[batch_start : self._mounted_start_idx]

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

            if not restored_turns:
                return

            # COMPENSATE FROM A STABLE ANCHOR, not from a container-height delta. The
            # previous approach compared virtual_size.height before and after, which was
            # calibrated for the prune case where the mounted region is large. Prepending a
            # batch onto a three-turn window is a much bigger relative jump, and measured
            # there it moved the reading position by 9 rows. The topmost previously-mounted
            # turn is a fixed point in the content the user is looking at, so the exact
            # amount everything shifted down is the change in ITS position — measured, never
            # predicted, so no height is cached or estimated.
            anchor_widget = self._scroll.children[0] if self._scroll.children else None
            anchor_before = anchor_widget.virtual_region.y if anchor_widget is not None else 0

            # Move restored turns above the existing content. REORDERED, not remounted:
            # Widget.remove() prunes the whole subtree, so removing each turn and mounting
            # it again produced an EMPTY container and silently discarded every restored
            # widget. Nothing caught it because the surrounding tests assert turn counts,
            # not turn contents.
            #
            # REVERSE ORDER against the CURRENT first child, not forward against a captured
            # one. Repeated move_child calls that all target one captured anchor interleave
            # the batch — measured, a 10-turn restore produced
            # [12, 24, 14, 15, ..., 22, 13, 23] — because each move shifts the indices the
            # next move is resolved against. Walking backwards and always inserting at the
            # front is order-preserving by construction and re-reads the position every time.
            for turn in reversed(restored_turns):
                first = self._scroll.children[0] if self._scroll.children else None
                if first is not None and first is not turn:
                    self._scroll.move_child(turn, before=first)

            # Update index BEFORE yield
            self._mounted_start_idx = batch_start
            committed = True

            # Yield for layout
            await asyncio.sleep(0)

            # Adjust scroll to prevent visual jump
            if anchor_widget is not None:
                shift = anchor_widget.virtual_region.y - anchor_before
                if shift:
                    self._scroll.scroll_to(
                        y=self._scroll.scroll_y + shift,
                        animate=False,
                        immediate=True,
                        release_anchor=False,
                    )
        except Exception:
            log.exception("Error restoring turns")
        finally:
            # ALL OR NOTHING. This method is not safe to interrupt part-way: it mounts turns
            # incrementally, so an interrupted run leaves PARTIALLY RENDERED turns behind, and
            # because `_mounted_start_idx` has not advanced the next run replays the same
            # batches on top of them. Measured before this rollback existed: interrupting
            # after one prompt had rendered, then re-running, produced 26 prompt bubbles for
            # 25 distinct prompts, with `prompt 12` duplicated and scrollback order corrupted.
            # Removing only EMPTY turns was not enough, because a half-rendered turn is not
            # empty.
            #
            # So on any exit that did not reach the commit point — exception, or cancellation
            # from a re-triggered worker or feed teardown — every turn this run mounted is
            # removed and the index is left exactly as it was found. Removal is NOT awaited:
            # this runs during cancellation, where awaiting raises immediately, and Textual
            # registers the removal synchronously.
            if self._scroll is not None:
                if committed:
                    stale = [
                        child
                        for child in self._scroll.children
                        if isinstance(child, TurnContainer) and not child.children
                    ]
                else:
                    stale = [
                        child
                        for child in self._scroll.children
                        if isinstance(child, TurnContainer) and child not in turns_before
                    ]
                    self._mounted_start_idx = saved_start_idx
                if stale:
                    with contextlib.suppress(Exception):
                        self._scroll.remove_children(stale)
            self._current_turn = saved_turn
            self._current_message = saved_message
            self._current_thought = saved_thought
            self._late_window_turn = saved_late_window
            self._loading_more = False

    async def _update_plan_impl(self, entries: list[object]) -> None:
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

    async def add_steered_message(self, from_agent: str | None, text: str) -> None:
        """Render a steered inter-agent message inside the RUNNING turn.

        A steered message is injected into a turn that is already in flight, so it
        mounts into the current turn container rather than starting a new one —
        splitting the turn would misrepresent when the message arrived.

        Args:
            from_agent: Sender agent ID, or None when unattributed.
            text: The message text as it was sent to the harness.
        """
        async with self.exclusive_render():
            await self._add_steered_message_impl(from_agent, text)

    async def _add_steered_message_impl(self, from_agent: str | None, text: str) -> None:
        """Mount the steered-message block into the current turn."""
        target = await self._mount_target()
        if target is None:
            return
        ts = datetime.now(UTC).strftime("%H:%M")
        container = Vertical(classes="mcp-msg")
        await target.mount(container)
        async with container.batch():
            await container.mount(CopyButton(lambda t=text: t))
            await container.mount(Markdown(text, open_links=False))
            await container.mount(
                Static(
                    f"[dim]◈ {escape(from_agent or 'agent')} → {escape(self._agent_id)}"
                    f"  {ts}[/dim]",
                    classes="bubble-ts",
                )
            )

    async def _add_hook_notification_impl(self, hook_name: str) -> None:
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

    async def _run_shell_command_impl(self, command: str) -> None:
        """Run a shell command and display the output in the feed.

        Args:
            command: Shell command string to execute.
        """
        import asyncio

        if self._scroll is None:
            return
        # A shell command is a real new-turn boundary: it starts its own turn and clears
        # _current_turn afterwards, so without the shared closer a `!cmd` after a late
        # reopen would replace _current_turn while leaving the reopened stream open.
        await self._close_late_window()
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
