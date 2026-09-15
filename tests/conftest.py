"""Shared test fixtures, including the performance replay harness.

``synthetic_journal`` builds a deterministic BrokerEvent stream matching the
proportions of a real 21-agent session, and ``perf_replay`` drives it into a
headless SynthApp while measuring loop lag, pump latency, GC pauses and widget
counts. Neither touches the user's database.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict
from textual.app import App
from textual.widgets import ContentSwitcher

from synth_acp.diagnostics import (
    GCPauseRecorder,
    LagStats,
    LoopLagSampler,
    ObjectCounters,
    PumpLatencyProbe,
    WidgetCounters,
    manual_gen2_pause_ms,
    object_counters,
    widget_counters,
)
from synth_acp.models.agent import AgentConfig, css_id
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentThoughtReceived,
    BrokerEvent,
    HookFired,
    MessageChunkReceived,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UserPromptSubmitted,
)
from synth_acp.ui.app import SynthApp
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.conversation import ConversationFeed
from synth_acp.ui.widgets.gradient_bar import GradientBarVisual


@pytest.fixture
def available_harness_binaries() -> Iterator[None]:
    """Make harness availability explicit instead of inheriting the developer's PATH."""
    with patch(
        "synth_acp.broker.lifecycle.shutil.which",
        side_effect=lambda binary: f"/test-bin/{binary}",
    ):
        yield


# ── synthetic journal ──────────────────────────────────────────────────────────

# Measured proportions from session SHScienceRetrieverDataset-86d5d716: 2560 events
# across 21 agents. The six counts sum to exactly 2560.
_REFERENCE_AGENTS = 21
_TOOL_CALLS = 1109
_MESSAGE_CHUNKS = 644
_THOUGHTS = 619
_PROMPTS = 86
_COMPLETES = 81
_HOOKS = 21
_DIFF_BEARING = 152

_MEAN_RAW_INPUT_BYTES = 15 * 1024
_MAX_DIFF_BYTES = int(42.5 * 1024)

# Two-point distribution whose mean is exactly _MEAN_RAW_INPUT_BYTES, so the mean is
# a property of the construction rather than of the seed.
_RAW_INPUT_SIZES = (_MEAN_RAW_INPUT_BYTES // 2, _MEAN_RAW_INPUT_BYTES * 3 // 2)

# Leads with the measured maximum, so even a fixture with a single diff reaches it and
# the benchmark cannot degenerate to all-tiny diffs.
_DIFF_SIZES = (_MAX_DIFF_BYTES, 2 * 1024, 8 * 1024)

# Fixed 20-slot kind mix, so the widgets each ToolCallBlock mounts (locations, raw
# input, raw output, text) reflect a real session rather than a bare header.
_TOOL_KIND_CYCLE = (
    "read",
    "read",
    "read",
    "read",
    "read",
    "read",
    "read",
    "read",
    "edit",
    "edit",
    "edit",
    "edit",
    "edit",
    "execute",
    "execute",
    "execute",
    "search",
    "search",
    "other",
    "other",
)

_BASE_TIMESTAMP = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)


def synthetic_agent_ids(agents: int = 21) -> list[str]:
    """Return the agent ids ``synthetic_journal`` uses for ``agents`` agents."""
    return [f"agent-{index:02d}" for index in range(agents)]


async def wait_for_transient_workers(app: App) -> None:
    """Wait for an app's workers, excluding the permanent runtime subsystems.

    ``SynthApp.on_mount`` starts a GC freeze worker in group "gc" — and, under
    SYNTH_DIAG, diagnostics workers in group "diag" — that run for the app's whole
    lifetime by design. A bare ``app.workers.wait_for_complete()`` therefore NEVER
    returns. Use this instead in any test that needs to wait for worker completion.

    The empty case must be skipped rather than passed through: Textual resolves the
    argument as ``workers or self`` (worker_manager.py:179), so handing it an empty list
    silently falls back to waiting on EVERY worker, including the permanent ones, and
    hangs exactly as the bare call does.
    """
    transient = [worker for worker in app.workers if worker.group not in {"gc", "diag"}]
    if transient:
        await app.workers.wait_for_complete(transient)


@pytest.fixture(autouse=True)
def _strict_render_lock():
    """Make an out-of-lock recording RAISE for the whole test suite.

    Production logs the violation instead, because turning a data-divergence guard into an
    exception on the UI message pump would trade a wrong batch for a dead app. In tests it must
    be fatal: four review rounds found a path that recorded outside the render lock, and the
    point of the invariant is that the fifth fails on its own rather than waiting for a
    reviewer to think of it.
    """
    from synth_acp.ui.widgets import conversation as conversation_module

    previous = conversation_module._STRICT_RENDER_LOCK
    conversation_module._STRICT_RENDER_LOCK = True
    try:
        yield
    finally:
        conversation_module._STRICT_RENDER_LOCK = previous


@contextmanager
def unfixed_timers() -> Iterator[None]:
    """Revert the UI phase's timer fix, for baseline measurement only.

    Restores exactly two pre-fix behaviours and nothing else: ``GradientBar.on_mount`` arms
    its 15Hz refresh unconditionally, and ``ExpandableSection.compose`` yields an eager
    inactive ActivityBar. Together those reproduce the leak of a periodic timer on a widget
    that is ``display: none`` from its first layout, which Textual never posts ``Hide`` to.

    Needed by two callers: the widget-count baseline in the UI gates, and the harness
    detection floor in ``test_perf_harness.py``, whose ``hidden_widgets_with_timers > 200``
    assertion is otherwise unsatisfiable once the fix lands. It lives here rather than in a
    test module so neither caller has to import from the other.
    """
    from textual.containers import VerticalScroll
    from textual.widgets import Static

    from synth_acp.ui.widgets.expandable_section import ExpandableSection, _Header, _ToggleLabel
    from synth_acp.ui.widgets.gradient_bar import ActivityBar, GradientBar

    def _unfixed_on_mount(self: GradientBar) -> None:
        self.auto_refresh = 1 / 15
        self._gradient = self._build_gradient()
        self._visual = GradientBarVisual(self._gradient)
        self.app.theme_changed_signal.subscribe(self, self._on_theme_changed)

    def _unfixed_compose(self: ExpandableSection):
        header = _Header(
            _ToggleLabel("▶ Expand", id="es-toggle"),
            Static("", id="es-preview"),
            classes="es-header",
        )
        activity = ActivityBar(classes="es-activity")
        activity.active = False
        body = VerticalScroll(*self._content_children, classes="es-body")
        if self._max_content_height != 20:
            body.styles.max_height = self._max_content_height
        if self._toggle_position == "top":
            yield header
            yield body
            yield activity
        else:
            yield activity
            yield body
            yield header

    with (
        patch.object(GradientBar, "on_mount", _unfixed_on_mount),
        patch.object(ExpandableSection, "compose", _unfixed_compose),
    ):
        yield


async def drain_pending_diffs(app: App, *, timeout: float = 60.0) -> None:
    """Wait until no conversation feed has diff render work outstanding.

    Diffs are prepared off-thread and mounted by a standing per-feed worker, so a snapshot
    taken once the screen goes quiet can still race pending mounts.

    This drains the WORK, not the worker. ``ConversationFeed._diff_executor`` is a standing
    worker that never completes while its feed is mounted, so awaiting it would hang
    exactly as a bare ``wait_for_complete()`` does against the permanent GC worker — which
    is why the "diffs" group stays excluded from ``wait_for_transient_workers``.

    Args:
        app: The running app.
        timeout: Seconds to wait before failing.

    Raises:
        AssertionError: if work is still outstanding when the timeout expires. Returning
            normally on timeout would reintroduce exactly the undercount this prevents.
    """
    from synth_acp.ui.widgets.conversation import ConversationFeed
    from synth_acp.ui.widgets.tool_call import ToolCallBlock

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        pending = [
            block
            for feed in app.query(ConversationFeed)
            for block in feed.query(ToolCallBlock)
            if block.has_pending_diffs()
        ]
        if not pending:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"diff work still pending after {timeout}s")


def _scaled(count: int, agents: int) -> int:
    """Scale a 21-agent reference count to ``agents`` agents."""
    if agents == _REFERENCE_AGENTS:
        return count
    return max(1, round(count * agents / _REFERENCE_AGENTS))


def synthetic_journal(*, agents: int = 21, seed: int = 20260729) -> list[BrokerEvent]:
    """Build a deterministic BrokerEvent sequence matching real measured proportions.

    Fixed seed => byte-identical output across runs and machines. Proportions from real
    session SHScienceRetrieverDataset-86d5d716: per 2560 events at 21 agents, ToolCallUpdated
    1109 (mean payload 15KB; 152 carrying a diff, max 42.5KB), MessageChunkReceived 644,
    AgentThoughtReceived 619, UserPromptSubmitted 86, TurnComplete 81, HookFired 21. Every
    AgentThoughtReceived run is length 1.

    Requires NO access to ~/.synth/synth.db.
    """
    rng = random.Random(seed)
    agent_ids = synthetic_agent_ids(agents)

    prompts = _scaled(_PROMPTS, agents)
    completes = _scaled(_COMPLETES, agents)
    hooks = _scaled(_HOOKS, agents)
    tool_calls = _scaled(_TOOL_CALLS, agents)
    chunks = _scaled(_MESSAGE_CHUNKS, agents)
    thoughts = _scaled(_THOUGHTS, agents)
    diff_bearing = _scaled(_DIFF_BEARING, agents)

    # 86 prompts across 21 agents is not divisible: agents below the remainder get one
    # extra turn. 86 - 81 = 5 turns are left unterminated, matching a capture that
    # ended mid-stream; deterministically those are the LAST turn of agents 0..4.
    base_turns, extra_turns = divmod(prompts, agents)
    turns_per_agent = [base_turns + (1 if index < extra_turns else 0) for index in range(agents)]
    unterminated = prompts - completes

    turn_slots: list[tuple[int, int]] = [
        (agent_index, turn_index)
        for agent_index, count in enumerate(turns_per_agent)
        for turn_index in range(count)
    ]
    total_turns = len(turn_slots)
    per_turn: dict[tuple[int, int], dict[str, int]] = {
        slot: {"tool": 0, "chunk": 0, "thought": 0} for slot in turn_slots
    }
    for key, total in (("tool", tool_calls), ("chunk", chunks), ("thought", thoughts)):
        for i in range(total):
            per_turn[turn_slots[i % total_turns]][key] += 1

    per_agent: list[list[BrokerEvent]] = [[] for _ in range(agents)]
    seq = 0
    tool_index = 0
    diff_index = 0

    def _stamp() -> datetime:
        nonlocal seq
        seq += 1
        return _BASE_TIMESTAMP + timedelta(milliseconds=seq)

    def _next_diff_index() -> int | None:
        """Select exactly ``diff_bearing`` of ``tool_calls`` tool calls, evenly spaced.

        Returns the diff's ordinal when this tool call carries one, else None.
        """
        nonlocal tool_index, diff_index
        before = (tool_index * diff_bearing) // tool_calls
        after = ((tool_index + 1) * diff_bearing) // tool_calls
        tool_index += 1
        if after == before:
            return None
        current = diff_index
        diff_index += 1
        return current

    for agent_index, agent_id in enumerate(agent_ids):
        events = per_agent[agent_index]
        if agent_index < hooks:
            events.append(
                HookFired(agent_id=agent_id, timestamp=_stamp(), hook_name="on_agent_startup")
            )
        for turn_index in range(turns_per_agent[agent_index]):
            events.append(
                UserPromptSubmitted(
                    agent_id=agent_id,
                    timestamp=_stamp(),
                    text=f"prompt {turn_index} for {agent_id}",
                )
            )
            counts = per_turn[(agent_index, turn_index)]

            # Non-thought events for this turn, in a fixed order.
            others: list[BrokerEvent] = []
            for n in range(counts["chunk"]):
                # The seed varies only streamed TEXT, never a payload size or kind, so it
                # cannot perturb the mean-payload or max-diff invariants.
                others.append(
                    MessageChunkReceived(
                        agent_id=agent_id,
                        timestamp=_stamp(),
                        chunk=f"chunk {n} " + "word " * rng.randint(1, 12),
                    )
                )
            for n in range(counts["tool"]):
                others.append(
                    _tool_call(
                        agent_id,
                        _stamp(),
                        f"{agent_id}-t{turn_index}-{n}",
                        tool_index=tool_index,
                        diff_index=_next_diff_index(),
                    )
                )

            # Thoughts are spread BETWEEN non-thought events so no two are ever
            # adjacent: every AgentThoughtReceived run has length 1. The measured
            # proportions give ~20 non-thought events per turn against ~7 thoughts,
            # so there is always room.
            events.extend(_interleave_thoughts(others, counts["thought"], agent_id, _stamp))

            is_last_turn = turn_index == turns_per_agent[agent_index] - 1
            if not (is_last_turn and agent_index < unterminated):
                events.append(
                    TurnComplete(agent_id=agent_id, timestamp=_stamp(), stop_reason="end_turn")
                )

    return _interleave(per_agent)


def _interleave_thoughts(
    others: list[BrokerEvent],
    thought_count: int,
    agent_id: str,
    stamp: Any,
) -> list[BrokerEvent]:
    """Weave ``thought_count`` thoughts among ``others`` with no two adjacent.

    Places one thought, then a block of non-thought events, repeating. The block size
    is chosen so the non-thought events last exactly as long as the thoughts do.

    Raises:
        AssertionError: if there are too few non-thought events to separate the
            thoughts, which would break the length-1 run invariant.
    """
    if thought_count == 0:
        return list(others)
    assert len(others) >= thought_count - 1, (
        f"{thought_count} thoughts cannot be separated by {len(others)} other events"
    )
    woven: list[BrokerEvent] = []
    remaining_others = list(others)
    for index in range(thought_count):
        woven.append(
            AgentThoughtReceived(agent_id=agent_id, timestamp=stamp(), chunk=f"reasoning {index}\n")
        )
        thoughts_left = thought_count - index - 1
        if thoughts_left:
            take = max(1, -(-len(remaining_others) // (thoughts_left + 1)))
        else:
            take = len(remaining_others)
        woven.extend(remaining_others[:take])
        remaining_others = remaining_others[take:]
    woven.extend(remaining_others)
    return woven


def _tool_call(
    agent_id: str,
    stamp: datetime,
    tool_call_id: str,
    *,
    tool_index: int,
    diff_index: int | None,
) -> ToolCallUpdated:
    """Build one ToolCallUpdated with a realistically sized payload.

    Every size and kind is chosen from the event's INDEX, not from the RNG, so the two
    properties the contract advertises hold BY CONSTRUCTION for any ``agents`` and any
    ``seed``: the mean serialized ``raw_input`` is the measured 15 KB, and the maximum
    diff reaches the measured 42.5 KB. Sampling them randomly made both properties true
    only for the large default fixture — at ``agents=1`` the mean drifted to 14.4 KB and
    some seeds produced no diff larger than 8 KB, silently making an alternate fixture
    materially lighter than the benchmark it claims to be.

    The ``kind`` cycle also matters: ``ToolCallBlock._raw_input_widgets`` renders a
    payload only under the ``command``/``cmd`` key and ``_raw_output_widgets`` fires only
    for execute/search/fetch, so a uniform ``read`` kind would generate the payload and
    then discard it, mounting a bare two-widget header.
    """
    # Alternating two-point distribution whose mean is exactly _MEAN_RAW_INPUT_BYTES.
    size = _RAW_INPUT_SIZES[tool_index % len(_RAW_INPUT_SIZES)]
    kind = _TOOL_KIND_CYCLE[tool_index % len(_TOOL_KIND_CYCLE)]
    locations = (
        [ToolCallLocation(path=f"src/{tool_call_id}.py", line=42)]
        if kind in {"read", "edit"}
        else []
    )
    raw_output: Any = None
    if kind in {"execute", "search", "fetch"}:
        raw_output = {"output": "line of captured output\n" * 40, "exitStatus": 0}
    text_content = ("notes\n\n" + "- observation\n" * 20) if kind == "other" else None
    diffs = [_diff(tool_call_id, diff_index)] if diff_index is not None else []
    return ToolCallUpdated(
        agent_id=agent_id,
        timestamp=stamp,
        tool_call_id=tool_call_id,
        title=f"{kind} {tool_call_id}",
        kind=kind,
        status="completed",
        locations=locations,
        raw_input={"command": "x" * size},
        raw_output=raw_output,
        diffs=diffs,
        text_content=text_content,
    )


def _diff(tool_call_id: str, diff_index: int) -> ToolCallDiff:
    """Build one ToolCallDiff whose size is chosen by index, not sampled.

    ``_DIFF_SIZES`` leads with the measured maximum, so the FIRST diff of any fixture
    reaches 42.5 KB however few diffs it has.
    """
    size = _DIFF_SIZES[diff_index % len(_DIFF_SIZES)]
    return ToolCallDiff(
        path=f"src/{tool_call_id}.py",
        old_text="old\n" * 8,
        new_text="n" * size,
    )


def _interleave(per_agent: list[list[BrokerEvent]]) -> list[BrokerEvent]:
    """Round-robin per-agent sequences into one flat journal."""
    flat: list[BrokerEvent] = []
    index = 0
    longest = max((len(events) for events in per_agent), default=0)
    while index < longest:
        for events in per_agent:
            if index < len(events):
                flat.append(events[index])
        index += 1
    return flat


# ── perf replay ────────────────────────────────────────────────────────────────


class PerfRun(BaseModel):
    """Result of one instrumented replay."""

    model_config = ConfigDict(frozen=True)
    widgets: WidgetCounters
    objects: ObjectCounters
    loop_lag: LagStats
    pump_latency: LagStats
    worst_auto_oldgen_pause_ms: (
        float  # worst interpreter-triggered pause; see max_automatic_pause_ms
    )
    manual_gen2_ms: float
    wall_ms: float
    first_switch_max_stall_ms: float
    first_switch_settle_ms: float


class _FakeBroker:
    """Minimal broker stand-in exposing only what SynthApp touches.

    A hand-written class rather than a MagicMock, so the fake does not distort the
    object counts this harness reports.
    """

    def __init__(self, agent_ids: Sequence[str], db_path: str) -> None:
        self._initial_agent = AgentConfig(agent_id=agent_ids[0], harness="kiro")
        self._db_path = db_path
        self.session_id = "perf-session"
        self._registry = _FakeRegistry()

    def set_composing_check(self, check: Any) -> None:
        self._composing_check = check

    async def handle(self, command: Any) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def events(self) -> Any:
        # Parks forever: the harness posts BrokerEventMessages directly.
        while True:
            await asyncio.sleep(3600)
            yield  # pragma: no cover

    async def load_journal(self, agent_id: str, session_id: str) -> list[BrokerEvent]:
        return []

    def get_agent_parent(self, agent_id: str) -> str | None:
        return None

    def get_agent_harness(self, agent_id: str) -> str:
        return "kiro"

    def get_agent_cwd(self, agent_id: str) -> str:
        return "."

    def get_agent_display_name(self, agent_id: str) -> str | None:
        return None

    def get_usage(self, agent_id: str) -> None:
        return None

    def get_discovered_agents(self, agent_id: str) -> list[Any]:
        return []

    def permission_position(self, agent_id: str) -> str:
        return "bottom"

    def is_permission_pending(self, agent_id: str) -> bool:
        return False


class _FakeRegistry:
    def get_agent_mode_target(self, agent_id: str) -> str:
        return "acp_mode"

    def get_agent_mode(self, agent_id: str) -> str | None:
        return None


async def _quiesce(app: App, *, quiet_polls: int = 3, timeout: float = 60.0) -> None:
    """Wait until the app has no pending render or message work.

    Replaces ``pilot.pause()``, which must never appear inside a timed window:
    ``Pilot._wait_for_screen`` posts one Callback per widget in the WHOLE app and
    awaits all of them, so it is O(total widgets) by construction and silently
    inflates every measurement.

    Raises:
        AssertionError: if the app is still dirty when the timeout expires.
            Returning normally on timeout would let a timed window close over
            outstanding layout/repaint work and UNDER-report stalls.
    """
    deadline = time.perf_counter() + timeout
    quiet = 0
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
        screen = app.screen
        busy = (
            screen._layout_required
            or screen._scroll_required
            or screen._repaint_required
            or bool(screen._dirty_widgets)
            or bool(screen._callbacks)
            or app._message_queue.qsize() > 0
        )
        if busy:
            quiet = 0
            continue
        quiet += 1
        if quiet >= quiet_polls:
            return
    raise AssertionError(f"app did not quiesce within {timeout}s")


async def _run_isolated(
    journal_events: list[BrokerEvent],
    agent_ids: Sequence[str],
    churn_seconds: float,
    size: tuple[int, int],
) -> PerfRun:
    """Execute one repetition on its own event loop in a dedicated thread.

    Delegates the isolation to ``_run_on_fresh_loop``; see it for why repetitions cannot
    share an event loop.
    """
    return await _run_on_fresh_loop(
        lambda: _one_run(journal_events, agent_ids, churn_seconds, size)
    )


async def _one_run(
    journal_events: list[BrokerEvent],
    agent_ids: Sequence[str],
    churn_seconds: float,
    size: tuple[int, int],
) -> PerfRun:
    """Replay the journal into one headless app instance and measure it."""
    panel_ids = list(dict.fromkeys([*agent_ids, *(e.agent_id for e in journal_events)]))
    broker = _FakeBroker(panel_ids, db_path=":memory:")
    app = SynthApp(
        broker,  # type: ignore[arg-type]
        SessionConfig(project="perf"),
        initial_agent=AgentConfig(agent_id=panel_ids[0], harness="kiro"),
    )

    sampler = LoopLagSampler(interval=0.02)
    probe = PumpLatencyProbe(app, interval=0.05)
    recorder = GCPauseRecorder()

    # Keeps _index_sessions away from any database.
    with patch("synth_acp.ui.app.embedding_available", return_value=False):
        async with app.run_test(headless=True, size=size):
            await app.select_agent(panel_ids[0])
            switcher = app.query_one("#right", ContentSwitcher)
            for agent_id in panel_ids[1:]:
                feed = ConversationFeed(
                    agent_id,
                    agent_id,
                    "perf",
                    harness="kiro",
                    cwd=".",
                    id=f"feed-{css_id(agent_id)}",
                )
                await switcher.add_content(feed, set_current=False)
                app._panels[agent_id] = feed

            recorder.install()
            sampler_task = asyncio.create_task(sampler.run())
            probe_task = asyncio.create_task(probe.run())
            try:
                started = time.perf_counter()
                total = len(journal_events)
                for index, event in enumerate(journal_events):
                    app.post_message(BrokerEventMessage(event))
                    target = started + (index + 1) * (churn_seconds / max(1, total))
                    now = time.perf_counter()
                    if now < target:
                        await asyncio.sleep(target - now)
                    else:
                        await asyncio.sleep(0)
                await _quiesce(app)
                wall_ms = (time.perf_counter() - started) * 1000.0

                # TIMINGS are read first, so they stay scoped to the churn window. The diff
                # drain below is a settling step for the COUNTS, not part of the workload
                # being measured: sampling across it would fold the whole render backlog
                # into loop lag and pump latency and measure the harness, not the app.
                loop_lag = sampler.stats()
                pump_latency = probe.stats()
                # Generation-agnostic and excludes our own freeze-worker collects, so
                # the metric survives the CPython 3.13 incremental-collector change.
                worst_auto_oldgen_ms = recorder.max_automatic_pause_ms()
                manual_ms = manual_gen2_pause_ms()

                # COUNTS come from the settled tree. Diffs are mounted by a standing worker
                # AFTER the screen goes quiet, so a snapshot taken on dirty flags alone
                # races those mounts and reports a nondeterministic total_widgets.
                await drain_pending_diffs(app)
                await _quiesce(app)

                widgets = widget_counters(app)
                objects = object_counters()

                renderable = sum(
                    1
                    for e in journal_events
                    if isinstance(
                        e,
                        (
                            MessageChunkReceived,
                            AgentThoughtReceived,
                            ToolCallUpdated,
                            UserPromptSubmitted,
                            HookFired,
                        ),
                    )
                )
                assert widgets.total_widgets >= 2 * renderable, (
                    f"replay built only {widgets.total_widgets} widgets for {renderable} "
                    "renderable events — the harness silently failed to build a tree"
                )

                # The switch is bracketed on its own, because the overall worst stall
                # and the first-switch stall are different quantities (measured 82.28ms
                # versus 305.6ms), so loop_lag.max_ms is not a substitute.
                switch_target = _switch_target(app, panel_ids)
                sampler.reset()
                switch_started = time.perf_counter()
                switcher.current = f"feed-{css_id(switch_target)}"
                await _quiesce(app)
                first_switch_settle_ms = (time.perf_counter() - switch_started) * 1000.0
                first_switch_max_stall_ms = sampler.stats().max_ms
            finally:
                sampler_task.cancel()
                probe_task.cancel()
                await sampler_task
                await probe_task
                recorder.remove()

    return PerfRun(
        widgets=widgets,
        objects=objects,
        loop_lag=loop_lag,
        pump_latency=pump_latency,
        worst_auto_oldgen_pause_ms=worst_auto_oldgen_ms,
        manual_gen2_ms=manual_ms,
        wall_ms=wall_ms,
        first_switch_max_stall_ms=first_switch_max_stall_ms,
        first_switch_settle_ms=first_switch_settle_ms,
    )


def _switch_target(app: SynthApp, panel_ids: Sequence[str]) -> str:
    """Return the populated, never-displayed feed with the most content.

    ``perf_replay`` guarantees at least two panels, so a candidate always exists — never
    falling back to the displayed feed, which would silently measure a re-switch.
    """
    candidates = [aid for aid in panel_ids[1:] if aid in app._panels]
    assert candidates, "no never-displayed feed available to switch to"
    return max(candidates, key=lambda aid: len(app._panels[aid]._tool_call_blocks))


def _median_lag(runs: list[LagStats], final: LagStats) -> LagStats:
    """Median every timing field; take ``samples`` from the final run.

    ``samples`` is a COUNT, and the contract takes counts from the final run.
    """
    return LagStats(
        samples=final.samples,
        p50_ms=statistics.median([r.p50_ms for r in runs]),
        p95_ms=statistics.median([r.p95_ms for r in runs]),
        p99_ms=statistics.median([r.p99_ms for r in runs]),
        max_ms=statistics.median([r.max_ms for r in runs]),
    )


def _effective_repetitions(repetitions: int) -> int:
    """Resolve how many repetitions to run, honouring the SYNTH_PERF_REPS override.

    Gates keep the noise-resistant default of 3; setting ``SYNTH_PERF_REPS=1`` lets local
    iteration skip the ~30 s per extra 21-agent replay.

    An unset, empty, unparseable OR non-positive value falls back to the caller's value
    unchanged. Clamping a bad value to 1 instead would let a typo silently reduce a
    three-run gate to a single noisy run.
    """
    raw = os.environ.get("SYNTH_PERF_REPS", "").strip()
    if not raw:
        return max(1, repetitions)
    try:
        override = int(raw)
    except ValueError:
        return max(1, repetitions)
    if override < 1:
        return max(1, repetitions)
    return override


async def perf_replay(
    journal_events: list[BrokerEvent],
    *,
    agent_ids: Sequence[str],
    churn_seconds: float = 15.0,
    size: tuple[int, int] = (120, 40),
    repetitions: int = 3,
) -> PerfRun:
    """Replay events into a headless App under churn and measure.

    Returns the MEDIAN of `repetitions` runs for every timing field; counts come from the
    final run.

    MUST NOT call pilot.pause() inside any timed window. Pilot._wait_for_screen
    (textual/pilot.py:490-505) posts one Callback per widget in the whole app and awaits all
    of them, so it is O(total widgets) BY CONSTRUCTION. Poll screen._layout_required /
    _scroll_required / _repaint_required / _dirty_widgets / _callbacks until quiet instead.

    After the churn window, performs ONE agent switch to a populated, never-displayed feed and
    measures that window separately. The switch is bracketed by resetting the LoopLagSampler
    immediately before setting the ContentSwitcher's current agent, then polling
    screen._layout_required / _scroll_required / _repaint_required / _dirty_widgets /
    _callbacks until quiet. `first_switch_settle_ms` is that wall duration and
    `first_switch_max_stall_ms` is the sampler's max over that window ONLY.

    These are reported separately from `loop_lag` because the two are different quantities:
    the measured references are 82.28 ms overall worst stall versus 305.6 ms first-switch
    stall, so `loop_lag.max_ms` is not a substitute.

    Requires at least two distinct agents across `agent_ids` and `journal_events`,
    since the switch measurement needs a populated feed that was never displayed.

    Raises:
        AssertionError: if fewer than two distinct agents are supplied, or if the
            replay produces fewer mounted widgets than expected for the given event
            count, indicating the harness silently failed to build a tree.
    """
    panel_ids = list(dict.fromkeys([*agent_ids, *(e.agent_id for e in journal_events)]))
    # The switch window is only meaningful with a populated feed that was NEVER
    # displayed, which needs at least two panels. Validated loudly rather than degrading
    # silently: with one panel _switch_target would return the ALREADY-DISPLAYED feed and
    # the two switch fields would quietly measure the wrong thing, and with none this
    # indexed out of range with an undeclared IndexError.
    assert len(panel_ids) >= 2, (
        f"perf_replay needs at least 2 distinct agents to measure a switch to a "
        f"never-displayed feed, got {len(panel_ids)}: {panel_ids}"
    )
    runs = [
        await _run_isolated(journal_events, agent_ids, churn_seconds, size)
        for _ in range(_effective_repetitions(repetitions))
    ]
    final = runs[-1]
    return PerfRun(
        widgets=final.widgets,
        objects=final.objects,
        loop_lag=_median_lag([r.loop_lag for r in runs], final.loop_lag),
        pump_latency=_median_lag([r.pump_latency for r in runs], final.pump_latency),
        worst_auto_oldgen_pause_ms=statistics.median([r.worst_auto_oldgen_pause_ms for r in runs]),
        manual_gen2_ms=statistics.median([r.manual_gen2_ms for r in runs]),
        wall_ms=statistics.median([r.wall_ms for r in runs]),
        first_switch_max_stall_ms=statistics.median([r.first_switch_max_stall_ms for r in runs]),
        first_switch_settle_ms=statistics.median([r.first_switch_settle_ms for r in runs]),
    )


def raw_input_bytes(event: ToolCallUpdated) -> int:
    """Return the serialized size of a tool call's raw_input payload."""
    return len(json.dumps(event.raw_input))


# ── first-paint fixture ───────────────────────────────────────────────────────

# Recorded composition of the real journal this feature was measured on: session
# SHScienceRetrieverDataset-86d5d716, agent code-planner, 779 events. The six counts sum to
# exactly 779, and 252 + 231 + 190 = 673 content events = 779 - 53 prompts - 52
# TurnCompletes - 1 hook.
_FP_AGENT = "code-planner"
_FP_TOOL_CALLS = 252
_FP_MESSAGE_CHUNKS = 231
_FP_THOUGHTS = 190
_FP_PROMPTS = 53
_FP_COMPLETES = 52
_FP_HOOKS = 1
_FP_DIFF_BEARING = 28
_FP_CONTENT_EVENTS = _FP_TOOL_CALLS + _FP_MESSAGE_CHUNKS + _FP_THOUGHTS

# Ratified turn-size construction. The cycle is skewed on purpose: with uniform turns the
# event budget would never bind before the turn cap, so FIRST_PAINT_EVENT_BUDGET — the bound
# that guards a pathological turn — would ship unexercised by any gate.
_FP_TURN_CYCLE = (4, 8, 12, 16, 24, 12, 8, 6)


def _first_paint_turn_sizes() -> list[int]:
    """Content-event counts per turn, fully determined by the ratified construction.

    The cycle is repeated across the 53 turns and the shortfall to 673 is distributed to the
    LARGEST turns ROUND-ROBIN across the tied-largest set, ties broken by ascending index.

    The round-robin reading is not a preference: it is selected by the criterion's own stated
    maximum. Always incrementing the first maximum piles the whole 69-event shortfall onto one
    turn, giving a maximum of 93 = 7.3x the mean, which contradicts "maximum roughly 3x the
    mean". Round-robin gives 34 = 2.68x, which matches.
    """
    sizes = [_FP_TURN_CYCLE[i % len(_FP_TURN_CYCLE)] for i in range(_FP_PROMPTS)]
    remaining = _FP_CONTENT_EVENTS - sum(sizes)
    while remaining:
        largest = max(sizes)
        for index in [i for i, size in enumerate(sizes) if size == largest]:
            if not remaining:
                break
            sizes[index] += 1
            remaining -= 1
    return sizes


def first_paint_journal() -> list[BrokerEvent]:
    """The single-agent journal the first-paint gate measures against.

    MATCHES the recorded shape of the real 779-event code-planner journal: the exact event
    type counts (ToolCallUpdated 252, MessageChunkReceived 231, AgentThoughtReceived 190,
    UserPromptSubmitted 53, TurnComplete 52, HookFired 1), 28 diff-bearing tool calls, 53
    turns of which 52 are closed and ONE is left open (the real 53-versus-52 shape), the
    ratified skewed turn-size distribution, and the payload SIZE distribution — ``_tool_call``
    and ``_diff`` already encode the measured 15 KB mean raw_input and 42.5 KB maximum diff by
    construction rather than by sampling.

    DOES NOT MATCH, and must not be read as if it did: the exact event ORDER within a turn,
    the real payload TEXT, the real coalescing behaviour (which depends on how chunks actually
    interleaved on the wire), and tool-call NESTING — every tool call here is top level, with
    no ``parent_tool_call_id``, because the real journal's nesting depth was never measured and
    inventing a ratio would make the fixture look more faithful than it is.

    Deterministic: no RNG, so it is byte-identical across runs and machines. Opens no database;
    the standing policy is that no test reads ~/.synth/synth.db.
    """
    turn_sizes = _first_paint_turn_sizes()
    events: list[BrokerEvent] = []
    seq = 0
    tool_index = 0
    diff_index = 0
    remaining = {
        "tool": _FP_TOOL_CALLS,
        "chunk": _FP_MESSAGE_CHUNKS,
        "thought": _FP_THOUGHTS,
    }

    def _stamp() -> datetime:
        nonlocal seq
        seq += 1
        return _BASE_TIMESTAMP + timedelta(milliseconds=seq)

    events.append(HookFired(agent_id=_FP_AGENT, timestamp=_stamp(), hook_name="on_agent_startup"))

    for turn_index, size in enumerate(turn_sizes):
        events.append(
            UserPromptSubmitted(agent_id=_FP_AGENT, timestamp=_stamp(), text=f"prompt {turn_index}")
        )
        # Round-robin the three content kinds so every turn holds a realistic mix, drawing
        # from fixed remaining budgets so the totals land exactly on the recorded counts.
        order = ("tool", "chunk", "thought")
        for slot in range(size):
            kind = order[slot % 3]
            if not remaining[kind]:
                kind = next((k for k in order if remaining[k]), "")
            if not kind:
                break
            remaining[kind] -= 1
            if kind == "tool":
                carries_diff = (
                    diff_index < _FP_DIFF_BEARING
                    and (tool_index * _FP_DIFF_BEARING) // _FP_TOOL_CALLS
                    != ((tool_index + 1) * _FP_DIFF_BEARING) // _FP_TOOL_CALLS
                )
                current_diff = diff_index if carries_diff else None
                if carries_diff:
                    diff_index += 1
                events.append(
                    _tool_call(
                        _FP_AGENT,
                        _stamp(),
                        f"tc-{tool_index}",
                        tool_index=tool_index,
                        diff_index=current_diff,
                    )
                )
                tool_index += 1
            elif kind == "chunk":
                events.append(
                    MessageChunkReceived(
                        agent_id=_FP_AGENT,
                        timestamp=_stamp(),
                        chunk=f"streamed text for turn {turn_index}\n\n",
                    )
                )
            else:
                events.append(
                    AgentThoughtReceived(
                        agent_id=_FP_AGENT,
                        timestamp=_stamp(),
                        chunk=f"reasoning for turn {turn_index}",
                    )
                )
        # The LAST turn is left OPEN: 53 prompts against 52 TurnCompletes, matching the real
        # journal and exercising the trailing-open-segment path in the tail selection.
        if turn_index < _FP_COMPLETES:
            events.append(
                TurnComplete(agent_id=_FP_AGENT, timestamp=_stamp(), stop_reason="end_turn")
            )
    return events


class FirstPaintRun(BaseModel):
    """One first-selection drain measurement."""

    model_config = ConfigDict(frozen=True)

    visible_ms: float
    rendered_ms: float
    diff_trail_ms: float
    mounted_turns: int
    mounted_widgets: int
    content_height: int
    viewport_height: int
    turn_batches: int


async def first_selection_paint(
    journal: list[BrokerEvent],
    *,
    agent_id: str,
    windowed: bool,
    size: tuple[int, int] = (120, 40),
) -> FirstPaintRun:
    """Buffer a journal, time ONE first-selection drain, and measure the result.

    Drives the real production path — it fills ``app._event_buffers`` and awaits
    ``app.select_agent`` — so ``visible_ms`` is the wall time the user actually waits before
    anything appears, not a proxy.

    The app is built with a DISTINCT DUMMY initial agent. ``SynthApp.on_mount`` selects the
    initial agent immediately and creates its panel, so making the target the initial agent
    would mean the panel already existed and ``_do_select_agent``'s first-selection drain —
    the entire thing under measurement — would never run.

    ``windowed=False`` raises the two window constants for the duration, forcing the whole
    feed through the IDENTICAL code path rather than adding a production flag for a test.

    Runs on a FRESH EVENT LOOP in a dedicated plain thread. ``SynthApp.on_unmount`` calls
    ``loop.shutdown_default_executor()``, so a second app on one loop fails as soon as Textual
    parses Markdown — which this does on every replayed message. ``asyncio.to_thread`` is
    unusable for the same reason: it goes through the executor being isolated against.
    """
    return await _run_on_fresh_loop(lambda: _one_first_paint(journal, agent_id, windowed, size))


async def _run_on_fresh_loop(factory: Callable[[], Any]) -> Any:
    """Run one coroutine factory on its own event loop in a dedicated plain thread.

    Extracted from ``_run_isolated`` so both callers share ONE implementation of this
    isolation contract rather than two that can drift. ``SynthApp.on_unmount`` calls
    ``loop.shutdown_default_executor()``, so a second app on the same loop fails as soon as
    anything needs the default executor — Textual parses Markdown there.
    ``asyncio.to_thread`` is unusable for the same reason: it goes through that executor.
    """
    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(factory())
        except BaseException as error:  # re-raised on the caller's loop below
            result["error"] = error

    thread = threading.Thread(target=_target, name="isolated-app-run")
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0.05)
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


async def _one_first_paint(
    journal: list[BrokerEvent],
    agent_id: str,
    windowed: bool,
    size: tuple[int, int],
) -> FirstPaintRun:
    """Measure one drain. Runs on its own loop; see first_selection_paint."""
    from textual.widgets import ContentSwitcher

    from synth_acp.models.agent import AgentConfig, css_id
    from synth_acp.models.config import SessionConfig
    from synth_acp.ui.app import DynamicAgentInfo, SynthApp
    from synth_acp.ui.widgets import conversation as conversation_module
    from synth_acp.ui.widgets.conversation import ConversationFeed, TurnContainer

    dummy = "dummy-initial"
    broker = _FakeBroker([dummy], db_path=":memory:")
    app = SynthApp(
        broker,  # type: ignore[arg-type]
        SessionConfig(project="perf"),
        initial_agent=AgentConfig(agent_id=dummy, harness="kiro"),
    )

    limits = {} if windowed else {"FIRST_PAINT_TURNS": 10**9, "FIRST_PAINT_EVENT_BUDGET": 10**9}
    patches = [patch.object(conversation_module, name, value) for name, value in limits.items()]

    async with app.run_test(headless=True, size=size):
        for active in patches:
            active.start()
        try:
            # Settle the pre-warm BEFORE the timed window, so its ~200ms of first-use lexer
            # loading is not charged to whichever configuration happens to run first.
            #
            # NOT wait_for_transient_workers: that helper excludes only the "gc" and "diag"
            # groups, and SynthApp also starts a standing "broker" consumer that never
            # completes, so awaiting transient workers here hangs forever. Waiting on the one
            # group that matters is both sufficient and safe.
            for _ in range(600):
                busy = [w for w in app.workers if w.group == "prewarm" and not w.is_finished]
                if not busy:
                    break
                await asyncio.sleep(0.01)

            app._dynamic_agents[agent_id] = DynamicAgentInfo(
                parent=None, task="", harness="kiro", cwd="."
            )
            app._event_buffers[agent_id] = list(journal)
            assert agent_id not in app._panels, "the drain under test must create the panel"

            started = time.perf_counter()
            await app.select_agent(agent_id)
            visible_ms = (time.perf_counter() - started) * 1000

            assert agent_id in app._panels, "the timed call did not create the panel"
            assert not app._event_buffers.get(agent_id), "the buffer was not drained"

            app.query_one("#right", ContentSwitcher).current = f"feed-{css_id(agent_id)}"
            await _quiesce(app)
            rendered_ms = (time.perf_counter() - started) * 1000

            trail_started = time.perf_counter()
            await drain_pending_diffs(app, timeout=300.0)
            diff_trail_ms = (time.perf_counter() - trail_started) * 1000

            # _quiesce, never pilot.pause(): Pilot._wait_for_screen posts one Callback per
            # widget in the whole app, so it is O(total widgets) by construction and has no
            # place in a benchmark module.
            await _quiesce(app)
            feed = app._panels[agent_id]
            assert feed._scroll is not None
            turns = [c for c in feed._scroll.children if isinstance(c, TurnContainer)]
            return FirstPaintRun(
                visible_ms=visible_ms,
                rendered_ms=rendered_ms,
                diff_trail_ms=diff_trail_ms,
                mounted_turns=len(turns),
                mounted_widgets=len(list(feed.query(ConversationFeed).nodes))
                + len(list(feed.query("*").nodes)),
                content_height=feed._scroll.virtual_size.height,
                viewport_height=feed._scroll.size.height,
                turn_batches=len(feed._turn_events),
            )
        finally:
            for active in reversed(patches):
                active.stop()


__all__ = [
    "FirstPaintRun",
    "PerfRun",
    "first_paint_journal",
    "first_selection_paint",
    "perf_replay",
    "raw_input_bytes",
    "synthetic_agent_ids",
    "synthetic_journal",
]
