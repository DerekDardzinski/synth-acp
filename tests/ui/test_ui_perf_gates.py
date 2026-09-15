"""Full-fixture gates for the UI layer, plus the two user non-negotiables.

These gate the whole UI layer against the shared performance harness rather than one
module, so they have no source mirror — the same reason ``tests/test_perf_harness.py``
exists for ``tests/conftest.py``.

The 21-agent replay costs roughly 30s per repetition, so the gates share cached runs
instead of each paying for one. ``PerfRun`` is a frozen model with no event-loop affinity,
so caching across function-scoped test loops is safe.
"""

from __future__ import annotations

import asyncio
import gc
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import ContentSwitcher

from synth_acp.diagnostics import PumpLatencyProbe
from synth_acp.models.agent import AgentConfig, css_id
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import MessageChunkReceived, ToolCallUpdated, TurnComplete
from synth_acp.ui.app import SynthApp
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.conversation import ConversationFeed, TurnContainer
from synth_acp.ui.widgets.gradient_bar import GradientBar
from tests import conftest as harness
from tests.conftest import (
    PerfRun,
    perf_replay,
    synthetic_agent_ids,
    synthetic_journal,
)

_AGENTS = 21

# (PerfRun, live GradientBar count observed at the counts snapshot), keyed by configuration.
_CACHE: dict[str, tuple[PerfRun, int]] = {}


@pytest.fixture(scope="module", autouse=True)
def _isolated_frozen_graph():
    """Start and leave the permanent generation clean.

    The full-fixture replays freeze millions of objects, and the production fix never
    reclaims them (an accepted tradeoff). Released on the way OUT so later tests in the
    process are unaffected, and on the way IN because another gate module may already have
    left millions of frozen objects behind — measured, that inflates this module's
    first-switch reading from ~290ms to ~550ms and has nothing to do with the code here.
    """
    gc.unfreeze()
    gc.collect()
    yield
    gc.unfreeze()
    gc.collect()


async def _cached_run(*, fixed: bool = True) -> tuple[PerfRun, int]:
    """Return the cached 21-agent replay and its live GradientBar count.

    The GradientBar count is recorded HERE, at cache-fill time, and stored beside the
    PerfRun. Recording it in a test body instead would make the observation order
    dependent: whichever gate ran first would fill the cache, and a later gate would
    receive a cached PerfRun with no live snapshot ever taken under its wrapper. The
    contracted ``PerfRun``/``WidgetCounters`` are deliberately left unchanged — they name no
    widget class by design — so the count is captured by wrapping the snapshot call.
    """
    assert fixed, "the unfixed_timers() baseline runs were removed; see test_stacked_ui_gate"
    key = "fixed"
    if key not in _CACHE:
        journal = synthetic_journal(agents=_AGENTS)
        agent_ids = synthetic_agent_ids(_AGENTS)
        observed: list[int] = []
        real_counters = harness.widget_counters

        def _recording(app: Any):
            observed.append(len(app.screen.query(GradientBar)))
            return real_counters(app)

        with patch.object(harness, "widget_counters", _recording):
            run = await perf_replay(journal, agent_ids=agent_ids)
        _CACHE[key] = (run, max(observed, default=0))
    return _CACHE[key]


@pytest.mark.perf
class TestStackedUIGate:
    """The A+B reading, which is the only one that corresponds to the user's experience.

    Marked `perf`: every test here replays the 21-agent journal, which costs minutes. Run with
    `uv run pytest -m perf`.
    """

    async def test_stacked_ui_gate(self) -> None:
        """Worst-case freeze must be gone with both phases' fixes active.

        Silent failure this catches: the four root causes each mask the others, so a
        mechanism can pass its own isolated gate while another phase's residual keeps the
        user's freeze in place. Only the stacked reading rules that out.

        p95 loop lag is deliberately NOT asserted to improve: it is expected to stay near
        59ms by design, because the residual is layout and mount CPU over ~1,800 widgets
        per feed, which this feature does not address.
        """
        run, _ = await _cached_run(fixed=True)

        assert run.widgets.hidden_widgets_with_timers == 0, (
            f"{run.widgets.hidden_widgets_with_timers} widgets hold a timer while hidden"
        )
        # ABSOLUTE WIDGET CEILING. Measured fixed value is 23,777. This is the gate that
        # protects agent-switch cost now that no timing assertion covers it: switch stall is
        # O(target feed size), and nothing else bounds widget MAGNITUDE in the fixed state —
        # the harness-detection floor is on the unfixed baseline, and the spread check below
        # asserts stability rather than size.
        #
        # An absolute bound is trustworthy HERE and almost nowhere else in this phase, because
        # synthetic_journal is seeded and deterministic so this count is exactly reproducible,
        # unlike every timing reading.
        #
        # RAISE THIS ONLY BY A DELIBERATE DECISION that records why the additional widgets are
        # justified. Never raise it to make a failing test pass: widget growth per event is the
        # dimension both recorded limitations point at — the ~86 events/sec sustained
        # throughput ceiling and the per-event diff-mount floor on worst-case pump latency —
        # and it is what the deferred widgets-per-event reduction exists to attack.
        assert run.widgets.total_widgets < 25000, (
            f"{run.widgets.total_widgets} widgets against a 25,000 ceiling (measured 23,777)"
        )
        assert run.loop_lag.max_ms < 150, f"worst loop stall {run.loop_lag.max_ms:.1f}ms"
        # The SAME ceiling as the diagnostics phase's own gate: a stacked gate must never be
        # stricter than the phase-only gate it contains, and this phase inherits that GC
        # residual unchanged.
        assert run.worst_auto_oldgen_pause_ms < 90, (
            f"worst interpreter pause {run.worst_auto_oldgen_pause_ms:.1f}ms"
        )
        # THE SWITCH WINDOW IS NOT ASSERTED HERE. Every form of it depended on measuring an
        # unfixed_timers() baseline, and that baseline deliberately reinstates 605 leaked 15Hz
        # timers which keep the screen permanently dirty — so _quiesce can never observe three
        # consecutive quiet polls and times out. The instability was in the knowingly-broken
        # comparison run, not in the switch path, and an absolute bound was separately shown to
        # sit inside a +/-200ms noise band. Removed on the user's instruction; measured
        # readings are recorded in the phase task for whoever revisits it: 308.2ms fixed
        # versus 732.9ms unfixed, settle 683.8ms versus 1201.2ms.

    async def test_pump_stays_responsive_below_render_throughput(self) -> None:
        """Below render throughput, keystrokes must NOT queue. Absolute, not relative.

        This is the question PumpLatencyProbe was introduced to answer, and it closes the gap
        the relative gate above leaves open: a regression that doubled per-event pump cost
        would pass a purely relative comparison, because both sides scale together.

        Drives ~30 events/sec — comfortably under the measured ~86 events/sec render
        throughput — for 4 seconds per mix, on a tree already built to 21-agent scale, so the
        reading reflects per-event cost rather than backlog growth. Both mixes are measured on
        ONE tree: building a second 21-agent tree cost ~4 minutes and enough memory pressure
        to destabilise the other full-fixture gates sharing the process. Every perf_replay
        default, churn_seconds included, is left untouched.

        GATES p50 ONLY, AT 15ms, AND THE MAX IS DELIBERATELY NOT ASSERTED. Do not restore a
        max assertion here; the reasoning is below rather than absent.

        WHY p50 AND WHY 15ms: measured p50 is 2.3-2.9ms across runs and across both event
        mixes, so 15ms leaves ~5x margin for host variation while still failing a 5x per-event
        regression. p50 is a median and therefore far less load-sensitive than the max. A
        looser bound would defeat the purpose of this gate, which exists to catch exactly the
        regression the relative gate cannot: one that doubles per-event cost, where both sides
        of a relative comparison scale together.

        WHY NOT THE MAX: pump max is bounded BELOW by the cost of the single most expensive
        SYNCHRONOUS render, at ANY arrival rate. That is structurally different from the
        saturation problem the relative gate above documents, where lowering the event rate
        would help — here nothing about the rate can lower the floor. Two contributors, and
        neither is what this metric exists to measure:
          - the LAYOUT MOUNT of one large DiffView. MEASURED: a single 42.5KB diff — the
            fixture's maximum — costs 206-225ms to mount and lay out, already above the 150ms
            bound on its own, with the tree-sitter highlight excluded because this phase moved
            it off-thread. That per-event floor is independent evidence for the deferred
            widgets-per-event reduction, alongside the ~86 events/sec sustained throughput
            ceiling: neither can be fixed by scheduling.
          - the GC freeze worker's collects over a freshly built 24k-widget tree.
        Measured max is 41-407ms diff-free and 244-503ms with diffs, against a 150ms bound.
        Three stabilisation attempts failed: a 3-second settle, a per-test unfreeze/collect
        (which also made this module 8 minutes slower and destabilised neighbouring gates),
        and module-level GC isolation. Any stable absolute max would have to sit near 600ms,
        which would pass even if per-event cost doubled. A flaky or meaningless gate is worse
        than an absent one, because it trains people to re-run and so silently disables
        itself.

        Silent failure this catches: with the pump saturated only under overload, no other
        gate here can distinguish "fast enough per event" from "slowly falling behind" — a
        regression doubling per-event pump cost shows up in p50 and nowhere else.
        """
        # Other gates in this module leave two 21-agent trees in the permanent generation,
        # which the production freeze worker never reclaims by design. That residue raises
        # per-event cost here — measured p50 16.2ms when this test runs straight after the
        # stacked gate, versus 2.3-2.9ms alone — and it is not part of what this gate measures,
        # which is the cost of events against THIS test's own tree.
        gc.unfreeze()
        gc.collect()

        journal = synthetic_journal(agents=_AGENTS)
        panel_ids = synthetic_agent_ids(_AGENTS)
        diff_free = [
            event for event in journal if not (isinstance(event, ToolCallUpdated) and event.diffs)
        ]
        app = SynthApp(
            _make_broker(),
            SessionConfig(project="perf"),
            initial_agent=AgentConfig(agent_id=panel_ids[0], harness="kiro"),
        )
        readings: dict[str, Any] = {}

        with patch("synth_acp.ui.app.embedding_available", return_value=False):
            async with app.run_test(headless=True, size=(120, 40)):
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

                # Build the tree to scale first, UNMEASURED: this deliberately runs the
                # fixture at its saturating rate, which is exactly what must not be inside a
                # measured window.
                for event in journal:
                    app.post_message(BrokerEventMessage(event))
                    await asyncio.sleep(0)
                # Quiesce FIRST: _quiesce waits on the App message queue, so draining before
                # it can return while events that will create more diff work are still
                # queued, leaving the fixture neither built nor settled.
                await harness._quiesce(app)
                await harness.drain_pending_diffs(app)
                await harness._quiesce(app)
                # Let the production GC freeze worker fold the newly built graph into the
                # permanent generation, so the readings reflect steady state, not startup.
                await asyncio.sleep(3.0)

                for mix, driven in (("without-diffs", diff_free), ("with-diffs", journal)):
                    probe = PumpLatencyProbe(app, interval=0.05)
                    probe_task = asyncio.create_task(probe.run())
                    try:
                        interval = 1 / 30
                        deadline = time.perf_counter() + 4.0
                        index = 0
                        while time.perf_counter() < deadline:
                            app.post_message(BrokerEventMessage(driven[index % len(driven)]))
                            index += 1
                            await asyncio.sleep(interval)
                        readings[mix] = probe.stats()
                    finally:
                        probe_task.cancel()
                        await probe_task
                    # Settle FULLY before the next window and before teardown: quiesce drains
                    # the App queue, the drain then clears the diff work those events created,
                    # and the final quiesce lays out what it released. Leaving events queued
                    # lets teardown shut the default executor underneath a pending
                    # Markdown.update.
                    await harness._quiesce(app)
                    await harness.drain_pending_diffs(app)
                    await harness._quiesce(app)

        for mix, stats in readings.items():
            assert stats.samples > 20, f"only {stats.samples} probe samples in {mix}"
            assert stats.p50_ms < 15, f"pump p50 {stats.p50_ms:.1f}ms in the {mix} mix"

    async def test_gradient_bars_are_bounded_per_agent(self) -> None:
        """Only the two per-agent ActivityBars may hold a gradient.

        Silent failure: hiding the inactive ExpandableSection bar instead of REMOVING it leaves
        all three widgets and their 15Hz timer alive while every functional activity assertion
        stays green. This counts the live tree at the snapshot instant, so it needs no baseline
        run — the relative total_widgets comparison it replaces required an unfixed_timers()
        baseline whose leaked timers made it unmeasurable.
        """
        _, gradient_bars = await _cached_run(fixed=True)

        # Strictly positive first, so a wrapper that never ran fails loudly here instead of
        # passing vacuously on an empty recording.
        assert gradient_bars > 0
        assert gradient_bars <= 2 * _AGENTS, (
            f"{gradient_bars} GradientBars for {_AGENTS} agents, expected at most two each "
            "(one agent tile, one input bar)"
        )

    async def test_widget_count_is_stable_across_replays(self) -> None:
        """Progressive diff rendering must not make the counts race the worker.

        Silent failure this catches is specifically FLAKINESS: without draining diff work
        before the snapshot, total_widgets varies with worker timing, so the gate passes
        locally and fails in review — and the diagnostics phase's own total_widgets floor
        becomes timing-dependent too.

        A small fixture, so three replays stay affordable.
        """
        journal = synthetic_journal(agents=3)
        agent_ids = synthetic_agent_ids(3)
        counts = [
            (
                await perf_replay(journal, agent_ids=agent_ids, churn_seconds=2.0, repetitions=1)
            ).widgets.total_widgets
            for _ in range(3)
        ]

        spread = (max(counts) - min(counts)) / max(counts)
        assert spread < 0.02, f"total_widgets varied by {spread:.1%} across replays: {counts}"


class TestDrainPendingDiffs:
    """The drain helper must not become a new way to hang or to under-report."""

    async def test_returns_promptly_when_nothing_is_pending(self) -> None:
        """Silent failure: a helper that waits on the standing diffs worker never returns.

        That worker loops on an Event for the feed's whole life, so awaiting it hangs the
        suite exactly as a bare wait_for_complete() does against the permanent GC worker.
        """
        app = SynthApp(_make_broker(), SessionConfig(project="test"))
        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("a1")

            await harness.drain_pending_diffs(app, timeout=2.0)

    async def test_raises_when_work_is_left_outstanding(self) -> None:
        """Silent failure: returning on timeout restores the undercount it exists to stop.

        A helper that gives up quietly is worse than one that hangs, because the gate then
        measures a half-built tree and still reads green.
        """
        from synth_acp.models.events import ToolCallDiff
        from synth_acp.ui.widgets.tool_call import DiffState, ToolCallBlock

        app = SynthApp(_make_broker(), SessionConfig(project="test"))
        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("a1")
            feed = app._panels["a1"]
            block = ToolCallBlock("tc1", "Edit", "edit", "completed")
            turn = await feed._start_turn()
            assert turn is not None
            await turn.mount(block)
            block.schedule_diffs([ToolCallDiff(path="a.py", old_text="a", new_text="b")])
            claim = block.claim_next_diff()
            assert claim is not None
            assert claim[1].state is DiffState.RENDERING

            with pytest.raises(AssertionError, match="still pending"):
                await harness.drain_pending_diffs(app, timeout=0.2)


class TestUserNonNegotiables:
    """Two things the user stated explicitly and this phase could have broken."""

    async def test_full_scrollback_is_preserved(self) -> None:
        """The oldest content must remain reachable and rendered.

        Silent failure: pruning or lazy gating that truncates the oldest history is
        invisible until the user scrolls up looking for something that is no longer there.
        """
        app = SynthApp(_make_broker(), SessionConfig(project="test"))
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await app.select_agent("a1")
            feed = app._panels["a1"]

            for index in range(41):
                await feed.add_prompt(f"prompt {index}")
                event = MessageChunkReceived(agent_id="a1", chunk=f"turn {index} body")
                feed.record_event(event)
                await feed.add_chunk(event.chunk)
                feed.record_event(TurnComplete(agent_id="a1", stop_reason="end_turn"))
                await feed.finalize_current_message()

            assert feed._mounted_start_idx > 0, "the fixture did not trigger pruning"

            # Scroll to the very top, restoring every pruned batch.
            while feed._mounted_start_idx > 0:
                await feed._restore_turns()
                await pilot.pause()

            assert feed._mounted_start_idx == 0
            assert feed._scroll is not None
            turns = [c for c in feed._scroll.children if isinstance(c, TurnContainer)]

            # CONTENT in ORDER, not counts. Asserting counts is exactly what let the
            # pre-existing defect survive: _restore_turns removed each replayed turn before
            # re-mounting it, which prunes the subtree, so every restored turn came back
            # EMPTY while the turn count stayed perfect.
            bodies = [
                "".join(message._chunks) for turn in turns for message in turn.query(AgentMessage)
            ]
            assert bodies == [f"turn {index} body" for index in range(41)]
            # Rendered, not merely mounted, and not gated behind an expand action.
            oldest_message = turns[0].query(AgentMessage).first()
            assert oldest_message.display is True

    async def test_hidden_feed_is_current_on_switch(self) -> None:
        """A hidden feed keeps rendering, so switching shows no rebuild.

        Silent failure: a feed that quietly stops rendering while hidden and rebuilds on
        switch reads as a fast switch in every timing metric while showing stale content.
        """
        app = SynthApp(_make_broker(), SessionConfig(project="test"))
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await app.select_agent("a1")
            hidden = ConversationFeed(
                "a2", "a2", "test", harness="kiro", cwd=".", id=f"feed-{css_id('a2')}"
            )
            switcher = app.query_one("#right", ContentSwitcher)
            await switcher.add_content(hidden, set_current=False)
            app._panels["a2"] = hidden

            for index in range(3):
                await hidden.add_prompt(f"prompt {index}")
                event = MessageChunkReceived(agent_id="a2", chunk=f"hidden body {index}")
                hidden.record_event(event)
                await hidden.add_chunk(event.chunk)
                hidden.record_event(TurnComplete(agent_id="a2", stop_reason="end_turn"))
                await hidden.finalize_current_message()
            await pilot.pause()

            assert hidden.display is False
            before = [id(m) for m in hidden.query(AgentMessage)]
            assert len(before) == 3

            switcher.current = f"feed-{css_id('a2')}"
            # No pause first: the content must already be present on switch.
            after = [id(m) for m in hidden.query(AgentMessage)]

            assert after == before, "the feed rebuilt its content on switch"
            await pilot.pause()
            texts = ["".join(m._chunks) for m in hidden.query(AgentMessage)]
            assert texts == ["hidden body 0", "hidden body 1", "hidden body 2"]


def _make_broker() -> MagicMock:
    """A broker stand-in exposing only what SynthApp touches."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    broker._initial_agent = AgentConfig(agent_id="a1", harness="kiro")
    broker.get_usage = MagicMock(return_value=None)

    async def _events():
        return
        yield  # pragma: no cover

    broker.events = _events
    return broker


@pytest.mark.perf
def test_prewarm_makes_the_first_diff_mount_cheaper() -> None:
    """AC12a: the pre-warm measurably cheapens the first REAL diff mount.

    Measured through a real mount via the feed (``add_tool_call`` plus the diff executor),
    not a bare ``DiffView.prepare()``, because AC12a is about the mount the user waits on.

    SUBPROCESSES ARE REQUIRED. The Pygments lexer index and lexer modules are cached
    process-globally, so any earlier highlight in the pytest session warms them and an
    in-process comparison reads a millisecond both ways and passes vacuously. The baseline
    subprocess also DISABLES the app's own pre-warm worker: without that it warms the cache
    while the baseline is being measured, which is a mistake that halved the apparent saving
    the first time this was run.

    The primary assertion is RELATIVE. An absolute saving floor was tried first at 150ms and
    flaked: the warm reading is bimodal on this host, so the saving spans 111-186ms while the
    ratio stays under 0.45. Neither bound may imply the pre-warm solves the multi-second
    first-selection cost, which tail-first windowing owns.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}

    def _probe(mode: str) -> tuple[float, str]:
        result = subprocess.run(
            [sys.executable, "-m", "tests._prewarm_probe", mode],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=env,
            check=True,
        )
        elapsed, state = result.stdout.strip().splitlines()[-1].split()
        return float(elapsed), state

    cold_ms, cold_state = _probe("cold")
    warm_ms, warm_state = _probe("warm")

    print(f"\nfirst diff mount: cold {cold_ms:.1f}ms -> warm {warm_ms:.1f}ms")

    # The diff must actually render in both, or the timing compares two failures.
    assert cold_state == "rendered"
    assert warm_state == "rendered"

    # RELATIVE IS PRIMARY, and this gate was RECALIBRATED after the absolute floor flaked.
    #
    # Measured over 10 pairs on this host: cold 200.9-256.9ms, warm 44.4-98.0ms, saving
    # 111.4-186.1ms, ratio 0.197-0.446. The warm reading is BIMODAL (~45ms or ~85-98ms),
    # so the absolute SAVING spans ~75ms run to run and a 150ms floor sat inside that noise
    # — it passed six times and then failed at 129.1ms. The ratio's upper bound is far more
    # stable, which is the same lesson every calibration error in this feature taught: a
    # relative bound self-calibrates to host speed, an absolute one does not.
    #
    # Ratio bound 0.60 against a worst observed 0.446. The saving floor is kept only to
    # reject a noise-level improvement and is set at 75ms, under the worst observed
    # 111.4ms. Neither figure implies the pre-warm solves the multi-second first-selection
    # cost, which tail-first windowing owns; per AC12a the expected saving is a few hundred
    # milliseconds once per process.
    assert warm_ms <= 0.60 * cold_ms, (
        f"warm {warm_ms:.1f}ms is not materially below cold {cold_ms:.1f}ms "
        f"(ratio {warm_ms / cold_ms:.3f})"
    )
    assert cold_ms - warm_ms >= 75, (
        f"pre-warm saved only {cold_ms - warm_ms:.1f}ms (cold {cold_ms:.1f}, warm {warm_ms:.1f})"
    )


# Fixture-calibrated absolute bound for AC1b, derived by an EXACT formula so a reviewer can
# reproduce the literal rather than judge whether a margin is "generous":
#
#     bound = ceil((3 * calibration_ms) / 100) * 100
#
# calibration_ms is the highest windowed visible_ms observed while calibrating on this
# fixture: 288.5ms (readings were 261.6, 268.4, 288.5). So
# ceil((3 * 288.5) / 100) * 100 = ceil(8.655) * 100 = 900ms.
# The gate compares a SINGLE windowed reading against this bound, not a median, because the
# bound already carries a 3x margin over the worst calibration reading.
#
# NOT COMPARABLE to the recorded real-journal figures of 231ms windowed / 7861ms unwindowed.
# This fixture is deliberately lighter than the real session — it matches type counts, turn
# shape and payload SIZES, but not real payload text or tool-call nesting — and it measures
# ~4800-5300ms unwindowed against the real 7861ms. Only the RELATIVE gate below is comparable
# across fixtures, which is why that one is primary.
_FIRST_PAINT_CALIBRATION_MS = 288.5
_FIRST_PAINT_ABSOLUTE_MS = math.ceil((3 * _FIRST_PAINT_CALIBRATION_MS) / 100) * 100

_FIRST_PAINT_CACHE: dict[str, tuple[Any, Any]] = {}


async def _first_paint_pair() -> tuple[Any, Any]:
    """Measure windowed and unwindowed first paint on the same fixture, in one session.

    A discarded warm-up runs first: the Pygments lexer cache and Textual's CSS parsing are
    process-global, so whichever configuration ran first would otherwise absorb their one-time
    cost and the comparison would flatter it.
    """
    if "pair" not in _FIRST_PAINT_CACHE:
        journal = harness.first_paint_journal()
        await harness.first_selection_paint(journal, agent_id=harness._FP_AGENT, windowed=True)
        windowed = await harness.first_selection_paint(journal, agent_id=harness._FP_AGENT, windowed=True)
        unwindowed = await harness.first_selection_paint(journal, agent_id=harness._FP_AGENT, windowed=False)
        _FIRST_PAINT_CACHE["pair"] = (windowed, unwindowed)
    return _FIRST_PAINT_CACHE["pair"]


@pytest.mark.perf
async def test_first_paint_windowed_versus_unwindowed() -> None:
    """AC1 primary (relative) and AC1b (fixture-calibrated absolute).

    The relative form is primary because it is robust to fixture weight BY CONSTRUCTION: both
    readings come from the same journal in the same run, so a lighter or heavier fixture moves
    them together. Every absolute threshold in the preceding feature was mis-calibrated at
    least once by inheriting a real-session number onto a different configuration.
    """
    windowed, unwindowed = await _first_paint_pair()
    ratio = windowed.visible_ms / unwindowed.visible_ms

    print(
        f"\nfirst paint: windowed {windowed.visible_ms:.1f}ms "
        f"vs unwindowed {unwindowed.visible_ms:.1f}ms = {ratio * 100:.1f}%"
        f"\n  windowed: {windowed.mounted_turns} turns mounted, "
        f"{windowed.turn_batches - windowed.mounted_turns} batches deferred, "
        f"{windowed.content_height} rows vs {windowed.viewport_height}-row viewport"
    )

    assert ratio <= 0.15, f"windowed first paint is {ratio * 100:.1f}% of unwindowed"
    assert windowed.visible_ms < _FIRST_PAINT_ABSOLUTE_MS

    # The recorded stream is complete regardless of what was mounted.
    assert windowed.turn_batches == 52
    # Deferral proven from the two declared fields rather than an extra one: fewer turns
    # mounted than batches recorded is exactly what windowing means.
    assert windowed.mounted_turns < windowed.turn_batches, (
        "nothing was deferred, so nothing was windowed"
    )


@pytest.mark.perf
async def test_windowed_first_paint_overfills_the_viewport() -> None:
    """AC2a: bottom-anchoring must be a real position, not a degenerate one.

    A window shorter than the viewport would make "lands at the bottom" vacuous and would
    strand the deferred history behind a scroll that cannot happen.
    """
    windowed, _ = await _first_paint_pair()
    assert windowed.content_height > windowed.viewport_height


@pytest.mark.perf
async def test_windowed_first_paint_shrinks_the_diff_trail() -> None:
    """AC17: diffs for unmounted turns are never scheduled, so the trail shrinks.

    Silent failure: windowing mounts fewer widgets but still schedules every diff, so the feed
    keeps churning after paint and the user still waits. A ratio near 1.0 means diffs are being
    scheduled for turns that were never mounted — a defect in the windowing, not a threshold to
    relax.
    """
    windowed, unwindowed = await _first_paint_pair()
    ratio = windowed.diff_trail_ms / max(unwindowed.diff_trail_ms, 0.001)

    print(
        f"\ndiff trail: windowed {windowed.diff_trail_ms:.1f}ms "
        f"vs unwindowed {unwindowed.diff_trail_ms:.1f}ms = {ratio * 100:.1f}%"
    )
    assert ratio <= 0.40
