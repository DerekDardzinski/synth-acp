"""Tests for the performance replay harness in tests/conftest.py.

``tests/conftest.py`` has no source mirror, so its tests live here rather than being
wedged into an unrelated module's mirror file.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import gc
import inspect
import itertools
import pathlib
import sqlite3
import statistics
from collections import Counter
from typing import Any
from unittest.mock import patch

import pytest
from textual.widgets import ContentSwitcher

from synth_acp.diagnostics import LagStats, ObjectCounters, WidgetCounters
from synth_acp.models.agent import css_id
from synth_acp.models.events import (
    AgentThoughtReceived,
    HookFired,
    MessageChunkReceived,
    ToolCallUpdated,
    TurnComplete,
    UserPromptSubmitted,
)
from synth_acp.runtime_gc import GCFreezeManager
from synth_acp.ui.widgets import conversation as conversation_module
from synth_acp.ui.widgets.conversation import (
    FIRST_PAINT_EVENT_BUDGET,
    FIRST_PAINT_TURNS,
    first_paint_window,
)
from tests import conftest as harness
from tests.conftest import (
    PerfRun,
    perf_replay,
    raw_input_bytes,
    synthetic_agent_ids,
    synthetic_journal,
    unfixed_timers,
)

_CONFTEST_SOURCE = pathlib.Path(inspect.getfile(harness)).read_text()

_MAX_DIFF_BYTES = int(42.5 * 1024)
_MEAN_RAW_INPUT_BYTES = 15 * 1024

# The two full-fixture replays are expensive (~30 s per repetition), so the three
# gates share them instead of running one each. PerfRun is a frozen pydantic model
# with no event-loop affinity, so caching across function-scoped test loops is safe.
_CACHE: dict[str, PerfRun] = {}


def _no_collect(_self: GCFreezeManager) -> None:
    """Stand-in for collect_and_freeze that disables the production fix."""


async def _cached_run(*, freeze_enabled: bool) -> PerfRun:
    """Return the cached 21-agent replay for one freeze configuration."""
    key = "active" if freeze_enabled else "unfrozen"
    if key not in _CACHE:
        journal = synthetic_journal(agents=21)
        agent_ids = synthetic_agent_ids(21)
        if freeze_enabled:
            _CACHE[key] = await perf_replay(journal, agent_ids=agent_ids)
        else:
            # AUTHORISED CROSS-PHASE CORRECTION. This baseline exists to prove the harness
            # can SEE the problems being fixed, so it must revert every fix it asserts a
            # floor for — not only the GC freeze. The UI phase's timer fix drives
            # hidden_widgets_with_timers to 0 by design, so without unfixed_timers() the
            # `> 200` floor below became unsatisfiable.
            with unfixed_timers(), patch.object(GCFreezeManager, "collect_and_freeze", _no_collect):
                _CACHE[key] = await perf_replay(journal, agent_ids=agent_ids)
    return _CACHE[key]


@pytest.fixture(scope="module", autouse=True)
def _release_frozen_graph():
    """Undo the permanent generation the gates leave behind.

    The full-fixture replays freeze roughly 4M objects. Leaving them in the permanent
    generation would change the GC behaviour every later test in the process observes.
    """
    yield
    gc.unfreeze()
    gc.collect()


# ── synthetic_journal ──


class TestSyntheticJournal:
    def test_is_deterministic_with_exact_proportions(self) -> None:
        """A nondeterministic fixture makes every A/B comparison meaningless."""
        first = synthetic_journal()
        second = synthetic_journal()

        assert first == second
        assert len(first) == 2560
        counts = Counter(type(event).__name__ for event in first)
        assert counts["ToolCallUpdated"] == 1109
        assert counts["MessageChunkReceived"] == 644
        assert counts["AgentThoughtReceived"] == 619
        assert counts["UserPromptSubmitted"] == 86
        assert counts["TurnComplete"] == 81
        assert counts["HookFired"] == 21

    def test_reads_no_database_and_no_synth_file(self) -> None:
        """AC17 for the generator: it must not fall back to the user's database."""
        with _no_synth_io():
            journal = synthetic_journal()

        assert len(journal) == 2560

    def test_thought_runs_are_length_one(self) -> None:
        """Consecutive thoughts coalesce into one widget, drifting every widget gate."""
        journal = synthetic_journal()
        per_agent: dict[str, list[object]] = {}
        for event in journal:
            per_agent.setdefault(event.agent_id, []).append(event)

        adjacent = [
            agent_id
            for agent_id, events in per_agent.items()
            for first, second in itertools.pairwise(events)
            if isinstance(first, AgentThoughtReceived) and isinstance(second, AgentThoughtReceived)
        ]

        assert adjacent == []

    def test_payload_fidelity(self) -> None:
        """Tiny payloads would pass a count-and-cap check while gutting the benchmark.

        The diff-highlight and GC costs this plan measures are payload-size driven.
        """
        tool_calls = [e for e in synthetic_journal() if isinstance(e, ToolCallUpdated)]
        diff_sizes = [len(d.new_text) for e in tool_calls for d in e.diffs]

        assert sum(1 for e in tool_calls if e.diffs) == 152
        assert max(diff_sizes) == _MAX_DIFF_BYTES
        mean_raw = statistics.mean(raw_input_bytes(e) for e in tool_calls)
        assert mean_raw == pytest.approx(_MEAN_RAW_INPUT_BYTES, rel=0.02)

    def test_scales_to_other_agent_counts(self) -> None:
        """The divmod turn distribution must stay exact for any agent count."""
        journal = synthetic_journal(agents=5)
        counts = Counter(type(event).__name__ for event in journal)

        assert counts["UserPromptSubmitted"] == 20
        assert counts["TurnComplete"] == 19
        assert counts["HookFired"] == 5

    def test_unterminated_turns_are_the_last_turn_of_the_first_five_agents(self) -> None:
        """WHICH turn lacks a TurnComplete is part of the design, not just how many.

        Splits each agent's own subsequence at UserPromptSubmitted boundaries and checks
        the POSITION of the gap. Silent failure: per-agent complete COUNTS are equally
        satisfied by omitting turn 0's TurnComplete instead of the last turn's, which
        would leave an early turn permanently open and every aggregate assertion green,
        so two fixtures claiming the same seed would render different structures.
        """
        journal = synthetic_journal(agents=21)
        per_agent: dict[str, list[object]] = {}
        for event in journal:
            per_agent.setdefault(event.agent_id, []).append(event)

        # For each agent, which turn indices lack a TurnComplete.
        gaps: dict[str, list[int]] = {}
        turn_counts: dict[str, int] = {}
        for agent_id, events in per_agent.items():
            completed: list[bool] = []
            for event in events:
                if isinstance(event, UserPromptSubmitted):
                    completed.append(False)
                elif isinstance(event, TurnComplete) and completed:
                    completed[-1] = True
            turn_counts[agent_id] = len(completed)
            gaps[agent_id] = [i for i, done in enumerate(completed) if not done]

        # divmod(86, 21) == (4, 2): agents 00 and 01 get 5 turns, the rest 4.
        assert turn_counts["agent-00"] == 5
        assert turn_counts["agent-01"] == 5
        assert turn_counts["agent-02"] == 4
        assert turn_counts["agent-20"] == 4

        # 86 - 81 == 5 unterminated, each the LAST turn of agents 00..04.
        for index in range(5):
            agent_id = f"agent-{index:02d}"
            assert gaps[agent_id] == [turn_counts[agent_id] - 1], agent_id
        for index in range(5, 21):
            agent_id = f"agent-{index:02d}"
            assert gaps[agent_id] == [], agent_id

    def test_agent_ids_match_the_generator(self) -> None:
        journal = synthetic_journal(agents=3)

        # Literal, not recomputed through synthetic_agent_ids: comparing the generator
        # against itself would pass even if the naming format changed.
        assert sorted({event.agent_id for event in journal}) == [
            "agent-00",
            "agent-01",
            "agent-02",
        ]
        assert synthetic_agent_ids(3) == ["agent-00", "agent-01", "agent-02"]

    @pytest.mark.parametrize(
        ("agents", "seed"),
        [(1, 1), (1, 20), (3, 7), (5, 999)],
    )
    def test_payload_invariants_hold_for_any_agents_and_seed(self, agents: int, seed: int) -> None:
        """The two advertised payload properties must hold BY CONSTRUCTION.

        Silent failure this catches: when sizes were sampled from the RNG, both held for
        the large default fixture but not otherwise — agents=1 drifted to a 14.4 KB mean
        and some seeds produced no diff above 8 KB, so an alternate deterministic fixture
        was materially lighter than the benchmark it claimed to be, and every gate
        asserted on it read healthy.
        """
        tool_calls = [
            e for e in synthetic_journal(agents=agents, seed=seed) if isinstance(e, ToolCallUpdated)
        ]
        diff_sizes = [len(d.new_text) for e in tool_calls for d in e.diffs]

        assert statistics.mean(raw_input_bytes(e) for e in tool_calls) == pytest.approx(
            _MEAN_RAW_INPUT_BYTES, rel=0.02
        )
        assert max(diff_sizes) == _MAX_DIFF_BYTES


# ── perf_replay structure ──


class TestPerfReplayStructure:
    def test_contains_no_pilot_pause(self) -> None:
        """pilot.pause() is O(total widgets) BY CONSTRUCTION.

        Pilot._wait_for_screen posts one Callback per widget in the whole app and
        awaits all of them, so using it inside a timed window silently inflates every
        measurement. This exact mistake already invalidated a full round of
        measurements on this feature.

        Asserted over the AST, not the raw text: the contract-mandated ``perf_replay``
        docstring itself says "MUST NOT call pilot.pause()", so a substring check
        would be self-defeating.
        """
        offenders = [
            node.lineno
            for node in ast.walk(ast.parse(_CONFTEST_SOURCE))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pause"
        ]

        assert offenders == []
        # Proves the walk resolves method calls at all rather than passing vacuously.
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "post_message"
            for node in ast.walk(ast.parse(_CONFTEST_SOURCE))
        )

    async def test_medians_timings_and_takes_counts_from_final_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every timing is a median; every count comes from the final run.

        Values are chosen so median and final DIFFER for every field asserted —
        otherwise the test could not tell the two reduction rules apart.
        """
        # Independent of the developer's SYNTH_PERF_REPS setting.
        monkeypatch.delenv("SYNTH_PERF_REPS", raising=False)
        # Order matters: the MEDIAN scale is 2.0 and the FINAL scale is 1.0, so the two
        # reduction rules produce different values for every field asserted below. With
        # the median run also last, returning the final run for every timing — or
        # medianing every count — would satisfy all of these literals.
        runs = [_fake_run(2.0), _fake_run(3.0), _fake_run(1.0)]
        calls = {"n": 0}

        async def _fake_isolated(*args: object, **kwargs: object) -> PerfRun:
            run = runs[calls["n"]]
            calls["n"] += 1
            return run

        with patch.object(harness, "_run_isolated", _fake_isolated):
            result = await perf_replay([], agent_ids=["agent-00", "agent-01"], repetitions=3)

        # Medians: the middle of scales 2.0 / 3.0 / 1.0 is the 2.0 run.
        assert result.wall_ms == 2000.0
        assert result.worst_auto_oldgen_pause_ms == 20.0
        assert result.manual_gen2_ms == 40.0
        assert result.first_switch_max_stall_ms == 60.0
        assert result.first_switch_settle_ms == 80.0
        for stats in (result.loop_lag, result.pump_latency):
            assert stats.p50_ms == 2.0
            assert stats.p95_ms == 4.0
            assert stats.p99_ms == 6.0
            assert stats.max_ms == 8.0

        # Counts: the FINAL run is the 1.0 run, so every one of these differs from the
        # median-scaled value above.
        assert result.widgets.total_widgets == 1000
        assert result.widgets.widgets_with_timers == 100
        assert result.widgets.hidden_widgets_with_timers == 50
        assert result.objects.gc_objects == 10000
        assert result.objects.unfrozen_gc_objects == 5000
        assert result.objects.frozen_gc_objects == 2000
        assert result.loop_lag.samples == 10
        assert result.pump_latency.samples == 10

    async def test_quiesce_timeout_fails_loudly(self) -> None:
        """A quiesce that returns on timeout closes timed windows over dirty work.

        Every stall in the remaining layout/repaint would then go unattributed, so
        gates would read healthy on an app that never settled.
        """
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        class _App(App):
            def compose(self) -> ComposeResult:
                yield Static("x")

        app = _App()
        async with app.run_test(headless=True):
            app.screen._repaint_required = True

            async def _stay_dirty() -> None:
                while True:
                    app.screen._repaint_required = True
                    await asyncio.sleep(0.001)

            keeper = asyncio.create_task(_stay_dirty())
            try:
                with pytest.raises(AssertionError):
                    await harness._quiesce(app, timeout=0.2)
            finally:
                keeper.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await keeper

    async def test_rejects_too_few_agents_for_a_switch(self) -> None:
        """The switch window needs a populated feed that was never displayed.

        Silent failure with one agent: `_switch_target` returned the ALREADY-DISPLAYED
        feed, so both switch fields measured a re-switch while still looking non-zero.
        With none it raised an undeclared IndexError from indexing panel_ids[0].
        """
        with pytest.raises(AssertionError, match="at least 2 distinct agents"):
            await perf_replay([], agent_ids=[], churn_seconds=0.0, repetitions=1)

        with pytest.raises(AssertionError, match="at least 2 distinct agents"):
            await perf_replay(
                synthetic_journal(agents=1),
                agent_ids=synthetic_agent_ids(1),
                churn_seconds=0.0,
                repetitions=1,
            )

    async def test_switch_targets_exactly_one_populated_never_displayed_feed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observe the switch MECHANICS, not just that two numbers are positive.

        Silent failure: assigning the already-current feed, or switching twice, still
        yields two positive values that differ from the overall maximum. This inspects
        the target at selection time — exactly one selection per replay, to a feed that
        is NOT displayed and DOES hold content — and pins the reset-before-switch order,
        since resetting after the assignment would fold churn into the switch window.
        """
        observed: list[tuple[str, bool, int, str | None]] = []
        real_switch_target = harness._switch_target

        def _spy(app: object, panel_ids: object) -> str:
            target = real_switch_target(app, panel_ids)  # type: ignore[arg-type]
            feed = app._panels[target]  # type: ignore[attr-defined]
            switcher = app.query_one("#right", ContentSwitcher)  # type: ignore[attr-defined]
            observed.append(
                (target, bool(feed.display), len(feed._tool_call_blocks), switcher.current)
            )
            return target

        monkeypatch.setattr(harness, "_switch_target", _spy)

        await perf_replay(
            synthetic_journal(agents=3),
            agent_ids=synthetic_agent_ids(3),
            churn_seconds=1.0,
            repetitions=1,
        )

        assert len(observed) == 1, "exactly ONE switch per replay"
        target, displayed, blocks, current = observed[0]
        assert displayed is False, "target feed must never have been displayed"
        assert current != f"feed-{css_id(target)}", "target must not already be the current feed"
        assert blocks > 0, "target feed must be populated"

        # Structural: the sampler is reset AFTER the target is chosen and BEFORE the
        # ContentSwitcher assignment, and the window closes on a dirty-flag poll.
        source = _CONFTEST_SOURCE
        pick = source.index("switch_target = _switch_target(")
        reset = source.index("sampler.reset()", pick)
        assign = source.index("switcher.current = ", reset)
        settle = source.index("await _quiesce(app)", assign)
        assert pick < reset < assign < settle

    async def test_harness_opens_no_database_and_no_synth_file(self) -> None:
        """AC17 verified BEHAVIOURALLY, not by grepping for a path.

        A source grep is unusable here: the contract-mandated synthetic_journal
        docstring itself contains the literal ``~/.synth/synth.db``, and the test
        would have to contain its own search literals. The criterion forbids OPENING
        the database, not naming it.
        """
        journal = synthetic_journal(agents=2)
        with _no_synth_io():
            run = await perf_replay(
                journal,
                agent_ids=synthetic_agent_ids(2),
                churn_seconds=1.0,
                repetitions=1,
            )

        assert run.widgets.total_widgets > 0


class TestWorkerWaitHelper:
    """AC25/AC26: the permanent 'gc' worker breaks every bare wait_for_complete()."""

    async def test_returns_promptly_with_no_transient_workers(self) -> None:
        """The empty case must be SKIPPED, not passed through.

        Textual resolves the argument as ``workers or self``, so handing it an empty
        list silently waits on every worker — including the permanent GC freeze worker,
        which never completes. Silent failure: the suite hangs forever instead of
        failing, which is how this was originally discovered (a 40-minute timeout).
        """
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        class _App(App):
            def compose(self) -> ComposeResult:
                yield Static("x")

        app = _App()
        async with app.run_test(headless=True):

            async def _forever() -> None:
                await asyncio.sleep(3600)

            app.run_worker(_forever(), group="gc", name="gc-freeze", exit_on_error=False)
            assert [w for w in app.workers if w.group not in {"gc", "diag"}] == []

            await asyncio.wait_for(harness.wait_for_transient_workers(app), timeout=2.0)

    async def test_waits_for_a_transient_worker(self) -> None:
        """It must still actually wait — a no-op helper would hide real races."""
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        class _App(App):
            def compose(self) -> ComposeResult:
                yield Static("x")

        app = _App()
        done: list[str] = []
        async with app.run_test(headless=True):

            async def _work() -> None:
                await asyncio.sleep(0.05)
                done.append("finished")

            async def _forever() -> None:
                await asyncio.sleep(3600)

            app.run_worker(_forever(), group="gc", name="gc-freeze", exit_on_error=False)
            app.run_worker(_work(), group="unit", name="unit", exit_on_error=False)

            await asyncio.wait_for(harness.wait_for_transient_workers(app), timeout=5.0)

            assert done == ["finished"]

    def test_no_bare_wait_for_complete_remains_in_the_suite(self) -> None:
        """A bare call anywhere in the suite is a latent hang.

        Swept across the whole suite rather than only the two call sites that happened
        to surface, since the hang depends on whether a test exercises the permanent
        worker, not on where the call lives.
        """
        tests_root = pathlib.Path(inspect.getfile(harness)).parent
        offenders: list[str] = []
        for path in sorted(tests_root.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "wait_for_complete"
                    and not node.args
                ):
                    offenders.append(f"{path.relative_to(tests_root)}:{node.lineno}")

        assert offenders == []


class TestRepetitionOverride:
    """AC27: gates keep the noise-resistant default; local iteration can opt out."""

    def test_default_is_unchanged_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SYNTH_PERF_REPS", raising=False)

        assert harness._effective_repetitions(3) == 3

    def test_override_reduces_repetitions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNTH_PERF_REPS", "1")

        assert harness._effective_repetitions(3) == 1

    @pytest.mark.parametrize("value", ["", "bogus", "0", "-2"])
    def test_unusable_values_fall_back_to_the_caller(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo must not silently weaken a three-run gate to a single noisy run."""
        monkeypatch.setenv("SYNTH_PERF_REPS", value)

        assert harness._effective_repetitions(3) == 3

    async def test_perf_replay_honours_the_override_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """perf_replay must actually CONSULT the override, not just expose the resolver.

        Silent failure: a one-line regression iterating `range(repetitions)` directly
        leaves both resolver tests green while the override does nothing and every gate
        silently keeps paying three full replays — or, worse, a gate runs once.
        """
        calls = {"n": 0}

        async def _counting(*args: object, **kwargs: object) -> PerfRun:
            calls["n"] += 1
            return _fake_run(1.0)

        monkeypatch.setattr(harness, "_run_isolated", _counting)
        agents = ["agent-00", "agent-01"]

        monkeypatch.delenv("SYNTH_PERF_REPS", raising=False)
        await perf_replay([], agent_ids=agents, repetitions=3)
        assert calls["n"] == 3

        calls["n"] = 0
        monkeypatch.setenv("SYNTH_PERF_REPS", "1")
        await perf_replay([], agent_ids=agents, repetitions=3)
        assert calls["n"] == 1


# ── Full-fixture gates ──


@pytest.mark.perf
class TestFullFixtureGates:
    async def test_detects_the_real_problem(self) -> None:
        """A harness that cannot see the bug it measures makes every gate vacuous.

        AN INTENTIONALLY-INVOKED DIAGNOSTIC, NOT A ROUTINE GATE. It is the only test proving
        the harness can observe the defects at all, so removing it would make every other
        fixture gate meaningless — which is why it keeps `unfixed_timers()` while the relative
        pump, switch and widget-drop gates that also needed that baseline were removed.

        IT USES AN UNSTABLE CONFIGURATION BY DESIGN, and its intermittency is ACCEPTED. The
        baseline reinstates ~605 leaked 15Hz timers, so the screen is almost never
        repaint-quiet and `_quiesce` cannot reliably observe its consecutive quiet polls; the
        run then times out at 60s. That is a race against the leaked timers the baseline exists
        to demonstrate, not a defect in this test or in `_quiesce`. `_quiesce` failing LOUDLY on
        timeout is the required behaviour: returning quietly would let a gate close its window
        over unsettled work and under-report.

        Accepted because this test is `perf`-marked and therefore runs on demand rather than on
        every invocation. If it times out, re-run it; do not "stabilise" it by weakening
        `_quiesce`, which would break test_quiesce_timeout_fails_loudly and silently weaken
        every other measurement.
        """
        run = await _cached_run(freeze_enabled=False)

        assert run.loop_lag.max_ms > 150
        # Strictly greater than 0 first, so a metric that silently reads 0.0 on some
        # interpreter fails loudly here instead of sailing past a "< 50ms" gate.
        assert run.worst_auto_oldgen_pause_ms > 0
        assert run.worst_auto_oldgen_pause_ms > 150
        assert run.widgets.hidden_widgets_with_timers > 200
        assert run.widgets.total_widgets > 15000

    async def test_gc_freeze_flattens_oldgen_pause(self) -> None:
        """AC12: relative primary, absolute as a loose sanity bound.

        The RELATIVE assertion is primary because it self-calibrates to host speed and
        fixture weight. An absolute ceiling carried over from a different workload is
        exactly what miscalibrated this gate before: 50 ms was measured on the real
        session, while this fixture is 16% heavier and runs saturated, so a correct
        implementation reads 36.8-73.3 ms.

        Both readings come from the same fixture in the same gate run — comparing across
        invocations would fold in host-load differences.
        """
        active = await _cached_run(freeze_enabled=True)
        unfrozen = await _cached_run(freeze_enabled=False)

        enabled_ms = active.worst_auto_oldgen_pause_ms
        disabled_ms = unfrozen.worst_auto_oldgen_pause_ms

        # 12a — relative, primary.
        assert enabled_ms <= disabled_ms / 3, (
            f"freeze worker did not flatten the old-gen pause: enabled={enabled_ms:.1f}ms, "
            f"disabled={disabled_ms:.1f}ms, ratio={enabled_ms / max(disabled_ms, 1e-9):.2f} "
            "(must be <= 0.33)"
        )
        # 12b — absolute, loose sanity bound calibrated against THIS fixture.
        assert enabled_ms < 90, f"enabled old-gen pause {enabled_ms:.1f}ms exceeded the 90ms bound"
        # 12c — the manual probe.
        assert active.manual_gen2_ms < 50

    async def test_freeze_worker_shrinks_unfrozen_object_count(self) -> None:
        active = await _cached_run(freeze_enabled=True)
        unfrozen = await _cached_run(freeze_enabled=False)

        assert active.objects.unfrozen_gc_objects < 0.25 * unfrozen.objects.unfrozen_gc_objects

    async def test_switch_window_is_bracketed_separately(self) -> None:
        """The switch window must be measured on its own, not aliased to the maximum.

        The measured references are 82.28 ms overall worst stall versus 305.6 ms
        first-switch stall, so loop_lag.max_ms is not a substitute.
        """
        run = await _cached_run(freeze_enabled=True)

        assert run.first_switch_max_stall_ms > 0
        assert run.first_switch_settle_ms > 0
        assert run.first_switch_max_stall_ms != run.loop_lag.max_ms


# ── helpers ──


def _no_synth_io() -> contextlib.ExitStack:
    """Return an entered context in which database or ~/.synth access raises."""
    real_open = builtins.open
    real_path_open = pathlib.Path.open

    def _guard_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if ".synth" in str(file):
            raise AssertionError(f"harness must not open {file!r}")
        return real_open(file, *args, **kwargs)

    def _guard_path_open(self: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        if ".synth" in str(self):
            raise AssertionError(f"harness must not open {self!r}")
        return real_path_open(self, *args, **kwargs)

    def _guard_connect(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("harness must not open any database")

    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(builtins, "open", _guard_open))
    stack.enter_context(patch.object(pathlib.Path, "open", _guard_path_open))
    stack.enter_context(patch.object(sqlite3, "connect", _guard_connect))
    return stack


def _fake_run(scale: float) -> PerfRun:
    """Build a PerfRun whose every field scales with ``scale``.

    Distinct per run, so a median and a final-run value can never coincide.
    """
    return PerfRun(
        widgets=WidgetCounters(
            total_widgets=int(1000 * scale),
            widgets_with_timers=int(100 * scale),
            hidden_widgets_with_timers=int(50 * scale),
        ),
        objects=ObjectCounters(
            gc_objects=int(10000 * scale),
            unfrozen_gc_objects=int(5000 * scale),
            frozen_gc_objects=int(2000 * scale),
            rss_mb=100.0 * scale,
        ),
        loop_lag=LagStats(
            samples=int(10 * scale),
            p50_ms=1.0 * scale,
            p95_ms=2.0 * scale,
            p99_ms=3.0 * scale,
            max_ms=4.0 * scale,
        ),
        pump_latency=LagStats(
            samples=int(10 * scale),
            p50_ms=1.0 * scale,
            p95_ms=2.0 * scale,
            p99_ms=3.0 * scale,
            max_ms=4.0 * scale,
        ),
        worst_auto_oldgen_pause_ms=10.0 * scale,
        manual_gen2_ms=20.0 * scale,
        wall_ms=1000.0 * scale,
        first_switch_max_stall_ms=30.0 * scale,
        first_switch_settle_ms=40.0 * scale,
    )


class TestFirstPaintJournal:
    """The committed fixture the first-paint gate measures against."""

    def test_matches_the_recorded_composition(self) -> None:
        """Literal event counts from the real code-planner journal.

        Silent failure: the fixture drifts, every number measured against it becomes
        incomparable to the recorded evidence, and the gate quietly changes meaning while
        still passing.
        """
        journal = harness.first_paint_journal()
        counts = Counter(type(event).__name__ for event in journal)

        assert len(journal) == 779
        assert counts["ToolCallUpdated"] == 252
        assert counts["MessageChunkReceived"] == 231
        assert counts["AgentThoughtReceived"] == 190
        assert counts["UserPromptSubmitted"] == 53
        assert counts["TurnComplete"] == 52
        assert counts["HookFired"] == 1
        assert sum(1 for e in journal if isinstance(e, ToolCallUpdated) and e.diffs) == 28

    def test_matches_the_ratified_turn_shape(self) -> None:
        """Literal turn-size distribution, derived from the JOURNAL itself.

        Reads turn sizes off `first_paint_journal()` rather than the private helper, so a
        journal builder that stopped following the ratified construction would fail here even
        if the helper still returned the right vector.

        Hardcoded, never recomputed with the construction's own logic. Silent failure: a
        uniform-turn fixture never makes the event budget bind, so FIRST_PAINT_EVENT_BUDGET
        ships unexercised by any gate while AC2 still passes on hand-built inputs.
        """
        journal = harness.first_paint_journal()

        # Content events per turn, counted from the journal's own structure.
        sizes: list[int] = []
        current = 0
        started = False
        for event in journal:
            if isinstance(event, UserPromptSubmitted):
                if started:
                    sizes.append(current)
                started, current = True, 0
            elif isinstance(event, TurnComplete):
                continue
            elif isinstance(event, HookFired):
                continue
            elif started:
                current += 1
        sizes.append(current)

        assert len(sizes) == 53
        assert sum(sizes) == 673
        assert min(sizes) == 4
        assert max(sizes) == 34
        assert sum(sizes) / len(sizes) == pytest.approx(12.698, abs=0.001)
        # The exact ratified vector, HARDCODED. Comparing against the production helper was
        # circular: swapping two of its outputs abandoned the index-order construction while
        # every summary statistic — count, sum, min, max, mean, tail — stayed satisfied, so a
        # permutation passed. Only a literal expectation can reject that.
        assert sizes == [
            4, 8, 12, 16, 34, 12, 8, 6,
            4, 8, 12, 16, 34, 12, 8, 6,
            4, 8, 12, 16, 34, 12, 8, 6,
            4, 8, 12, 16, 34, 12, 8, 6,
            4, 8, 12, 16, 34, 12, 8, 6,
            4, 8, 12, 16, 34, 12, 8, 6,
            4, 8, 12, 16, 33,
        ]

    def test_payload_sizes_match_the_measured_session(self) -> None:
        """The 15KB mean raw_input and 42.5KB maximum diff, asserted on the fixture.

        Silent failure: a fixture with tiny payloads mounts far cheaper widgets, so first
        paint looks better than the real session and every absolute reading becomes
        meaningless while the relative gate still passes.
        """
        journal = harness.first_paint_journal()
        tool_calls = [e for e in journal if isinstance(e, ToolCallUpdated)]
        mean_bytes = sum(raw_input_bytes(e) for e in tool_calls) / len(tool_calls)
        max_diff = max(
            len(diff.new_text) for e in tool_calls for diff in (e.diffs or [])
        )

        assert mean_bytes == pytest.approx(15 * 1024, rel=0.02)
        assert max_diff == int(42.5 * 1024)

    def test_ends_in_an_open_turn_and_is_flat(self) -> None:
        """53 prompts against 52 TurnCompletes, and no tool-call nesting.

        The trailing OPEN turn is the real journal's shape and is what exercises the
        trailing-open-segment branch of the tail selection. Nesting is flat because the real
        depth was never measured; the docstring says so rather than the fixture pretending.
        """
        journal = harness.first_paint_journal()

        assert isinstance(
            journal[-1], (ToolCallUpdated, MessageChunkReceived, AgentThoughtReceived)
        ), "the journal must end mid-turn, with no trailing TurnComplete"
        assert all(e.parent_tool_call_id is None for e in journal if isinstance(e, ToolCallUpdated))

    def test_three_tail_turns_exceed_the_event_budget(self) -> None:
        """AC1a-i: the skew is load-bearing, so assert it rather than assume it.

        With uniform turns three tail turns would sit under the budget and the bound guarding
        a pathological turn would never be exercised by the gate.
        """
        journal = harness.first_paint_journal()
        boundaries = [i for i, e in enumerate(journal) if isinstance(e, TurnComplete)]
        # Three whole turns from the tail start just after the third-from-last TurnComplete.
        tail_events = len(journal) - (boundaries[-3] + 1)

        # The LITERAL from the ratified construction (turn events 14 + 18 + 34), not merely
        # "over the budget": a construction change that still cleared 40 by a hair would
        # leave the budget cap barely exercised and this assertion would not notice.
        assert tail_events == 66
        assert tail_events > FIRST_PAINT_EVENT_BUDGET

    def test_both_caps_are_reachable_on_this_fixture(self) -> None:
        """AC1a-i: each bound must be exercised by at least one case.

        At production constants the BUDGET binds on this fixture, so the turn cap needs a
        raised-budget case. A single case would leave one bound untested end to end.
        """
        journal = harness.first_paint_journal()

        budget_bound = first_paint_window(journal)
        logical = sum(
            1 for e in journal[budget_bound.start_index :] if isinstance(e, UserPromptSubmitted)
        )
        assert logical < FIRST_PAINT_TURNS, "expected the budget to bind at production constants"

        with patch.object(conversation_module, "FIRST_PAINT_EVENT_BUDGET", 10**9):
            turn_bound = first_paint_window(journal)
        logical_turns = sum(
            1 for e in journal[turn_bound.start_index :] if isinstance(e, UserPromptSubmitted)
        )
        assert logical_turns == FIRST_PAINT_TURNS

    def test_is_deterministic(self) -> None:
        """A flaky fixture is indistinguishable from a real regression."""
        assert harness.first_paint_journal() == harness.first_paint_journal()

    def test_opens_no_database(self) -> None:
        """The standing policy, asserted behaviourally rather than by reading source.

        Silent failure: someone later "improves" fidelity by reading the real journal, it
        passes on their machine, and it fails or leaks private session content everywhere else.
        """
        with (
            patch("sqlite3.connect", side_effect=AssertionError("opened a database")),
            patch.object(pathlib.Path, "home", side_effect=AssertionError("read the home dir")),
        ):
            assert len(harness.first_paint_journal()) == 779

    def test_docstring_records_what_it_does_not_match(self) -> None:
        """The fixture must not be mistaken for the real journal.

        Its absolute numbers are fixture-calibrated; a reader who assumes otherwise would
        compare them against the recorded 231ms/7861ms and draw the wrong conclusion.
        """
        doc = harness.first_paint_journal.__doc__ or ""
        lowered = doc.lower()
        for omission in ("order", "payload text", "coalescing", "nesting"):
            assert omission in lowered, f"docstring does not disclaim {omission}"


class TestFirstSelectionPaintHarness:
    """The measurement helper itself, in the DEFAULT suite.

    These guard the harness rather than the product, and they belong here rather than behind
    the perf mark: the perf gates are deselected by default, so a harness that silently
    measured nothing would go unnoticed on every ordinary run.
    """

    async def test_measures_a_real_first_selection_drain(self) -> None:
        """The timed call must create the panel and drain the buffer.

        Silent failure: the target is already selected, so `_do_select_agent` short-circuits
        and the harness reports a spectacular first-paint number that measures nothing. That
        is exactly what happens if the dummy initial agent is dropped.
        """
        journal = harness.first_paint_journal()[:40]
        run = await harness.first_selection_paint(
            journal, agent_id=harness._FP_AGENT, windowed=True
        )

        assert run.visible_ms > 0
        assert run.turn_batches > 0
        assert run.mounted_turns > 0
        assert run.viewport_height > 0

    async def test_runs_each_app_on_a_fresh_event_loop(self) -> None:
        """Two consecutive measurements in one test must both succeed.

        `SynthApp.on_unmount` shuts down the current loop's default executor, so without
        per-run isolation the SECOND app dies inside Markdown parsing. The failure is not
        silent but it IS misattributed — it surfaces as an unrelated-looking Textual executor
        error, which is how much time this would cost to diagnose later.
        """
        journal = harness.first_paint_journal()[:40]
        first = await harness.first_selection_paint(
            journal, agent_id=harness._FP_AGENT, windowed=True
        )
        second = await harness.first_selection_paint(
            journal, agent_id=harness._FP_AGENT, windowed=True
        )

        assert first.turn_batches == second.turn_batches
