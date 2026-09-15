"""Tests for synth_acp.diagnostics — the measurement API every later phase gates on."""

from __future__ import annotations

import ast
import asyncio
import gc
import inspect
import logging
import time
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from synth_acp import diagnostics
from synth_acp.diagnostics import (
    DIAG_SUMMARY_INTERVAL_SECONDS,
    GCPause,
    GCPauseRecorder,
    LagStats,
    LoopLagSampler,
    PumpLatencyProbe,
    diagnostics_enabled,
    manual_gen2_pause_ms,
    object_counters,
    start_diagnostics,
    widget_counters,
)

_SOURCE = Path(inspect.getfile(diagnostics)).read_text()


class _BareApp(App):
    """Minimal app used to exercise pump-level and widget-level metrics."""

    def compose(self) -> ComposeResult:
        yield Static("bare")


# ── Module boundary (AC1) ──


def _imported_modules(source: str) -> list[str]:
    """Return every module path an import statement in ``source`` binds.

    ``from textual import events`` yields both ``textual`` and ``textual.events``,
    so a banned-root check catches ``from textual import widget`` as well as
    ``import textual.widget``.
    """
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


class TestModuleBoundary:
    def test_module_performs_no_process_mutation(self) -> None:
        """No gc mutation in CODE.

        Asserted over the AST rather than the raw text, because the module's own
        docstring names the forbidden calls in order to explain the boundary.
        """
        banned = {"freeze", "unfreeze", "set_threshold"}
        offenders = [
            node.attr
            for node in ast.walk(ast.parse(_SOURCE))
            if isinstance(node, ast.Attribute)
            and node.attr in banned
            and isinstance(node.value, ast.Name)
            and node.value.id == "gc"
        ]

        assert offenders == []
        # Proves the walk resolves gc attribute access at all.
        assert any(
            isinstance(node, ast.Attribute)
            and node.attr == "get_freeze_count"
            and isinstance(node.value, ast.Name)
            and node.value.id == "gc"
            for node in ast.walk(ast.parse(_SOURCE))
        )

    def test_module_references_no_widget_class(self) -> None:
        """No import may bind a widget class — the metrics must stay mechanism-generic.

        A substring check for ``synth_acp.ui`` is not enough: ``from textual.widget
        import Widget`` would slip through while still coupling the metric to a
        widget type.
        """
        banned_roots = ("textual.widget", "textual.widgets", "synth_acp.ui")
        imported = _imported_modules(_SOURCE)

        offenders = [
            module
            for module in imported
            if any(module == root or module.startswith(root + ".") for root in banned_roots)
        ]
        assert offenders == []
        # Proves the walk saw real import nodes rather than passing vacuously.
        assert "textual.events" in imported


# ── LoopLagSampler (AC3) ──


class TestLoopLagSampler:
    async def test_sampler_sees_blocking_sleep(self) -> None:
        sampler = LoopLagSampler(interval=0.005)
        task = asyncio.create_task(sampler.run())
        await asyncio.sleep(0.02)
        time.sleep(0.1)  # blocks the loop thread
        await asyncio.sleep(0.02)
        task.cancel()
        await task

        assert sampler.stats().max_ms >= 90

    def test_stats_empty_is_all_zero(self) -> None:
        assert LoopLagSampler().stats() == LagStats(
            samples=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0
        )

    def test_percentiles_use_nearest_rank(self) -> None:
        """Nearest rank is the CEILING of fraction * count.

        Silent failure: `round()` under-reports — with 5 samples it lands on index 1
        rather than 2, so p50 of [1..5] reads 2.0 instead of 3.0 and every latency
        percentile in the plan is quietly optimistic.
        """
        sampler = LoopLagSampler()
        sampler._samples.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = sampler.stats()

        assert stats == LagStats(samples=5, p50_ms=3.0, p95_ms=5.0, p99_ms=5.0, max_ms=5.0)

    def test_percentiles_on_a_single_sample(self) -> None:
        sampler = LoopLagSampler()
        sampler._samples.append(7.5)

        assert sampler.stats() == LagStats(
            samples=1, p50_ms=7.5, p95_ms=7.5, p99_ms=7.5, max_ms=7.5
        )

    async def test_reset_discards_samples(self) -> None:
        sampler = LoopLagSampler(interval=0.001)
        task = asyncio.create_task(sampler.run())
        await asyncio.sleep(0.03)
        task.cancel()
        await task
        assert sampler.stats().samples > 0

        sampler.reset()

        assert sampler.stats().samples == 0


# ── The two metrics are different quantities (AC4) ──


class TestPumpLatencyProbe:
    async def test_pump_latency_and_loop_lag_measure_different_things(self) -> None:
        """A pump handler awaiting off-thread work delays messages but not the loop.

        This is the whole reason both instruments exist: loop-lag reads ~0 while
        queued keystrokes still wait.
        """

        class _SlowHandlerApp(App):
            def compose(self) -> ComposeResult:
                yield Static("slow")

            async def on_paste(self, event: object) -> None:
                await asyncio.sleep(0.1)

        app = _SlowHandlerApp()
        async with app.run_test(headless=True) as pilot:
            sampler = LoopLagSampler(interval=0.005)
            probe = PumpLatencyProbe(app, interval=0.005)
            sampler_task = asyncio.create_task(sampler.run())
            probe_task = asyncio.create_task(probe.run())
            try:
                from textual.events import Paste

                app.post_message(Paste(""))
                await asyncio.sleep(0.25)
                del pilot
            finally:
                sampler_task.cancel()
                probe_task.cancel()
                await sampler_task
                await probe_task

        # The pump handler awaits 0.1s off-loop, so the probe must see it...
        assert probe.stats().max_ms >= 90
        # ...while the loop stays free. Asserted RELATIVELY, not against a wall-clock
        # bound: an absolute ceiling here is load-sensitive and was observed to flake
        # under whole-suite load while passing in isolation. The contract being proven
        # is that the two instruments measure DIFFERENT quantities, which a ratio
        # states directly and a fixed millisecond figure only approximates.
        assert probe.stats().max_ms > sampler.stats().max_ms * 3


# ── GCPauseRecorder (AC5) ──


class TestGCPauseRecorder:
    def test_records_gen2_pause(self) -> None:
        recorder = GCPauseRecorder()
        recorder.install()
        try:
            gc.collect(2)
        finally:
            recorder.remove()

        assert any(pause.generation == 2 for pause in recorder.pauses())

    def test_install_and_remove_are_idempotent(self) -> None:
        baseline = list(gc.callbacks)
        recorder = GCPauseRecorder()

        recorder.install()
        recorder.install()
        assert len(gc.callbacks) == len(baseline) + 1

        recorder.remove()
        recorder.remove()
        assert gc.callbacks == baseline

    def test_pauses_min_ms_filters_and_max_pause_ms_defaults(self) -> None:
        recorder = GCPauseRecorder()
        recorder.install()
        try:
            gc.collect(2)
        finally:
            recorder.remove()
        assert recorder.pauses(min_ms=1e9) == []

        assert GCPauseRecorder().max_pause_ms() == 0.0

    def test_reset_discards_pauses_without_uninstalling(self) -> None:
        recorder = GCPauseRecorder()
        recorder.install()
        try:
            gc.collect(2)
            assert recorder.pauses()
            recorder.reset()

            assert recorder.pauses() == []
            gc.collect(2)
            assert recorder.pauses()
        finally:
            recorder.remove()


class TestAutomaticPauseMetric:
    """max_automatic_pause_ms must survive the CPython 3.13 collector change."""

    def test_is_generation_agnostic(self) -> None:
        """Pauses recorded as generation 1 AND generation 2 are both eligible.

        On 3.12 automatic collections report generations 0, 1 and 2; on 3.13+ the
        incremental collector reports only 0 and 1, and generation 2 appears solely for
        an explicit gc.collect(). A metric keyed to a generation number is therefore
        satisfiable on one interpreter and silently vacuous on the other, and
        requires-python = ">=3.12" permits both.
        """
        recorder = GCPauseRecorder()
        recorder._pauses.extend(
            [
                GCPause(
                    generation=1, duration_ms=11.0, collected=0, uncollectable=0, explicit=False
                ),
                GCPause(
                    generation=2, duration_ms=22.0, collected=0, uncollectable=0, explicit=False
                ),
            ]
        )

        assert recorder.max_automatic_pause_ms() == 22.0

        recorder.reset()
        recorder._pauses.append(
            GCPause(generation=1, duration_ms=33.0, collected=0, uncollectable=0, explicit=False)
        )
        assert recorder.max_automatic_pause_ms() == 33.0

    def test_excludes_our_own_explicit_collects(self) -> None:
        """Our mitigation's own collect cost must not be reported as the problem."""
        recorder = GCPauseRecorder()
        recorder._pauses.extend(
            [
                GCPause(
                    generation=2, duration_ms=900.0, collected=0, uncollectable=0, explicit=True
                ),
                GCPause(
                    generation=1, duration_ms=12.0, collected=0, uncollectable=0, explicit=False
                ),
            ]
        )

        assert recorder.max_automatic_pause_ms() == 12.0

    def test_returns_zero_when_only_explicit_pauses_recorded(self) -> None:
        recorder = GCPauseRecorder()
        recorder._pauses.append(
            GCPause(generation=2, duration_ms=500.0, collected=0, uncollectable=0, explicit=True)
        )

        assert recorder.max_automatic_pause_ms() == 0.0

    def test_records_the_explicit_flag_from_the_attribution_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorder must LABEL pauses from the attribution predicate, not guess.

        Patched at the module boundary rather than importing runtime_gc, so this file
        keeps testing exactly one source module. Silent failure: a recorder that hard-codes
        explicit=False would let our own collects into the primary metric, and one that
        hard-codes True would empty the metric entirely — both read plausibly.
        """
        recorder = GCPauseRecorder()
        monkeypatch.setattr(diagnostics, "in_explicit_collection", lambda: True)
        recorder.install()
        try:
            gc.collect(2)
            marked = recorder.pauses()
        finally:
            recorder.remove()

        assert marked
        assert all(pause.explicit for pause in marked)
        assert recorder.max_automatic_pause_ms() == 0.0

        recorder.reset()
        monkeypatch.setattr(diagnostics, "in_explicit_collection", lambda: False)
        recorder.install()
        try:
            gc.collect(2)
            unmarked = recorder.pauses()
        finally:
            recorder.remove()

        assert unmarked
        assert not any(pause.explicit for pause in unmarked)
        assert recorder.max_automatic_pause_ms() > 0.0

    def test_manual_probe_times_a_real_gen2_collection(self) -> None:
        """manual_gen2_pause_ms must actually run gc.collect(2), not merely take time.

        Silent failure: any positive-duration operation satisfies an `elapsed > 0`
        assertion, so the edge contract would have no falsifiable test. A generation-2
        entry is the observable proof, and it is portable — an EXPLICIT collect(2) reports
        generation 2 on 3.12 and on 3.13+ alike.
        """
        recorder = GCPauseRecorder()
        recorder.install()
        try:
            elapsed = manual_gen2_pause_ms()
            recorded = recorder.pauses()
        finally:
            recorder.remove()

        assert elapsed > 0.0
        assert recorded, "no GC pause recorded — manual_gen2_pause_ms ran no collection"
        assert any(pause.generation == 2 for pause in recorded)
        # It is OUR deliberate probe, so it must not pollute the automatic metric.
        assert all(pause.explicit for pause in recorded)
        assert recorder.max_automatic_pause_ms() == 0.0

class TestWidgetCounters:
    async def test_counts_planted_hidden_timers(self) -> None:
        class _PlantedApp(App):
            def compose(self) -> ComposeResult:
                with Vertical():
                    for index in range(4):
                        yield Static(f"planted-{index}", id=f"planted-{index}")

        app = _PlantedApp()
        async with app.run_test(headless=True) as pilot:
            for index in range(4):
                widget = app.query_one(f"#planted-{index}", Static)
                widget.auto_refresh = 1 / 15
                # Exactly three of the four are hidden.
                widget.display = index >= 3
            await pilot.pause()

            counters = widget_counters(app)

        assert counters.hidden_widgets_with_timers == 3
        assert counters.widgets_with_timers == 4

    def test_without_screen_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            widget_counters(_BareApp())


class TestObjectCounters:
    def test_freeze_moves_objects_from_unfrozen_to_frozen(self) -> None:
        """gc.get_objects() must exclude the permanent generation.

        If it did not, the freeze worker's whole effect would be invisible and the
        unfrozen-object gate could never be satisfied by a correct implementation.
        """
        gc.collect()
        before = object_counters()
        try:
            gc.freeze()
            after = object_counters()
        finally:
            gc.unfreeze()
            gc.collect()

        assert after.frozen_gc_objects > before.frozen_gc_objects
        assert after.unfrozen_gc_objects < before.unfrozen_gc_objects


# ── Gating (AC2) ──


class TestDiagnosticsGating:
    @pytest.mark.parametrize("value", ["", "0", "FALSE", "off"])
    def test_falsey_values(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """One case per boundary: empty, numeric zero, case-insensitivity, word form.

        Unset is not a distinct path — os.environ.get defaults to "" — and is covered
        observably by test_start_diagnostics_disabled_is_fully_inert.
        """
        monkeypatch.setenv("SYNTH_DIAG", value)
        assert diagnostics_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true"])
    def test_truthy_values(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNTH_DIAG", value)
        assert diagnostics_enabled() is True

    async def test_start_diagnostics_disabled_is_fully_inert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SYNTH_DIAG", raising=False)
        app = _BareApp()
        async with app.run_test(headless=True):
            baseline = list(gc.callbacks)

            handle = start_diagnostics(app)

            assert handle is None
            assert [w for w in app.workers if w.group == "diag"] == []
            assert gc.callbacks == baseline

    async def test_start_diagnostics_enabled_starts_workers_and_logs_summary(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The default is the measured 10 s; the summary cadence is shortened here so
        # the assertion lands well inside the criterion's 11 s ceiling.
        assert DIAG_SUMMARY_INTERVAL_SECONDS == 10.0
        monkeypatch.setenv("SYNTH_DIAG", "1")
        monkeypatch.setattr(diagnostics, "DIAG_SUMMARY_INTERVAL_SECONDS", 0.02)
        app = _BareApp()
        async with app.run_test(headless=True):
            handle = start_diagnostics(app)
            assert handle is not None
            try:
                with caplog.at_level(logging.INFO, logger="synth_acp.diagnostics"):
                    await asyncio.sleep(0.15)
                    summaries = [r for r in caplog.records if "diag summary" in r.getMessage()]
                assert len([w for w in app.workers if w.group == "diag"]) == 3
                assert summaries
            finally:
                handle.stop()


# ── Teardown (AC19 mechanism, AC20) ──


class TestDiagnosticsTeardown:
    async def test_instruments_are_read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A consumer must not be able to swap out the instrument being recorded from.

        Silent failure: reassignment would detach the summary worker and every later
        phase from the sampler actually collecting, so all reported metrics would read
        empty while collection carried on.
        """
        monkeypatch.setenv("SYNTH_DIAG", "1")
        app = _BareApp()
        async with app.run_test(headless=True):
            handle = start_diagnostics(app)
            assert handle is not None
            try:
                for name in ("loop_lag", "pump_latency", "gc_pauses"):
                    with pytest.raises(AttributeError):
                        setattr(handle, name, "reassigned")
            finally:
                handle.stop()

    async def test_stop_is_idempotent_and_restores_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SYNTH_DIAG", "1")
        app = _BareApp()
        async with app.run_test(headless=True):
            baseline = list(gc.callbacks)
            handle = start_diagnostics(app)
            assert handle is not None

            handle.stop()
            handle.stop()

            # Asserted with NO intervening yield: the contract says that when stop()
            # RETURNS, gc.callbacks is back to its pre-start contents and no worker
            # remains in group "diag". cancel_group alone would leave three RUNNING
            # workers registered here.
            assert gc.callbacks == baseline
            assert [w for w in app.workers if w.group == "diag"] == []

    async def test_sampler_finally_invokes_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unexpected sampler exit must not leave the process-global hook installed."""
        monkeypatch.setenv("SYNTH_DIAG", "1")

        async def _boom(self: LoopLagSampler) -> None:
            raise RuntimeError("sampler died")

        monkeypatch.setattr(LoopLagSampler, "run", _boom)
        app = _BareApp()
        async with app.run_test(headless=True):
            baseline = list(gc.callbacks)
            handle = start_diagnostics(app)
            assert handle is not None

            for _ in range(200):
                if gc.callbacks == baseline:
                    break
                await asyncio.sleep(0.005)

            assert gc.callbacks == baseline

    async def test_partial_start_failure_rolls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure after the first worker registers must not leak the hook."""
        monkeypatch.setenv("SYNTH_DIAG", "1")
        app = _BareApp()
        async with app.run_test(headless=True):
            baseline = list(gc.callbacks)
            real_run_worker = type(app).run_worker
            calls = {"n": 0}

            def _flaky(self: App, *args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                if calls["n"] > 1:
                    raise RuntimeError("worker registration failed")
                return real_run_worker(self, *args, **kwargs)

            monkeypatch.setattr(type(app), "run_worker", _flaky)

            handle = start_diagnostics(app)

            assert handle is None
            assert gc.callbacks == baseline
            assert [w for w in app.workers if w.group == "diag"] == []
