"""Runtime observation primitives for diagnosing event-loop starvation.

This module OBSERVES only. It never mutates process-wide garbage-collection
state — no ``gc.freeze``, ``gc.unfreeze`` or ``gc.set_threshold`` appears here.
Process mutation lives in :mod:`synth_acp.runtime_gc`, because observation and
process-wide mutation have different reasons to change.

Every metric is mechanism-generic: ``hidden_widgets_with_timers`` counts any
widget holding a live periodic refresh timer while not displayed, so it names no
widget class and stays valid as the UI evolves.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from textual import events

# Attribution only: lets a pause be labelled as one WE triggered rather than one the
# interpreter chose. The dependency points observer -> mutator because the reverse
# would put a measurement import inside the mutation module.
from synth_acp.runtime_gc import explicit_collection, in_explicit_collection

log = logging.getLogger(__name__)

DIAG_SUMMARY_INTERVAL_SECONDS: float = 10.0
"""Seconds between diagnostic summary log lines."""

_DIAG_WORKER_GROUP = "diag"

_FALSEY_ENV = frozenset({"", "0", "false", "no", "off"})


class LagStats(BaseModel):
    """Distribution of a latency measurement, in milliseconds."""

    model_config = ConfigDict(frozen=True)
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def _distribution(samples: list[float]) -> LagStats:
    """Summarise millisecond samples as a LagStats.

    Uses nearest-rank percentiles over a sorted copy, so the caller's list may
    keep growing concurrently.

    Args:
        samples: Millisecond measurements. May be empty.

    Returns:
        The distribution, all-zero with ``samples=0`` when the input is empty.
    """
    if not samples:
        return LagStats(samples=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0)
    ordered = sorted(samples)
    count = len(ordered)

    def _rank(fraction: float) -> float:
        # Nearest-rank percentile uses the CEILING of fraction * count. `round` would
        # under-report — with 5 samples, round(0.5 * 5) - 1 lands on index 1 rather
        # than the correct index 2, so p50 of [1..5] would read 2.0 instead of 3.0.
        index = min(count - 1, max(0, math.ceil(fraction * count) - 1))
        return ordered[index]

    return LagStats(
        samples=count,
        p50_ms=_rank(0.50),
        p95_ms=_rank(0.95),
        p99_ms=_rank(0.99),
        max_ms=ordered[-1],
    )


class LoopLagSampler:
    """Measures event-loop scheduling delay via sleep overshoot.

    Detects work that BLOCKS the loop (synchronous CPU, GC pauses). Does NOT detect a pump
    handler awaiting off-thread work — use PumpLatencyProbe for that.
    """

    def __init__(self, interval: float = 0.02) -> None:
        """Args: interval: seconds to sleep between samples."""
        self._interval = interval
        self._samples: list[float] = []

    async def run(self) -> None:
        """Sample until cancelled. Returns cleanly on CancelledError; never re-raises it."""
        try:
            while True:
                started = time.perf_counter()
                await asyncio.sleep(self._interval)
                overshoot = (time.perf_counter() - started) - self._interval
                self._samples.append(max(0.0, overshoot * 1000.0))
        except asyncio.CancelledError:
            return

    def stats(self) -> LagStats:
        """Current distribution. Safe while running. All-zero with samples=0 if none taken."""
        return _distribution(self._samples)

    def reset(self) -> None:
        """Discard accumulated samples."""
        self._samples.clear()


class PumpLatencyProbe:
    """Measures how long a message waits in an App message pump before being handled.

    Posts a lightweight probe Message and records enqueue-to-handle delay. This is the
    quantity corresponding to keystroke responsiveness, because the driver posts Key events
    to the same queue as BrokerEventMessage.
    """

    def __init__(self, app: Any, interval: float = 0.05) -> None:
        """Args: app: the running App. interval: seconds between probes."""
        self._app = app
        self._interval = interval
        self._samples: list[float] = []

    async def run(self) -> None:
        """Probe until cancelled. Returns cleanly on CancelledError; never re-raises it."""
        try:
            while True:
                await asyncio.sleep(self._interval)
                # post_message puts the probe on the pump's FIFO queue, so the
                # recorded delay is true queue wait. call_next is deliberately NOT
                # used: it appends to _next_callbacks, which is flushed out-of-band
                # when the queue drains, and would under-measure.
                self._app.post_message(
                    events.Callback(callback=partial(self._record, time.perf_counter()))
                )
        except asyncio.CancelledError:
            return

    def _record(self, sent_at: float) -> None:
        """Record the delay for a probe enqueued at ``sent_at``."""
        self._samples.append((time.perf_counter() - sent_at) * 1000.0)

    def stats(self) -> LagStats:
        """Current distribution. Safe while running. All-zero with samples=0 if none taken."""
        return _distribution(self._samples)

    def reset(self) -> None:
        """Discard accumulated samples."""
        self._samples.clear()


class GCPause(BaseModel):
    """One garbage-collection pause."""

    model_config = ConfigDict(frozen=True)
    generation: int
    duration_ms: float
    collected: int
    uncollectable: int
    explicit: bool  # True when caused by our own collect_and_freeze(), not by the interpreter


class GCPauseRecorder:
    """Records gen-0/1/2 pause durations via gc.callbacks."""

    def __init__(self) -> None:
        self._pauses: list[GCPause] = []
        self._callback: Any = None
        self._started_at: float | None = None

    def install(self) -> None:
        """Register the gc callback. Idempotent — a second call adds no second callback."""
        if self._callback is not None:
            return
        self._callback = self._on_gc
        gc.callbacks.append(self._callback)

    def remove(self) -> None:
        """Unregister. Idempotent — safe when not installed."""
        if self._callback is None:
            return
        with contextlib.suppress(ValueError):
            gc.callbacks.remove(self._callback)
        self._callback = None

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        """gc.callbacks hook recording the duration of each collection."""
        if phase == "start":
            self._started_at = time.perf_counter()
            return
        if self._started_at is None:
            return
        duration_ms = (time.perf_counter() - self._started_at) * 1000.0
        self._started_at = None
        self._pauses.append(
            GCPause(
                generation=int(info.get("generation", 0)),
                duration_ms=duration_ms,
                collected=int(info.get("collected", 0)),
                uncollectable=int(info.get("uncollectable", 0)),
                # The "stop" callback still runs inside gc.collect(), so a collect we
                # triggered ourselves is still marked at this point.
                explicit=in_explicit_collection(),
            )
        )

    def pauses(self, min_ms: float = 0.0) -> list[GCPause]:
        """Recorded pauses with duration_ms >= min_ms, in occurrence order."""
        return [pause for pause in self._pauses if pause.duration_ms >= min_ms]

    def max_pause_ms(self, generation: int | None = None) -> float:
        """Worst recorded pause, optionally for one generation. 0.0 if none recorded."""
        durations = [
            pause.duration_ms
            for pause in self._pauses
            if generation is None or pause.generation == generation
        ]
        return max(durations) if durations else 0.0

    def max_automatic_pause_ms(self) -> float:
        """Worst INTERPRETER-TRIGGERED pause, any generation, 0.0 if none recorded.

        Generation-agnostic ON PURPOSE. CPython 3.13 replaced the three-generation collector
        with an incremental one: on 3.12 automatic collections report generations 0, 1 AND 2,
        while on 3.13+ they report only 0 and 1 and generation 2 appears solely for an explicit
        gc.collect(). A metric keyed to generation 2 is therefore satisfiable on 3.12 and
        vacuous on 3.13+, and `requires-python = ">=3.12"` permits both.

        EXCLUDES pauses where `explicit` is True. Without that exclusion, enabling the freeze
        worker would make this metric measure our own mitigation's collect cost instead of the
        uncontrolled pause it exists to eliminate. Our own cost is reported separately as
        GCFreezeStats.max_collect_ms.
        """
        durations = [pause.duration_ms for pause in self._pauses if not pause.explicit]
        return max(durations) if durations else 0.0

    def reset(self) -> None:
        """Discard recorded pauses. Does not uninstall."""
        self._pauses.clear()
        self._started_at = None


class ObjectCounters(BaseModel):
    """Process-level object and memory counts. No UI knowledge."""

    model_config = ConfigDict(frozen=True)
    gc_objects: int
    unfrozen_gc_objects: int
    frozen_gc_objects: int
    rss_mb: float


class WidgetCounters(BaseModel):
    """Widget-tree counts. Mechanism-generic: names no specific widget class."""

    model_config = ConfigDict(frozen=True)
    total_widgets: int
    widgets_with_timers: int
    hidden_widgets_with_timers: int


def _rss_mb() -> float:
    """Return resident set size in MiB.

    Reads ``/proc/self/statm`` when available, which reports CURRENT RSS. On
    platforms without procfs (macOS included) it falls back to
    ``resource.getrusage``, which reports PEAK RSS for the process — a high-water
    mark rather than a live value.

    Returns:
        Resident set size in MiB, or 0.0 if it cannot be determined.
    """
    statm = Path("/proc/self/statm")
    try:
        if statm.exists():
            pages = int(statm.read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return 0.0
    # ru_maxrss is bytes on Darwin and KiB on Linux and the BSDs.
    divisor = 1024.0 * 1024.0 if os.uname().sysname == "Darwin" else 1024.0
    return peak / divisor


def object_counters() -> ObjectCounters:
    """Snapshot process object/memory counts. Requires no App."""
    # gc.get_objects() excludes the permanent generation, so it IS the unfrozen
    # count — which is the quantity gen-2 pause duration is linear in.
    unfrozen = len(gc.get_objects())
    frozen = gc.get_freeze_count()
    return ObjectCounters(
        gc_objects=unfrozen + frozen,
        unfrozen_gc_objects=unfrozen,
        frozen_gc_objects=frozen,
        rss_mb=_rss_mb(),
    )


def widget_counters(app: Any) -> WidgetCounters:
    """Snapshot widget-tree counts for a running App.

    hidden_widgets_with_timers counts widgets where a live auto_refresh coexists with
    display False — a leaked periodic timer regardless of widget class.

    Raises:
        RuntimeError: if the App has no active screen.
    """
    try:
        screen = app.screen
    except Exception as error:  # textual raises ScreenStackError, not RuntimeError
        raise RuntimeError("App has no active screen") from error

    nodes = screen.walk_children(with_self=True)
    with_timers = 0
    hidden_with_timers = 0
    for node in nodes:
        # Duck-typed so this module references no widget class: a LIVE timer is a
        # non-None _auto_refresh_timer, which is distinct from the interval value.
        if getattr(node, "_auto_refresh_timer", None) is None:
            continue
        with_timers += 1
        if not getattr(node, "display", True):
            hidden_with_timers += 1
    return WidgetCounters(
        total_widgets=len(nodes),
        widgets_with_timers=with_timers,
        hidden_widgets_with_timers=hidden_with_timers,
    )


def manual_gen2_pause_ms() -> float:
    """Time a full gc.collect(2) and return elapsed milliseconds."""
    started = time.perf_counter()
    # Marked so this deliberate probe can never be mistaken for an
    # interpreter-triggered pause, whatever order a caller reads the metrics in.
    with explicit_collection():
        gc.collect(2)
    return (time.perf_counter() - started) * 1000.0


def diagnostics_enabled() -> bool:
    """True when the SYNTH_DIAG environment variable is set to a truthy value."""
    return os.environ.get("SYNTH_DIAG", "").strip().lower() not in _FALSEY_ENV


class DiagnosticsHandle:
    """Ownership handle for the started diagnostics workers and gc hook."""

    def __init__(
        self,
        app: Any,
        loop_lag: LoopLagSampler,
        pump_latency: PumpLatencyProbe,
        gc_pauses: GCPauseRecorder,
    ) -> None:
        self._loop_lag = loop_lag
        self._pump_latency = pump_latency
        self._gc_pauses = gc_pauses
        self._app = app
        self._stopped = False

    @property
    def loop_lag(self) -> LoopLagSampler:
        """The loop-lag sampler this handle owns. Read-only."""
        return self._loop_lag

    @property
    def pump_latency(self) -> PumpLatencyProbe:
        """The pump-latency probe this handle owns. Read-only."""
        return self._pump_latency

    @property
    def gc_pauses(self) -> GCPauseRecorder:
        """The GC pause recorder this handle owns. Read-only.

        Read-only so a consumer cannot swap out the instrument the summary worker and
        later phases read from, which would silently detach every reported metric from
        the one actually recording.
        """
        return self._gc_pauses

    def stop(self) -> None:
        """Cancel the diag workers and uninstall the gc.callbacks hook. Idempotent.

        After this returns, gc.callbacks contains exactly its pre-start contents and no worker
        remains in group 'diag'. Also invoked from the sampler worker's finally block so an
        unexpected worker exit cannot leave the hook installed.
        """
        if self._stopped:
            return
        self._stopped = True
        # The gc hook is process-global, so it comes off first: a failure while
        # cancelling workers must not leave it installed.
        self.gc_pauses.remove()
        with contextlib.suppress(Exception):
            cancelled = self._app.workers.cancel_group(self._app, _DIAG_WORKER_GROUP)
            # cancel_group only REQUESTS cancellation: a Worker's state becomes CANCELLED
            # when its task later processes the CancelledError, and the manager drops it
            # only in the task's done-callback. So the workers would still be present and
            # RUNNING when this returns, contradicting the contract above. Deregister them
            # now so the group is observably empty on return; the done-callback's discard
            # is idempotent, and the tasks unwind on their own afterwards.
            for worker in cancelled:
                self._app.workers._remove_worker(worker)

    async def _run_sampler(self) -> None:
        """Run the loop-lag sampler, tearing diagnostics down if it ever exits."""
        try:
            await self.loop_lag.run()
        finally:
            self.stop()

    async def _run_summary(self) -> None:
        """Log a diagnostics summary every DIAG_SUMMARY_INTERVAL_SECONDS."""
        try:
            while True:
                await asyncio.sleep(DIAG_SUMMARY_INTERVAL_SECONDS)
                log.info(
                    "diag summary: loop_lag=%s pump_latency=%s gen2_max_ms=%.1f objects=%s",
                    self.loop_lag.stats(),
                    self.pump_latency.stats(),
                    self.gc_pauses.max_pause_ms(2),
                    object_counters(),
                )
        except asyncio.CancelledError:
            return


def start_diagnostics(app: Any) -> DiagnosticsHandle | None:
    """Start LoopLagSampler, PumpLatencyProbe and GCPauseRecorder as App workers.

    Worker group 'diag'. No-op returning None when diagnostics_enabled() is False: registers
    no worker and installs no gc callback. Logs a summary every 10 s. Never raises —
    instrumentation failure must not take down the app.

    Returns a handle the app MUST stop on unmount. The gc.callbacks hook is process-global, so
    a recorder left installed survives repeated run_test app instances, retains state,
    duplicates measurements, and changes the callback baseline observed by later
    SYNTH_DIAG-off tests.
    """
    if not diagnostics_enabled():
        return None
    # Build the handle BEFORE anything is registered, so the rollback path below
    # always has something to stop.
    handle = DiagnosticsHandle(app, LoopLagSampler(), PumpLatencyProbe(app), GCPauseRecorder())
    try:
        handle.gc_pauses.install()
        # CALLABLES, not coroutine objects. A worker cancelled before its task starts
        # never invokes the callable, so nothing is left un-awaited. Passing coroutines
        # here leaks "coroutine was never awaited" warnings whenever the sampler exits
        # early and its finally block cancels the siblings.
        app.run_worker(
            handle._run_sampler,
            name="diag-loop-lag",
            group=_DIAG_WORKER_GROUP,
            exit_on_error=False,
        )
        app.run_worker(
            handle.pump_latency.run,
            name="diag-pump-latency",
            group=_DIAG_WORKER_GROUP,
            exit_on_error=False,
        )
        app.run_worker(
            handle._run_summary,
            name="diag-summary",
            group=_DIAG_WORKER_GROUP,
            exit_on_error=False,
        )
    except Exception:
        # Partial construction is reachable: the gc callback is installed and one
        # or two workers may already be running. Returning None without rollback
        # would leave a process-global hook and orphan workers with no handle for
        # the app to stop. stop() is idempotent, so a later normal stop is safe.
        log.debug("Diagnostics failed to start", exc_info=True)
        handle.stop()
        return None
    return handle
