"""Process-wide garbage-collection tuning for the SYNTH runtime.

This module MUTATES only. It exposes no measurement API and knows nothing about
observation, which lives in :mod:`synth_acp.diagnostics`. The two are separate
because observation and process-wide mutation have different reasons to change.

Gen-2 collection pause duration is linear in the number of UNFROZEN live tracked
objects (measured at roughly 0.21 us per object: 730 ms at 3.39M objects, 172 ms
at 604k, 16 ms at 26k). Freezing the stabilized object graph on a short timer
therefore flattens the stop-the-world pause without unmounting anything — the
measured auto gen-2 maximum drops from 714.1 ms to 19.9 ms and the worst loop
stall from 785 ms to 90.5 ms.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import statistics
import threading
import time
from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

GC_FREEZE_INTERVAL_SECONDS: float = 1.0
"""Seconds between collect+freeze cycles.

DECIDED FROM MEASUREMENT — do not tune. 0.5 s gave loop max 110.5 ms / gen-2
22.1 ms, 1.0 s gave 91.1 / 14.5, and 2.0 s gave 82.3 / 19.9, with total collect
CPU roughly constant at 660-720 ms across the build. The differences are within
run-to-run noise, so further tuning buys nothing.
"""


_explicit_state = threading.local()
"""Per-thread nesting depth of deliberate, self-triggered collections.

THREAD-LOCAL, not process-global. CPython runs a collection — and therefore the
``gc.callbacks`` hook — on whichever thread triggered it. A process-global flag would
label an unrelated collection on ANOTHER thread as ours while this thread merely sat
inside ``explicit_collection()``, so that pause would be wrongly excluded from
``max_automatic_pause_ms()`` and the primary GC metric would under-report. The perf
harness runs each replay on its own thread, so this is reachable, not theoretical.

A depth counter rather than a flag, so a nested collect — ``unfreeze()`` collects inside
its own mark — cannot clear the mark early.
"""


@contextlib.contextmanager
def explicit_collection() -> Iterator[None]:
    """Mark the enclosing block as a collection WE triggered, not the interpreter.

    :class:`synth_acp.diagnostics.GCPauseRecorder` reads this while handling the
    ``gc.callbacks`` hook, so pauses caused by our own mitigation are recorded with
    ``explicit=True`` and excluded from ``max_automatic_pause_ms()``. Without it,
    enabling the freeze worker would make that metric partly measure our own fix
    instead of the uncontrolled pause it exists to eliminate.

    The mark is scoped to the CALLING THREAD and released in a ``finally`` block, so
    neither an exception mid-collection nor a collection on another thread can leave
    later interpreter pauses misattributed as ours.

    The dependency deliberately points this way — the observer imports the mutator.
    The reverse would put a measurement import inside this module, which the phase
    boundary forbids.
    """
    depth = getattr(_explicit_state, "depth", 0)
    _explicit_state.depth = depth + 1
    try:
        yield
    finally:
        _explicit_state.depth = depth


def in_explicit_collection() -> bool:
    """True while THIS THREAD is inside a collection it deliberately triggered."""
    return getattr(_explicit_state, "depth", 0) > 0


class GCFreezeStats(BaseModel):
    """Cumulative statistics for the freeze loop."""

    model_config = ConfigDict(frozen=True)
    freeze_count: int
    frozen_objects: int
    unfrozen_objects: int
    max_collect_ms: float
    median_collect_ms: float


class GCFreezeManager:
    """Periodically collects then freezes the object graph to flatten gen-2 pauses."""

    def __init__(self, interval: float = GC_FREEZE_INTERVAL_SECONDS) -> None:
        """Args: interval: seconds between collect+freeze cycles."""
        self._interval = interval
        self._freeze_count = 0
        self._collect_ms: list[float] = []

    async def run(self) -> None:
        """Loop forever: sleep(interval) then collect_and_freeze().

        Returns cleanly on CancelledError. Never propagates an exception from
        collect_and_freeze — a GC failure must not take down the app.
        """
        try:
            while True:
                await asyncio.sleep(self._interval)
                try:
                    self.collect_and_freeze()
                except Exception:
                    # One GC hiccup must not kill the worker for the rest of the
                    # session, which would silently disable the fix entirely.
                    # CancelledError derives from BaseException, so cancellation
                    # still propagates to the handler below.
                    log.debug("collect_and_freeze failed", exc_info=True)
        except asyncio.CancelledError:
            return

    def collect_and_freeze(self) -> None:
        """Full gc.collect() followed by gc.freeze().

        MUST call gc.collect() with NO generation argument.

        Marks the collect as OURS for the duration so GCPauseRecorder records
        `explicit=True` and max_automatic_pause_ms() excludes it. The mark must be cleared in
        a finally block so an exception cannot leave subsequent interpreter pauses
        misattributed.
        """
        started = time.perf_counter()
        with explicit_collection():
            # NO GENERATION ARGUMENT. gc.collect(0) collects only gen-0, so gen-1 and
            # gen-2 GARBAGE would be frozen permanently — measured at +344 MB RSS and
            # a gen-2 maximum of 136 ms instead of 16 ms.
            gc.collect()
            gc.freeze()
        self._collect_ms.append((time.perf_counter() - started) * 1000.0)
        self._freeze_count += 1

    def unfreeze(self) -> None:
        """Escape hatch: gc.unfreeze() then a full gc.collect().

        Costs one full-cost gen-2 pause (~400-450 ms). NOT called automatically anywhere.
        """
        # Marked too: this collect is ours, so it must not be misread as an
        # interpreter-triggered pause either.
        with explicit_collection():
            gc.unfreeze()
            gc.collect()

    def stats(self) -> GCFreezeStats:
        """Current cumulative statistics."""
        return GCFreezeStats(
            freeze_count=self._freeze_count,
            frozen_objects=gc.get_freeze_count(),
            unfrozen_objects=len(gc.get_objects()),
            max_collect_ms=max(self._collect_ms) if self._collect_ms else 0.0,
            median_collect_ms=(
                statistics.median(self._collect_ms) if self._collect_ms else 0.0
            ),
        )
