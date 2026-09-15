"""Tests for synth_acp.runtime_gc — the gen-2 pause mitigation."""

from __future__ import annotations

import ast
import asyncio
import gc
import inspect
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest

from synth_acp import runtime_gc
from synth_acp.runtime_gc import GC_FREEZE_INTERVAL_SECONDS, GCFreezeManager

_SOURCE = Path(inspect.getfile(runtime_gc)).read_text()
_SRC_ROOT = Path(inspect.getfile(runtime_gc)).parent


class _Node:
    """Participates in a reference cycle and supports weak references."""

    def __init__(self) -> None:
        self.peer: _Node | None = None


class TestCollectAndFreeze:
    def test_reclaims_older_generation_garbage(self) -> None:
        """A full collect must run BEFORE the freeze.

        Builds a reference cycle, promotes it to an older generation while it is
        still strongly reachable, then drops the last strong reference and freezes.
        A ``gc.collect(0)`` implementation collects only gen-0, so the cycle would
        survive and be frozen PERMANENTLY — the measured +344 MB RSS and 136 ms
        gen-2 regression. No mock-based assertion can tell the two apart.
        """
        manager = GCFreezeManager()
        try:
            first = _Node()
            second = _Node()
            first.peer = second
            second.peer = first
            ref = weakref.ref(first)

            # Promote the cycle past gen-0 and gen-1 while still reachable.
            gc.collect()
            gc.collect()
            assert ref() is not None

            del first, second

            manager.collect_and_freeze()

            assert ref() is None
        finally:
            manager.unfreeze()
            gc.collect()

    def test_stats_reports_collect_timings(self) -> None:
        manager = GCFreezeManager()
        try:
            manager.collect_and_freeze()
            manager.collect_and_freeze()
            stats = manager.stats()
        finally:
            manager.unfreeze()
            gc.collect()

        assert stats.freeze_count == 2
        assert stats.median_collect_ms > 0.0
        assert stats.max_collect_ms >= stats.median_collect_ms

    def test_stats_on_fresh_manager_has_zero_timings(self) -> None:
        stats = GCFreezeManager().stats()

        assert stats.freeze_count == 0
        assert stats.max_collect_ms == 0.0
        assert stats.median_collect_ms == 0.0


class TestInterval:
    def test_default_is_one_second(self) -> None:
        assert GC_FREEZE_INTERVAL_SECONDS == 1.0
        assert GCFreezeManager()._interval == 1.0


class TestRunLoop:
    async def test_survives_a_failing_collect(self) -> None:
        """One GC failure must not kill the worker for the rest of the session."""
        manager = GCFreezeManager(interval=0.005)
        calls = {"n": 0}

        def _boom() -> None:
            calls["n"] += 1
            raise RuntimeError("collect exploded")

        manager.collect_and_freeze = _boom  # type: ignore[method-assign]
        task = asyncio.create_task(manager.run())
        await asyncio.sleep(0.06)

        assert task.done() is False
        assert calls["n"] > 1

        task.cancel()
        await task

    async def test_returns_cleanly_on_cancellation(self) -> None:
        manager = GCFreezeManager(interval=0.005)
        task = asyncio.create_task(manager.run())
        await asyncio.sleep(0.01)

        task.cancel()
        await task

        assert task.cancelled() is False


def _gc_attribute_calls(attr: str) -> list[str]:
    """Return ``file:line`` for every ``gc.<attr>()`` call under ``src/``.

    AST-based rather than a text grep, because module docstrings legitimately name
    these calls in order to explain the observation/mutation boundary.
    """
    hits: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == attr
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "gc"
            ):
                hits.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}")
    return hits


class TestExplicitCollectionMark:
    """The attribution mark must be scoped to the thread that triggered the collect.

    Observed through this file's OWN gc.callbacks hook rather than
    ``diagnostics.GCPauseRecorder``, so the file keeps testing one source module.
    """

    @staticmethod
    def _marks_during_collection() -> tuple[list[bool], Any]:
        """Return a list that receives in_explicit_collection() per collection, and the hook."""
        marks: list[bool] = []

        def hook(phase: str, info: dict[str, Any]) -> None:
            if phase == "stop":
                marks.append(runtime_gc.in_explicit_collection())

        return marks, hook

    def test_collect_and_freeze_marks_its_own_collection(self) -> None:
        """AC23: the pause caused by collect_and_freeze must be attributable to us."""
        manager = GCFreezeManager()
        marks, hook = self._marks_during_collection()
        gc.callbacks.append(hook)
        try:
            manager.collect_and_freeze()
        finally:
            gc.callbacks.remove(hook)
            manager.unfreeze()
            gc.collect()

        assert marks, "collect_and_freeze triggered no collection"
        assert all(marks)

    def test_raising_collect_clears_the_mark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC23: a FAILING collect_and_freeze must not poison later attribution.

        Drives the real manager failure path — ``gc.collect`` itself raising inside
        ``collect_and_freeze`` — rather than only the helper's own finally. Silent
        failure: one failed collect would leave the mark set for the rest of the thread's
        life, so every later interpreter pause would be recorded as ours and excluded,
        and the primary GC metric would read 0.0 forever.
        """
        manager = GCFreezeManager()

        def _boom(*args: object, **kwargs: object) -> int:
            raise RuntimeError("collect exploded")

        monkeypatch.setattr(gc, "collect", _boom)
        with pytest.raises(RuntimeError):
            manager.collect_and_freeze()
        monkeypatch.undo()

        assert runtime_gc.in_explicit_collection() is False

        # A subsequent real collection must be attributed to the interpreter.
        marks, hook = self._marks_during_collection()
        gc.callbacks.append(hook)
        try:
            gc.collect(1)
        finally:
            gc.callbacks.remove(hook)

        assert marks
        assert not any(marks)

    def test_mark_does_not_leak_to_other_threads(self) -> None:
        """A collection on ANOTHER thread must not be labelled as ours.

        CPython runs a collection, and therefore the gc.callbacks hook, on whichever
        thread triggered it. With a process-global flag, an interpreter-triggered
        collection on thread B while thread A merely sat inside explicit_collection()
        was recorded as ours and then EXCLUDED from the primary metric. The perf harness
        runs each replay on its own thread, so this is reachable rather than theoretical.
        """
        observed: list[bool] = []
        checked = threading.Event()

        def _other_thread() -> None:
            observed.append(runtime_gc.in_explicit_collection())
            checked.set()

        with runtime_gc.explicit_collection():
            assert runtime_gc.in_explicit_collection() is True
            worker = threading.Thread(target=_other_thread)
            worker.start()
            assert checked.wait(timeout=5.0)
            worker.join()

        assert observed == [False]
        assert runtime_gc.in_explicit_collection() is False

    def test_mark_nests(self) -> None:
        """unfreeze() collects inside its own mark, so an inner exit must not clear it."""
        with runtime_gc.explicit_collection():
            with runtime_gc.explicit_collection():
                assert runtime_gc.in_explicit_collection() is True
            assert runtime_gc.in_explicit_collection() is True

        assert runtime_gc.in_explicit_collection() is False


class TestInterpreterPin:
    """AC24: the plan's whole evidence base was measured on CPython 3.12."""

    def test_python_version_pins_312_and_readme_says_so(self) -> None:
        """Without a pin, two worktrees of this repo produce non-comparable numbers.

        Silent failure: a fresh worktree builds a 3.13+ venv, the incremental collector
        changes both pause timings and the generation numbers reported to gc.callbacks,
        and every measurement silently stops being comparable to the recorded baseline.
        """
        repo_root = _SRC_ROOT.parent.parent
        pin = (repo_root / ".python-version").read_text().strip()

        assert pin == "3.12"
        readme = (repo_root / "README.md").read_text()
        assert ".python-version" in readme
        # requires-python stays permissive; the pin is for reproducibility only.
        assert 'requires-python = ">=3.12"' in (repo_root / "pyproject.toml").read_text()


class TestSourceBoundaries:
    def test_set_threshold_is_not_called_in_src(self) -> None:
        """Threshold tuning measured as a wash and is out of scope.

        It cut gen-0 count 4177 -> 52 but RAISED gen-0 total 368 -> 689 ms and
        worsened gen-0 max 14.1 -> 23.4 ms, leaving net loop max unchanged.
        """
        assert _gc_attribute_calls("set_threshold") == []
        # Proves the walk resolves gc calls at all rather than passing vacuously.
        assert _gc_attribute_calls("freeze") != []

    def test_unfreeze_has_no_call_site_in_src(self) -> None:
        """No automatic reclaim hook (user decision, root Decision 4).

        A reclaim hook buys ~55 MB at the cost of a ~400-450 ms stop-the-world
        pause per agent termination — the exact symptom this feature removes. The
        only permitted ``unfreeze`` call is ``gc.unfreeze()`` inside
        ``GCFreezeManager.unfreeze`` itself; any other receiver means something in
        ``src/`` invoked the escape hatch.
        """
        offenders: list[str] = []
        for path in sorted(_SRC_ROOT.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "unfreeze":
                    receiver = func.value
                    if isinstance(receiver, ast.Name) and receiver.id == "gc":
                        continue
                    offenders.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}")
                elif isinstance(func, ast.Name) and func.id == "unfreeze":
                    offenders.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}")

        assert offenders == []
        # The permitted gc.unfreeze() implementation exists exactly once.
        assert len(_gc_attribute_calls("unfreeze")) == 1

    def test_module_contains_no_measurement_api(self) -> None:
        """runtime_gc must not USE the measurement API.

        Checked over the AST, not the raw text: the module docstring legitimately
        cross-references ``synth_acp.diagnostics.GCPauseRecorder`` to explain which
        direction the attribution dependency points, and naming a class in prose is not
        depending on it.
        """
        tree = ast.parse(_SOURCE)
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)

        measurement_api = {
            "LagStats",
            "LoopLagSampler",
            "PumpLatencyProbe",
            "GCPauseRecorder",
            "GCPause",
            "object_counters",
            "widget_counters",
            "manual_gen2_pause_ms",
            "max_automatic_pause_ms",
        }
        assert referenced & measurement_api == set()

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
                imported.extend(f"{node.module}.{alias.name}" for alias in node.names)

        assert "synth_acp.diagnostics" not in imported
        assert "pydantic" in imported


@pytest.fixture(autouse=True)
def _leave_gc_unfrozen():
    """Never leak a frozen permanent generation into another test."""
    yield
    gc.unfreeze()
