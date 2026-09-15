"""Tests for DiffView widget."""

from __future__ import annotations

import inspect
import pathlib
from typing import cast
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.containers import Vertical

from synth_acp.ui.widgets import diff_view as diff_view_module
from synth_acp.ui.widgets.diff_view import (
    DiffCode,
    DiffView,
    LineContent,
    fill_lists,
    loop_last,
    prewarm_highlighting,
)

_BEFORE = "alpha = 1\nbeta = 2\ngamma = 3\n"
_AFTER = "alpha = 1\nbeta = 99\ngamma = 3\n"


class _Host(App):
    """Hosts one DiffView inside a container of a known, non-zero width."""

    def __init__(self, view: DiffView, width: int) -> None:
        super().__init__()
        self._view = view
        self._width = width

    def compose(self) -> ComposeResult:
        container = Vertical(self._view)
        container.styles.width = self._width
        yield container


def _rendered_text(view: DiffView) -> str:
    """Join the plain text of every rendered code line."""
    parts: list[str] = []
    for code in view.query(DiffCode):
        visual = cast(LineContent, code._render())
        parts.extend("" if line is None else line.plain for line in visual.code_lines)
    return "\n".join(parts)


class TestLoopLast:
    def test_loop_last_yields_last_flag_on_final_item(self) -> None:
        result = list(loop_last([1, 2, 3]))
        assert result == [(False, 1), (False, 2), (True, 3)]

    def test_loop_last_empty_iterable_yields_nothing(self) -> None:
        assert list(loop_last([])) == []


class TestFillLists:
    def test_fill_lists_pads_shorter_list(self) -> None:
        a, b = [1, 2, 3], [4]
        fill_lists(a, b, 0)
        assert len(a) == len(b) == 3
        assert b == [4, 0, 0]


class TestDiffViewProperties:
    def test_counts_returns_additions_and_removals(self) -> None:
        dv = DiffView("f.py", "f.py", "a\nb\nc", "a\nx\nc\nd")
        adds, rems = dv.counts
        assert adds > 0
        assert rems > 0

    def test_grouped_opcodes_empty_when_no_changes(self) -> None:
        dv = DiffView("f.py", "f.py", "same", "same")
        assert dv.grouped_opcodes == []


class TestSplitResolution:
    def test_split_threshold_never_highlights(self) -> None:
        """The threshold must be computed from PLAIN lines.

        Silent failure: reading `highlighted_code_lines` here reintroduces the ~186ms
        synchronous tree-sitter highlight on the message pump, and every functional and
        content test would still pass.
        """
        dv = DiffView("f.py", "f.py", "ab\n", "abcd\n")

        threshold = dv._split_threshold()

        assert dv._highlighted_code_lines is None
        # 4 cells (longest line "abcd") * 2, + 4, + 2 * 1 line-number digit, + 2 (no
        # annotations) == 16.
        assert threshold == 16

    def test_resolve_split_picks_unified_when_narrow_and_split_when_wide(self) -> None:
        """Auto-split must still choose correctly after the refactor.

        Silent failure: a threshold regression silently forces every diff into one mode,
        which reads as a styling preference rather than a bug.
        """
        dv = DiffView("f.py", "f.py", _BEFORE, _AFTER)

        assert dv._resolve_split(20) is False
        assert dv._resolve_split(200) is True

    def test_width_zero_is_skipped(self) -> None:
        """At width 0 the computation must be skipped entirely.

        Silent failure: this is the actual cause of the double compose — a width-0 pass
        resolves False, then the real width flips it back to True and the whole diff is
        recomposed.
        """
        dv = DiffView("f.py", "f.py", _BEFORE, _AFTER)
        dv.set_reactive(DiffView.split, True)

        assert dv._resolve_split(0) is None
        dv._check_auto_split(0)

        assert dv.split is True

    async def test_compose_runs_once_when_split_resolved_before_mount(self) -> None:
        """A prepared, pre-resolved DiffView must compose exactly once.

        Silent failure: recomposing a 42KB diff twice costs a second full render pass with
        no visible difference in the result, so only a compose counter can see it.
        """
        view = DiffView("f.py", "f.py", _BEFORE, _AFTER)
        await view.prepare()
        view.resolve_split(120)
        assert view.split is True

        calls = {"n": 0}
        original = DiffView.compose

        def _counting(self: DiffView):
            calls["n"] += 1
            return original(self)

        with patch.object(DiffView, "compose", _counting):
            app = _Host(view, width=120)
            async with app.run_test(headless=True, size=(140, 40)) as pilot:
                await pilot.pause()
                assert calls["n"] == 1
                assert view.split is True

    async def test_content_is_correct_in_both_modes(self) -> None:
        """Both render modes must show the changed lines.

        Silent failure: a split refactor that renders the wrong pane drops the new text
        while still producing a plausible-looking diff widget.
        """
        for split in (True, False):
            view = DiffView("f.py", "f.py", _BEFORE, _AFTER)
            await view.prepare()
            view.set_reactive(DiffView.split, split)
            view.auto_split = False

            app = _Host(view, width=120)
            async with app.run_test(headless=True, size=(140, 40)) as pilot:
                await pilot.pause()
                text = _rendered_text(view)
                assert "beta = 99" in text, f"split={split}"
                assert "beta = 2" in text, f"split={split}"


class TestHighlightPrewarm:
    def test_prewarm_never_raises_when_highlighting_fails(self) -> None:
        """A failed optimisation must not surface.

        This broad guard is the ONLY error handling in the pre-warm, so if it does not hold
        a Pygments failure takes down app startup on some machine we never tested.
        """
        with patch.object(
            diff_view_module.highlight, "highlight", side_effect=RuntimeError("boom")
        ):
            assert prewarm_highlighting() is None

    def test_prewarm_warms_the_calls_the_diff_path_reads(self) -> None:
        """The warm-up must target the entry points DiffView actually uses.

        Silent failure: it warms a different cache than the diff path reads, so it burns
        thread time at every startup and buys nothing — and the perf gate would be the only
        thing that ever noticed.
        """
        with (
            patch.object(
                diff_view_module.highlight, "guess_language", return_value="python"
            ) as guess,
            patch.object(diff_view_module.highlight, "highlight") as hl,
        ):
            prewarm_highlighting()

        assert guess.call_count == 1
        assert hl.call_count == 1
        code, path = guess.call_args.args
        assert code.strip() and path.endswith(".py")
        assert hl.call_args.kwargs["language"] == "python"

    def test_prewarm_has_no_availability_branch(self) -> None:
        """No tree-sitter reference and no ImportError guard.

        The mechanism is Pygments, a hard Textual dependency, so an availability branch would
        be dead code inviting the next reader to re-derive the wrong explanation. The broad
        `except Exception` is expected and is deliberately not what this checks.
        """
        source = inspect.getsource(prewarm_highlighting)
        assert "tree_sitter" not in source
        assert "ImportError" not in source
        assert "ModuleNotFoundError" not in source
        module_source = pathlib.Path(diff_view_module.__file__).read_text()
        assert "tree_sitter" not in module_source
