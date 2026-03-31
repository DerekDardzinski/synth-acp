"""Tests for DiffView widget."""

from __future__ import annotations

from synth_acp.ui.widgets.diff_view import DiffView, fill_lists, loop_last


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
