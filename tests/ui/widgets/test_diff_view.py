"""Tests for DiffView widget's _unified_diff_markup function."""

from __future__ import annotations

from synth_acp.ui.widgets.diff_view import _unified_diff_markup


class TestUnifiedDiffMarkup:
    """Tests for _unified_diff_markup pure function."""

    def test_unified_diff_markup_when_lines_added_contains_success_style(self) -> None:
        result = _unified_diff_markup("f.py", "a", "a\nb")
        assert "[$success]" in result

    def test_unified_diff_markup_when_lines_removed_contains_error_style(self) -> None:
        result = _unified_diff_markup("f.py", "a\nb", "a")
        assert "[$error]" in result

    def test_unified_diff_markup_when_no_changes_shows_placeholder(self) -> None:
        result = _unified_diff_markup("f.py", "same", "same")
        assert "(no changes)" in result
