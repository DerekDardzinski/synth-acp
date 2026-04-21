"""Tests for input_bar helper functions."""

from __future__ import annotations

from pathlib import Path

from synth_acp.ui.widgets.input_bar import _short_path


class TestShortPath:
    def test_short_path_when_relative_dot_resolves_to_absolute(self) -> None:
        """Relative '.' must resolve, never appear as literal '.' in the UI."""
        result = _short_path(".")
        assert result != "."
        assert result.startswith("~/") or Path(result).is_absolute()

    def test_short_path_when_home_dir_returns_tilde(self) -> None:
        """Home directory must display as '~', not '~/.'."""
        result = _short_path(str(Path.home()))
        assert result == "~"
