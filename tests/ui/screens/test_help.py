"""Tests for HelpScreen modal."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

from textual.binding import Binding

from synth_acp.ui.screens.help import HelpScreen


class TestHelpScreen:
    def test_help_screen_when_bindings_change_reflects_new_bindings(self) -> None:
        """Bindings text is built dynamically from app.BINDINGS, not hardcoded."""
        screen = HelpScreen()

        mock_app = MagicMock()
        mock_app.BINDINGS = [
            Binding("x", "do_x", "Do X"),
            Binding("y", "do_y", "Do Y"),
            Binding("z", "do_z", "Do Z"),
        ]

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            text = screen._build_bindings_text()

        # All 3 mock bindings present
        for key, desc in [("x", "Do X"), ("y", "Do Y"), ("z", "Do Z")]:
            assert key in text
            assert desc in text

        # Count binding lines (non-header lines with content)
        binding_lines = [ln for ln in text.splitlines() if ln.startswith("  ")]
        assert len(binding_lines) == 3
