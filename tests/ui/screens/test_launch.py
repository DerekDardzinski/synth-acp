"""Tests for LaunchAgentScreen modal."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from textual.widgets import Input, Select

from synth_acp.models.agent import AgentConfig
from synth_acp.ui.screens.launch import LaunchAgentScreen, _detect_harnesses


class TestLaunchScreen:
    def test_launch_screen_when_escape_pressed_dismisses_none(self) -> None:
        """Escape action dismisses with None."""
        screen = LaunchAgentScreen()
        with patch.object(screen, "dismiss") as mock_dismiss:
            screen.action_dismiss_none()
        mock_dismiss.assert_called_once_with(None)

    def test_launch_screen_when_missing_agent_id_notifies_warning(self) -> None:
        """Submit with empty agent_id shows a warning notification."""
        screen = LaunchAgentScreen()

        mock_select = MagicMock(spec=Select)
        mock_select.value = "kiro"
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "  "
        mock_mode_input = MagicMock(spec=Input)
        mock_mode_input.value = ""
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp"

        with (
            patch.object(
                screen,
                "query_one",
                side_effect=[mock_select, mock_id_input, mock_mode_input, mock_cwd_input],
            ),
            patch.object(screen, "notify") as mock_notify,
            patch.object(screen, "dismiss") as mock_dismiss,
        ):
            event = MagicMock()
            event.button.id = "launch-submit"
            screen.on_button_pressed(event)

        mock_notify.assert_called_once()
        assert "required" in mock_notify.call_args[0][0].lower()
        mock_dismiss.assert_not_called()

    def test_launch_screen_when_no_harness_selected_notifies_warning(self) -> None:
        """Submit with blank harness shows a warning notification."""
        screen = LaunchAgentScreen()

        mock_select = MagicMock(spec=Select)
        mock_select.value = Select.BLANK
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "my-agent"
        mock_mode_input = MagicMock(spec=Input)
        mock_mode_input.value = ""
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp"

        with (
            patch.object(
                screen,
                "query_one",
                side_effect=[mock_select, mock_id_input, mock_mode_input, mock_cwd_input],
            ),
            patch.object(screen, "notify") as mock_notify,
            patch.object(screen, "dismiss") as mock_dismiss,
        ):
            event = MagicMock()
            event.button.id = "launch-submit"
            screen.on_button_pressed(event)

        mock_notify.assert_called_once()
        mock_dismiss.assert_not_called()

    def test_launch_screen_when_valid_form_dismisses_with_config(self) -> None:
        """Valid form submission dismisses with an AgentConfig."""
        screen = LaunchAgentScreen()

        mock_select = MagicMock(spec=Select)
        mock_select.value = "kiro"
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "my-agent"
        mock_mode_input = MagicMock(spec=Input)
        mock_mode_input.value = "code"
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp/project"

        with (
            patch.object(
                screen,
                "query_one",
                side_effect=[mock_select, mock_id_input, mock_mode_input, mock_cwd_input],
            ),
            patch.object(screen, "dismiss") as mock_dismiss,
        ):
            event = MagicMock()
            event.button.id = "launch-submit"
            screen.on_button_pressed(event)

        mock_dismiss.assert_called_once()
        config = mock_dismiss.call_args[0][0]
        assert isinstance(config, AgentConfig)
        assert config.agent_id == "my-agent"
        assert config.harness == "kiro"
        assert config.agent_mode == "code"


class TestDetectHarnesses:
    def test_detect_harnesses_filters_by_path(self) -> None:
        """Only harnesses with binaries in PATH are returned."""
        with patch("synth_acp.ui.screens.launch.shutil.which") as mock_which:
            mock_which.side_effect = lambda b: "/usr/bin/kiro-cli" if b == "kiro-cli" else None
            result = _detect_harnesses()
        names = [h.short_name for h in result]
        assert "kiro" in names
        assert "claude" not in names
