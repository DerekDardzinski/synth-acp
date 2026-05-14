"""Tests for LaunchAgentScreen modal."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from textual.widgets import Input, OptionList, Select

from synth_acp.discovery import DiscoveredAgent
from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import HarnessEntry
from synth_acp.ui.screens.launch import LaunchAgentScreen, _detect_harnesses


def _make_harness(short_name: str = "kiro", identity: str = "kiro") -> HarnessEntry:
    return HarnessEntry(
        identity=identity,
        name="Kiro CLI",
        short_name=short_name,
        binary_names=["kiro-cli"],
        run_cmd="kiro-cli acp",
    )


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

        mock_harness_select = MagicMock(spec=Select)
        mock_harness_select.value = "kiro"
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "  "
        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = ""
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp"

        def query_one_side_effect(selector, widget_type=None):
            mapping = {
                "#harness-select": mock_harness_select,
                "#agent-id-input": mock_id_input,
                "#agent-filter-input": mock_filter_input,
                "#cwd-input": mock_cwd_input,
            }
            return mapping[selector]

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
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

        mock_harness_select = MagicMock(spec=Select)
        mock_harness_select.value = Select.BLANK
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "my-agent"
        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = ""
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp"

        def query_one_side_effect(selector, widget_type=None):
            mapping = {
                "#harness-select": mock_harness_select,
                "#agent-id-input": mock_id_input,
                "#agent-filter-input": mock_filter_input,
                "#cwd-input": mock_cwd_input,
            }
            return mapping[selector]

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
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
        screen._selected_agent = "code-planner"

        mock_harness_select = MagicMock(spec=Select)
        mock_harness_select.value = "kiro"
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "my-agent"
        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = "code-planner"
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp/project"

        def query_one_side_effect(selector, widget_type=None):
            mapping = {
                "#harness-select": mock_harness_select,
                "#agent-id-input": mock_id_input,
                "#agent-filter-input": mock_filter_input,
                "#cwd-input": mock_cwd_input,
            }
            return mapping[selector]

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
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
        assert config.agent_mode == "code-planner"


class TestHarnessChange:
    def test_harness_change_populates_agent_list(self) -> None:
        """Harness selection triggers discovery and populates agent OptionList."""
        screen = LaunchAgentScreen()
        screen._harnesses = [_make_harness()]

        agents = [
            DiscoveredAgent(qualified_name="plan", name="plan", description="", source="user"),
            DiscoveredAgent(qualified_name="code", name="code", description="", source="user"),
        ]

        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = ""
        mock_option_list = MagicMock(spec=OptionList)

        def query_one_side_effect(selector, widget_type=None):
            if selector == "#agent-filter-input":
                return mock_filter_input
            if selector == "#agent-option-list":
                return mock_option_list
            raise ValueError(f"Unexpected selector: {selector}")

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
            patch("synth_acp.ui.screens.launch.discover_agents", return_value=agents),
        ):
            event = MagicMock()
            event.select.id = "harness-select"
            event.value = "kiro"
            screen.on_select_changed(event)

        assert screen._agents == agents
        assert mock_filter_input.display is True
        assert mock_option_list.display is True

    def test_harness_change_no_agents_shows_filter_input(self) -> None:
        """No agents discovered shows filter input for manual entry."""
        screen = LaunchAgentScreen()
        screen._harnesses = [_make_harness()]

        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = ""
        mock_option_list = MagicMock(spec=OptionList)

        def query_one_side_effect(selector, widget_type=None):
            if selector == "#agent-filter-input":
                return mock_filter_input
            if selector == "#agent-option-list":
                return mock_option_list
            raise ValueError(f"Unexpected selector: {selector}")

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
            patch("synth_acp.ui.screens.launch.discover_agents", return_value=[]),
        ):
            event = MagicMock()
            event.select.id = "harness-select"
            event.value = "kiro"
            screen.on_select_changed(event)

        assert mock_filter_input.display is True
        assert mock_option_list.display is False


class TestSubmitAgentMode:
    def test_submit_uses_selected_agent_as_agent_mode(self) -> None:
        """Selected agent qualified_name becomes agent_mode."""
        screen = LaunchAgentScreen()
        screen._selected_agent = "local-plugin:code-planner"

        mock_harness_select = MagicMock(spec=Select)
        mock_harness_select.value = "kiro"
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "my-agent"
        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = "code-planner"
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp"

        def query_one_side_effect(selector, widget_type=None):
            mapping = {
                "#harness-select": mock_harness_select,
                "#agent-id-input": mock_id_input,
                "#agent-filter-input": mock_filter_input,
                "#cwd-input": mock_cwd_input,
            }
            return mapping[selector]

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
            patch.object(screen, "dismiss") as mock_dismiss,
        ):
            event = MagicMock()
            event.button.id = "launch-submit"
            screen.on_button_pressed(event)

        config = mock_dismiss.call_args[0][0]
        assert config.agent_mode == "local-plugin:code-planner"

    def test_submit_uses_filter_input_when_no_selection(self) -> None:
        """Raw filter input value used as agent_mode when nothing selected."""
        screen = LaunchAgentScreen()
        screen._selected_agent = None

        mock_harness_select = MagicMock(spec=Select)
        mock_harness_select.value = "kiro"
        mock_id_input = MagicMock(spec=Input)
        mock_id_input.value = "my-agent"
        mock_filter_input = MagicMock(spec=Input)
        mock_filter_input.value = "custom-mode"
        mock_cwd_input = MagicMock(spec=Input)
        mock_cwd_input.value = "/tmp"

        def query_one_side_effect(selector, widget_type=None):
            mapping = {
                "#harness-select": mock_harness_select,
                "#agent-id-input": mock_id_input,
                "#agent-filter-input": mock_filter_input,
                "#cwd-input": mock_cwd_input,
            }
            return mapping[selector]

        with (
            patch.object(screen, "query_one", side_effect=query_one_side_effect),
            patch.object(screen, "dismiss") as mock_dismiss,
        ):
            event = MagicMock()
            event.button.id = "launch-submit"
            screen.on_button_pressed(event)

        config = mock_dismiss.call_args[0][0]
        assert config.agent_mode == "custom-mode"


class TestDetectHarnesses:
    def test_detect_harnesses_filters_by_path(self) -> None:
        """Only harnesses with binaries in PATH are returned."""
        with patch("synth_acp.ui.screens.launch.shutil.which") as mock_which:
            mock_which.side_effect = lambda b: "/usr/bin/kiro-cli" if b == "kiro-cli" else None
            result = _detect_harnesses()
        names = [h.short_name for h in result]
        assert "kiro" in names
        assert "claude" not in names
