"""Tests for LaunchAgentScreen modal."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from textual.app import App, ComposeResult
from textual.widgets import Button

from synth_acp.models.agent import AgentState
from synth_acp.ui.screens.launch import _RUNNING_STATES, LaunchAgentScreen


class TestLaunchScreen:
    def test_launch_screen_when_agent_selected_dismisses_with_id(self) -> None:
        """Pressing a terminated agent's button dismisses with that agent's ID."""
        agents = [
            ("agent-1", "Agent 1", AgentState.TERMINATED),
            ("agent-2", "Agent 2", AgentState.BUSY),
        ]
        screen = LaunchAgentScreen(agents)

        event = MagicMock()
        event.button.id = "launch-agent-1"

        with patch.object(screen, "dismiss") as mock_dismiss:
            screen.on_button_pressed(event)

        mock_dismiss.assert_called_once_with("agent-1")

    def test_launch_screen_when_running_agent_shown_disabled(self) -> None:
        """Running states are classified correctly — all active states disabled, others not."""
        assert {
            AgentState.INITIALIZING,
            AgentState.IDLE,
            AgentState.BUSY,
            AgentState.AWAITING_PERMISSION,
        } == _RUNNING_STATES
        assert AgentState.TERMINATED not in _RUNNING_STATES
        assert AgentState.UNSTARTED not in _RUNNING_STATES

    def test_launch_screen_when_escape_pressed_dismisses_none(self) -> None:
        """Escape action dismisses with None."""
        agents = [("agent-1", "Agent 1", AgentState.IDLE)]
        screen = LaunchAgentScreen(agents)

        with patch.object(screen, "dismiss") as mock_dismiss:
            screen.action_dismiss_none()

        mock_dismiss.assert_called_once_with(None)


class TestLaunchScreenDynamicAgents:
    async def test_launch_screen_when_dynamic_agents_exist_shows_them(self) -> None:
        """Both config-defined and dynamic agents appear as buttons."""
        agents = [
            ("config-agent", "Config Agent", AgentState.TERMINATED),
            ("dynamic-agent", "dynamic-agent", AgentState.TERMINATED),
        ]

        class ShellApp(App):
            def compose(self) -> ComposeResult:
                yield Button("open", id="open-btn")

        app = ShellApp()
        async with app.run_test(headless=True, size=(120, 40)):
            screen = LaunchAgentScreen(agents)
            await app.push_screen(screen)
            assert screen.query_one("#launch-config-agent", Button)
            assert screen.query_one("#launch-dynamic-agent", Button)
