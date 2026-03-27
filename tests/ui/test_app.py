"""Tests for SynthApp: event routing, panel switching, modals, loading states."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.worker import WorkerState

from synth_acp.models.agent import AgentState
from synth_acp.models.commands import LaunchAgent
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerEvent,
    MessageChunkReceived,
    UsageUpdated,
)
from synth_acp.ui.app import SynthApp
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.gradient_bar import ActivityBar
from synth_acp.ui.widgets.message_queue import MessageQueue


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        project="test",
        agents=[{"id": aid, "cmd": ["echo"]} for aid in agent_ids],
    )


def _make_broker(events: list[BrokerEvent] | None = None) -> MagicMock:
    """Create a mock broker with async stubs and optional event iterator."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()

    async def _events():
        for e in events or []:
            yield e

    broker.events = _events
    return broker


def _make_app(*agent_ids: str) -> SynthApp:
    """Create a SynthApp with a mock broker and given agents."""
    return SynthApp(_make_broker(), _make_config(*agent_ids))


# ── Broker event bridge ──


class TestConsumeEvents:
    async def test_consume_broker_events_when_event_emitted_posts_message(self) -> None:
        event = AgentStateChanged(agent_id="a", old_state="idle", new_state="busy")
        broker = _make_broker([event])
        app = SynthApp(broker, _make_config("a"))

        posted: list[BrokerEventMessage] = []
        app.post_message = MagicMock(side_effect=lambda m: posted.append(m))  # type: ignore[method-assign]

        await app._consume_broker_events()

        assert len(posted) == 1
        assert isinstance(posted[0], BrokerEventMessage)
        assert posted[0].event is event


class TestCLIModeSelection:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_main_when_headless_flag_calls_async_run(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".synth.toml"
        config_file.write_text('project = "s"\n\n[[agents]]\nid = "a"\ncmd = ["echo"]\n')

        with (
            patch("synth_acp.cli.asyncio.run") as mock_run,
            patch(
                "synth_acp.cli.sys.argv",
                ["synth", "-c", str(config_file), "--headless"],
            ),
            pytest.raises(SystemExit, match="0"),
        ):
            from synth_acp.cli import main

            main()

        mock_run.assert_called_once()
        mock_run.call_args[0][0].close()

    def test_main_when_default_calls_tui(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".synth.toml"
        config_file.write_text('project = "s"\n\n[[agents]]\nid = "a"\ncmd = ["echo"]\n')

        with (
            patch("synth_acp.cli._run_tui") as mock_tui,
            patch(
                "synth_acp.cli.sys.argv",
                ["synth", "-c", str(config_file)],
            ),
            pytest.raises(SystemExit, match="0"),
        ):
            from synth_acp.cli import main

            main()

        mock_tui.assert_called_once()


# ── Event routing ──


class TestRouteEventThought:
    async def test_route_event_when_thought_received_adds_to_feed(self) -> None:
        """AgentThoughtReceived routes to feed.add_thought_chunk."""
        app = _make_app("a")
        feed = MagicMock()
        feed.add_thought_chunk = AsyncMock()
        event = AgentThoughtReceived(agent_id="a", chunk="text")

        await app._route_event_to_feed(feed, event)

        feed.add_thought_chunk.assert_called_once_with("text")


class TestRouteEventUsage:
    async def test_route_event_when_usage_updated_calls_handler(self) -> None:
        """UsageUpdated routes to _update_usage_display."""
        app = _make_app("a")
        feed = MagicMock()
        event = UsageUpdated(agent_id="a", size=128000, used=32000, cost_amount=0.14)

        with patch.object(app, "_update_usage_display") as mock_handler:
            await app._route_event_to_feed(feed, event)

        mock_handler.assert_called_once_with(event)


class TestWorkerErrorHandling:
    def test_worker_state_changed_when_error_notifies_and_restarts(self) -> None:
        """Broker consumer error triggers notification and restart."""
        app = _make_app("a")

        mock_worker = MagicMock()
        mock_worker.name = "broker-consumer"
        mock_worker.error = RuntimeError("test")

        mock_event = MagicMock()
        mock_event.worker = mock_worker
        mock_event.state = WorkerState.ERROR

        with (
            patch.object(app, "notify") as mock_notify,
            patch.object(app, "run_worker") as mock_run_worker,
        ):
            app.on_worker_state_changed(mock_event)

        mock_notify.assert_called_once()
        assert mock_notify.call_args.kwargs["severity"] == "error"
        mock_run_worker.assert_called_once()
        assert mock_run_worker.call_args.kwargs["name"] == "broker-consumer"


# ── ContentSwitcher panel switching ──


class TestSelectAgentContentSwitcher:
    async def test_select_agent_when_first_visit_mounts_feed_in_switcher(self) -> None:
        """First visit creates panel, mounts into ContentSwitcher, sets reactive."""
        app = _make_app("agent-1")
        app._event_buffers["agent-1"] = []

        mock_switcher = AsyncMock()
        with (
            patch.object(app, "query_one", return_value=mock_switcher),
            patch.object(app, "watch_selected_agent"),
        ):
            await app.select_agent("agent-1")

        assert "agent-1" in app._panels
        mock_switcher.mount.assert_called_once()
        assert app.selected_agent == "agent-1"

    async def test_select_agent_when_revisit_sets_switcher_current(self) -> None:
        """Revisit skips mount, just sets the reactive (watcher handles switch)."""
        app = _make_app("agent-1")
        app._event_buffers["agent-1"] = []

        mock_switcher = AsyncMock()
        with (
            patch.object(app, "query_one", return_value=mock_switcher),
            patch.object(app, "watch_selected_agent"),
        ):
            await app.select_agent("agent-1")
            mock_switcher.reset_mock()
            await app.select_agent("agent-1")

        mock_switcher.mount.assert_not_called()
        assert app.selected_agent == "agent-1"

    async def test_select_agent_when_buffered_events_drains_before_switch(self) -> None:
        """Buffered events are drained via _replay_event before reactive is set."""
        app = _make_app("agent-1")
        events = [
            MessageChunkReceived(agent_id="agent-1", chunk="hello"),
            MessageChunkReceived(agent_id="agent-1", chunk=" world"),
        ]
        app._event_buffers["agent-1"] = list(events)

        mock_switcher = AsyncMock()
        replayed: list[object] = []

        async def _track_replay(feed, event):  # type: ignore[no-untyped-def]
            replayed.append(event)

        with (
            patch.object(app, "query_one", return_value=mock_switcher),
            patch.object(app, "_replay_event", side_effect=_track_replay),
            patch.object(app, "watch_selected_agent"),
        ):
            await app.select_agent("agent-1")

        assert replayed == events
        assert app._event_buffers["agent-1"] == []


class TestWatchSelectedAgent:
    def test_watch_selected_agent_when_empty_string_skips_switch(self) -> None:
        """Empty string guard prevents crash on initial reactive value."""
        app = _make_app("agent-1")
        mock_query_one = MagicMock()
        with patch.object(app, "query_one", mock_query_one):
            app.watch_selected_agent("")

        mock_query_one.assert_not_called()


class TestShowMessagesContentSwitcher:
    async def test_show_messages_when_first_call_mounts_mcp_panel(self) -> None:
        """First call creates MessageQueue with id='messages' and mounts it."""
        app = _make_app("agent-1")

        mock_switcher = SimpleNamespace(current=None, mount=AsyncMock())
        with (
            patch.object(app, "query_one", return_value=mock_switcher),
            patch.object(app, "query", return_value=[]),
        ):
            await app.show_messages()

        assert app._mcp_panel is not None
        assert isinstance(app._mcp_panel, MessageQueue)
        mock_switcher.mount.assert_called_once()
        mounted = mock_switcher.mount.call_args[0][0]
        assert mounted.id == "messages"
        assert mock_switcher.current == "messages"


# ── Modal screens ──


class TestActionLaunchModal:
    async def test_action_launch_when_modal_returns_id_sends_launch_command(self) -> None:
        """Selecting an agent in the modal triggers broker.handle(LaunchAgent(...))."""
        app = _make_app("agent-1")

        with patch.object(app, "push_screen_wait", new_callable=AsyncMock, return_value="agent-1"):
            await app._do_launch()

        app.broker.handle.assert_called_once_with(LaunchAgent(agent_id="agent-1"))

    async def test_action_launch_when_modal_returns_none_skips_launch(self) -> None:
        """Escape from modal (None result) does not call broker.handle."""
        app = _make_app("agent-1")

        with patch.object(app, "push_screen_wait", new_callable=AsyncMock, return_value=None):
            await app._do_launch()

        app.broker.handle.assert_not_called()


# ── ActivityBar ──


class TestActivityBar:
    async def test_activity_bar_inactive_when_agent_idle(self) -> None:
        """InputBar's ActivityBar is inactive when agent is idle."""
        app = _make_app("agent-1")

        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]
            bar = feed.input_bar.query_one(ActivityBar)
            assert bar.active is False

    async def test_set_busy_false_deactivates_activity_bar(self) -> None:
        """set_busy(False) sets ActivityBar.active to False."""
        app = _make_app("agent-1")

        async with app.run_test(headless=True, size=(120, 40)):
            app._agent_states["agent-1"] = AgentState.INITIALIZING
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]
            bar = feed.input_bar.query_one(ActivityBar)

            event = AgentStateChanged(
                agent_id="agent-1",
                old_state=AgentState.INITIALIZING,
                new_state=AgentState.IDLE,
            )
            await app.on_broker_event_message(BrokerEventMessage(event))
            assert bar.active is False

    async def test_set_busy_true_activates_activity_bar(self) -> None:
        """set_busy(True) sets ActivityBar.active to True."""
        app = _make_app("agent-1")

        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]
            bar = feed.input_bar.query_one(ActivityBar)

            busy_event = AgentStateChanged(
                agent_id="agent-1",
                old_state=AgentState.IDLE,
                new_state=AgentState.BUSY,
            )
            await app.on_broker_event_message(BrokerEventMessage(busy_event))
            assert bar.active is True

    async def test_route_event_idle_calls_set_busy_false(self) -> None:
        """AgentStateChanged(IDLE) calls set_busy(False) on input_bar."""
        app = _make_app("a")
        feed = MagicMock()
        feed.input_bar = MagicMock()
        event = AgentStateChanged(
            agent_id="a", old_state=AgentState.INITIALIZING, new_state=AgentState.IDLE
        )

        await app._route_event_to_feed(feed, event)
        feed.input_bar.set_busy.assert_called_once_with(False)


class TestReplayEventSkipsSpinner:
    async def test_replay_event_when_state_changed_skips(self) -> None:
        """_replay_event does not call spinner logic for AgentStateChanged."""
        app = _make_app("a")
        feed = MagicMock()
        feed.add_chunk = AsyncMock()
        feed.add_thought_chunk = AsyncMock()
        feed.finalize_current_message = AsyncMock()
        event = AgentStateChanged(
            agent_id="a", old_state=AgentState.INITIALIZING, new_state=AgentState.IDLE
        )

        await app._replay_event(feed, event)

        feed.query_one.assert_not_called()
