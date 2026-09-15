"""Tests for SynthApp: event routing, panel switching, modals, loading states."""

from __future__ import annotations

import ast
import asyncio
import pathlib
import sqlite3
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import ContentSwitcher
from textual.worker import WorkerState

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.commands import LaunchAgent
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentHandedOff,
    AgentStateChanged,
    AgentThoughtReceived,
    AvailableCommandsReceived,
    BrokerEvent,
    MessageChunkReceived,
    SessionRestoreComplete,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
    UserPromptSubmitted,
)
from synth_acp.ui import app as synth_app_module
from synth_acp.ui.app import DynamicAgentInfo, SynthApp, _coalesce_events
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets import conversation as conversation_module, diff_view as diff_view_module
from synth_acp.ui.widgets.agent_list import AgentList
from synth_acp.ui.widgets.agent_message import AgentMessage
from synth_acp.ui.widgets.conversation import RESTORE_BATCH, ConversationFeed
from synth_acp.ui.widgets.gradient_bar import ActivityBar
from synth_acp.ui.widgets.prompt_bubble import PromptBubble
from tests import conftest as harness


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        project="test",
    )


def _make_broker(
    events: list[BrokerEvent] | None = None, agent_ids: list[str] | None = None
) -> MagicMock:
    """Create a mock broker with async stubs and optional event iterator."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    # Default initial agent
    first_id = (agent_ids or ["agent-1"])[0] if agent_ids else "agent-1"
    broker._initial_agent = AgentConfig(agent_id=first_id, harness="kiro")
    broker.get_agent_parent = MagicMock(return_value=None)
    broker.get_agent_harness = MagicMock(return_value="kiro")
    broker.get_agent_cwd = MagicMock(return_value=".")
    broker.get_agent_display_name = MagicMock(return_value=None)
    broker.get_usage = MagicMock(return_value=None)

    async def _events():
        for e in events or []:
            yield e

    broker.events = _events
    return broker


def _make_app(*agent_ids: str) -> SynthApp:
    """Create a SynthApp with a mock broker and given agents."""
    ids = list(agent_ids) if agent_ids else ["agent-1"]
    broker = _make_broker(agent_ids=ids)
    initial_agent = AgentConfig(agent_id=ids[0], harness="kiro")
    return SynthApp(broker, _make_config(*agent_ids), initial_agent=initial_agent)


# ── Broker event bridge ──


class TestConsumeEvents:
    async def test_consume_broker_events_when_event_emitted_posts_message(self) -> None:
        event = AgentStateChanged(agent_id="a", old_state="idle", new_state="busy")
        broker = _make_broker([event], agent_ids=["a"])
        app = SynthApp(broker, _make_config("a"))

        posted: list[BrokerEventMessage] = []
        app.post_message = MagicMock(side_effect=posted.append)  # type: ignore[method-assign]

        await app._consume_broker_events()

        assert len(posted) == 1
        assert isinstance(posted[0], BrokerEventMessage)
        assert posted[0].event is event


class TestCLIModeSelection:
    def test_main_when_default_calls_tui(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".synth.json"
        config_file.write_text('{"project": "s", "agents": [{"agent_id": "a", "harness": "kiro"}]}')

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
    async def test_usage_updated_dispatched_in_pre_routing(self) -> None:
        """UsageUpdated fires _update_usage_display in pre-routing (no panel needed)."""
        app = _make_app("a")
        app._agent_states["a"] = AgentState.IDLE
        app._dynamic_agents["a"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
        event = UsageUpdated(agent_id="a", size=128000, used=32000, cost_amount=0.14)

        with patch.object(app, "_update_usage_display") as mock_handler:
            await app.on_broker_event_message(BrokerEventMessage(event))

        mock_handler.assert_called_once_with(event)


class TestFormatCost:
    def test_format_cost_usd(self) -> None:
        """USD amounts get dollar-sign prefix."""
        app = _make_app("a")
        assert app._format_cost(1.23, "USD") == "$1.23"

    def test_format_cost_usd_case_insensitive(self) -> None:
        """USD matching is case-insensitive."""
        app = _make_app("a")
        assert app._format_cost(0.50, "usd") == "$0.50"

    def test_format_cost_other_currency(self) -> None:
        """Non-USD currencies appear as suffix."""
        app = _make_app("a")
        assert app._format_cost(1.23, "EUR") == "1.23 EUR"

    def test_format_cost_none_amount(self) -> None:
        """None amount returns empty string."""
        app = _make_app("a")
        assert app._format_cost(None, None) == ""

    def test_format_cost_no_currency(self) -> None:
        """Amount without currency returns just the formatted number."""
        app = _make_app("a")
        assert app._format_cost(1.23, None) == "1.23"


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


class TestWatchSelectedAgent:
    def test_watch_selected_agent_when_empty_string_skips_switch(self) -> None:
        """Empty string guard prevents crash on initial reactive value."""
        app = _make_app("agent-1")
        mock_query_one = MagicMock()
        with patch.object(app, "query_one", mock_query_one):
            app.watch_selected_agent("", "")

        mock_query_one.assert_not_called()


# ── Modal screens ──


class TestActionLaunchModal:
    async def test_action_launch_when_modal_returns_config_sends_launch_command(self) -> None:
        """Selecting an agent in the modal triggers broker.handle(LaunchAgent(...))."""
        app = _make_app("agent-1")
        config = AgentConfig(agent_id="new-agent", harness="kiro")

        with (
            patch.object(app, "push_screen_wait", new_callable=AsyncMock, return_value=config),
            patch.object(app, "select_agent", new_callable=AsyncMock) as mock_select,
        ):
            await app._do_launch()

        app.broker.handle.assert_called_once_with(LaunchAgent(agent_id="new-agent", config=config))
        mock_select.assert_called_once_with("new-agent")

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
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
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
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
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


class TestCoalesceEvents:
    def test_consecutive_message_chunks_merged(self) -> None:
        """Consecutive MCR events with same agent_id merge into one."""
        events: list[BrokerEvent] = [
            MessageChunkReceived(agent_id="a", chunk="x"),
            MessageChunkReceived(agent_id="a", chunk="y"),
            ToolCallUpdated(
                agent_id="a", tool_call_id="t1", title="t", kind="read", status="completed"
            ),
            MessageChunkReceived(agent_id="a", chunk="z"),
        ]
        result = _coalesce_events(events)
        assert len(result) == 3
        assert isinstance(result[0], MessageChunkReceived)
        assert result[0].chunk == "xy"
        assert isinstance(result[1], ToolCallUpdated)
        assert isinstance(result[2], MessageChunkReceived)
        assert result[2].chunk == "z"

    def test_consecutive_thought_chunks_merged(self) -> None:
        """Consecutive ATR events with same agent_id merge into one."""
        events: list[BrokerEvent] = [
            AgentThoughtReceived(agent_id="a", chunk="p"),
            AgentThoughtReceived(agent_id="a", chunk="q"),
            MessageChunkReceived(agent_id="a", chunk="r"),
        ]
        result = _coalesce_events(events)
        assert len(result) == 2
        assert isinstance(result[0], AgentThoughtReceived)
        assert result[0].chunk == "pq"
        assert isinstance(result[1], MessageChunkReceived)
        assert result[1].chunk == "r"

    def test_empty_buffer_returns_empty(self) -> None:
        assert _coalesce_events([]) == []

    def test_different_agent_ids_not_merged(self) -> None:
        """MCR events with different agent_ids stay separate."""
        events: list[BrokerEvent] = [
            MessageChunkReceived(agent_id="a", chunk="x"),
            MessageChunkReceived(agent_id="b", chunk="y"),
        ]
        result = _coalesce_events(events)
        assert len(result) == 2
        assert result[0].chunk == "x"  # type: ignore[union-attr]
        assert result[1].chunk == "y"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Race condition reproducers: select_agent reentrancy and drain serialization
# ---------------------------------------------------------------------------


class TestSelectAgentReentrancy:
    """Verify that concurrent select_agent calls don't create duplicate panels."""

    async def test_concurrent_select_agent_same_id_no_duplicate(self) -> None:
        """Two concurrent select_agent calls for the same agent_id create only one panel."""
        app = _make_app()
        app._event_buffers["agent-1"] = []
        app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

        async with app.run_test(headless=True, size=(120, 40)):
            t1 = asyncio.ensure_future(app.select_agent("agent-1"))
            t2 = asyncio.ensure_future(app.select_agent("agent-1"))
            await asyncio.gather(t1, t2)

            assert "agent-1" in app._panels
            assert len([k for k in app._panels if k == "agent-1"]) == 1


class TestDrainSerialization:
    """Verify that events arriving during drain are buffered and replayed in order."""

    async def test_select_agent_drain_races_with_live_event_routing(self) -> None:
        """Events arriving during drain are buffered and replayed in order."""
        app = _make_app()
        app._event_buffers["agent-2"] = [
            MessageChunkReceived(agent_id="agent-2", chunk="first"),
        ]
        app._dynamic_agents["agent-2"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

        chunks_received: list[str] = []
        original_replay = app._replay_event

        async def _tracking_replay(feed, event):
            if isinstance(event, MessageChunkReceived):
                chunks_received.append(event.chunk)
                if event.chunk == "first" and "agent-2" in app._draining:
                    app._event_buffers.setdefault("agent-2", []).append(
                        MessageChunkReceived(agent_id="agent-2", chunk="mid-drain")
                    )
            await original_replay(feed, event)

        async with app.run_test(headless=True, size=(120, 40)):
            with patch.object(app, "_replay_event", side_effect=_tracking_replay):
                await app.select_agent("agent-2")

            assert "first" in chunks_received
            assert "mid-drain" in chunks_received
            assert chunks_received.index("first") < chunks_received.index("mid-drain")

    async def test_live_event_during_drain_is_buffered(self) -> None:
        """on_broker_event_message buffers events when agent is draining."""
        app = _make_app()
        app._event_buffers["agent-1"] = []
        app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("agent-1")

            app._draining["agent-1"] = asyncio.Event()
            app._event_buffers["agent-1"] = []

            event = MessageChunkReceived(agent_id="agent-1", chunk="during-drain")
            await app.on_broker_event_message(BrokerEventMessage(event))

            assert event in app._event_buffers["agent-1"]
            app._draining.pop("agent-1", None)


class TestDisabledStates:
    """Verify _DISABLED_STATES only contains AWAITING_PERMISSION."""

    def test_disabled_states_only_awaiting_permission(self) -> None:
        """_DISABLED_STATES reduced to {AWAITING_PERMISSION} only."""
        from synth_acp.ui.app import _DISABLED_STATES

        assert {AgentState.AWAITING_PERMISSION} == _DISABLED_STATES


class TestConfigOptionsHandling:
    """Tests for ConfigOptionsReceived and ConfigOptionChanged event handling."""

    def _make_select_option(
        self,
        opt_id: str,
        name: str,
        category: str | None,
        current_value: str,
        options: list[tuple[str, str]],
    ):
        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        return SessionConfigOptionSelect(
            id=opt_id,
            name=name,
            category=category,
            type="select",
            current_value=current_value,
            options=[SessionConfigSelectOption(value=v, name=n) for v, n in options],
        )

    async def test_config_options_received_stores_and_updates_bar(self) -> None:
        """ConfigOptionsReceived stores options and calls input_bar.update_config_options."""
        from synth_acp.models.events import ConfigOptionsReceived
        from synth_acp.ui.messages import BrokerEventMessage

        app = _make_app("agent-1")
        app._agent_states["agent-1"] = AgentState.IDLE

        mode_opt = self._make_select_option(
            "mode", "Mode", "mode", "code", [("code", "Code"), ("plan", "Plan")]
        )
        model_opt = self._make_select_option(
            "model", "Model", "model", "gpt-4", [("gpt-4", "GPT-4")]
        )
        event = ConfigOptionsReceived(agent_id="agent-1", config_options=[mode_opt, model_opt])

        mock_bar = MagicMock()
        mock_feed = MagicMock()
        mock_feed.input_bar = mock_bar
        app._panels["agent-1"] = mock_feed

        mock_tile = MagicMock()
        app._tiles["agent-1"] = mock_tile

        await app.on_broker_event_message(BrokerEventMessage(event))

        assert "agent-1" in app._agent_config_options
        assert len(app._agent_config_options["agent-1"]) == 2
        mock_bar.update_config_options.assert_called_once()
        mock_tile.update_mode.assert_called_once_with("Code")

    async def test_config_option_changed_updates_stored_value(self) -> None:
        """ConfigOptionChanged updates the stored current_value."""
        from synth_acp.models.events import ConfigOptionChanged
        from synth_acp.ui.messages import BrokerEventMessage

        app = _make_app("agent-1")
        app._agent_states["agent-1"] = AgentState.IDLE

        mode_opt = self._make_select_option(
            "mode", "Mode", "mode", "code", [("code", "Code"), ("plan", "Plan")]
        )
        app._agent_config_options["agent-1"] = [mode_opt]

        mock_bar = MagicMock()
        mock_feed = MagicMock()
        mock_feed.input_bar = mock_bar
        app._panels["agent-1"] = mock_feed
        app._tiles["agent-1"] = MagicMock()

        event = ConfigOptionChanged(agent_id="agent-1", config_id="mode", value="plan")
        await app.on_broker_event_message(BrokerEventMessage(event))

        updated_opt = app._agent_config_options["agent-1"][0]
        assert updated_opt.current_value == "plan"
        mock_bar.update_config_option_value.assert_called_once_with("mode", "plan")

    async def test_terminated_clears_config_options(self) -> None:
        """Termination clears _agent_config_options for the agent."""
        from synth_acp.ui.messages import BrokerEventMessage

        app = _make_app("agent-1")
        app._agent_states["agent-1"] = AgentState.IDLE
        app._agent_config_options["agent-1"] = [
            self._make_select_option("mode", "Mode", "mode", "code", [("code", "Code")])
        ]
        app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

        mock_tile = MagicMock()
        app._tiles["agent-1"] = mock_tile

        mock_feed = MagicMock()
        mock_feed.input_bar = MagicMock()
        mock_feed.remove = AsyncMock()
        app._panels["agent-1"] = mock_feed

        event = AgentStateChanged(
            agent_id="agent-1", old_state=AgentState.IDLE, new_state=AgentState.TERMINATED
        )
        await app.on_broker_event_message(BrokerEventMessage(event))

        assert "agent-1" not in app._agent_config_options


# ── Background indexer ──


class TestIndexSessions:
    def test_index_sessions_skips_when_unavailable(self) -> None:
        """Worker exits immediately when embedding deps not installed."""
        app = _make_app("agent-1")

        with patch("synth_acp.ui.app.embedding_available", return_value=False) as mock_avail:
            app._do_index_sessions()

        mock_avail.assert_called_once()
        assert app._indexing_complete is False
        assert app._embedding_engine is None

    def test_index_sessions_embeds_unembedded_sessions(self, tmp_path: Path) -> None:
        """Worker processes unembedded agents and stores embeddings."""
        app = _make_app("agent-1")
        app.broker._db_path = tmp_path / "test.db"

        mock_embedding = MagicMock()
        mock_embedding.tobytes.return_value = b"\x00" * 1536

        mock_engine = MagicMock()
        mock_engine.embed.return_value = mock_embedding

        with (
            patch("synth_acp.ui.app.embedding_available", return_value=True),
            patch("synth_acp.ui.app.EmbeddingEngine", return_value=mock_engine),
            patch(
                "synth_acp.ui.app.get_unembedded_agents_sync",
                return_value=[("sess-1", "a1"), ("sess-2", "a2")],
            ),
            patch.object(
                SynthApp, "_query_agent_text", return_value="hello world this is a test prompt"
            ),
            patch("synth_acp.ui.app.store_embedding_sync") as mock_store,
        ):
            app._do_index_sessions()

        assert app._indexing_complete is True
        assert app._embedding_engine is mock_engine
        mock_engine.ensure_model.assert_called_once()
        assert mock_store.call_count == 2
        call_args = mock_store.call_args_list[0]
        assert call_args[0][1] == "sess-1"
        assert call_args[0][2] == "a1"
        assert call_args[0][4] == b"\x00" * 1536

    def test_index_sessions_stores_sentinel_for_short_text(self, tmp_path: Path) -> None:
        """Worker stores empty sentinel when _query_agent_text returns None."""
        app = _make_app("agent-1")
        app.broker._db_path = tmp_path / "test.db"

        mock_engine = MagicMock()

        with (
            patch("synth_acp.ui.app.embedding_available", return_value=True),
            patch("synth_acp.ui.app.EmbeddingEngine", return_value=mock_engine),
            patch("synth_acp.ui.app.get_unembedded_agents_sync", return_value=[("sess-1", "a1")]),
            patch.object(SynthApp, "_query_agent_text", return_value=None),
            patch("synth_acp.ui.app.store_embedding_sync") as mock_store,
        ):
            app._do_index_sessions()

        assert app._indexing_complete is True
        mock_engine.embed.assert_not_called()
        mock_store.assert_called_once()
        call_args = mock_store.call_args[0]
        assert call_args[1] == "sess-1"
        assert call_args[2] == "a1"
        assert call_args[3] == ""
        assert call_args[4] == b""

    def test_index_sessions_handles_embed_error(self, tmp_path: Path) -> None:
        """Exception during engine.embed() does not propagate."""
        app = _make_app("agent-1")
        app.broker._db_path = tmp_path / "test.db"

        mock_engine = MagicMock()
        mock_engine.embed.side_effect = RuntimeError("model failed")

        with (
            patch("synth_acp.ui.app.embedding_available", return_value=True),
            patch("synth_acp.ui.app.EmbeddingEngine", return_value=mock_engine),
            patch("synth_acp.ui.app.get_unembedded_agents_sync", return_value=[("sess-1", "a1")]),
            patch.object(
                SynthApp, "_query_agent_text", return_value="hello world this is a test prompt"
            ),
        ):
            # Should not raise
            app._do_index_sessions()

        assert app._indexing_complete is False


# ── Session picker plumbing ──


class TestShowSessionPickerIndexingState:
    async def test_show_session_picker_passes_indexing_state(self) -> None:
        """_show_session_picker passes db_path, engine, and indexing_complete to SessionPickerScreen."""
        app = _make_app("agent-1")
        app.broker._db_path = Path("/tmp/test.db")
        engine = MagicMock()
        app._embedding_engine = engine
        app._indexing_complete = True

        with (
            patch.object(
                ACPBroker, "list_restorable_sessions", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(
                app, "push_screen_wait", new_callable=AsyncMock, return_value=None
            ) as mock_push,
        ):
            await app._show_session_picker(from_startup=False)

        screen = mock_push.call_args[0][0]
        assert screen._db_path == Path("/tmp/test.db")
        assert screen._engine is engine
        assert screen._indexing_complete is True


# ── Record event tracking ──


class TestRecordEvent:
    async def test_route_event_calls_record_event(self) -> None:
        """_route_event_to_feed records renderable events in _current_turn_events."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]
            event = MessageChunkReceived(agent_id="agent-1", chunk="hi")
            await app._route_event_to_feed(feed, event)
            assert event in feed._current_turn_events

    async def test_route_event_skips_non_renderable(self) -> None:
        """_route_event_to_feed does not record non-renderable events."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]
            event = AgentStateChanged(
                agent_id="agent-1", old_state=AgentState.BUSY, new_state=AgentState.IDLE
            )
            await app._route_event_to_feed(feed, event)
            assert feed._current_turn_events == []


class TestSynthesizeAgentOption:
    def test_returns_select_for_meta_agent(self) -> None:
        """Picker appears for Claude Code agents with discovered agents.
        Silent failure: picker never appears."""
        from unittest.mock import MagicMock

        from acp.schema import SessionConfigOptionSelect

        from synth_acp.discovery import DiscoveredAgent

        broker = _make_broker(agent_ids=["agent-1"])
        # Set up registry mock
        registry = MagicMock()
        registry.get_agent_mode_target = MagicMock(return_value="meta_agent")
        registry.get_agent_mode = MagicMock(return_value="plugin-x:code-planner")
        broker._registry = registry

        fake_agents = [
            DiscoveredAgent(
                qualified_name="plugin-x:code-planner",
                name="code-planner",
                description="Plans code",
                source="plugin:plugin-x",
            ),
            DiscoveredAgent(
                qualified_name="reviewer", name="reviewer", description="Reviews", source="user"
            ),
        ]
        broker.get_discovered_agents = MagicMock(return_value=fake_agents)

        app = SynthApp(broker, _make_config("agent-1"))
        result = app._synthesize_agent_option("agent-1")

        assert result is not None
        assert isinstance(result, SessionConfigOptionSelect)
        assert result.id == "agent"
        assert result.name == "Agent"
        assert result.category == "agent"
        assert result.current_value == "plugin-x:code-planner"
        assert len(result.options) == 3
        assert result.options[0].name == "Default"
        assert result.options[0].value == ""
        assert result.options[1].name == "code-planner"
        assert result.options[1].value == "plugin-x:code-planner"
        assert result.options[2].name == "reviewer"
        assert result.options[2].value == "reviewer"

    def test_returns_none_for_non_meta_agent(self) -> None:
        """Non-meta_agent harnesses are unaffected.
        Silent failure: Kiro/OpenCode get a spurious Agent picker."""
        from unittest.mock import MagicMock

        broker = _make_broker(agent_ids=["agent-1"])
        registry = MagicMock()
        registry.get_agent_mode_target = MagicMock(return_value="acp_mode")
        broker._registry = registry

        app = SynthApp(broker, _make_config("agent-1"))
        result = app._synthesize_agent_option("agent-1")

        assert result is None


# ── Runtime subsystems: GC freeze + diagnostics ──


class TestRuntimeSubsystems:
    """Composition of synth_acp.runtime_gc and synth_acp.diagnostics into the app."""

    @staticmethod
    async def _await_groups_drained(app: SynthApp, timeout: float = 3.0) -> list[object]:
        """Wait for every "diag"/"gc" worker to be reaped, returning any stragglers.

        ``WorkerManager._remove_worker`` is registered as the task done-callback, so a
        cancelled worker is discarded from the manager only once its task completes.
        """
        import time

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            remaining = [w for w in app.workers if w.group in {"diag", "gc"}]
            if not remaining:
                return []
            await asyncio.sleep(0.005)
        return [w for w in app.workers if w.group in {"diag", "gc"}]

    async def test_gc_freeze_worker_starts_unconditionally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The freeze worker is a production fix, not diagnostics.

        Silent failure: gating it on SYNTH_DIAG means the largest win in the plan
        never runs for the user, while every instrumented test still passes.
        """
        from synth_acp.runtime_gc import GCFreezeManager

        monkeypatch.delenv("SYNTH_DIAG", raising=False)
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            assert isinstance(app._gc_freeze, GCFreezeManager)
            assert [w.name for w in app.workers if w.group == "gc"] == ["gc-freeze"]
            assert [w for w in app.workers if w.group == "diag"] == []

    async def test_diagnostics_started_when_enabled_and_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handle must be STORED, or on_unmount has nothing to stop."""
        monkeypatch.setenv("SYNTH_DIAG", "1")
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            assert app._diagnostics is not None
            assert [w for w in app.workers if w.group == "diag"] != []

    async def test_diagnostics_absent_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SYNTH_DIAG", raising=False)
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            assert app._diagnostics is None

    async def test_teardown_restores_gc_callbacks_and_leaves_no_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Teardown must return the process to its pre-start state.

        Silent failure: a leaked gc.callbacks hook is process-global, so it survives
        into every later app instance in the same process — retaining state,
        duplicating measurements, and corrupting the baseline that SYNTH_DIAG-off
        code observes. A retained cancelled worker also keeps its captured frames
        and instruments alive, which is why the assertion is "no worker in either
        group at all" rather than "none running".
        """
        import gc as gc_module

        baseline = list(gc_module.callbacks)

        monkeypatch.setenv("SYNTH_DIAG", "1")
        first = _make_app("agent-1")
        async with first.run_test(headless=True, size=(120, 40)):
            assert first._diagnostics is not None
            assert len(gc_module.callbacks) == len(baseline) + 1

        assert gc_module.callbacks == baseline
        assert await self._await_groups_drained(first) == []
        assert first._diagnostics is None
        assert first._gc_freeze is None

        # A SECOND instance in the same PROCESS with diagnostics off must observe the
        # ORIGINAL baseline, proving nothing leaked across app instances. It runs on
        # its own event loop in a plain thread because SynthApp.on_unmount shuts down
        # the running loop's default executor: a same-loop second app would need it
        # (Markdown parses in an executor), and even asyncio.to_thread would fail for
        # the same reason. The criterion is about the process-global gc.callbacks
        # list, so a separate loop tests it faithfully.
        import threading

        monkeypatch.delenv("SYNTH_DIAG", raising=False)
        captured: dict[str, object] = {}

        def _run_second_app() -> None:
            second = _make_app("agent-2")

            async def _body() -> None:
                async with second.run_test(headless=True, size=(120, 40)):
                    captured["during"] = list(gc_module.callbacks)
                    captured["handle"] = second._diagnostics
                captured["after"] = list(gc_module.callbacks)

            try:
                asyncio.run(_body())
            except BaseException as error:  # surfaced below so the test fails loudly
                captured["error"] = error

        thread = threading.Thread(target=_run_second_app)
        thread.start()
        while thread.is_alive():
            await asyncio.sleep(0.01)
        thread.join()

        assert captured.get("error") is None, captured.get("error")
        assert captured["during"] == baseline
        assert captured["handle"] is None
        assert captured["after"] == baseline


# ── tail-first windowing on the first-selection drain ─────────────────────────


def _w_prompt(agent_id: str = "target") -> UserPromptSubmitted:
    return UserPromptSubmitted(agent_id=agent_id, text="p")


def _w_chunk(text: str = "c", agent_id: str = "target") -> MessageChunkReceived:
    return MessageChunkReceived(agent_id=agent_id, chunk=text)


def _w_tc(agent_id: str = "target") -> TurnComplete:
    return TurnComplete(agent_id=agent_id, stop_reason="end_turn")


def _w_turn(index: int, size: int, *, closed: bool = True) -> list[BrokerEvent]:
    """One turn whose chunks are individually identifiable by turn index.

    Chunks are made DISTINCT per turn so batching and scroll-up fidelity can be asserted
    on CONTENT rather than on counts — a count-only assertion passes while a turn is
    silently duplicated or dropped.
    """
    events: list[BrokerEvent] = [_w_prompt()]
    events.extend(_w_chunk(f"t{index}-c{n} ") for n in range(size))
    if closed:
        events.append(_w_tc())
    return events


def _w_feed(sizes: list[int], *, open_size: int | None = None) -> list[BrokerEvent]:
    events: list[BrokerEvent] = []
    for index, size in enumerate(sizes):
        events.extend(_w_turn(index, size))
    if open_size is not None:
        events.extend(_w_turn(len(sizes), open_size, closed=False))
    return events


def _w_tall_feed(turns: int) -> list[BrokerEvent]:
    """A feed whose turns render MANY rows, as real agent output does.

    A turn of short chunks coalesces into one one-line message, which would make the
    viewport-overfill assertion fail for want of content rather than for a real defect. The
    measured real journal renders 429 rows for three turns against a 32-row viewport.
    """
    events: list[BrokerEvent] = []
    for index in range(turns):
        events.append(_w_prompt())
        body = "\n\n".join(f"paragraph {index}-{n} of streamed output" for n in range(6))
        events.append(_w_chunk(body))
        events.append(_w_tc())
    return events


async def _drain(app: SynthApp, events: list[BrokerEvent], *, agent_id: str = "target"):
    """Buffer events for an UNSELECTED agent and run the real first-selection drain."""
    app._dynamic_agents[agent_id] = DynamicAgentInfo(parent=None, task="", harness="kiro", cwd=".")
    app._event_buffers[agent_id] = list(events)
    assert agent_id not in app._panels, "the drain under test must create the panel"
    await app.select_agent(agent_id)
    return app._panels[agent_id]


async def _drain_recorded_stream(
    events: list[BrokerEvent], *, windowed: bool
) -> list[list[BrokerEvent]]:
    """Run ONE drain on its own event loop and return the recorded `_turn_events`.

    Each app needs a fresh loop: `SynthApp.on_unmount` calls
    `loop.shutdown_default_executor()`, so a second app on the same loop fails as soon as
    Textual parses Markdown — which it does for every replayed message.
    """

    async def _run() -> list[list[BrokerEvent]]:
        app = _make_app("dummy")
        limits = {} if windowed else {"FIRST_PAINT_TURNS": 10**9, "FIRST_PAINT_EVENT_BUDGET": 10**9}
        async with app.run_test(headless=True, size=(120, 40)):
            with ExitStack() as stack:
                for name, value in limits.items():
                    stack.enter_context(patch.object(conversation_module, name, value))
                feed = await _drain(app, events)
            return [list(batch) for batch in feed._turn_events]

    return await harness._run_on_fresh_loop(_run)


def _mounted_turns(feed):
    from synth_acp.ui.widgets.conversation import TurnContainer

    assert feed._scroll is not None
    return [c for c in feed._scroll.children if isinstance(c, TurnContainer)]


class TestWindowedFirstSelectionDrain:
    async def test_records_every_renderable_event_but_mounts_only_the_tail(self) -> None:
        """Eight turns are all recorded; only the last three are mounted.

        Silent failure: skipped events never reach `_turn_events`, so scroll-up loses them
        permanently while first paint looks great.
        """
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _drain(app, _w_feed([2] * 8))

            assert len(feed._turn_events) == 8
            assert feed._mounted_start_idx == 5
            assert len(_mounted_turns(feed)) == 3

    async def test_recorded_stream_is_byte_identical_to_an_unwindowed_drain(self) -> None:
        """AC3, the invariant everything else follows from.

        Compares the two PRODUCTION paths directly: one windowed drain and one with the
        window constants raised so the whole feed mounts, each on its own event loop, and
        asserts the complete nested `_turn_events` structures are equal batch for batch and
        event for event. Separate loops are required because `SynthApp.on_unmount` calls
        `loop.shutdown_default_executor()`, so a second app on one loop dies as soon as
        Textual parses Markdown.

        The journal deliberately INTERLEAVES non-renderable events among the renderable ones,
        so the comparison also pins the boundary the planner amended: the `_RENDERABLE_EVENTS`
        filter must be applied identically in the mounted and skipped regions.

        Silent failure: recording diverges between the regions, so a scroll-up shows content
        that never happened or drops content that did, and every count-based assertion in this
        file still passes.
        """
        events: list[BrokerEvent] = []
        for index in range(8):
            events.append(_w_prompt())
            events.append(AvailableCommandsReceived(agent_id="target", commands=[f"/c{index}"]))
            events.append(_w_chunk(f"t{index}-body "))
            events.append(_w_tc())

        windowed = await _drain_recorded_stream(events, windowed=True)
        unwindowed = await _drain_recorded_stream(events, windowed=False)

        assert windowed == unwindowed
        assert not any(
            isinstance(event, AvailableCommandsReceived) for batch in windowed for event in batch
        )

    async def test_non_renderable_events_are_not_recorded_but_still_apply(self) -> None:
        """Non-renderables stay out of the batches while their side effects land.

        Silent failure has two halves: recorded non-renderables are unreplayable by any
        scroll-up, and dropping their side effects silently loses slash commands and leaves
        the input bar stuck busy after a restore.
        """
        turns = _w_feed([2] * 8)
        # BOTH cheap side effects, INTERLEAVED into the skipped region: they are emitted early
        # in a session so that is where they land, and dropping either silently loses slash
        # commands or leaves the input bar stuck busy after a restore.
        events: list[BrokerEvent] = [
            AvailableCommandsReceived(agent_id="target", commands=["/effort"]),
            *turns[:6],
            SessionRestoreComplete(agent_id="target"),
            *turns[6:],
        ]
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)):
            # The agent is cached as BUSY, so _do_select_agent's own post-drain branch sets
            # busy TRUE. That makes SessionRestoreComplete's set_busy(False) the ONLY thing
            # that can clear it — without this the assertion is vacuous, because the drain
            # clears busy by itself whenever no BUSY state is cached.
            feed = await _drain(app, events)

            recorded = [e for batch in feed._turn_events for e in batch]
            assert not any(isinstance(e, AvailableCommandsReceived) for e in recorded)
            assert not any(isinstance(e, SessionRestoreComplete) for e in recorded)
            assert feed.input_bar is not None
            assert "/effort" in feed.input_bar._slash_commands

    async def test_batches_match_their_turns_exactly(self) -> None:
        """AC3a, per batch rather than in aggregate.

        Silent failure: an off-by-one drops or duplicates a whole turn on scroll-up.
        """
        sizes = [2, 5, 3, 4, 2, 6, 3, 2]
        events = _w_feed(sizes)
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _drain(app, events)

            assert len(feed._turn_events) == sum(1 for e in events if isinstance(e, TurnComplete))
            for index, (batch, size) in enumerate(zip(feed._turn_events, sizes, strict=True)):
                # Consecutive chunks are COALESCED before recording, so one turn's
                # streamed text arrives as a single concatenated event.
                chunks = [e.chunk for e in batch if isinstance(e, MessageChunkReceived)]
                assert chunks == ["".join(f"t{index}-c{n} " for n in range(size))]
                assert isinstance(batch[-1], TurnComplete)

    async def test_lands_at_the_bottom_with_the_viewport_overfilled(self) -> None:
        """AC5 and AC2a together, on a live DOM.

        The overfill half is what stops AC5 being vacuous: anchoring to the bottom of
        content shorter than the viewport is a degenerate position, not the newest content.
        """
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            feed = await _drain(app, _w_tall_feed(8))
            # A panel that is not CURRENT has size (0, 0), so the overfill assertion would
            # be vacuous without switching to it first.
            app.query_one("#right", ContentSwitcher).current = "feed-target"
            await pilot.pause()
            assert feed._scroll is not None

            assert feed._scroll.virtual_size.height > feed._scroll.size.height
            assert feed._scroll.scroll_y == pytest.approx(feed._scroll.max_scroll_y, abs=1.0)

    async def test_late_window_turn_is_the_last_mounted_turn(self) -> None:
        """AC10: the reopen target must be a MOUNTED turn, not a skipped one.

        Silent failure: `_late_window_turn` points at a turn that was recorded but never
        constructed, so the late-chunk path targets a widget that does not exist.
        """
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _drain(app, _w_feed([2] * 8))
            assert feed._late_window_turn is _mounted_turns(feed)[-1]

            reopen_target = feed._late_window_turn.query(AgentMessage).last()
            before = len(feed.query(AgentMessage))
            await app._replay_event(feed, _w_chunk("late "))

            # The SAME widget was reopened rather than a second bubble created.
            assert len(feed.query(AgentMessage)) == before
            assert feed._current_message is reopen_target
            assert feed._current_turn is feed._late_window_turn

    async def test_events_arriving_during_the_drain_are_replayed_in_full(self) -> None:
        """Only the FIRST pass is windowed; later passes carry live events.

        Silent failure: agent output vanishes whenever it races a selection.
        """
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)):
            app._dynamic_agents["target"] = DynamicAgentInfo(
                parent=None, task="", harness="kiro", cwd="."
            )
            app._event_buffers["target"] = _w_feed([2] * 8)

            original = app._replay_event
            injected = False

            async def _inject(feed, event):
                nonlocal injected
                await original(feed, event)
                if not injected and isinstance(event, TurnComplete):
                    injected = True
                    app._event_buffers["target"].extend(_w_turn(99, 1))

            with patch.object(app, "_replay_event", _inject):
                await app.select_agent("target")

            feed = app._panels["target"]
            # 3 windowed turns plus the one that arrived mid-drain. Had the second pass been
            # windowed too, the injected turn would have been recorded but never mounted.
            assert len(_mounted_turns(feed)) == 4
            assert len(feed._turn_events) == 9


    async def test_record_only_applies_both_cheap_side_effects(self) -> None:
        """Each non-renderable side effect, asserted where it is actually observable.

        Deliberately NOT asserted at the end of a drain: `_do_select_agent` sets the busy
        state itself after the loop from the cached agent state, so a post-drain assertion is
        either overwritten or vacuous — a mutation probe that removed the SessionRestoreComplete
        branch left an end-of-drain assertion green. Driving `_record_only` directly is the
        only place the branch is the sole cause.

        Silent failure: slash commands vanish after every restore, or the input bar stays
        disabled and the user cannot type.
        """
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _drain(app, _w_feed([2] * 4))
            assert feed.input_bar is not None
            feed.input_bar.update_slash_commands([])
            feed.input_bar.set_busy(True)

            app._record_only(
                feed, AvailableCommandsReceived(agent_id="target", commands=["/effort"])
            )
            assert "/effort" in feed.input_bar._slash_commands

            app._record_only(feed, SessionRestoreComplete(agent_id="target"))
            assert feed.input_bar._busy is False

class TestWindowingScope:
    def test_windowing_is_unreachable_from_the_live_render_path(self) -> None:
        """AC8: windowing belongs to the first-selection drain and nowhere else.

        Silent failure: the tail selection leaks into live streaming, where
        `_tool_call_blocks` holds only tail blocks and an update for an older tool call
        would mount a duplicate at the bottom instead of updating in place.

        The drain now lives in `_ensure_feed`, extracted from `_do_select_agent` so a
        handoff can mount the retired agent's transcript without copying this loop. The
        guard is unchanged in substance: exactly ONE function may window.
        """
        app_source = pathlib.Path(synth_app_module.__file__).read_text()
        tree = ast.parse(app_source)
        callers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(inner, ast.Name) and inner.id == "first_paint_window"
                for inner in ast.walk(node)
            )
        }
        assert callers == {"_ensure_feed"}

        feed_source = pathlib.Path(conversation_module.__file__).read_text()
        feed_tree = ast.parse(feed_source)
        for node in ast.walk(feed_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "add_chunk",
                "add_tool_call",
                "record_event",
            }:
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                assert not names & {
                    "first_paint_window",
                    "FIRST_PAINT_TURNS",
                    "FIRST_PAINT_EVENT_BUDGET",
                }, f"{node.name} reaches windowing state"


class TestHighlightPrewarmWorker:
    async def test_prewarm_runs_off_thread_in_its_own_group(self) -> None:
        """AC12: own group, real thread, and startup does not wait on it.

        The external `textual.highlight.highlight` call is LATCHED so the worker cannot
        finish before the assertions run. Without the latch this test is timing-dependent and
        silently vacuous: in a process where Pygments is already warm the call returns in
        about a millisecond and Textual removes the completed worker, so a pre-warm
        accidentally moved ONTO the loop thread would still pass.
        """
        released = threading.Event()
        entered = threading.Event()
        real_highlight = diff_view_module.highlight.highlight

        def _latched(*args, **kwargs):
            entered.set()
            released.wait(5)
            return real_highlight(*args, **kwargs)

        app = _make_app("a")
        with patch.object(diff_view_module.highlight, "highlight", _latched):
            async with app.run_test(headless=True, size=(80, 24)):
                for _ in range(200):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert entered.is_set(), "the pre-warm never ran"

                # The app is READY while the pre-warm is still blocked, which is the property
                # that proves startup does not wait on it.
                workers = [w for w in app.workers if w.group == "prewarm"]
                assert len(workers) == 1
                assert workers[0].is_running
                assert app.screen is not None

                released.set()
                for _ in range(200):
                    if workers[0].is_finished:
                        break
                    await asyncio.sleep(0.01)
                assert workers[0].is_finished


class TestWindowedDrainThenPrune:
    async def test_mounted_start_idx_survives_a_windowed_drain_then_prune(self) -> None:
        """AC11: both writers of `_mounted_start_idx` must agree on one meaning.

        The drain initialises it to the batches it SKIPPED; `_check_prune` increments it by
        the turns it UNMOUNTS. If either disagreed, scroll-up would restore the wrong slice —
        silently showing the user history that is not the history they scrolled past, which no
        count-based assertion detects.
        """
        app = _make_app("dummy")
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            feed = await _drain(app, _w_feed([2] * 8))
            skipped = feed._mounted_start_idx
            assert skipped == 5

            # Drive enough further turns to trip HIGH_MARK and force a prune.
            for index in range(ConversationFeed.HIGH_MARK + 2):
                for event in _w_turn(100 + index, 1):
                    await app._replay_event(feed, event)
            await pilot.pause()

            pruned_idx = feed._mounted_start_idx
            assert pruned_idx > skipped, "the prune writer did not advance the index"
            assert pruned_idx <= len(feed._turn_events)

            # Scroll-up must restore the batches immediately BELOW the mounted region, by
            # content rather than by count.
            expected_batch = feed._turn_events[max(0, pruned_idx - RESTORE_BATCH) : pruned_idx]
            expected_prompts = [
                event.text
                for batch in expected_batch
                for event in batch
                if isinstance(event, UserPromptSubmitted)
            ]
            await feed._restore_turns()
            await pilot.pause()

            restored_prompts = [
                str(bubble._text)
                for turn in _mounted_turns(feed)
                for bubble in turn.query(PromptBubble)
            ][: len(expected_prompts)]
            assert restored_prompts == expected_prompts


class _HandoffUIHarness:
    """Shared setup for the handoff UI tests. NOT collected: the name has no Test prefix.

    A previous revision had the failure-path class subclass the main test class, which made
    pytest re-collect and re-run all six of its live Textual tests. Sharing helpers through
    an uncollected base keeps one execution per test.
    """

    @staticmethod
    def _events(original: str, retired: str, parent: str | None = None) -> list:
        return [
            AgentStateChanged(
                agent_id=original, old_state=AgentState.BUSY, new_state=AgentState.TERMINATED
            ),
            AgentHandedOff(
                agent_id=original, retired_agent_id=retired, parent=parent, task="the task"
            ),
            AgentStateChanged(
                agent_id=original, old_state=AgentState.UNSTARTED, new_state=AgentState.INITIALIZING
            ),
        ]

    @staticmethod
    async def _post(app: SynthApp, pilot: Any, event: object) -> None:
        """Deliver one event and let the message pump settle.

        Each broker event is its own pump cycle in production, and Textual's mount() is
        not awaited by add_agent_tile, so a tile's children do not exist until the pump
        runs. Posting several events back to back is not a realistic sequence.
        """
        await app.on_broker_event_message(BrokerEventMessage(event))
        await pilot.pause()

    @staticmethod
    def _app(*agent_ids: str, journal: list | None = None) -> SynthApp:
        app = _make_app(*agent_ids)
        app.broker.load_journal = AsyncMock(return_value=journal or [])
        app.broker.session_id = "sess-1"
        return app

    @staticmethod
    async def _bring_up(app: SynthApp, pilot: Any, agent_id: str) -> None:
        """Put the agent in the state a live agent is in before it terminates.

        A handoff always happens to an agent that was running, so starting from a bare id
        would test a state the sequence cannot reach. The state is seeded directly rather
        than driven through an INITIALIZING event because app.py mounts a tile and calls
        update_state on it within the same pump cycle, and the tile's children do not
        exist yet at that point -- a pre-existing fragility this feature does not touch.
        """
        app._dynamic_agents[agent_id] = DynamicAgentInfo(
            parent=None, task="", harness="kiro", cwd="."
        )
        app._event_buffers.setdefault(agent_id, [])
        tile = app.query_one(AgentList).add_agent_tile(agent_id)
        app._tiles[agent_id] = tile
        await pilot.pause()
        app._agent_states[agent_id] = AgentState.IDLE
        tile.update_state(AgentState.IDLE)


class TestAgentHandedOff(_HandoffUIHarness):
    """The retirement rebuild, driven by the REAL three-event sequence.

    A handoff reaches the UI as three events in this order, and the middle one is the only
    one this feature adds:

        AgentStateChanged(original, TERMINATED)   <- tears the original id down
        AgentHandedOff(original, retired)         <- rebuilds the predecessor's view
        AgentStateChanged(original, INITIALIZING) <- the successor mounts normally

    Testing only the last two hides the defects: the first event is what destroys the
    original id's widgets and moves the selection.
    """

    async def test_mounts_retired_tile_and_feed_and_frees_the_original_id(self) -> None:
        journal = [MessageChunkReceived(agent_id="worker.h0000dead", chunk="predecessor said this")]
        app = self._app("worker", journal=journal)
        first, handed_off, initializing = self._events("worker", "worker.h0000dead")

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            await self._post(app, pilot, first)
            await self._post(app, pilot, handed_off)

            # The predecessor is readable under its new id...
            assert app.query_one("#tile-worker-h0000dead") is not None
            assert app.query_one("#feed-worker-h0000dead") is not None
            app.broker.load_journal.assert_awaited_once_with("worker.h0000dead", "sess-1")
            # ...and the original id is free for the successor.
            assert app._tiles.get("worker") is None

            await self._post(app, pilot, initializing)

            # No DuplicateIds: Textual swallows that into a logged error, so the symptom
            # would be a successor that silently never gets a tile.
            assert "worker" in app._tiles
            assert app.query_one("#tile-worker-h0000dead") is not None

    async def test_original_id_is_absent_from_every_per_agent_dict(self) -> None:
        """_agent_states is the one dict the TERMINATED handler does not clear. Left in
        place, INITIALIZING reads prev_state == TERMINATED and takes the RESURRECTION
        branch: it reloads a journal that is now empty for the original id and skips
        registering fresh metadata, with nothing raised."""
        app = self._app("worker")
        first, handed_off, initializing = self._events("worker", "worker.h0000dead")

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            await self._post(app, pilot, first)
            await self._post(app, pilot, handed_off)

            for name, mapping in (
                ("_agent_states", app._agent_states),
                ("_tiles", app._tiles),
                ("_panels", app._panels),
                ("_event_buffers", app._event_buffers),
                ("_dynamic_agents", app._dynamic_agents),
                ("_agent_config_options", app._agent_config_options),
            ):
                assert "worker" not in mapping, f"{name} still holds the original id"

            app.broker.load_journal.reset_mock()
            await self._post(app, pilot, initializing)

            # New-agent branch, not resurrection: no journal load for the original id.
            for call in app.broker.load_journal.await_args_list:
                assert call.args[0] != "worker"
            assert "worker" in app._dynamic_agents

    async def test_retired_tile_renders_as_retired_and_survives_the_successor(self) -> None:
        app = self._app("worker")
        first, handed_off, initializing = self._events("worker", "worker.h0000dead")

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            for event in (first, handed_off, initializing):
                await self._post(app, pilot, event)

            tile = app._tiles["worker.h0000dead"]
            markup = tile._build_markup()
            assert "retired" in markup
            assert "terminated" not in markup
            assert len(app.query("#tile-worker-h0000dead")) == 1

    async def test_selection_returns_to_the_original_id_when_another_agent_is_live(self) -> None:
        """With a child live, the TERMINATED event moves the selection to the child, so
        "is the original still selected?" is already false. A one-agent test cannot see
        this: the user's view sits on the wrong agent while the successor does the work."""
        app = self._app("worker", "kid")
        first, handed_off, initializing = self._events("worker", "worker.h0000dead")

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            app._agent_states["kid"] = AgentState.IDLE
            app._dynamic_agents["kid"] = DynamicAgentInfo(
                parent=None, task="", harness="kiro", cwd="."
            )
            await app.select_agent("worker")
            await pilot.pause()

            await self._post(app, pilot, first)
            assert app.selected_agent == "kid", "precondition: TERMINATED moved the selection"

            await self._post(app, pilot, handed_off)
            # Still on the child: selecting the original id here would create its panel
            # before the successor exists.
            assert app.selected_agent == "kid"

            await self._post(app, pilot, initializing)

            assert app.selected_agent == "worker"

    async def test_selection_is_not_stolen_after_an_explicit_user_selection(self) -> None:
        app = self._app("worker", "kid")
        first, handed_off, initializing = self._events("worker", "worker.h0000dead")

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            app._agent_states["kid"] = AgentState.IDLE
            app._dynamic_agents["kid"] = DynamicAgentInfo(
                parent=None, task="", harness="kiro", cwd="."
            )
            await app.select_agent("worker")
            await pilot.pause()
            await self._post(app, pilot, first)

            # The user deliberately picks the child between the two events.
            await app.select_agent("kid")
            await self._post(app, pilot, handed_off)
            await self._post(app, pilot, initializing)

            assert app.selected_agent == "kid"

    async def test_resurrecting_the_retired_agent_takes_the_resurrection_branch(self) -> None:
        """Root AC 2 requires the predecessor to stay resurrectable. Without
        _agent_states[retired] this falls into the new-agent branch and calls
        add_agent_tile for a tile already mounted, raising DuplicateIds -- which app.py
        swallows into a log line, so the retired agent is silently unresurrectable."""
        app = self._app("worker")
        first, handed_off, initializing = self._events("worker", "worker.h0000dead")

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            for event in (first, handed_off, initializing):
                await self._post(app, pilot, event)

            assert app._agent_states["worker.h0000dead"] == AgentState.TERMINATED
            agent_list = app.query_one(AgentList)
            with patch.object(
                agent_list, "add_agent_tile", wraps=agent_list.add_agent_tile
            ) as spy:
                await self._post(
                    app,
                    pilot,
                    AgentStateChanged(
                        agent_id="worker.h0000dead",
                        old_state=AgentState.UNSTARTED,
                        new_state=AgentState.INITIALIZING,
                    ),
                )
                spy.assert_not_called()

            assert len(app.query("#tile-worker-h0000dead")) == 1


class TestAgentHandedOffFailurePaths(_HandoffUIHarness):
    """Mounting the retired view is mandatory, so its failures must be visible.

    Both operations sit behind a broad except. Logging alone is not enough: an empty
    retired feed looks exactly like a handoff that discarded the predecessor's history,
    and a missing sidebar entry looks exactly like it discarded the agent.
    """

    async def test_a_journal_read_failure_is_surfaced_not_shown_as_an_empty_transcript(
        self,
    ) -> None:
        app = self._app("worker")
        app.broker.load_journal = AsyncMock(side_effect=sqlite3.OperationalError("disk I/O error"))
        first, handed_off, _ = self._events("worker", "worker.h0000dead")
        notices: list[tuple[str, str]] = []
        app.notify = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda msg, **kw: notices.append((msg, kw.get("severity", "")))
        )

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            await self._post(app, pilot, first)
            await self._post(app, pilot, handed_off)

            # The agent stays reachable, and the user is told the transcript did not load.
            assert len(app.query("#feed-worker-h0000dead")) == 1
            assert len(app.query("#tile-worker-h0000dead")) == 1
            assert [n for n in notices if n[1] == "error" and "worker.h0000dead" in n[0]]

    async def test_a_tile_mount_failure_is_surfaced(self) -> None:
        app = self._app("worker")
        first, handed_off, _ = self._events("worker", "worker.h0000dead")
        notices: list[tuple[str, str]] = []
        app.notify = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda msg, **kw: notices.append((msg, kw.get("severity", "")))
        )

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._bring_up(app, pilot, "worker")
            await self._post(app, pilot, first)
            agent_list = app.query_one(AgentList)
            with patch.object(
                agent_list, "add_agent_tile", side_effect=RuntimeError("mount failed")
            ):
                await self._post(app, pilot, handed_off)

            assert [n for n in notices if n[1] == "error" and "worker.h0000dead" in n[0]]


class TestInputBarStateIsRederivedNotRemembered:
    """The input bar's busy and disabled properties come from AgentStateChanged, but the
    live route drops those events whenever the agent has no panel or is draining, and
    _replay_event_locked deliberately ignores them.  Feed creation must therefore
    re-derive BOTH from _agent_states, which is written on every event.
    """

    async def test_awaiting_permission_shows_busy_on_the_live_route(self) -> None:
        """AWAITING_PERMISSION previously matched neither branch, so set_busy was not
        called at all and the bar kept whatever it had -- no cancel button while a
        permission was pending."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]

            await app.on_broker_event_message(
                BrokerEventMessage(
                    AgentStateChanged(
                        agent_id="agent-1",
                        old_state=AgentState.BUSY,
                        new_state=AgentState.AWAITING_PERMISSION,
                    )
                )
            )

            assert feed.input_bar.query_one(ActivityBar).active is True
            assert feed.input_bar.query_one("#cancel-btn").display is True

    async def test_feed_creation_rederives_busy_and_disabled_from_agent_states(self) -> None:
        """A feed created while the agent is AWAITING_PERMISSION previously got
        set_busy(False) from the tail's else branch and no set_disabled at all: no cancel
        button, no activity bar, and typing enabled, while the broker correctly refused
        to prompt because the agent was not IDLE.  That combination is what made a
        prompt appear to 'go straight to the queue' on a seemingly idle agent."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            app._agent_states["agent-1"] = AgentState.AWAITING_PERMISSION
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

            await app.select_agent("agent-1")
            # The sync is deferred to after refresh, because the feed's on_mount is what
            # assigns input_bar and it runs on the message pump.
            await pilot.pause()
            feed = app._panels["agent-1"]

            assert feed.input_bar.query_one(ActivityBar).active is True
            assert feed.input_bar.query_one("#cancel-btn").display is True
            assert feed.input_bar.query_one("#prompt-input").disabled is True


class TestTerminationDoesNotStealSelection:
    async def test_marker_is_not_armed_when_no_handoff_is_in_flight(self) -> None:
        """_auto_selected_from exists so a handoff can put the user back on the id that
        continues the work.  Armed on every termination it never expires, so a later
        INITIALIZING for that same id -- an ordinary resurrect, or a launch reusing a
        terminated id -- would yank the selection out from under the user."""
        app = _make_app("agent-1", "agent-2")
        app.broker.handoff_in_flight = MagicMock(return_value=False)
        async with app.run_test(headless=True, size=(120, 40)):
            app._agent_states["agent-2"] = AgentState.IDLE
            for aid in ("agent-1", "agent-2"):
                app._dynamic_agents[aid] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            await app.select_agent("agent-1")

            await app.on_broker_event_message(
                BrokerEventMessage(
                    AgentStateChanged(
                        agent_id="agent-1",
                        old_state=AgentState.IDLE,
                        new_state=AgentState.TERMINATED,
                    )
                )
            )

            assert app._auto_selected_from is None
            assert app.selected_agent == "agent-2", "selection still moves off the dead agent"
