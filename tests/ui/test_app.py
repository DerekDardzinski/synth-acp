"""Tests for SynthApp: event routing, panel switching, modals, loading states."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.worker import WorkerState

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.commands import LaunchAgent
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerEvent,
    McpMessageDelivered,
    MessageChunkReceived,
    ToolCallUpdated,
    UsageUpdated,
)
from synth_acp.ui.app import DynamicAgentInfo, SynthApp, _coalesce_events
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.gradient_bar import ActivityBar
from synth_acp.ui.widgets.message_queue import MessageQueue


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        project="test",
    )


def _make_broker(events: list[BrokerEvent] | None = None, agent_ids: list[str] | None = None) -> MagicMock:
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


class TestWatchSelectedAgent:
    def test_watch_selected_agent_when_empty_string_skips_switch(self) -> None:
        """Empty string guard prevents crash on initial reactive value."""
        app = _make_app("agent-1")
        mock_query_one = MagicMock()
        with patch.object(app, "query_one", mock_query_one):
            app.watch_selected_agent("", "")

        mock_query_one.assert_not_called()


class TestShowMessagesContentSwitcher:
    async def test_show_messages_when_first_call_mounts_mcp_panel(self) -> None:
        """First call creates MessageQueue with id='messages' and mounts it."""
        app = _make_app("agent-1")

        mock_switcher = SimpleNamespace(current=None, add_content=AsyncMock())
        with (
            patch.object(app, "query_one", return_value=mock_switcher),
            patch.object(app, "_tiles", {}),
        ):
            await app.show_messages()

        assert app._mcp_panel is not None
        assert isinstance(app._mcp_panel, MessageQueue)
        mock_switcher.add_content.assert_called_once()
        mounted = mock_switcher.add_content.call_args[0][0]
        assert mounted.id == "messages"
        assert mock_switcher.current == "messages"


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
            ToolCallUpdated(agent_id="a", tool_call_id="t1", title="t", kind="read", status="completed"),
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


class TestMcpMessageDeliveredBuffering:
    """Verify that McpMessageDelivered for unknown recipients is buffered."""

    async def test_mcp_message_for_unknown_recipient_buffered(self) -> None:
        """McpMessageDelivered for unknown recipient is buffered, not dropped."""
        app = _make_app()
        assert "unknown-agent" not in app._panels
        assert "unknown-agent" not in app._event_buffers

        async with app.run_test(headless=True, size=(120, 40)):
            event = McpMessageDelivered(
                agent_id="sender",
                from_agent="sender",
                to_agent="unknown-agent",
                preview="hello",
            )
            await app.on_broker_event_message(BrokerEventMessage(event))

            assert "unknown-agent" in app._event_buffers
            assert event in app._event_buffers["unknown-agent"]


class TestMcpInterceptionRouting:
    """Verify MCP messages route to queue when composing, to feed otherwise."""

    async def test_mcp_message_held_routes_to_queue(self) -> None:
        """McpMessageHeld calls enqueue on the input bar."""
        from synth_acp.models.events import McpMessageHeld

        app = _make_app("agent-1")
        feed = MagicMock()
        feed.input_bar = MagicMock()
        feed.input_bar.enqueue = MagicMock()
        app._panels["agent-1"] = feed

        event = McpMessageHeld(
            agent_id="agent-1",
            from_agent="other",
            preview="hello from other",
        )
        await app.on_broker_event_message(BrokerEventMessage(event))

        feed.input_bar.enqueue.assert_called_once_with("hello from other", "mcp", "other")

    async def test_mcp_message_routes_to_feed_when_not_composing(self) -> None:
        """McpMessageDelivered calls add_mcp_message when is_composing is False."""
        app = _make_app("agent-1")
        app._event_buffers["agent-1"] = []
        app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

        feed = MagicMock()
        feed.input_bar = MagicMock()
        feed.input_bar.is_composing = False
        feed.add_mcp_message = AsyncMock()
        app._panels["agent-1"] = feed

        event = McpMessageDelivered(
            agent_id="other",
            from_agent="other",
            to_agent="agent-1",
            preview="hello from other",
        )
        await app.on_broker_event_message(BrokerEventMessage(event))

        feed.add_mcp_message.assert_called_once_with("other", "agent-1", "hello from other")
        feed.input_bar.enqueue.assert_not_called()


class TestAttemptDrain:
    """Verify _attempt_drain dispatches SendPrompt or is a no-op."""

    async def test_attempt_drain_dispatches_send_prompt_when_queue_has_item(self) -> None:
        """_attempt_drain calls broker.handle(SendPrompt) when drain_next returns an item."""
        from synth_acp.models.commands import SendPrompt
        from synth_acp.ui.widgets.prompt_queue import QueuedPrompt

        app = _make_app("agent-1")
        feed = MagicMock()
        feed.input_bar = MagicMock()
        feed.input_bar.drain_next = MagicMock(return_value=QueuedPrompt(text="queued msg"))
        feed.input_bar.is_composing = False
        feed.input_bar.has_queue_items = True
        app._panels["agent-1"] = feed

        with patch.object(app, "run_worker") as mock_run:
            app._attempt_drain("agent-1")

        mock_run.assert_called_once()
        # The argument to run_worker is a coroutine from broker.handle(SendPrompt(...))
        app.broker.handle.assert_called_once_with(SendPrompt(agent_id="agent-1", text="queued msg"))

    async def test_attempt_drain_noop_when_queue_empty(self) -> None:
        """_attempt_drain does nothing when drain_next returns None."""
        app = _make_app("agent-1")
        feed = MagicMock()
        feed.input_bar = MagicMock()
        feed.input_bar.drain_next = MagicMock(return_value=None)
        app._panels["agent-1"] = feed

        with patch.object(app, "run_worker") as mock_run:
            app._attempt_drain("agent-1")

        mock_run.assert_not_called()

    async def test_turn_complete_does_not_drain(self) -> None:
        """TurnComplete does NOT trigger _attempt_drain (drain is on IDLE only)."""
        from synth_acp.models.events import TurnComplete

        app = _make_app("agent-1")
        feed = MagicMock()
        feed.input_bar = MagicMock()
        feed.finalize_current_message = AsyncMock()
        app._panels["agent-1"] = feed

        event = TurnComplete(agent_id="agent-1", stop_reason="end_turn")

        with patch.object(app, "_attempt_drain") as mock_drain:
            await app._route_event_to_feed(feed, event)

        mock_drain.assert_not_called()

    async def test_agent_state_idle_triggers_drain(self) -> None:
        """AgentStateChanged(IDLE) triggers _attempt_drain."""
        app = _make_app("agent-1")
        feed = MagicMock()
        feed.input_bar = MagicMock()
        app._panels["agent-1"] = feed

        event = AgentStateChanged(
            agent_id="agent-1",
            old_state=AgentState.BUSY,
            new_state=AgentState.IDLE,
        )

        with patch.object(app, "_attempt_drain") as mock_drain:
            await app._route_event_to_feed(feed, event)

        mock_drain.assert_called_once_with("agent-1")


class TestDrainReady:
    """Verify on_input_bar_drain_ready triggers drain only when idle."""

    async def test_drain_ready_triggers_drain_when_idle(self) -> None:
        """DrainReady calls _attempt_drain when agent state is IDLE."""
        app = _make_app("agent-1")
        app._agent_states["agent-1"] = AgentState.IDLE

        event = MagicMock()
        event.agent_id = "agent-1"

        with patch.object(app, "_attempt_drain") as mock_drain:
            app.on_input_bar_drain_ready(event)

        mock_drain.assert_called_once_with("agent-1")

    async def test_drain_ready_skipped_when_busy(self) -> None:
        """DrainReady does NOT call _attempt_drain when agent is BUSY."""
        app = _make_app("agent-1")
        app._agent_states["agent-1"] = AgentState.BUSY

        event = MagicMock()
        event.agent_id = "agent-1"

        with patch.object(app, "_attempt_drain") as mock_drain:
            app.on_input_bar_drain_ready(event)

        mock_drain.assert_not_called()


class TestDisabledStates:
    """Verify _DISABLED_STATES only contains AWAITING_PERMISSION."""

    def test_disabled_states_only_awaiting_permission(self) -> None:
        """_DISABLED_STATES reduced to {AWAITING_PERMISSION} only."""
        from synth_acp.ui.app import _DISABLED_STATES

        assert {AgentState.AWAITING_PERMISSION} == _DISABLED_STATES


class TestConfigOptionsHandling:
    """Tests for ConfigOptionsReceived and ConfigOptionChanged event handling."""

    def _make_select_option(self, opt_id: str, name: str, category: str | None, current_value: str, options: list[tuple[str, str]]):
        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption
        return SessionConfigOptionSelect(
            id=opt_id, name=name, category=category, type="select",
            current_value=current_value,
            options=[SessionConfigSelectOption(value=v, name=n) for v, n in options],
        )

    async def test_config_options_received_stores_and_updates_bar(self) -> None:
        """ConfigOptionsReceived stores options and calls input_bar.update_config_options."""
        from synth_acp.models.events import ConfigOptionsReceived
        from synth_acp.ui.messages import BrokerEventMessage

        app = _make_app("agent-1")
        app._agent_states["agent-1"] = AgentState.IDLE

        mode_opt = self._make_select_option("mode", "Mode", "mode", "code", [("code", "Code"), ("plan", "Plan")])
        model_opt = self._make_select_option("model", "Model", "model", "gpt-4", [("gpt-4", "GPT-4")])
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

        mode_opt = self._make_select_option("mode", "Mode", "mode", "code", [("code", "Code"), ("plan", "Plan")])
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
        app._agent_config_options["agent-1"] = [self._make_select_option("mode", "Mode", "mode", "code", [("code", "Code")])]
        app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")

        mock_tile = MagicMock()
        app._tiles["agent-1"] = mock_tile

        mock_feed = MagicMock()
        mock_feed.input_bar = MagicMock()
        mock_feed.remove = AsyncMock()
        app._panels["agent-1"] = mock_feed

        event = AgentStateChanged(agent_id="agent-1", old_state=AgentState.IDLE, new_state=AgentState.TERMINATED)
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
        """Worker processes unembedded sessions and stores embeddings."""
        app = _make_app("agent-1")
        app.broker._db_path = tmp_path / "test.db"

        mock_embedding = MagicMock()
        mock_embedding.tobytes.return_value = b"\x00" * 1536

        mock_engine = MagicMock()
        mock_engine.embed.return_value = mock_embedding

        with (
            patch("synth_acp.ui.app.embedding_available", return_value=True),
            patch("synth_acp.ui.app.EmbeddingEngine", return_value=mock_engine),
            patch("synth_acp.ui.app.get_unembedded_sessions_sync", return_value=["sess-1", "sess-2"]),
            patch.object(SynthApp, "_query_session_metadata", return_value={"agents": ["a"], "cwd": "/tmp", "tasks": ["t"], "first_messages": ["hello"]}),
            patch("synth_acp.ui.app._build_embedding_text", return_value="hello a tmp t"),
            patch("synth_acp.ui.app._text_hash", return_value="abc123"),
            patch("synth_acp.ui.app.store_embedding_sync") as mock_store,
        ):
            app._do_index_sessions()

        assert app._indexing_complete is True
        assert app._embedding_engine is mock_engine
        mock_engine.ensure_model.assert_called_once()
        assert mock_store.call_count == 2
        # Verify correct args for first call
        call_args = mock_store.call_args_list[0]
        assert call_args[0][1] == "sess-1"
        assert call_args[0][2] == "abc123"

    def test_index_sessions_handles_embed_error(self, tmp_path: Path) -> None:
        """Exception during engine.embed() does not propagate."""
        app = _make_app("agent-1")
        app.broker._db_path = tmp_path / "test.db"

        mock_engine = MagicMock()
        mock_engine.embed.side_effect = RuntimeError("model failed")

        with (
            patch("synth_acp.ui.app.embedding_available", return_value=True),
            patch("synth_acp.ui.app.EmbeddingEngine", return_value=mock_engine),
            patch("synth_acp.ui.app.get_unembedded_sessions_sync", return_value=["sess-1"]),
            patch.object(SynthApp, "_query_session_metadata", return_value={"agents": [], "cwd": None, "tasks": [], "first_messages": []}),
            patch("synth_acp.ui.app._build_embedding_text", return_value="text"),
            patch("synth_acp.ui.app._text_hash", return_value="hash"),
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
            patch.object(ACPBroker, "list_restorable_sessions", new_callable=AsyncMock, return_value=[]),
            patch.object(app, "push_screen_wait", new_callable=AsyncMock, return_value=None) as mock_push,
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
            event = AgentStateChanged(agent_id="agent-1", old_state=AgentState.BUSY, new_state=AgentState.IDLE)
            await app._route_event_to_feed(feed, event)
            assert feed._current_turn_events == []

    async def test_mcp_delivered_early_return_records_event(self) -> None:
        """McpMessageDelivered in the early-return block is recorded."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)):
            await app.select_agent("agent-1")
            feed = app._panels["agent-1"]
            event = McpMessageDelivered(agent_id="agent-1", from_agent="other", to_agent="agent-1", preview="hello")
            await app.on_broker_event_message(BrokerEventMessage(event))
            assert event in feed._current_turn_events
