"""Tests for AgentTile and AgentList sidebar widgets."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import AgentStateChanged
from synth_acp.ui.app import DynamicAgentInfo, SynthApp
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.agent_list import AgentList, AgentTile


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        project="test",
    )


def _make_broker(*agent_ids: str) -> MagicMock:
    """Create a mock broker with async stubs."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    first_id = agent_ids[0] if agent_ids else "agent-1"
    broker._initial_agent = AgentConfig(agent_id=first_id, harness="kiro")
    broker.get_agent_harness = MagicMock(return_value="kiro")
    broker.get_agent_parent = MagicMock(return_value=None)
    broker.get_agent_cwd = MagicMock(return_value=".")

    async def _events():
        return
        yield

    broker.events = _events
    return broker


class TestAgentTileStateChange:
    async def test_agent_tile_when_state_changes_to_awaiting_permission_adds_warning_class(
        self,
    ) -> None:
        """AWAITING_PERMISSION state adds tile-permission class for visual alert."""
        broker = _make_broker("agent-1")
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            app._event_buffers.setdefault("agent-1", [])
            agent_list = app.query_one(AgentList)
            tile = agent_list.add_agent_tile("agent-1")
            app._tiles["agent-1"] = tile
            await pilot.pause()

            assert not tile.has_class("tile-permission")
            tile.update_state(AgentState.AWAITING_PERMISSION)
            assert tile.has_class("tile-permission")

    async def test_agent_tile_when_state_changes_from_permission_to_busy_removes_warning_class(
        self,
    ) -> None:
        """Transitioning away from AWAITING_PERMISSION removes tile-permission class."""
        broker = _make_broker()
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            app._event_buffers.setdefault("agent-1", [])
            agent_list = app.query_one(AgentList)
            tile = agent_list.add_agent_tile("agent-1")
            app._tiles["agent-1"] = tile
            await pilot.pause()

            tile.update_state(AgentState.AWAITING_PERMISSION)
            assert tile.has_class("tile-permission")

            tile.update_state(AgentState.BUSY)
            assert not tile.has_class("tile-permission")


class TestAgentTileClick:
    async def test_agent_tile_when_clicked_calls_select_agent(self) -> None:
        """Clicking a tile switches the selected agent."""
        broker = _make_broker("agent-1")
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            app._event_buffers.setdefault("agent-1", [])
            agent_list = app.query_one(AgentList)
            tile = agent_list.add_agent_tile("agent-1")
            app._tiles["agent-1"] = tile
            await pilot.pause()

            # Reset selected_agent so the click actually triggers a change
            app.selected_agent = ""
            await pilot.click("#tile-agent-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.selected_agent == "agent-1"


class TestAgentStateChangedRouting:
    async def test_on_broker_event_message_when_state_changed_updates_tile(self) -> None:
        """AgentStateChanged event routes to the matching tile's update_state."""
        broker = _make_broker()
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            app._event_buffers.setdefault("agent-1", [])
            agent_list = app.query_one(AgentList)
            tile = agent_list.add_agent_tile("agent-1")
            app._tiles["agent-1"] = tile
            await pilot.pause()

            event = AgentStateChanged(
                agent_id="agent-1",
                old_state=AgentState.IDLE,
                new_state=AgentState.AWAITING_PERMISSION,
            )
            await app.on_broker_event_message(BrokerEventMessage(event))

            assert tile.has_class("tile-permission")

    async def test_on_broker_event_message_when_state_changed_buffers_event(self) -> None:
        """AgentStateChanged events are buffered for agents without panels."""
        broker = _make_broker("agent-1")
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)):
            # Pre-register agent-2 as dynamic so tile creation doesn't fire
            from synth_acp.ui.app import DynamicAgentInfo
            app._dynamic_agents["agent-2"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            app._event_buffers.setdefault("agent-2", [])
            event = AgentStateChanged(
                agent_id="agent-2",
                old_state=AgentState.IDLE,
                new_state=AgentState.BUSY,
            )
            await app.on_broker_event_message(BrokerEventMessage(event))

            assert len(app._event_buffers["agent-2"]) == 1
            assert app._event_buffers["agent-2"][0] is event


class TestAddAgentTile:
    async def test_agent_list_when_add_agent_tile_called_mounts_new_tile(self) -> None:
        """Dynamic tile appears in the DOM after add_agent_tile is called."""
        broker = _make_broker()
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)):
            from synth_acp.ui.widgets.agent_list import AgentList

            agent_list = app.query_one(AgentList)
            agent_list.add_agent_tile("new-agent")
            await app.workers.wait_for_complete()
            tile = app.query_one("#tile-new-agent", AgentTile)
            assert tile._agent_id == "new-agent"
