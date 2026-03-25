"""Tests for AgentTile and AgentList sidebar widgets."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from synth_acp.models.agent import AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import AgentStateChanged
from synth_acp.ui.app import SynthApp
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.agent_list import AgentTile


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        session="test",
        agents=[{"id": aid, "binary": "echo"} for aid in agent_ids],
    )


def _make_broker() -> MagicMock:
    """Create a mock broker with async stubs."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()

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
        broker = _make_broker()
        config = _make_config("agent-1", "agent-2")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)):
            tile = app.query_one("#tile-agent-1", AgentTile)
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

        async with app.run_test(headless=True, size=(120, 40)):
            tile = app.query_one("#tile-agent-1", AgentTile)
            tile.update_state(AgentState.AWAITING_PERMISSION)
            assert tile.has_class("tile-permission")

            tile.update_state(AgentState.BUSY)
            assert not tile.has_class("tile-permission")


class TestAgentTileClick:
    async def test_agent_tile_when_clicked_calls_select_agent(self) -> None:
        """Clicking a tile switches the selected agent."""
        broker = _make_broker()
        config = _make_config("agent-1", "agent-2")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.click("#tile-agent-2")
            assert app.selected_agent == "agent-2"
            tile = app.query_one("#tile-agent-2", AgentTile)
            assert tile.has_class("tile-active")


class TestAgentStateChangedRouting:
    async def test_on_broker_event_message_when_state_changed_updates_tile(self) -> None:
        """AgentStateChanged event routes to the matching tile's update_state."""
        broker = _make_broker()
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)):
            event = AgentStateChanged(
                agent_id="agent-1",
                old_state=AgentState.IDLE,
                new_state=AgentState.AWAITING_PERMISSION,
            )
            await app.on_broker_event_message(BrokerEventMessage(event))

            tile = app.query_one("#tile-agent-1", AgentTile)
            assert tile.has_class("tile-permission")

    async def test_on_broker_event_message_when_state_changed_buffers_event(self) -> None:
        """AgentStateChanged events are buffered per agent."""
        broker = _make_broker()
        config = _make_config("agent-1")
        app = SynthApp(broker, config)

        async with app.run_test(headless=True, size=(120, 40)):
            event = AgentStateChanged(
                agent_id="agent-1",
                old_state=AgentState.IDLE,
                new_state=AgentState.BUSY,
            )
            await app.on_broker_event_message(BrokerEventMessage(event))

            assert len(app._event_buffers["agent-1"]) == 1
            assert app._event_buffers["agent-1"][0] is event
