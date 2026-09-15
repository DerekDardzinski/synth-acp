"""Tests for AgentTile and AgentList sidebar widgets."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from textual.widgets import Static

from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import AgentStateChanged
from synth_acp.ui.app import DynamicAgentInfo, SynthApp
from synth_acp.ui.messages import BrokerEventMessage
from synth_acp.ui.widgets.agent_list import AgentList, AgentTile
from tests.conftest import wait_for_transient_workers


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
    broker.get_usage = MagicMock(return_value=None)

    async def _events():
        return
        yield

    broker.events = _events
    return broker


def _make_app(*agent_ids: str) -> SynthApp:
    """Create a SynthApp with a mock broker."""
    return SynthApp(_make_broker(*agent_ids), _make_config(*agent_ids))


class TestAgentTileActivityBar:
    async def test_inactive_tile_activity_bar_keeps_usage_bar(self) -> None:
        """An INACTIVE tile ActivityBar must still carry a working UsageBar.

        The lazy-bar change belongs to ExpandableSection only. Silent failure: applying
        it to the shared ActivityBar blanks every tile's context/cost readout, and every
        ExpandableSection test would still pass.
        """
        from synth_acp.ui.widgets.gradient_bar import ActivityBar, UsageBar

        app = SynthApp(_make_broker("agent-1"), _make_config("agent-1"))
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            app._dynamic_agents["agent-1"] = DynamicAgentInfo(parent=None, task="", harness="kiro")
            app._event_buffers.setdefault("agent-1", [])
            tile = app.query_one(AgentList).add_agent_tile("agent-1")
            await pilot.pause()

            bar = tile.query_one(ActivityBar)
            assert bar.active is False

            usage_bar = bar.query_one(UsageBar)
            tile.update_usage(500, 1000, "$1.23")
            assert usage_bar._used == 500
            assert usage_bar._context_size == 1000
            assert usage_bar._cost_text == "$1.23"


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
            await wait_for_transient_workers(app)
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
            await wait_for_transient_workers(app)
            tile = app.query_one("#tile-new-agent", AgentTile)
            assert tile._agent_id == "new-agent"


class TestRetiredTile:
    def test_retired_tile_renders_distinctly_from_terminated(self) -> None:
        """A retired predecessor is not a dead agent: it keeps its full transcript and
        stays resurrectable, so it stays in the sidebar to be read. Rendering it as
        terminated would tell the user its history is gone.
        """
        retired = AgentTile("worker.h0000dead", AgentState.TERMINATED, retired=True)
        terminated = AgentTile("worker", AgentState.TERMINATED)

        retired_markup = retired._build_markup()

        assert "retired" in retired_markup
        assert "terminated" not in retired_markup
        assert "terminated" in terminated._build_markup()

    def test_resurrecting_a_retired_tile_clears_the_retired_presentation(self) -> None:
        """The only state a retired tile can receive is a resurrection, and a resurrected
        agent is live again."""
        tile = AgentTile("worker.h0000dead", AgentState.TERMINATED, retired=True)

        tile.update_state(AgentState.INITIALIZING)

        assert tile._retired is False
        assert "retired" not in tile._build_markup()


class TestTileUpdatesBeforeCompose:
    """Callers mount a tile and update it in the same message-pump cycle, so both
    mutators must survive being called before the tile's children exist -- and every
    piece of state recorded then must be realized once it composes."""

    async def test_permission_highlight_survives_an_update_before_compose(self) -> None:
        """has_permission drives the tile-permission CSS class and is the one property
        compose() cannot recover, because compose only renders the label.  Without
        on_mount re-applying it, a tile whose first state is AWAITING_PERMISSION shows the
        warning glyph but never the highlight, for as long as no further state arrives."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            agent_list = app.query_one(AgentList)
            tile = agent_list.add_agent_tile("probe-perm")
            assert tile.is_mounted is False, "precondition: still inside the guard window"
            tile.update_state(AgentState.AWAITING_PERMISSION)
            await pilot.pause()

            assert tile.has_permission is True
            assert tile.has_class("tile-permission")

    async def test_update_mode_before_compose_does_not_raise(self) -> None:
        """An exception out of a Textual message handler breaks the message loop, so a
        raising update_mode would kill the whole UI rather than skip one render."""
        app = _make_app("agent-1")
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            agent_list = app.query_one(AgentList)
            tile = agent_list.add_agent_tile("probe-mode")
            assert tile.is_mounted is False

            tile.update_mode("Plan")  # must not raise
            await pilot.pause()

            assert tile._current_mode == "Plan"
            # The label was actually re-rendered once the tile composed, not merely recorded.
            assert "Plan" in str(tile.query_one(".tile-label", Static).content)
