"""Tests for Phase 4 UI: thought routing, usage display, worker error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from textual.worker import WorkerState

from synth_acp.models.config import SessionConfig
from synth_acp.models.events import AgentThoughtReceived, UsageUpdated
from synth_acp.ui.app import SynthApp


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        project="test",
        agents=[{"id": aid, "cmd": ["echo"]} for aid in agent_ids],
    )


def _make_app(*agent_ids: str) -> SynthApp:
    """Create a SynthApp with a mock broker."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    return SynthApp(broker, _make_config(*agent_ids))


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

        # Build a mock WorkerStateChanged event
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
