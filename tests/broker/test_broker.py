"""Tests for ACPBroker command dispatch and permission integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from acp.schema import PermissionOption

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentState
from synth_acp.models.commands import RespondPermission, SendPrompt
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError, PermissionRequested, UsageUpdated
from synth_acp.models.permissions import PermissionDecision


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig with the given agent IDs."""
    return SessionConfig(
        project="test-session",
        agents=[{"id": aid, "cmd": ["echo"], "cwd": "."} for aid in agent_ids],
    )


def _make_broker(*agent_ids: str, tmp_path: Path) -> ACPBroker:
    """Create a broker with a temp db."""
    config = _make_config(*agent_ids)
    return ACPBroker(
        config=config,
        db_path=tmp_path / "synth.db",
    )


class TestBrokerDispatch:
    async def test_handle_when_respond_permission_resolves_future(self, tmp_path: Path) -> None:
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        async def noop_sink(event: object) -> None:
            pass

        from synth_acp.acp.session import ACPSession

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=noop_sink,
        )
        session.state = AgentState.BUSY

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        session._permission_future = future

        broker._sessions["agent-1"] = session

        await broker.handle(RespondPermission(agent_id="agent-1", option_id="opt-allow"))

        assert future.done()
        assert future.result() == "opt-allow"

    async def test_handle_when_send_prompt_to_idle_agent_prompts(self, tmp_path: Path) -> None:
        broker = _make_broker("agent-1", "agent-2", tmp_path=tmp_path)

        from synth_acp.acp.session import ACPSession

        idle_session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        idle_session.state = AgentState.IDLE
        idle_session.prompt = AsyncMock()  # type: ignore[method-assign]

        busy_session = ACPSession(
            agent_id="agent-2",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        busy_session.state = AgentState.BUSY

        broker._sessions["agent-1"] = idle_session
        broker._sessions["agent-2"] = busy_session

        await broker.handle(SendPrompt(agent_id="agent-1", text="hello"))
        await asyncio.sleep(0)
        idle_session.prompt.assert_awaited_once_with("hello")

        await broker.handle(SendPrompt(agent_id="agent-2", text="hello"))
        event = broker._event_queue.get_nowait()
        assert isinstance(event, BrokerError)
        assert "agent-2" in event.message

    async def test_resolve_permission_when_always_option_persists_rule(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        # Set up a fake session so resolve_permission can call session.resolve_permission
        from synth_acp.acp.session import ACPSession

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        session.state = AgentState.BUSY
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        session._permission_future = future
        broker._sessions["agent-1"] = session

        # Store a pending permission event
        broker._pending_permissions["agent-1"] = PermissionRequested(
            agent_id="agent-1",
            request_id="req-1",
            title="Run command",
            kind="execute",
            options=[
                PermissionOption(kind="allow_always", option_id="opt-1", name="Always allow"),
                PermissionOption(kind="reject_once", option_id="opt-2", name="Reject"),
            ],
        )

        with patch.object(broker._permission_engine, "persist") as mock_persist:
            broker._resolve_permission("agent-1", "opt-1")

        mock_persist.assert_called_once()
        rule = mock_persist.call_args[0][0]
        assert rule.agent_id == "agent-1"
        assert rule.tool_kind == "execute"
        assert rule.session_id == broker._session_id
        assert rule.decision == PermissionDecision.allow_always


class TestBrokerUsageAccumulation:
    async def test_broker_get_usage_when_multiple_updates_keeps_latest(
        self, tmp_path: Path
    ) -> None:
        """SDK cost is already cumulative — broker must store latest, not sum."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        event1 = UsageUpdated(
            agent_id="agent-1", size=128000, used=20000, cost_amount=0.10, cost_currency="USD"
        )
        event2 = UsageUpdated(
            agent_id="agent-1", size=128000, used=32000, cost_amount=0.15, cost_currency="USD"
        )

        broker._accumulate_usage(event1)
        broker._accumulate_usage(event2)

        result = broker.get_usage("agent-1")
        assert result is not None
        assert result.cost_amount == pytest.approx(0.15)
        assert result.size == 128000
        assert result.used == 32000
        assert result.cost_currency == "USD"
