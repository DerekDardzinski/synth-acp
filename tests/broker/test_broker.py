"""Tests for ACPBroker command dispatch and permission integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentState
from synth_acp.models.commands import RespondPermission, SendPrompt
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig with the given agent IDs."""
    return SessionConfig(
        session="test-session",
        agents=[{"id": aid, "binary": "echo", "args": [], "cwd": "."} for aid in agent_ids],
    )


def _make_broker(*agent_ids: str, tmp_path: Path) -> ACPBroker:
    """Create a broker with a temp rules file."""
    config = _make_config(*agent_ids)
    return ACPBroker(
        config=config,
        db_path=tmp_path / "synth.db",
        rules_path=tmp_path / "rules.json",
    )


class TestBrokerDispatch:
    async def test_handle_when_respond_permission_resolves_future(self, tmp_path: Path) -> None:
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        # Create a fake session with a pending future
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

        # Manually set up a permission future
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        session._permission_future = future

        broker._sessions["agent-1"] = session

        await broker.handle(RespondPermission(agent_id="agent-1", option_id="opt-allow"))

        assert future.done()
        assert future.result() == "opt-allow"

    async def test_handle_when_send_prompt_to_idle_agent_prompts(self, tmp_path: Path) -> None:
        broker = _make_broker("agent-1", "agent-2", tmp_path=tmp_path)

        # Create two sessions: one IDLE, one BUSY
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

        # Prompt the idle agent — should succeed (fire-and-forget task)
        await broker.handle(SendPrompt(agent_id="agent-1", text="hello"))
        await asyncio.sleep(0)  # let the prompt task run
        idle_session.prompt.assert_awaited_once_with("hello")

        # Prompt the busy agent — should emit BrokerError
        await broker.handle(SendPrompt(agent_id="agent-2", text="hello"))
        event = broker._event_queue.get_nowait()
        assert isinstance(event, BrokerError)
        assert "agent-2" in event.message
