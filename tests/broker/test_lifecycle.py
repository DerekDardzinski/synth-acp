"""Tests for AgentLifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from synth_acp.broker.lifecycle import AgentLifecycle
from synth_acp.broker.registry import AgentRegistry
from synth_acp.models.agent import AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError


def _config(*ids: str) -> SessionConfig:
    return SessionConfig(
        project="test",
        agents=[{"agent_id": aid, "harness": "kiro"} for aid in ids],
    )


class TestTaskCleanup:
    async def test_run_task_removed_after_agent_exits(self) -> None:
        config = _config("a")
        reg = AgentRegistry(config)
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session.agent_id = "a"

        async def fake_run() -> None:
            pass

        mock_session.run = fake_run
        reg.register("a", mock_session)

        task = lc._make_run_task("a", mock_session)
        lc._tasks["a"] = task
        await task
        await asyncio.sleep(0)  # Let done callback fire
        assert "a" not in lc._tasks

    async def test_prompt_task_removed_after_completion(self) -> None:
        config = _config("a")
        reg = AgentRegistry(config)

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        async def fake_prompt() -> None:
            pass

        task = lc._make_prompt_task("a", fake_prompt())
        lc._tasks["prompt-a"] = task
        await task
        await asyncio.sleep(0)
        assert "prompt-a" not in lc._tasks


class TestPromptGuard:
    async def test_prompt_rejects_non_idle_agent(self) -> None:
        config = _config("a")
        reg = AgentRegistry(config)
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.BUSY
        reg.register("a", mock_session)

        await lc.prompt("a", "hello")
        assert any(isinstance(e, BrokerError) for e in events)


class TestLifecycleShutdown:
    async def test_shutdown_terminates_all_then_cancels_tasks(self) -> None:
        """Shutdown must cancel BUSY agents, terminate all non-TERMINATED,
        then cancel remaining tasks."""
        config = _config("busy-agent", "idle-agent")
        reg = AgentRegistry(config)
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        busy = AsyncMock()
        busy.state = AgentState.BUSY
        busy.agent_id = "busy-agent"
        busy.cancel = AsyncMock()
        busy.terminate = AsyncMock()
        reg.register("busy-agent", busy)

        idle = AsyncMock()
        idle.state = AgentState.IDLE
        idle.agent_id = "idle-agent"
        idle.terminate = AsyncMock()
        reg.register("idle-agent", idle)

        await lc.shutdown()

        busy.cancel.assert_awaited_once()
        busy.terminate.assert_awaited()
        idle.terminate.assert_awaited()

    async def test_terminate_times_out_on_unresponsive_agent(self, tmp_path: Path) -> None:
        """If session.terminate() hangs, lifecycle must not block forever."""
        config = _config("stuck")
        reg = AgentRegistry(config)

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")
        lc._terminate_timeout = 0.1  # Fast timeout for testing

        stuck_session = AsyncMock()
        stuck_session.state = AgentState.IDLE
        stuck_session.agent_id = "stuck"

        async def hang_forever() -> None:
            await asyncio.sleep(60)

        stuck_session.terminate = hang_forever
        reg.register("stuck", stuck_session)

        # Mock DB so we don't need schema
        mock_db = AsyncMock()
        lc._db = mock_db

        t0 = asyncio.get_event_loop().time()
        await lc.terminate("stuck")
        elapsed = asyncio.get_event_loop().time() - t0
        assert elapsed < 1.0
