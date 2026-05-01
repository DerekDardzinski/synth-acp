"""Tests for AgentLifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
        """Shutdown must force_kill all agents, then cancel remaining tasks."""
        config = _config("busy-agent", "idle-agent")
        reg = AgentRegistry(config)
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        busy = AsyncMock()
        busy.state = AgentState.BUSY
        busy.agent_id = "busy-agent"
        busy.force_kill = MagicMock()
        reg.register("busy-agent", busy)

        idle = AsyncMock()
        idle.state = AgentState.IDLE
        idle.agent_id = "idle-agent"
        idle.force_kill = MagicMock()
        reg.register("idle-agent", idle)

        await lc.shutdown()

        busy.force_kill.assert_called_once()
        idle.force_kill.assert_called_once()

    async def test_shutdown_terminates_agents_concurrently(self) -> None:
        """force_kill is sync so shutdown should complete near-instantly for N agents."""
        config = _config("a", "b", "c")
        reg = AgentRegistry(config)

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        killed: list[str] = []
        for aid in ("a", "b", "c"):
            session = AsyncMock()
            session.state = AgentState.IDLE
            session.agent_id = aid
            session.force_kill = MagicMock(side_effect=lambda _aid=aid: killed.append(_aid))
            reg.register(aid, session)

        t0 = asyncio.get_event_loop().time()
        await lc.shutdown()
        elapsed = asyncio.get_event_loop().time() - t0

        assert set(killed) == {"a", "b", "c"}
        # force_kill is sync — shutdown should be well under 1s
        assert elapsed < 0.5, f"Shutdown took {elapsed:.2f}s — unexpectedly slow"

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


class TestResurrect:
    """Tests for handle_resurrect_command."""

    async def _make_lifecycle(self, tmp_path: Path) -> tuple[AgentLifecycle, list]:
        config = _config()
        reg = AgentRegistry(config)
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")
        db = await lc._ensure_db()
        from synth_acp.db import ensure_schema_async

        await ensure_schema_async(db)
        return lc, events

    async def _insert_agent(
        self,
        lc: AgentLifecycle,
        agent_id: str,
        *,
        status: str = "inactive",
        parent: str | None = None,
        harness: str = "kiro",
        acp_session_id: str | None = None,
        cwd: str = "/tmp",
    ) -> None:
        db = await lc._ensure_db()
        await db.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent, harness, acp_session_id, cwd) "
            "VALUES (?, ?, ?, 1000, ?, ?, ?, ?)",
            (agent_id, lc._session_id, status, parent, harness, acp_session_id, cwd),
        )
        await db.commit()

    async def _insert_command(self, lc: AgentLifecycle, cmd_id: int = 1) -> int:
        db = await lc._ensure_db()
        await db.execute(
            "INSERT INTO agent_commands (id, session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, ?, 'test', 'resurrect', '{}', 'pending', 1000)",
            (cmd_id, lc._session_id),
        )
        await db.commit()
        return cmd_id

    async def _get_command_status(self, lc: AgentLifecycle, cmd_id: int) -> tuple[str, str | None]:
        db = await lc._ensure_db()
        cursor = await db.execute(
            "SELECT status, error FROM agent_commands WHERE id = ?", (cmd_id,)
        )
        row = await cursor.fetchone()
        return (row[0], row[1]) if row else ("not_found", None)

    async def _get_agent_status(self, lc: AgentLifecycle, agent_id: str) -> str | None:
        db = await lc._ensure_db()
        cursor = await db.execute(
            "SELECT status FROM agents WHERE agent_id = ? AND session_id = ?",
            (agent_id, lc._session_id),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def test_resurrect_rejects_wrong_parent(self, tmp_path: Path) -> None:
        lc, events = await self._make_lifecycle(tmp_path)
        try:
            await self._insert_agent(lc, "child", parent="parent-agent")
            cmd_id = await self._insert_command(lc)

            await lc.handle_resurrect_command(cmd_id, "other-agent", {"agent_id": "child"})

            status, error = await self._get_command_status(lc, cmd_id)
            assert status == "rejected"
            assert "Not authorized" in (error or "")
        finally:
            await lc.close_db()

    async def test_resurrect_rejects_non_inactive_agent(self, tmp_path: Path) -> None:
        lc, events = await self._make_lifecycle(tmp_path)
        try:
            await self._insert_agent(lc, "child", status="active", parent="parent-agent")
            cmd_id = await self._insert_command(lc)

            await lc.handle_resurrect_command(cmd_id, "parent-agent", {"agent_id": "child"})

            errors = [e for e in events if isinstance(e, BrokerError)]
            assert len(errors) == 1
            assert "not inactive" in errors[0].message
        finally:
            await lc.close_db()

    async def test_resurrect_success_calls_restore_and_updates_status(self, tmp_path: Path) -> None:
        lc, events = await self._make_lifecycle(tmp_path)
        try:
            await self._insert_agent(
                lc, "child",
                status="inactive",
                parent="parent-agent",
                harness="kiro",
                acp_session_id="sess-123",
                cwd="/tmp",
            )
            cmd_id = await self._insert_command(lc)

            lc.restore = AsyncMock()

            await lc.handle_resurrect_command(cmd_id, "parent-agent", {"agent_id": "child"})

            lc.restore.assert_awaited_once_with(
                agent_id="child",
                acp_session_id="sess-123",
                harness="kiro",
                agent_mode=None,
                cwd="/tmp",
                parent="parent-agent",
            )
            status, error = await self._get_command_status(lc, cmd_id)
            assert status == "processed"
            assert error is None
            assert await self._get_agent_status(lc, "child") == "active"
        finally:
            await lc.close_db()
