"""Tests for ACPBroker command dispatch and permission integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from acp.schema import PermissionOption

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentState
from synth_acp.models.commands import LaunchAgent, RespondPermission, SendPrompt
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


class TestProcessCommands:
    """Tests for broker command processing (Phase 3)."""

    async def _init_broker_db(self, broker: ACPBroker) -> None:
        """Initialize the broker DB with schema."""
        await broker._register_agents()

    def _insert_command(
        self,
        broker: ACPBroker,
        from_agent: str,
        command: str,
        payload: str,
        cmd_id: int = 1,
    ) -> int:
        """Insert a pending command into the DB and return its ID."""
        import sqlite3
        import time

        conn = sqlite3.connect(str(broker._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        now = int(time.time() * 1000)
        cursor = conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (broker._session_id, from_agent, command, payload, now),
        )
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id  # type: ignore[return-value]

    def _get_command_status(self, broker: ACPBroker, cmd_id: int) -> tuple[str, str | None]:
        """Read command status and error from DB."""
        import sqlite3

        conn = sqlite3.connect(str(broker._db_path))
        row = conn.execute(
            "SELECT status, error FROM agent_commands WHERE id = ?", (cmd_id,)
        ).fetchone()
        conn.close()
        return (row[0], row[1]) if row else ("not_found", None)

    async def test_process_commands_when_launch_with_valid_harness_spawns_session(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            # Mock ACPSession to avoid real subprocess
            mock_session = AsyncMock()
            mock_session.state = AgentState.IDLE
            mock_session.run = AsyncMock()

            import json

            payload = json.dumps(
                {
                    "agent_id": "worker-1",
                    "agent_name": "implementor",
                    "harness": "kiro",
                    "cwd": "/tmp",
                    "task": "Fix auth",
                    "message": "",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            with patch("synth_acp.broker.broker.ACPSession", return_value=mock_session):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            # Session created
            assert "worker-1" in broker._sessions
            # Command processed
            status, error = self._get_command_status(broker, cmd_id)
            assert status == "processed"
            assert error is None
            # Parentage tracked
            assert broker._agent_parents["worker-1"] == "orchestrator"
        finally:
            if broker._db:
                await broker._db.close()

    async def test_process_commands_when_launch_with_unknown_harness_rejects(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            import json

            payload = json.dumps(
                {
                    "agent_id": "worker-1",
                    "agent_name": "implementor",
                    "harness": "nonexistent",
                    "cwd": ".",
                    "task": "",
                    "message": "",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            status, error = self._get_command_status(broker, cmd_id)
            assert status == "rejected"
            assert "Unknown harness" in (error or "")
        finally:
            if broker._db:
                await broker._db.close()

    async def test_process_commands_when_terminate_by_non_parent_rejects(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            # Set up child agent with parent
            mock_session = AsyncMock()
            mock_session.state = AgentState.IDLE
            broker._sessions["child"] = mock_session
            broker._agent_parents["child"] = "orchestrator"

            import json

            payload = json.dumps({"agent_id": "child"})
            cmd_id = self._insert_command(broker, "stranger", "terminate", payload)

            await broker._process_commands([(cmd_id, "stranger", "terminate", payload)])

            status, error = self._get_command_status(broker, cmd_id)
            assert status == "rejected"
            assert "Not authorized" in (error or "")
            # Session still alive
            assert "child" in broker._sessions
        finally:
            if broker._db:
                await broker._db.close()

    async def test_process_commands_when_slot_opens_processes_queued_launch(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            # Set up 1 active agent at capacity
            mock_active = AsyncMock()
            mock_active.state = AgentState.IDLE
            broker._sessions["orchestrator"] = mock_active

            import json

            payload = json.dumps(
                {
                    "agent_id": "worker-1",
                    "agent_name": "implementor",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "",
                    "message": "",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            # At capacity — should leave as pending
            with patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            status, _ = self._get_command_status(broker, cmd_id)
            assert status == "pending"

            # Terminate the active agent to free a slot
            mock_active.state = AgentState.TERMINATED

            mock_new = AsyncMock()
            mock_new.state = AgentState.IDLE
            mock_new.run = AsyncMock()

            with (
                patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}),
                patch("synth_acp.broker.broker.ACPSession", return_value=mock_new),
            ):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            assert "worker-1" in broker._sessions
            status, _ = self._get_command_status(broker, cmd_id)
            assert status == "processed"
        finally:
            if broker._db:
                await broker._db.close()

    async def test_process_commands_when_agent_reaches_idle_sends_initial_message(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            mock_session = AsyncMock()
            mock_session.state = AgentState.IDLE
            mock_session.run = AsyncMock()
            mock_session.prompt = AsyncMock()

            import json

            payload = json.dumps(
                {
                    "agent_id": "worker-1",
                    "agent_name": "implementor",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Fix auth",
                    "message": "Start working on auth",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            with patch("synth_acp.broker.broker.ACPSession", return_value=mock_session):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            # Pending initial prompt stored
            assert "worker-1" in broker._pending_initial_prompts

            # Simulate IDLE transition
            from synth_acp.models.events import AgentStateChanged

            await broker._sink(
                AgentStateChanged(
                    agent_id="worker-1",
                    old_state=AgentState.INITIALIZING,
                    new_state=AgentState.IDLE,
                )
            )
            await asyncio.sleep(0)  # Let create_task fire

            mock_session.prompt.assert_awaited_once_with("Start working on auth")
            assert "worker-1" not in broker._pending_initial_prompts
        finally:
            if broker._db:
                await broker._db.close()

    async def test_join_broadcast_when_agent_registered_sends_to_visible_agents(
        self, tmp_path: Path
    ) -> None:
        from synth_acp.models.config import CommunicationMode, SettingsConfig

        config = SessionConfig(
            project="test-session",
            agents=[{"id": "orchestrator", "cmd": ["echo"], "cwd": "."}],
            settings=SettingsConfig(communication_mode=CommunicationMode.LOCAL),
        )
        broker = ACPBroker(config=config, db_path=tmp_path / "synth.db")
        await self._init_broker_db(broker)
        try:

            # Set up orchestrator as active
            mock_orch = AsyncMock()
            mock_orch.state = AgentState.IDLE
            broker._sessions["orchestrator"] = mock_orch

            mock_worker = AsyncMock()
            mock_worker.state = AgentState.IDLE
            mock_worker.run = AsyncMock()

            import json
            import sqlite3

            payload = json.dumps(
                {
                    "agent_id": "worker",
                    "agent_name": "implementor",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Fix auth",
                    "message": "",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            with patch("synth_acp.broker.broker.ACPSession", return_value=mock_worker):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            # Check messages table for join broadcast
            conn = sqlite3.connect(str(broker._db_path))
            rows = conn.execute(
                "SELECT from_agent, to_agent, body FROM messages WHERE session_id = ?",
                (broker._session_id,),
            ).fetchall()
            conn.close()

            assert len(rows) == 1
            assert rows[0][0] == "system"
            assert rows[0][1] == "orchestrator"
            assert rows[0][2] == '[System] Agent "worker" has joined. Task: Fix auth.'
        finally:
            if broker._db:
                await broker._db.close()


class TestRelaunchTerminatedAgent:
    async def test_launch_when_agent_terminated_relaunches_without_error(self, tmp_path: Path) -> None:
        """Re-launching a terminated agent should succeed without BrokerError."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        from synth_acp.acp.session import ACPSession

        # First session — mark as terminated
        old_session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        old_session.state = AgentState.TERMINATED
        broker._sessions["agent-1"] = old_session

        # Re-launch should clean up old session and create a new one
        mock_session = AsyncMock()
        mock_session.state = AgentState.INITIALIZING
        mock_session.run = AsyncMock()

        with patch("synth_acp.broker.broker.ACPSession", return_value=mock_session):
            await broker.handle(LaunchAgent(agent_id="agent-1"))

        # New session replaced the old one
        assert broker._sessions["agent-1"] is mock_session
        # No BrokerError emitted
        assert broker._event_queue.empty()
