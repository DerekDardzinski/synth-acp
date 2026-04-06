"""Tests for ACPBroker command dispatch and permission integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from acp.schema import PermissionOption

from synth_acp.acp.session import ACPSession
from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentState
from synth_acp.models.commands import (
    LaunchAgent,
    RespondPermission,
    SendPrompt,
    SetAgentMode,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError, BrokerEvent, PermissionRequested, UsageUpdated
from synth_acp.models.permissions import PermissionDecision


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig with the given agent IDs."""
    return SessionConfig(
        project="test-session",
        agents=[{"agent_id": aid, "harness": "kiro"} for aid in agent_ids],
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
        session._sm._state = AgentState.BUSY

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        session._permission_futures["req-1"] = future

        broker._registry._sessions["agent-1"] = session

        await broker.handle(RespondPermission(agent_id="agent-1", request_id="req-1", option_id="opt-allow"))

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
        idle_session._sm._state = AgentState.IDLE
        idle_session.prompt = AsyncMock()  # type: ignore[method-assign]

        busy_session = ACPSession(
            agent_id="agent-2",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        busy_session._sm._state = AgentState.BUSY

        broker._registry._sessions["agent-1"] = idle_session
        broker._registry._sessions["agent-2"] = busy_session

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
        session._sm._state = AgentState.BUSY
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        session._permission_futures["req-1"] = future
        broker._registry._sessions["agent-1"] = session

        # Store a pending permission event
        broker._pending_permissions["req-1"] = PermissionRequested(
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
            await broker._resolve_permission("agent-1", "req-1", "opt-1")

        mock_persist.assert_called_once()
        rule = mock_persist.call_args[0][0]
        assert rule.agent_id == "agent-1"
        assert rule.tool_kind == "execute"
        assert rule.session_id == broker._session_id
        assert rule.decision == PermissionDecision.allow_always

    async def test_set_agent_mode_when_idle_calls_session_set_mode(
        self, tmp_path: Path
    ) -> None:
        """SetAgentMode must forward to session.set_mode() when agent is IDLE."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        session._sm._state = AgentState.IDLE
        session.set_mode = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        await broker.handle(SetAgentMode(agent_id="agent-1", mode_id="architect"))

        # The broker delegates to session — the call IS the contract.
        # session.set_mode handles state transitions and event emission.
        session.set_mode.assert_awaited_once_with("architect")

    async def test_set_agent_mode_when_not_idle_emits_broker_error(
        self, tmp_path: Path
    ) -> None:
        """SetAgentMode on a non-idle agent must emit BrokerError and not call set_mode."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        events: list[BrokerEvent] = []
        broker._sink = AsyncMock(side_effect=events.append)  # type: ignore[method-assign]

        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.TERMINATED
        session.set_mode = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        await broker.handle(SetAgentMode(agent_id="agent-1", mode_id="code"))

        session.set_mode.assert_not_awaited()
        assert any(isinstance(e, BrokerError) for e in events)


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

        broker._registry.update_usage(event1)
        broker._registry.update_usage(event2)

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
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle.register_agents()

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
                    "agent_mode": "",
                    "harness": "kiro",
                    "cwd": "/tmp",
                    "task": "Fix auth",
                    "message": "",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            # Session created
            assert "worker-1" in broker._registry._sessions
            # Command processed
            status, error = self._get_command_status(broker, cmd_id)
            assert status == "processed"
            assert error is None
            # Parentage tracked
            assert broker._registry._parents["worker-1"] == "orchestrator"
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()
            if broker._lifecycle:
                await broker._lifecycle.close_db()

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
                    "agent_mode": "",
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
            if broker._message_bus:
                await broker._message_bus.stop()
            if broker._lifecycle:
                await broker._lifecycle.close_db()

    async def test_process_commands_when_terminate_by_non_parent_rejects(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            # Set up child agent with parent
            mock_session = AsyncMock()
            mock_session.state = AgentState.IDLE
            broker._registry._sessions["child"] = mock_session
            broker._registry._parents["child"] = "orchestrator"

            import json

            payload = json.dumps({"agent_id": "child"})
            cmd_id = self._insert_command(broker, "stranger", "terminate", payload)

            await broker._process_commands([(cmd_id, "stranger", "terminate", payload)])

            status, error = self._get_command_status(broker, cmd_id)
            assert status == "rejected"
            assert "Not authorized" in (error or "")
            # Session still alive
            assert "child" in broker._registry._sessions
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()
            if broker._lifecycle:
                await broker._lifecycle.close_db()

    async def test_process_commands_when_slot_opens_processes_queued_launch(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:

            # Set up 1 active agent at capacity
            mock_active = AsyncMock()
            mock_active.state = AgentState.IDLE
            broker._registry._sessions["orchestrator"] = mock_active

            import json

            payload = json.dumps(
                {
                    "agent_id": "worker-1",
                    "agent_mode": "",
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
                patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_new),
            ):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            assert "worker-1" in broker._registry._sessions
            status, _ = self._get_command_status(broker, cmd_id)
            assert status == "processed"
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()
            if broker._lifecycle:
                await broker._lifecycle.close_db()

    async def test_process_commands_when_agent_reaches_idle_sends_initial_message(
        self, tmp_path: Path
    ) -> None:
        broker = _make_broker("orchestrator", tmp_path=tmp_path)
        await self._init_broker_db(broker)
        try:
            await broker._start_message_bus()
            await broker._ensure_lifecycle()

            mock_session = AsyncMock()
            mock_session.state = AgentState.IDLE
            mock_session.run = AsyncMock()
            mock_session.prompt = AsyncMock()

            import json

            payload = json.dumps(
                {
                    "agent_id": "worker-1",
                    "agent_mode": "",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Fix auth",
                    "message": "Start working on auth",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            assert broker._message_bus is not None
            assert broker._message_bus._pending.get("worker-1") is not None

            from synth_acp.models.events import AgentStateChanged

            broker._registry._sessions["worker-1"] = mock_session
            await broker._sink(
                AgentStateChanged(
                    agent_id="worker-1",
                    old_state=AgentState.INITIALIZING,
                    new_state=AgentState.IDLE,
                )
            )
            await asyncio.sleep(0)

            mock_session.prompt.assert_awaited_once()
            assert broker._message_bus._pending.get("worker-1") is None
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()
            if broker._lifecycle:
                await broker._lifecycle.close_db()

    async def test_join_broadcast_when_agent_registered_sends_to_visible_agents(
        self, tmp_path: Path
    ) -> None:
        from synth_acp.models.config import CommunicationMode, SettingsConfig

        config = SessionConfig(
            project="test-session",
            agents=[{"agent_id": "orchestrator", "harness": "kiro"}],
            settings=SettingsConfig(communication_mode=CommunicationMode.LOCAL),
        )
        broker = ACPBroker(config=config, db_path=tmp_path / "synth.db")
        await self._init_broker_db(broker)
        try:

            # Set up orchestrator as active
            mock_orch = AsyncMock()
            mock_orch.state = AgentState.IDLE
            broker._registry._sessions["orchestrator"] = mock_orch

            mock_worker = AsyncMock()
            mock_worker.state = AgentState.IDLE
            mock_worker.run = AsyncMock()

            import json
            import sqlite3

            payload = json.dumps(
                {
                    "agent_id": "worker",
                    "agent_mode": "",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Fix auth",
                    "message": "",
                }
            )
            cmd_id = self._insert_command(broker, "orchestrator", "launch", payload)

            with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_worker):
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
            if broker._message_bus:
                await broker._message_bus.stop()
            if broker._lifecycle:
                await broker._lifecycle.close_db()


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
        old_session._sm._state = AgentState.TERMINATED
        broker._registry._sessions["agent-1"] = old_session

        # Re-launch should clean up old session and create a new one
        mock_session = AsyncMock()
        mock_session.state = AgentState.INITIALIZING
        mock_session.run = AsyncMock()

        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session):
            await broker.handle(LaunchAgent(agent_id="agent-1"))

        # New session replaced the old one
        assert broker._registry._sessions["agent-1"] is mock_session
        # No BrokerError emitted
        assert broker._event_queue.empty()


class TestSelfTerminate:
    async def test_self_terminate_emits_terminated_event(self, tmp_path: Path) -> None:
        """self_terminate command must transition agent to TERMINATED and emit
        AgentStateChanged. Without this, the TUI never learns the agent left."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.IDLE
        broker._registry._sessions["agent-1"] = session

        # Ensure lifecycle + schema exist so update_command_status works
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle.register_agents()

        try:
            await broker._handle_self_terminate_command(cmd_id=1, from_agent="agent-1")

            assert session.state == AgentState.TERMINATED
            events = []
            while not broker._event_queue.empty():
                events.append(broker._event_queue.get_nowait())
            from synth_acp.models.events import AgentStateChanged

            assert any(
                isinstance(e, AgentStateChanged) and e.new_state == AgentState.TERMINATED
                for e in events
            )
        finally:
            await lifecycle.close_db()

    async def test_self_terminate_does_not_call_session_terminate(self, tmp_path: Path) -> None:
        """self_terminate must use force_terminate(), not terminate().
        terminate() kills the process — wrong for an agent voluntarily leaving."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.IDLE
        session.terminate = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        lifecycle = await broker._ensure_lifecycle()
        await lifecycle.register_agents()
        try:
            await broker._handle_self_terminate_command(cmd_id=1, from_agent="agent-1")
            session.terminate.assert_not_awaited()
            assert session.state == AgentState.TERMINATED
        finally:
            await lifecycle.close_db()

    async def test_two_pending_messages_before_idle_both_delivered(self, tmp_path: Path) -> None:
        """Two messages enqueued before IDLE must both appear in the prompt.
        Catches the old dict-overwrite bug where only the last message survived."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        await broker._start_message_bus()
        lifecycle = await broker._ensure_lifecycle()

        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.IDLE
        session.prompt = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        assert broker._message_bus is not None
        broker._message_bus.enqueue_pending("agent-1", "sender-a", "msg1")
        broker._message_bus.enqueue_pending("agent-1", "sender-b", "msg2")

        from synth_acp.models.events import AgentStateChanged

        try:
            await broker._sink(
                AgentStateChanged(
                    agent_id="agent-1",
                    old_state=AgentState.INITIALIZING,
                    new_state=AgentState.IDLE,
                )
            )
            await asyncio.sleep(0)

            session.prompt.assert_awaited_once()
            prompt_text = session.prompt.call_args[0][0]
            assert "msg1" in prompt_text
            assert "msg2" in prompt_text
        finally:
            await broker._message_bus.stop()
            await lifecycle.close_db()


class TestShutdownOrdering:
    async def test_shutdown_phases_in_order(self, tmp_path: Path) -> None:
        """Shutdown must call lifecycle.shutdown() → message_bus.stop() → lifecycle.close_db().
        Wrong ordering causes zombie DB connections or writes after close."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        call_log: list[str] = []

        mock_lifecycle = AsyncMock()
        mock_lifecycle.shutdown = AsyncMock(side_effect=lambda: call_log.append("lifecycle.shutdown"))
        mock_lifecycle.close_db = AsyncMock(side_effect=lambda: call_log.append("lifecycle.close_db"))
        broker._lifecycle = mock_lifecycle

        mock_bus = AsyncMock()
        mock_bus.stop = AsyncMock(side_effect=lambda: call_log.append("bus.stop"))
        broker._message_bus = mock_bus

        await broker.shutdown()

        assert call_log == ["lifecycle.shutdown", "bus.stop", "lifecycle.close_db"]


class TestEventQueueBackpressure:
    async def test_queue_full_drops_chunk_events_not_state_events(self, tmp_path: Path) -> None:
        """When the event queue is full, MessageChunkReceived must be dropped
        but AgentStateChanged must block until space is available.
        Without this, state events are silently lost."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        broker._event_queue = asyncio.Queue(maxsize=2)

        from synth_acp.models.events import AgentStateChanged, MessageChunkReceived

        # Fill the queue
        await broker._event_queue.put(
            AgentStateChanged(agent_id="agent-1", old_state=AgentState.UNSTARTED, new_state=AgentState.INITIALIZING)
        )
        await broker._event_queue.put(
            AgentStateChanged(agent_id="agent-1", old_state=AgentState.INITIALIZING, new_state=AgentState.IDLE)
        )
        assert broker._event_queue.full()

        # Chunk should be silently dropped
        await broker._sink(MessageChunkReceived(agent_id="agent-1", chunk="hello"))
        assert broker._event_queue.qsize() == 2  # unchanged

        # State event should block — drain one first to make room
        broker._event_queue.get_nowait()
        await broker._sink(
            AgentStateChanged(agent_id="agent-1", old_state=AgentState.IDLE, new_state=AgentState.BUSY)
        )
        assert broker._event_queue.qsize() == 2  # filled back up

