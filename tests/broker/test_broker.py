"""Tests for ACPBroker command dispatch and permission integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from acp.schema import PermissionOption

from synth_acp.acp.session import ACPSession
from synth_acp.broker.broker import ACPBroker
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.commands import (
    LaunchAgent,
    RespondPermission,
    SendPrompt,
    SetAgentMode,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError, BrokerEvent, PermissionRequested, UsageUpdated
from synth_acp.models.permissions import PermissionDecision


def _make_config() -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(project="test-session")


def _make_broker(*agent_ids: str, tmp_path: Path) -> ACPBroker:
    """Create a broker with a temp db."""
    config = _make_config()
    first_id = agent_ids[0] if agent_ids else "agent-1"
    initial_agent = AgentConfig(agent_id=first_id, harness="kiro")
    return ACPBroker(
        config=config,
        initial_agent=initial_agent,
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
        idle_session.prompt.assert_awaited_once()
        prompt_text = idle_session.prompt.call_args[0][0]
        assert "hello" in prompt_text
        assert "orchestration_context" in prompt_text

        await broker.handle(SendPrompt(agent_id="agent-2", text="hello"))
        # Drain any HookFired events to find the BrokerError
        events = []
        while not broker._event_queue.empty():
            events.append(broker._event_queue.get_nowait())
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "agent-2" in errors[0].message

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

        with patch.object(broker._permission_engine, "persist_async") as mock_persist:
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
        """SetAgentMode must route through set_config_option to session."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        session._sm._state = AgentState.IDLE
        session.set_config_option = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        await broker.handle(SetAgentMode(agent_id="agent-1", mode_id="architect"))

        session.set_config_option.assert_awaited_once_with("mode", "architect")

    async def test_set_agent_mode_when_not_idle_emits_broker_error(
        self, tmp_path: Path
    ) -> None:
        """SetAgentMode on a non-idle agent must emit BrokerError and not call set_config_option."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        events: list[BrokerEvent] = []
        broker._sink = AsyncMock(side_effect=events.append)  # type: ignore[method-assign]

        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.TERMINATED
        session.set_config_option = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        await broker.handle(SetAgentMode(agent_id="agent-1", mode_id="code"))

        session.set_config_option.assert_not_awaited()
        assert any(isinstance(e, BrokerError) for e in events)

    async def test_set_agent_model_routes_through_set_config_option(
        self, tmp_path: Path
    ) -> None:
        """SetAgentModel must route through set_config_option with config_id='model'."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        session._sm._state = AgentState.IDLE
        session.set_config_option = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        from synth_acp.models.commands import SetAgentModel

        await broker.handle(SetAgentModel(agent_id="agent-1", model_id="claude-4"))

        session.set_config_option.assert_awaited_once_with("model", "claude-4")

    async def test_set_config_option_command_routes_to_lifecycle(
        self, tmp_path: Path
    ) -> None:
        """SetConfigOption must route to lifecycle.set_config_option with all fields."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        session._sm._state = AgentState.IDLE
        session.set_config_option = AsyncMock()  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        from synth_acp.models.commands import SetConfigOption

        await broker.handle(SetConfigOption(agent_id="agent-1", config_id="effort", value="high"))

        session.set_config_option.assert_awaited_once_with("effort", "high")


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
        from synth_acp.db import ensure_schema_sync
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle._db_op(ensure_schema_sync)

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

    async def test_process_commands_when_at_capacity_rejects_with_error(
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

            # At capacity — should reject with descriptive error
            with patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}):
                await broker._process_commands([(cmd_id, "orchestrator", "launch", payload)])

            status, error = self._get_command_status(broker, cmd_id)
            assert status == "rejected"
            assert "Max agents" in error
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()



    async def test_join_broadcast_when_agent_registered_sends_to_visible_agents(
        self, tmp_path: Path
    ) -> None:
        from synth_acp.models.config import (
            CommunicationMode,
            SettingsConfig,
        )

        config = SessionConfig(
            project="test-session",
            settings=SettingsConfig(communication_mode=CommunicationMode.LOCAL),
        )
        initial_agent = AgentConfig(agent_id="orchestrator", harness="kiro")
        broker = ACPBroker(config=config, initial_agent=initial_agent, db_path=tmp_path / "synth.db")
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

            # Default recipients is "none" — no join broadcast messages
            conn = sqlite3.connect(str(broker._db_path))
            rows = conn.execute(
                "SELECT from_agent, to_agent, body FROM messages WHERE session_id = ?",
                (broker._session_id,),
            ).fetchall()
            conn.close()

            assert len(rows) == 0
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()


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

        try:
            with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session):
                await broker.handle(LaunchAgent(agent_id="agent-1", config=AgentConfig(agent_id="agent-1", harness="kiro")))

            # New session replaced the old one
            assert broker._registry._sessions["agent-1"] is mock_session
            # No BrokerError emitted
            assert broker._event_queue.empty()
        finally:
            if broker._message_bus:
                await broker._message_bus.stop()


class TestSelfTerminate:
    async def test_two_pending_messages_before_idle_wake_event_set(self, tmp_path: Path) -> None:
        """Two messages enqueued before IDLE — _sink must wake the bus (not deliver directly)."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        await broker._start_message_bus()
        await broker._ensure_lifecycle()

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

            # _sink only wakes the bus — does not deliver directly
            assert broker._message_bus._wake_event.is_set()
            # Messages still in pending (bus loop delivers them)
            assert broker._message_bus._pending.get("agent-1") is not None
        finally:
            await broker._message_bus.stop()


class TestShutdownOrdering:
    async def test_shutdown_phases_in_order(self, tmp_path: Path) -> None:
        """Shutdown must call lifecycle.shutdown() → message_bus.stop() → flush.
        Wrong ordering causes zombie DB connections or writes after close."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        call_log: list[str] = []

        mock_lifecycle = AsyncMock()
        mock_lifecycle.shutdown = AsyncMock(side_effect=lambda: call_log.append("lifecycle.shutdown"))
        mock_lifecycle.journal_ui_events = AsyncMock()
        broker._lifecycle = mock_lifecycle

        mock_bus = AsyncMock()
        mock_bus.stop = AsyncMock(side_effect=lambda: call_log.append("bus.stop"))
        broker._message_bus = mock_bus

        await broker.shutdown()

        assert call_log == ["lifecycle.shutdown", "bus.stop"]


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



class TestSinkLockGuard:
    async def test_sink_idle_handler_wakes_message_bus(self, tmp_path: Path) -> None:
        """_sink IDLE handler must only call wake() — no pop_pending, no prompt."""
        from unittest.mock import MagicMock

        broker = _make_broker("agent-1", tmp_path=tmp_path)

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session.agent_id = "agent-1"
        broker._registry.register("agent-1", mock_session)

        mock_bus = MagicMock()
        broker._message_bus = mock_bus

        mock_lifecycle = AsyncMock()
        broker._lifecycle = mock_lifecycle

        from synth_acp.models.events import AgentStateChanged

        event = AgentStateChanged(agent_id="agent-1", new_state=AgentState.IDLE, old_state=AgentState.BUSY)
        await broker._sink(event)

        mock_bus.wake.assert_called_once_with("agent-1")
        mock_bus.pop_pending.assert_not_called()
        mock_lifecycle.prompt.assert_not_called()


# ---------------------------------------------------------------------------
# Race condition reproducer: permission flush active-slot race
# ---------------------------------------------------------------------------


class TestPermissionFlushActiveSlotRace:
    """Verify that _flush_permission_queue holds the active slot during auto-resolve awaits."""

    async def test_flush_permission_queue_overwrites_active_slot_set_by_concurrent_sink(
        self, tmp_path: Path,
    ) -> None:
        """During flush's auto-resolve await, a new PermissionRequested via _sink
        must NOT be able to claim the active slot."""
        from synth_acp.models.events import PermissionAutoResolved, PermissionRequested

        config = SessionConfig(project="t")
        initial = AgentConfig(agent_id="a1", harness="kiro")
        broker = ACPBroker(config=config, initial_agent=initial, db_path=tmp_path / "synth.db")

        session = AsyncMock()
        session.resolve_permission = lambda _rid, _oid: None
        broker._registry._sessions["a1"] = session

        object.__setattr__(broker._config.settings, "auto_approve_tools", ("Tool first",))

        first = PermissionRequested(
            agent_id="a1",
            request_id="first",
            title="Tool first",
            kind="execute",
            options=[
                PermissionOption(kind="allow_once", option_id="first-allow", name="Allow"),
                PermissionOption(kind="reject_once", option_id="first-reject", name="Reject"),
            ],
        )
        second = PermissionRequested(
            agent_id="a1",
            request_id="second",
            title="Tool second",
            kind="execute",
            options=[
                PermissionOption(kind="allow_once", option_id="second-allow", name="Allow"),
                PermissionOption(kind="reject_once", option_id="second-reject", name="Reject"),
            ],
        )
        broker._pending_permissions[first.request_id] = first
        broker._pending_permissions[second.request_id] = second
        broker._permission_queue["a1"] = [first, second]
        broker._permission_counter["a1"] = (1, 2)

        real_put = broker._event_queue.put
        auto_resolved_put_started = asyncio.Event()
        auto_resolved_put_unblock = asyncio.Event()

        async def gated_put(event: BrokerEvent) -> None:
            if isinstance(event, PermissionAutoResolved):
                auto_resolved_put_started.set()
                await auto_resolved_put_unblock.wait()
            await real_put(event)

        broker._event_queue.put = gated_put  # type: ignore[method-assign]

        flush_task = asyncio.create_task(broker._flush_permission_queue("a1"))
        await asyncio.wait_for(auto_resolved_put_started.wait(), timeout=1.0)

        concurrent = PermissionRequested(
            agent_id="a1",
            request_id="concurrent",
            title="Tool concurrent",
            kind="execute",
            options=[
                PermissionOption(kind="allow_once", option_id="concurrent-allow", name="Allow"),
                PermissionOption(kind="reject_once", option_id="concurrent-reject", name="Reject"),
            ],
        )
        await broker._sink(concurrent)

        assert broker._active_permission.get("a1") == "first"
        assert concurrent in broker._permission_queue.get("a1", [])

        auto_resolved_put_unblock.set()
        await flush_task

        assert broker._active_permission.get("a1") == "second"
        assert broker._permission_queue.get("a1") == [concurrent]


class TestListRestorableSessions:
    async def test_list_restorable_sessions_returns_enriched_fields(self, tmp_path: Path) -> None:
        """list_restorable_sessions must return cwd, tasks, first_messages."""
        import json
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "synth.db"
        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)

        # Root agent with cwd
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, cwd, task) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("root", "sess-1", "restorable", 1000, "/home/user/project", "build feature"),
        )
        # Child agent (terminated — should still appear in agents list)
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, task) "
            "VALUES (?, ?, ?, ?, ?)",
            ("child", "sess-1", "terminated", 2000, "fix tests"),
        )
        # UserPromptSubmitted events
        for i, text in enumerate(["hello world", "do the thing", "third msg", "fourth"]):
            payload = json.dumps({"agent_id": "root", "text": text})
            conn.execute(
                "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("sess-1", "root", i, "UserPromptSubmitted", payload, 1000 + i),
            )
        conn.commit()
        conn.close()

        results = await ACPBroker.list_restorable_sessions(db_path)
        assert len(results) == 1
        sess = results[0]
        assert set(sess["agents"]) == {"root", "child"}
        assert sess["cwd"] == "/home/user/project"
        assert sess["tasks"] == ["build feature", "fix tests"]
        assert sess["first_messages"] == ["hello world", "do the thing", "third msg"]

    async def test_list_restorable_sessions_includes_initial_prompts(self, tmp_path: Path) -> None:
        """initial_prompts populated from InitialPromptDelivered events."""
        import json
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "synth.db"
        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)

        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, cwd) "
            "VALUES (?, ?, ?, ?, ?)",
            ("root", "sess-1", "restorable", 1000, "/home/user/project"),
        )
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) "
            "VALUES (?, ?, ?, ?)",
            ("child", "sess-1", "restorable", 2000),
        )
        # InitialPromptDelivered for child (seq=0, fires before user prompt)
        conn.execute(
            "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("sess-1", "child", 0, "InitialPromptDelivered",
             json.dumps({"agent_id": "child", "text": "You are a code reviewer"}), 1000),
        )
        # UserPromptSubmitted for root (no InitialPromptDelivered)
        conn.execute(
            "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("sess-1", "root", 1, "UserPromptSubmitted",
             json.dumps({"agent_id": "root", "text": "Fix the auth bug"}), 1001),
        )
        conn.commit()
        conn.close()

        results = await ACPBroker.list_restorable_sessions(db_path)
        assert results[0]["initial_prompts"] == {
            "child": "You are a code reviewer",
            "root": "Fix the auth bug",
        }

    async def test_list_restorable_sessions_initial_prompts_empty_when_no_events(self, tmp_path: Path) -> None:
        """initial_prompts is always present as empty dict when no qualifying events."""
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "synth.db"
        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)

        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, cwd) "
            "VALUES (?, ?, ?, ?, ?)",
            ("root", "sess-1", "restorable", 1000, "/tmp"),
        )
        conn.commit()
        conn.close()

        results = await ACPBroker.list_restorable_sessions(db_path)
        assert results[0]["initial_prompts"] == {}
