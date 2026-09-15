"""Tests for ACPBroker command dispatch and permission integration."""

from __future__ import annotations

import ast
import asyncio
import inspect
import sqlite3
import textwrap
import time
import warnings
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.contrib.session_state import SessionAccumulator
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    McpServerStdio,
    PermissionOption,
    PlanEntry,
    SessionNotification,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)

from synth_acp.acp.session import DRAIN_PASSES, ACPSession, _StderrTail
from synth_acp.broker.broker import ACPBroker
from synth_acp.broker.prompt_queue import QueuedItem
from synth_acp.db import ensure_schema_sync
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.commands import (
    LaunchAgent,
    RespondPermission,
    SendPrompt,
    SetAgentMode,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentHandedOff,
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerError,
    BrokerEvent,
    MessageChunkReceived,
    MessageSteered,
    PermissionRequested,
    PlanReceived,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
    UserPromptSubmitted,
)
from synth_acp.models.permissions import PermissionDecision


def _make_config() -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(project="test-session")


def _make_broker(
    *agent_ids: str, tmp_path: Path, event_queue_maxsize: int = 2000
) -> ACPBroker:
    """Create a broker with a temp db.

    Args:
        agent_ids: Agent ids; the first becomes the initial agent.
        tmp_path: Directory for the temp database.
        event_queue_maxsize: Event queue bound. Lower it to force real QueueFull.
    """
    config = _make_config()
    first_id = agent_ids[0] if agent_ids else "agent-1"
    initial_agent = AgentConfig(agent_id=first_id, harness="kiro")
    return ACPBroker(
        config=config,
        initial_agent=initial_agent,
        db_path=tmp_path / "synth.db",
        event_queue_maxsize=event_queue_maxsize,
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

        # Sending to busy agent enqueues instead of erroring (unified prompt queue)
        await broker.handle(SendPrompt(agent_id="agent-2", text="hello"))
        assert not broker._prompt_queue.is_empty("agent-2")

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


class TestPromptQueueDrainOnIdle:
    async def test_queued_message_drains_on_idle(self, tmp_path: Path) -> None:
        """MCP messages queued while busy drain on IDLE with signed text.

        Enqueues via _on_mcp_message (the real bus->broker seam) so the queued
        item carries the already-signed text; the drain path must deliver it
        verbatim. Guards signed delivery through the DRAIN path.
        """
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        await broker._start_message_bus()
        lifecycle = await broker._ensure_lifecycle()
        # Isolate the formatting seam from lifecycle startup-context prepending.
        lifecycle._first_prompted.add("agent-1")

        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.BUSY
        prompt_mock = AsyncMock()
        session.prompt = prompt_mock  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session

        from synth_acp.models.events import AgentStateChanged

        try:
            # Deliver while busy — goes to queue as signed text
            await broker._on_mcp_message("agent-1", "msg1", "sender-a", "chat")
            assert not broker._prompt_queue.is_empty("agent-1")
            prompt_mock.assert_not_awaited()

            # Transition to IDLE — triggers drain
            session._sm._state = AgentState.IDLE
            await broker._sink(
                AgentStateChanged(
                    agent_id="agent-1",
                    old_state=AgentState.BUSY,
                    new_state=AgentState.IDLE,
                )
            )

            # Queue drained, prompt dispatched (as a task) with signed text
            assert broker._prompt_queue.is_empty("agent-1")
            await asyncio.sleep(0)  # let the prompt task run
            prompt_mock.assert_awaited_once_with("[Message from sender-a]: msg1")
        finally:
            await broker._message_bus.stop()


class TestMcpMessageFormatting:
    """Formatting is applied once at _on_mcp_message and bound to that seam only."""

    def _idle_session(self, broker: ACPBroker) -> AsyncMock:
        session = ACPSession(
            agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
        )
        session._sm._state = AgentState.IDLE
        prompt_mock = AsyncMock()
        session.prompt = prompt_mock  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session
        return prompt_mock

    async def test_on_mcp_message_direct_delivery_signs_text(self, tmp_path: Path) -> None:
        """Idle recipient: _on_mcp_message delivers signed text via the direct path."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        prompt = self._idle_session(broker)

        await broker._on_mcp_message("agent-1", "hello", "agent-a", "chat")
        await asyncio.sleep(0)

        prompt.assert_awaited_once_with("[Message from agent-a]: hello")

    async def test_on_mcp_message_system_kind_uses_no_action_wrapper(
        self, tmp_path: Path
    ) -> None:
        """kind='system' selects the system_template (no-action wrapper)."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        prompt = self._idle_session(broker)

        await broker._on_mcp_message("agent-1", "joined", "system", "system")
        await asyncio.sleep(0)

        prompt.assert_awaited_once_with(
            "[System notification — no action required]: joined"
        )

    async def test_user_prompt_not_prefixed(self, tmp_path: Path) -> None:
        """User prompts (source='user') are delivered unchanged — never signed."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        prompt = self._idle_session(broker)

        await broker.submit_prompt("agent-1", "plain", "user")
        await asyncio.sleep(0)

        prompt.assert_awaited_once_with("plain")

    async def test_startup_bypass_submit_prompt_mcp_not_prefixed(
        self, tmp_path: Path
    ) -> None:
        """source='mcp' sent directly via submit_prompt (bypassing _on_mcp_message)
        is NOT signed — proves formatting is bound to _on_mcp_message, not source.
        This is the dynamic-child startup path (lifecycle._submit_prompt)."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        prompt = self._idle_session(broker)

        await broker.submit_prompt("agent-1", "raw startup", "mcp", "parent")
        await asyncio.sleep(0)

        prompt.assert_awaited_once_with("raw startup")


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

        # Chunk should be silently dropped from the live queue, but counted
        await broker._sink(MessageChunkReceived(agent_id="agent-1", chunk="hello"))
        assert broker._event_queue.qsize() == 2  # unchanged
        assert broker.chunk_loss_stats().dropped_total == 1

        # State event should block — drain one first to make room.  The drop
        # above armed a warning, which is delivered by an awaited put AFTER the
        # state event, so the sink needs a second slot to finish.
        broker._event_queue.get_nowait()
        task = asyncio.create_task(
            broker._sink(
                AgentStateChanged(
                    agent_id="agent-1", old_state=AgentState.IDLE, new_state=AgentState.BUSY
                )
            )
        )
        for _ in range(4):
            await asyncio.sleep(0)
        assert broker._event_queue.qsize() == 2  # state event took the free slot
        assert not task.done()  # armed warning is parked on the full queue

        broker._event_queue.get_nowait()
        await asyncio.wait_for(task, timeout=2.0)
        assert broker._event_queue.qsize() == 2  # filled back up by the warning



class TestInTurnSteering:
    """In-turn steering delivery for MCP messages on a steer-capable harness."""

    class _Fake(NamedTuple):
        session: ACPSession
        prompt: AsyncMock
        steer: AsyncMock

    def _busy_session(
        self,
        broker: ACPBroker,
        *,
        steer_protocol: str | None = "kiro",
        steer_result: bool = True,
        state: AgentState = AgentState.BUSY,
    ) -> TestInTurnSteering._Fake:
        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
            steer_protocol=steer_protocol,
        )
        session._sm._state = state
        prompt_mock = AsyncMock()
        steer_mock = AsyncMock(return_value=steer_result)
        session.prompt = prompt_mock  # type: ignore[method-assign]
        session.steer = steer_mock  # type: ignore[method-assign]
        broker._registry._sessions["agent-1"] = session
        return self._Fake(session, prompt_mock, steer_mock)

    def _steered(self, broker: ACPBroker) -> list[MessageSteered]:
        events = []
        while not broker._event_queue.empty():
            event = broker._event_queue.get_nowait()
            if isinstance(event, MessageSteered):
                events.append(event)
        return events

    async def test_busy_agent_receives_message_by_steer_not_queue(self, tmp_path: Path) -> None:
        """The whole point of the feature: a message arriving mid-turn is injected
        into that turn. If it were also enqueued, Kiro would deliver it twice."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker)

        await broker._on_mcp_message("agent-1", "hello", "agent-a", "chat")

        fake.steer.assert_awaited_once_with("[Message from agent-a]: hello")
        assert broker._prompt_queue.is_empty("agent-1")
        fake.prompt.assert_not_awaited()
        assert [e.text for e in self._steered(broker)] == ["[Message from agent-a]: hello"]

    async def test_failed_steer_falls_back_to_queue_and_drains_once(
        self, tmp_path: Path
    ) -> None:
        """The queue is the guaranteed floor. A rejected steer must lose nothing and
        must deliver exactly once when the turn ends."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        await broker._start_message_bus()
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker, steer_result=False)
        bus = broker._message_bus
        assert bus is not None

        try:
            await broker._on_mcp_message("agent-1", "hello", "agent-a", "chat")
            assert [i.text for i in broker._prompt_queue.items("agent-1")] == [
                "[Message from agent-a]: hello"
            ]
            assert self._steered(broker) == []

            fake.session._sm._state = AgentState.IDLE
            await broker._sink(
                AgentStateChanged(
                    agent_id="agent-1",
                    old_state=AgentState.BUSY,
                    new_state=AgentState.IDLE,
                )
            )
            await asyncio.sleep(0)

            assert broker._prompt_queue.is_empty("agent-1")
            fake.prompt.assert_awaited_once_with("[Message from agent-a]: hello")
        finally:
            await bus.stop()

    async def test_stalled_steer_releases_message_bus_and_agent_lock(
        self, tmp_path: Path
    ) -> None:
        """A missing Kiro response must not wedge the poller or agent lock.

        The existing failed-steer test below proves the later IDLE drain. This
        test stops at the silent failure boundary: the poll callback returns,
        marks its row delivered, and leaves one signed fallback item.
        """
        from synth_acp.broker.message_bus import MessageBus

        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
            steer_protocol="kiro",
        )
        session._sm._state = AgentState.BUSY
        session._session_id = "sess-1"

        async def ext_method(method: str, params: dict[str, Any]) -> dict[str, Any]:
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable")

        session._conn = MagicMock(ext_method=ext_method)
        broker._registry._sessions["agent-1"] = session

        with sqlite3.connect(str(broker._db_path)) as conn:
            ensure_schema_sync(conn)
            conn.execute(
                "INSERT INTO messages "
                "(session_id, from_agent, to_agent, body, kind, status, created_at) "
                "VALUES (?, 'agent-a', 'agent-1', 'PRIVATE_STALLED_STEER', "
                "'chat', 'pending', 100)",
                (broker.session_id,),
            )
            conn.commit()

        bus = MessageBus(
            broker._db_path,
            broker.session_id,
            broker._on_mcp_message,
            None,
        )
        with patch("synth_acp.acp.session._STEER_ACCEPTANCE_TIMEOUT", 0.01):
            await asyncio.wait_for(bus._poll_messages(), timeout=1.0)

        with sqlite3.connect(str(broker._db_path)) as conn:
            row = conn.execute(
                "SELECT status, delivered_at FROM messages "
                "WHERE body = 'PRIVATE_STALLED_STEER'"
            ).fetchone()
        assert row is not None
        assert row[0] == "delivered"
        assert row[1] is not None

        queued = broker._prompt_queue.items("agent-1")
        assert len(queued) == 1
        assert queued[0].text == "[Message from agent-a]: PRIVATE_STALLED_STEER"
        assert queued[0].from_agent == "agent-a"
        assert queued[0].steerable is True

        events: list[BrokerEvent] = []
        while not broker._event_queue.empty():
            events.append(broker._event_queue.get_nowait())
        assert [event for event in events if isinstance(event, MessageSteered)] == []

        async def acquire_same_lock() -> None:
            async with broker._registry.agent_lock("agent-1"):
                pass

        await asyncio.wait_for(acquire_same_lock(), timeout=1.0)

    async def test_backlog_is_sent_in_one_call_and_consumed(self, tmp_path: Path) -> None:
        """A steer following a failed one must carry the backlog, joined and consumed
        together — otherwise the earlier bodies are delivered a second time later."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker)
        broker._prompt_queue.enqueue(
            "agent-1", QueuedItem(text="older", source="mcp", steerable=True)
        )

        await broker._on_mcp_message("agent-1", "newer", "agent-a", "chat")

        fake.steer.assert_awaited_once_with(
            "# Message 1\n\nolder\n\n# Message 2\n\n[Message from agent-a]: newer"
        )
        assert broker._prompt_queue.is_empty("agent-1")

    async def test_queued_user_prompt_is_not_swept_into_a_later_steer(
        self, tmp_path: Path
    ) -> None:
        """A user prompt queued before an eligible message arrives must still not be
        steered. The queue stores rendered text, so without a per-item eligibility
        flag the user's text rides along in the steer payload and is consumed."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker)
        await broker.submit_prompt("agent-1", "USER NEXT TURN", "user")

        await broker._on_mcp_message("agent-1", "chat body", "agent-a", "chat")

        fake.steer.assert_awaited_once_with("[Message from agent-a]: chat body")
        assert [i.text for i in broker._prompt_queue.items("agent-1")] == ["USER NEXT TURN"]

    async def test_queued_system_message_is_not_swept_into_a_later_steer(
        self, tmp_path: Path
    ) -> None:
        """Same for a system notification: 'no action required' must not be injected
        into a running turn just because an eligible chat arrived after it."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker)
        await broker._on_mcp_message("agent-1", "joined", "system", "system")

        await broker._on_mcp_message("agent-1", "chat body", "agent-a", "chat")

        fake.steer.assert_awaited_once_with("[Message from agent-a]: chat body")
        assert [i.text for i in broker._prompt_queue.items("agent-1")] == [
            "[System notification — no action required]: joined"
        ]

    async def test_editing_item_is_neither_sent_nor_consumed(self, tmp_path: Path) -> None:
        """An item under edit blocks the drain point, so it must not be swept into a
        steer — that would send half-typed text and discard the user's edit."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker)
        broker._prompt_queue.enqueue(
            "agent-1", QueuedItem(id="q-edit", text="draft", source="user")
        )
        broker._prompt_queue.mark_editing("agent-1", "q-edit")

        await broker._on_mcp_message("agent-1", "newer", "agent-a", "chat")

        fake.steer.assert_awaited_once_with("[Message from agent-a]: newer")
        assert [i.text for i in broker._prompt_queue.items("agent-1")] == ["draft"]

    async def test_failed_steer_restores_backlog_ahead_of_new_arrival(
        self, tmp_path: Path
    ) -> None:
        """A failed steer must leave the queue as it found it, in order, so the IDLE
        drain still delivers the backlog before the message that failed."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        self._busy_session(broker, steer_result=False)
        broker._prompt_queue.enqueue(
            "agent-1", QueuedItem(text="older", source="mcp", steerable=True)
        )

        await broker._on_mcp_message("agent-1", "newer", "agent-a", "chat")

        assert [i.text for i in broker._prompt_queue.items("agent-1")] == [
            "older",
            "[Message from agent-a]: newer",
        ]

    async def test_user_prompt_is_never_steered(self, tmp_path: Path) -> None:
        """The user has the cancel button; their prompt must queue as it does today."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker)

        await broker.submit_prompt("agent-1", "user text", "user")

        fake.steer.assert_not_awaited()
        assert [i.text for i in broker._prompt_queue.items("agent-1")] == ["user text"]

    async def test_idle_agent_is_never_steered(self, tmp_path: Path) -> None:
        """An idle Kiro steer is held and rides the next prompt, so steering an idle
        agent would deliver the message twice."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        lifecycle._first_prompted.add("agent-1")
        fake = self._busy_session(broker, state=AgentState.IDLE)

        await broker._on_mcp_message("agent-1", "hello", "agent-a", "chat")
        await asyncio.sleep(0)

        fake.steer.assert_not_awaited()
        fake.prompt.assert_awaited_once_with("[Message from agent-a]: hello")

    async def test_steer_eligible_rejects_system_kind(self, tmp_path: Path) -> None:
        """Join/exit notifications are explicitly 'no action required'."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        self._busy_session(broker)
        assert broker._steer_eligible("agent-1", "system") is False
        assert broker._steer_eligible("agent-1", "chat") is True

    async def test_steer_eligible_honors_messages_interrupt(self, tmp_path: Path) -> None:
        """messages_interrupt=false must restore today's behavior exactly."""
        config = SessionConfig.model_validate(
            {"project": "t", "settings": {"messages_interrupt": False}}
        )
        broker = ACPBroker(
            config=config,
            initial_agent=AgentConfig(agent_id="agent-1", harness="kiro"),
            db_path=tmp_path / "synth.db",
        )
        self._busy_session(broker)
        assert broker._steer_eligible("agent-1", "chat") is False

    async def test_steer_eligible_rejects_harness_without_protocol(
        self, tmp_path: Path
    ) -> None:
        """Claude's steering takes a different payload shape and would fail with
        -32602; opencode and gemini have none at all."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        self._busy_session(broker, steer_protocol=None)
        assert broker._steer_eligible("agent-1", "chat") is False


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
        """initial_prompts populated from InitialPromptDelivered/UserPromptSubmitted events (legacy compat)."""
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


class TestBrokerAgentRouting:
    """Tests for SetConfigOption(config_id='agent') routing to set_agent."""

    async def test_broker_routes_agent_config_to_set_agent_for_meta_agent(self, tmp_path: Path) -> None:
        """config_id='agent' with meta_agent target must route to lifecycle.set_agent.
        Silent failure: goes through set_config_option which doesn't know how to fork."""
        from synth_acp.models.commands import SetConfigOption

        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()

        # Register a session with agent_mode_target="meta_agent"
        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
            agent_mode_target="meta_agent",
        )
        session._sm._state = AgentState.IDLE
        broker._registry._sessions["agent-1"] = session

        lifecycle.set_agent = AsyncMock()  # type: ignore[method-assign]
        lifecycle.set_config_option = AsyncMock()  # type: ignore[method-assign]

        await broker.handle(SetConfigOption(agent_id="agent-1", config_id="agent", value="code-planner"))

        lifecycle.set_agent.assert_awaited_once_with("agent-1", "code-planner")
        lifecycle.set_config_option.assert_not_awaited()

    async def test_broker_routes_non_agent_config_to_set_config_option(self, tmp_path: Path) -> None:
        """config_id='mode' must still route to set_config_option unchanged.
        Silent failure: all config options accidentally routed to set_agent."""
        from synth_acp.models.commands import SetConfigOption

        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
            agent_mode_target="meta_agent",
        )
        session._sm._state = AgentState.IDLE
        broker._registry._sessions["agent-1"] = session

        lifecycle.set_agent = AsyncMock()  # type: ignore[method-assign]
        lifecycle.set_config_option = AsyncMock()  # type: ignore[method-assign]

        await broker.handle(SetConfigOption(agent_id="agent-1", config_id="mode", value="trust"))

        lifecycle.set_config_option.assert_awaited_once_with("agent-1", "mode", "trust")
        lifecycle.set_agent.assert_not_awaited()


class TestDiscoveryCache:
    async def test_get_discovered_agents_delegates_to_lifecycle(self, tmp_path: Path) -> None:
        """Broker is a pure delegate: it forwards to lifecycle and returns its result.
        Silent failure: broker re-implements discovery or drops the agent_id."""
        from unittest.mock import MagicMock

        from synth_acp.discovery import DiscoveredAgent

        broker = _make_broker("agent-1", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()

        fake_agents = [
            DiscoveredAgent(qualified_name="planner", name="planner", description="", source="user")
        ]
        lifecycle.get_discovered_agents = MagicMock(return_value=fake_agents)  # type: ignore[method-assign]

        result = broker.get_discovered_agents("agent-1")

        assert result == fake_agents
        lifecycle.get_discovered_agents.assert_called_once_with("agent-1")

    async def test_get_discovered_agents_returns_empty_without_lifecycle(
        self, tmp_path: Path
    ) -> None:
        """Broker returns [] when the lifecycle has not been initialized."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        assert broker._lifecycle is None
        assert broker.get_discovered_agents("agent-1") == []

    async def test_get_discovered_agents_returns_empty_for_unknown_harness(self, tmp_path: Path) -> None:
        """Unknown harness returns empty list gracefully.
        Silent failure: KeyError crash."""
        broker = _make_broker("agent-1", tmp_path=tmp_path)
        await broker._ensure_lifecycle()

        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        broker._registry._sessions["agent-1"] = session
        broker._registry.set_harness("agent-1", "nonexistent")

        result = broker.get_discovered_agents("agent-1")
        assert result == []


# ---------------------------------------------------------------------------
# Live-UI chunk loss
# ---------------------------------------------------------------------------


def _loss_broker(tmp_path: Path, maxsize: int = 2) -> ACPBroker:
    """Broker whose event queue is small enough to force a real QueueFull."""
    return _make_broker("agent-1", tmp_path=tmp_path, event_queue_maxsize=maxsize)


def _fill_queue(broker: ACPBroker) -> None:
    """Put filler events until the queue is provably full."""
    while True:
        try:
            broker._event_queue.put_nowait(
                BrokerError(agent_id="filler", message="filler", severity="warning")
            )
        except asyncio.QueueFull:
            return


def _take(broker: ACPBroker, n: int) -> list[BrokerEvent]:
    """Remove exactly ``n`` events, freeing ``n`` slots."""
    return [broker._event_queue.get_nowait() for _ in range(n)]


def _drain_all(broker: ACPBroker) -> list[BrokerEvent]:
    taken: list[BrokerEvent] = []
    while not broker._event_queue.empty():
        taken.append(broker._event_queue.get_nowait())
    return taken


def _consumer(broker: ACPBroker) -> tuple[asyncio.Task[None], list[BrokerEvent]]:
    """Start a task that drains the queue into a list until cancelled."""
    seen: list[BrokerEvent] = []

    async def run() -> None:
        while True:
            seen.append(await broker._event_queue.get())

    return asyncio.create_task(run()), seen


def _warnings(events: list[BrokerEvent]) -> list[BrokerError]:
    return [
        e
        for e in events
        if isinstance(e, BrokerError) and e.severity == "warning" and e.agent_id != "filler"
    ]


def _chunk(agent_id: str = "agent-1", text: str = "hi") -> MessageChunkReceived:
    return MessageChunkReceived(agent_id=agent_id, chunk=text)


async def _pump(times: int = 4) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class TestChunkLossCounters:
    async def test_drops_are_counted_per_agent(self, tmp_path: Path) -> None:
        """Dropped chunks must be counted, or output vanishes from the live feed
        with nothing anywhere recording that it happened."""
        broker = _loss_broker(tmp_path)
        _fill_queue(broker)
        before = time.time()

        for _ in range(60):
            await broker._sink(_chunk("agent-1"))
        for _ in range(40):
            await broker._sink(_chunk("agent-2"))

        stats = broker.chunk_loss_stats()
        assert stats.dropped_total == 100
        assert stats.dropped_by_agent == {"agent-1": 60, "agent-2": 40}
        assert stats.last_drop_at is not None and stats.last_drop_at >= before

    async def test_record_chunk_drop_is_synchronous_and_delivers_nothing(
        self, tmp_path: Path
    ) -> None:
        """An await on the chunk path would let a later chunk overtake an
        earlier one; a delivery attempt here is pointless because the queue is
        provably full."""
        assert not inspect.iscoroutinefunction(ACPBroker.record_chunk_drop)
        assert not inspect.iscoroutinefunction(ACPBroker.try_deliver_drop_warning)
        for fn in (ACPBroker.record_chunk_drop, ACPBroker.try_deliver_drop_warning):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            assert not any(isinstance(n, ast.Await) for n in ast.walk(tree))

        broker = _loss_broker(tmp_path)
        _fill_queue(broker)
        size = broker._event_queue.qsize()

        broker.record_chunk_drop("agent-1")

        assert broker._event_queue.qsize() == size
        assert broker.chunk_loss_stats().armed_agents == frozenset({"agent-1"})


class TestChunkLossStructure:
    """AST guards for the no-await chunk path and the absence of stamping."""

    @staticmethod
    def _sink_parts() -> tuple[list[ast.stmt], ast.If, ast.If]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(ACPBroker._sink)))
        node = tree.body[0]
        assert isinstance(node, ast.AsyncFunctionDef)
        body = node.body

        def names(n: ast.AST) -> set[str]:
            return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}

        permission = next(
            s for s in body if isinstance(s, ast.If) and "PermissionRequested" in names(s.test)
        )
        chunk = next(
            s
            for s in body
            if isinstance(s, ast.If) and "MessageChunkReceived" in names(s.test)
        )
        return body, permission, chunk

    @staticmethod
    def _put_nowait_index(stmts: list[ast.stmt]) -> int:
        for i, stmt in enumerate(stmts):
            attrs = {n.attr for n in ast.walk(stmt) if isinstance(n, ast.Attribute)}
            if "put_nowait" in attrs:
                return i
        raise AssertionError("no put_nowait on the chunk path")

    def test_no_await_on_the_chunk_path(self) -> None:
        """An await here reorders chunks under saturation, which no behavioural
        test can see because it only manifests when the queue is full."""
        body, permission, chunk = self._sink_parts()
        prelude = [s for s in body[: body.index(chunk)] if s is not permission]
        upto = chunk.body[: self._put_nowait_index(chunk.body) + 1]

        def has_await(nodes: list[ast.stmt]) -> bool:
            return any(isinstance(n, ast.Await) for s in nodes for n in ast.walk(s))

        assert not has_await(prelude)
        assert not has_await(upto)
        # Non-vacuity: the excluded subtrees really do await.
        assert has_await([permission])
        assert has_await(chunk.orelse)

    def test_chunk_enqueue_stays_put_nowait(self) -> None:
        """An awaited put fast-paths to put_nowait when not full, so a new
        producer can overtake one parked in _putters — worse than dropping."""
        _, _, chunk = self._sink_parts()
        upto = chunk.body[: self._put_nowait_index(chunk.body) + 1]
        attrs = {n.attr for s in upto for n in ast.walk(s) if isinstance(n, ast.Attribute)}
        assert "put_nowait" in attrs
        assert "put" not in attrs

    def test_queue_full_handler_counts_the_drop(self) -> None:
        """A drop path that only logs at debug level is invisible in
        production — the exact defect this phase fixes."""
        _, _, chunk = self._sink_parts()
        handlers = [
            h
            for s in chunk.body
            if isinstance(s, ast.Try)
            for h in s.handlers
            if h.type is not None and "QueueFull" in ast.dump(h.type)
        ]
        assert len(handlers) == 1
        called = {
            n.func.attr
            for n in ast.walk(handlers[0])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "record_chunk_drop" in called

    def test_no_sequence_stamping_on_events(self) -> None:
        """A counter allocated at the sink is tautologically monotonic: it looks
        like an ordering guarantee while detecting nothing."""
        from synth_acp.models import events as ev

        def subclasses(cls: type[ev.BrokerEvent]) -> list[type[ev.BrokerEvent]]:
            out: list[type[ev.BrokerEvent]] = []
            for sub in cls.__subclasses__():
                out.append(sub)
                out.extend(subclasses(sub))
            return out

        for cls in subclasses(ev.BrokerEvent):
            assert not {"seq", "sequence"} & set(cls.model_fields)

        event_names = {c.__name__ for c in subclasses(ev.BrokerEvent)}
        src = Path(__file__).resolve().parents[2] / "src" / "synth_acp"
        for path in (src / "broker" / "broker.py", src / "acp" / "session.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                is_event = isinstance(node.func, ast.Name) and node.func.id in event_names
                is_copy = isinstance(node.func, ast.Attribute) and node.func.attr == "model_copy"
                if not (is_event or is_copy):
                    continue
                keys = {kw.arg for kw in node.keywords} | {
                    k.value
                    for kw in node.keywords
                    if isinstance(kw.value, ast.Dict)
                    for k in kw.value.keys
                    if isinstance(k, ast.Constant)
                }
                assert not {"seq", "sequence"} & keys, ast.dump(node)


class TestDropWarningDelivery:
    @pytest.mark.parametrize("trigger_kind", ["turn_complete", "state_changed", "usage"])
    async def test_awaited_trigger_delivers_the_only_warning(
        self, tmp_path: Path, trigger_kind: str
    ) -> None:
        """Each awaited event must be able to surface a drop by itself.

        Two silent failures guarded: wiring the warning to TurnComplete alone
        never surfaces a drop mid-turn, and using put_nowait for delivery fails
        at exactly the saturation where drops happen.
        """
        broker = _loss_broker(tmp_path)
        triggers: dict[str, BrokerEvent] = {
            "turn_complete": TurnComplete(agent_id="agent-1", stop_reason="end_turn"),
            "state_changed": AgentStateChanged(
                agent_id="agent-1", old_state=AgentState.BUSY, new_state=AgentState.IDLE
            ),
            "usage": UsageUpdated(agent_id="agent-1", size=1, used=1),
        }
        _fill_queue(broker)

        await broker._sink(_chunk("agent-1"))
        assert broker._event_queue.full()
        assert broker.chunk_loss_stats().armed_agents == frozenset({"agent-1"})

        task = asyncio.create_task(broker._sink(triggers[trigger_kind]))
        await _pump()
        assert not task.done()  # the trigger's own put is parked

        taken = _take(broker, 1)
        await _pump()
        # The trigger took the freed slot; the warning's awaited put is now
        # parked on a full queue. A put_nowait implementation would have
        # re-armed here and completed with nothing delivered.
        assert not task.done()
        assert _warnings(taken) == []

        taken += _take(broker, 1)
        await asyncio.wait_for(task, timeout=2.0)
        taken += _drain_all(broker)

        delivered = _warnings(taken)
        assert len(delivered) == 1
        # Filler events carry agent_id 'filler'; only agent-1's events matter here.
        mine = [type(e).__name__ for e in taken if e.agent_id == "agent-1"]
        assert mine == [type(triggers[trigger_kind]).__name__, "BrokerError"]
        assert broker.chunk_loss_stats().warned_agents == frozenset({"agent-1"})

    async def test_no_warning_when_drop_arms_after_the_last_awaited_event(
        self, tmp_path: Path
    ) -> None:
        """ACCEPTED RESIDUAL. A trigger cannot observe an arming event that
        happens after it runs, and guaranteeing delivery would require the
        chunk-before-TurnComplete invariant this phase retires. Pinned so a
        future 'fix' cannot quietly claim a guarantee the mechanism lacks.
        """
        broker = _loss_broker(tmp_path)

        # TurnComplete first, on an empty queue, with nothing armed.
        await broker._sink(TurnComplete(agent_id="agent-1", stop_reason="end_turn"))
        # Then the late runner's chunk hits a full queue and arms.
        _fill_queue(broker)
        await broker._sink(_chunk("agent-1", "late-chunk"))

        stats = broker.chunk_loss_stats()
        assert _warnings(_drain_all(broker)) == []
        assert stats.armed_agents == frozenset({"agent-1"})
        assert stats.warned_agents == frozenset()
        assert stats.dropped_total == 1
        buffered = "".join(
            e.chunk
            for _, e in broker._turn_buffer.get("agent-1", [])
            if isinstance(e, MessageChunkReceived)
        )
        assert "late-chunk" in buffered

    async def test_later_successful_chunk_delivers_one_warning_after_it(
        self, tmp_path: Path
    ) -> None:
        """A warning enqueued ahead of a chunk would claim loss before the
        content that follows it."""
        broker = _loss_broker(tmp_path)
        _fill_queue(broker)
        await broker._sink(_chunk("agent-1", "dropped"))
        _take(broker, 2)

        await broker._sink(_chunk("agent-1", "landed"))

        taken = _drain_all(broker)
        assert [type(e).__name__ for e in taken] == ["MessageChunkReceived", "BrokerError"]
        assert len(_warnings(taken)) == 1

    async def test_both_paths_eligible_yields_exactly_one_warning(
        self, tmp_path: Path
    ) -> None:
        """Duplicate warnings for one drop train the user to ignore them."""
        broker = _loss_broker(tmp_path)
        _fill_queue(broker)
        await broker._sink(_chunk("agent-1", "dropped"))
        _take(broker, 2)
        await broker._sink(_chunk("agent-1", "landed"))  # sync path delivers

        task, seen = _consumer(broker)
        try:
            await asyncio.wait_for(
                broker._sink(TurnComplete(agent_id="agent-1", stop_reason="end_turn")),
                timeout=2.0,
            )
            await _pump()
        finally:
            task.cancel()

        assert len(_warnings(seen + _drain_all(broker))) == 1

    async def test_no_warning_storm_after_delivery(self, tmp_path: Path) -> None:
        """One warning per dropped chunk would flood the feed at exactly the
        moment it is least able to absorb it."""
        broker = _loss_broker(tmp_path)
        _fill_queue(broker)
        await broker._sink(_chunk("agent-1", "dropped"))
        _take(broker, 2)
        await broker._sink(_chunk("agent-1", "landed"))
        assert len(_warnings(_drain_all(broker))) == 1

        _fill_queue(broker)
        for _ in range(100):
            await broker._sink(_chunk("agent-1"))

        stats = broker.chunk_loss_stats()
        assert _warnings(_drain_all(broker)) == []
        assert stats.dropped_total == 101
        assert stats.warned_agents == frozenset({"agent-1"})
        assert stats.armed_agents == frozenset()

    async def test_concurrent_sinks_for_one_agent_deliver_one_warning(
        self, tmp_path: Path
    ) -> None:
        """Two concurrent sinks both delivering is unreproducible by hand and
        only shows up under real multi-agent load."""
        broker = _loss_broker(tmp_path)
        _fill_queue(broker)
        await broker._sink(_chunk("agent-1"))
        _take(broker, 2)

        task, seen = _consumer(broker)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    broker._sink(
                        AgentStateChanged(
                            agent_id="agent-1",
                            old_state=AgentState.BUSY,
                            new_state=AgentState.CONFIGURING,
                        )
                    ),
                    broker._sink(
                        AgentStateChanged(
                            agent_id="agent-1",
                            old_state=AgentState.CONFIGURING,
                            new_state=AgentState.BUSY,
                        )
                    ),
                ),
                timeout=2.0,
            )
            await _pump()
        finally:
            task.cancel()

        assert len(_warnings(seen + _drain_all(broker))) == 1


class TestChunkLossJournal:
    async def test_dropped_chunk_text_survives_into_the_journal(
        self, tmp_path: Path
    ) -> None:
        """If a drop also skipped the journal, the text would be gone for good
        rather than reappearing on restore — a UI gap becoming data loss."""
        broker = _loss_broker(tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle._db_op(ensure_schema_sync)
        _fill_queue(broker)

        await broker._sink(_chunk("agent-1", "dropped-text"))

        buffered = "".join(
            e.chunk
            for _, e in broker._turn_buffer["agent-1"]
            if isinstance(e, MessageChunkReceived)
        )
        assert "dropped-text" in buffered

        _drain_all(broker)
        await broker._sink(TurnComplete(agent_id="agent-1", stop_reason="end_turn"))
        await asyncio.gather(*list(broker._pending_flushes), return_exceptions=True)

        replayed = await broker.load_journal("agent-1", broker.session_id)
        text = "".join(e.chunk for e in replayed if isinstance(e, MessageChunkReceived))
        assert "dropped-text" in text


class TestToolCallMerge:
    """Merging repeated updates for one tool_call_id must not degrade types.

    ``model_copy(update=...)`` does not validate, so feeding it dumped values
    silently leaves the dataclass fields holding plain dicts. Nothing crashes —
    the journal JSON is even byte-identical — so the only observable symptom is
    a pydantic serializer warning at flush time and a merged event that breaks
    attribute access for any future consumer.
    """

    @staticmethod
    def _update(status: str, **extra: Any) -> ToolCallUpdated:
        return ToolCallUpdated(
            agent_id="agent-1",
            tool_call_id="tc-1",
            title="Edit",
            kind="edit",
            status=status,
            locations=[ToolCallLocation(path="/repo/f.py", line=7)],
            **extra,
        )

    def test_merged_event_keeps_dataclass_instances(self, tmp_path: Path) -> None:
        broker = _make_broker("agent-1", tmp_path=tmp_path)

        broker._buffer_journal_event(self._update("pending"))
        broker._buffer_journal_event(
            self._update(
                "completed",
                diffs=[ToolCallDiff(path="/repo/f.py", old_text=None, new_text="x")],
            )
        )

        (_, merged), = broker._turn_buffer["agent-1"]
        assert isinstance(merged, ToolCallUpdated)
        assert merged.status == "completed"
        assert merged.locations == [ToolCallLocation(path="/repo/f.py", line=7)]
        assert merged.diffs == [ToolCallDiff(path="/repo/f.py", old_text=None, new_text="x")]

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert '"path":"/repo/f.py"' in merged.model_dump_json()


# ---------------------------------------------------------------------------
# Accumulator bypass, end to end through the broker sink and the journal
#
# These drive an ACPSession and assert on what the BROKER queues and journals,
# so per the AGENTS.md testing convention they belong in the mirrored file of
# the module under ASSERTION — this one.
# ---------------------------------------------------------------------------

REPLAY_EVENT_TYPES = (
    MessageChunkReceived,
    AgentThoughtReceived,
    ToolCallUpdated,
    PlanReceived,
    UserPromptSubmitted,
)


def _history_updates(n: int) -> list[Any]:
    """A realistic mix of replayed history notifications."""
    updates: list[Any] = []
    for i in range(n):
        kind = i % 5
        if kind == 0:
            updates.append(
                AgentMessageChunk(
                    content=TextContentBlock(type="text", text=f"hist-{i}"),
                    message_id="m1",
                    session_update="agent_message_chunk",
                )
            )
        elif kind == 1:
            updates.append(
                AgentThoughtChunk(
                    content=TextContentBlock(type="text", text=f"thought-{i}"),
                    message_id="m1",
                    session_update="agent_thought_chunk",
                )
            )
        elif kind == 2:
            updates.append(
                ToolCallStart(
                    tool_call_id=f"tc-{i}",
                    title="Read",
                    kind="read",
                    status="pending",
                    content=None,
                    locations=None,
                    raw_input=None,
                    raw_output=None,
                    session_update="tool_call",
                )
            )
        elif kind == 3:
            updates.append(
                ToolCallProgress(
                    tool_call_id=f"tc-{i - 1}",
                    title="Read",
                    kind="read",
                    status="completed",
                    content=None,
                    locations=None,
                    raw_input=None,
                    raw_output=None,
                    session_update="tool_call_update",
                )
            )
        else:
            updates.append(
                AgentPlanUpdate(
                    entries=[PlanEntry(content="step", priority="medium", status="pending")],
                    session_update="plan",
                )
            )
    return updates


def _replay_session(broker: ACPBroker, session_id: str = "sess-saved") -> ACPSession:
    session = ACPSession(
        agent_id="agent-1", binary="echo", args=[], cwd=".", event_sink=broker._sink
    )
    session._session_id = session_id
    return session


def _spawn_patch(conn: Any, proc: Any) -> Any:
    """Patch _spawn_isolated_agent to yield the supplied conn/proc plus a stderr tail."""

    @asynccontextmanager
    async def fake_spawn(*args: Any, **kwargs: Any) -> Any:
        yield conn, proc, _StderrTail()

    return patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn)


def _handshake_conn() -> Any:
    conn = AsyncMock()
    conn.initialize.return_value = MagicMock(agent_capabilities=None)
    response = MagicMock(session_id="sess-saved", modes=None, models=None, config_options=None)
    conn.new_session.return_value = response
    return conn


async def _ui_event_count(broker: ACPBroker) -> int:
    lifecycle = await broker._ensure_lifecycle()

    def _count(conn: sqlite3.Connection) -> int:
        return conn.execute("SELECT COUNT(*) FROM ui_events").fetchone()[0]

    return await lifecycle._db_op(_count)


class TestSuppressedHistoryReplay:
    """The relocated guard, end to end. Without it, every restore and every
    mode switch replays history into the live UI and duplicates the journal."""

    async def test_run_restored_replay_emits_nothing_and_journals_nothing(
        self, tmp_path: Path
    ) -> None:
        broker = _loss_broker(tmp_path, maxsize=2000)
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle._db_op(ensure_schema_sync)
        session = _replay_session(broker)
        window: dict[str, int] = {}

        conn = _handshake_conn()

        async def load_session(**kwargs: Any) -> Any:
            window["before"] = broker._event_queue.qsize()
            for update in _history_updates(50):
                await session.session_update("sess-saved", update)
            # Falsifiability yield: any emission task a broken guard created has
            # actually RUN by now.
            await asyncio.sleep(0)
            # AC6b sample point: the LAST action before load_session returns.
            window["after"] = broker._event_queue.qsize()
            return MagicMock(session_id="sess-saved", modes=None, models=None, config_options=None)

        conn.load_session = load_session
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)

        with _spawn_patch(conn, proc):
            await session.run_restored("sess-saved")

        queued = _drain_all(broker)
        assert [e for e in queued if isinstance(e, REPLAY_EVENT_TYPES)] == []
        assert window["after"] == window["before"]
        assert [type(e).__name__ for e in queued] == ["AgentStateChanged"] * 3
        assert [e.new_state for e in queued if isinstance(e, AgentStateChanged)] == [
            AgentState.INITIALIZING,
            AgentState.IDLE,
            AgentState.TERMINATED,
        ]

        await broker._flush_turn_buffer_all()
        assert broker._turn_buffer.get("agent-1", []) == []
        assert await _ui_event_count(broker) == 0

    async def test_mode_switch_mcp_restore_emits_nothing_and_journals_nothing(
        self, tmp_path: Path
    ) -> None:
        """Kiro takes this path on EVERY mode switch, so it fires far more often
        than restore."""
        broker = _loss_broker(tmp_path, maxsize=2000)
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle._db_op(ensure_schema_sync)
        session = _replay_session(broker)
        session._mcp_servers = [McpServerStdio(name="synth-mcp", command="echo", args=[], env=[])]
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        window: dict[str, int] = {}

        conn = AsyncMock()

        async def load_session(**kwargs: Any) -> Any:
            window["before"] = broker._event_queue.qsize()
            for update in _history_updates(50):
                await session.session_update("sess-saved", update)
            await asyncio.sleep(0)
            window["after"] = broker._event_queue.qsize()
            return MagicMock(session_id="sess-saved", modes=None, models=None, config_options=None)

        conn.load_session = load_session
        session._conn = conn
        _drain_all(broker)

        await session.set_mode("plan")

        queued = _drain_all(broker)
        assert [e for e in queued if isinstance(e, REPLAY_EVENT_TYPES)] == []
        assert window["after"] == window["before"]
        assert [type(e).__name__ for e in queued] == [
            "AgentStateChanged",
            "AgentModeChanged",
            "AgentStateChanged",
        ]
        assert [e.new_state for e in queued if isinstance(e, AgentStateChanged)] == [
            AgentState.CONFIGURING,
            AgentState.IDLE,
        ]

        await broker._flush_turn_buffer_all()
        assert await _ui_event_count(broker) == 0


class TestReplayGuardSettles:
    """The flag must not clear while replay runners are still queued.

    A runner created during the replay can first reach session_update AFTER the
    finally clears the flag, so a slice of history is emitted live and journalled
    while every other suppression assertion still passes.  Draining
    _pending_emissions does not help: a suppressed notification never registers
    an emission task.
    """

    @staticmethod
    def _ghost() -> AgentMessageChunk:
        return AgentMessageChunk(
            content=TextContentBlock(type="text", text="ghost"),
            message_id="m1",
            session_update="agent_message_chunk",
        )

    async def test_late_replay_runner_is_still_suppressed_on_restore(
        self, tmp_path: Path
    ) -> None:
        broker = _loss_broker(tmp_path, maxsize=2000)
        session = _replay_session(broker)
        runners: list[asyncio.Task[None]] = []
        conn = _handshake_conn()

        async def load_session(**kwargs: Any) -> Any:
            # Created, NOT started: no await between here and the return.
            runners.append(
                asyncio.create_task(session.session_update("sess-saved", self._ghost()))
            )
            return MagicMock(session_id="sess-saved", modes=None, models=None, config_options=None)

        conn.load_session = load_session
        proc = MagicMock()
        proc.returncode = 0
        # Block in proc.wait so the assertion happens while the session is LIVE.
        # Letting run_restored finish first would set _shutting_down, whose guard
        # would suppress the ghost for the wrong reason and mask a missing settle.
        exited = asyncio.Event()
        proc.wait = exited.wait

        with _spawn_patch(conn, proc):
            restore = asyncio.create_task(session.run_restored("sess-saved"))
            try:
                for _ in range(200):
                    if runners and runners[0].done():
                        break
                    await asyncio.sleep(0)
                assert runners and runners[0].done(), "the replay runner never ran"
                assert session._shutting_down is False
                # A suppressed notification registers NO emission task, so this
                # is the direct observation: an unsettled clear would have let
                # the runner past the guard and registered one.
                assert session._pending_emissions == set()
                await _pump(6)  # let any emission task that WAS created run
                queued = _drain_all(broker)
                assert [e for e in queued if isinstance(e, MessageChunkReceived)] == []
            finally:
                exited.set()
                await restore

    async def test_late_replay_runner_is_still_suppressed_on_mode_switch(
        self, tmp_path: Path
    ) -> None:
        broker = _loss_broker(tmp_path, maxsize=2000)
        session = _replay_session(broker)
        session._mcp_servers = [McpServerStdio(name="synth-mcp", command="echo", args=[], env=[])]
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        runners: list[asyncio.Task[None]] = []

        conn = AsyncMock()

        async def load_session(**kwargs: Any) -> Any:
            runners.append(
                asyncio.create_task(session.session_update("sess-saved", self._ghost()))
            )
            return MagicMock(session_id="sess-saved", modes=None, models=None, config_options=None)

        conn.load_session = load_session
        session._conn = conn
        _drain_all(broker)

        await session.set_mode("plan")
        await asyncio.gather(*runners)

        assert session._pending_emissions == set()
        await _pump(6)
        queued = _drain_all(broker)
        assert [e for e in queued if isinstance(e, MessageChunkReceived)] == []

    async def test_settle_yields_exactly_drain_passes_times(self, tmp_path: Path) -> None:
        """An unbounded or multiplied wait for runners that may never exist would
        hang or slow every restore and every mode switch."""
        broker = _loss_broker(tmp_path, maxsize=2000)
        session = _replay_session(broker)
        calls: list[Any] = []
        real_sleep = asyncio.sleep

        async def counting_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
            calls.append(delay)
            return await real_sleep(delay, *args, **kwargs)

        with patch("synth_acp.acp.session.asyncio.sleep", counting_sleep):
            await session._settle_replay_guard()

        assert calls == [0] * DRAIN_PASSES

    def test_settle_shape_is_a_single_bounded_loop(self) -> None:
        """The loop shape alone cannot detect two awaits per iteration, so the
        count test above is paired with an exact shape assertion here."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(ACPSession._settle_replay_guard)))
        fn = tree.body[0]
        assert isinstance(fn, ast.AsyncFunctionDef)
        loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
        assert len(loops) == 1
        loop = loops[0]
        assert isinstance(loop, ast.For)
        assert isinstance(loop.iter, ast.Call)
        assert isinstance(loop.iter.func, ast.Name) and loop.iter.func.id == "range"
        assert "DRAIN_PASSES" in {
            n.id for n in ast.walk(loop.iter) if isinstance(n, ast.Name)
        }
        assert len(loop.body) == 1
        assert isinstance(loop.body[0], ast.Expr)
        assert len([n for n in ast.walk(fn) if isinstance(n, ast.Await)]) == 1

    @pytest.mark.parametrize("method_name", ["run_restored", "_restore_mcp_servers"])
    def test_both_clear_sites_settle_inside_a_nested_finally(self, method_name: str) -> None:
        """BOTH clear sites must settle first, inside an inner finally.

        Behavioural coverage cannot reach this: flattening only ONE site's
        nested finally leaves every other test in this class green, so restore
        could be fixed while mode switch still leaks (or vice versa).
        """
        src = textwrap.dedent(inspect.getsource(getattr(ACPSession, method_name)))
        fn = ast.parse(src).body[0]
        assert isinstance(fn, ast.AsyncFunctionDef)

        def clears_flag(node: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute) and t.attr == "_suppress_history_replay"
                    for t in n.targets
                )
                and isinstance(n.value, ast.Constant)
                and n.value.value is False
                for n in ast.walk(node)
            )

        def settles(node: ast.AST) -> bool:
            """True when the settle coroutine is AWAITED, not merely called.

            An unawaited call satisfies the nested-finally shape while doing
            nothing at all, so the Await is the load-bearing part.
            """
            return any(
                isinstance(n, ast.Await)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "_settle_replay_guard"
                for n in ast.walk(node)
            )

        # The clear lives in the finalbody of a Try whose own body settles, and
        # that Try is itself inside the finalbody that owns the flag.
        nested = [
            inner
            for outer in ast.walk(fn)
            if isinstance(outer, ast.Try) and outer.finalbody
            for inner in outer.finalbody
            if isinstance(inner, ast.Try)
            and settles(ast.Module(body=inner.body, type_ignores=[]))
            and clears_flag(ast.Module(body=inner.finalbody, type_ignores=[]))
        ]
        assert len(nested) == 1, f"{method_name} must settle then clear in a nested finally"

    async def test_restore_with_no_runner_completes_promptly(self, tmp_path: Path) -> None:
        """A runner the dispatcher never created must not be waited for."""
        broker = _loss_broker(tmp_path, maxsize=2000)
        session = _replay_session(broker)
        conn = _handshake_conn()
        conn.load_session.return_value = MagicMock(
            session_id="sess-saved", modes=None, models=None, config_options=None
        )
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)

        with _spawn_patch(conn, proc):
            await asyncio.wait_for(session.run_restored("sess-saved"), timeout=2.0)

        assert session._suppress_history_replay is False

    @pytest.mark.parametrize("failure", [RuntimeError("settle blew up"), asyncio.CancelledError()])
    async def test_suppress_flag_clears_when_settling_fails(
        self, tmp_path: Path, failure: BaseException
    ) -> None:
        """A flag left set silences the agent for the whole session — worse than
        the duplication being fixed. A raising load_session alone would pass a
        flat finally, so the failure is injected DURING settling.
        """
        broker = _loss_broker(tmp_path, maxsize=2000)
        session = _replay_session(broker)

        async def failing_settle() -> None:
            raise failure

        session._settle_replay_guard = failing_settle  # type: ignore[method-assign]
        conn = _handshake_conn()
        conn.load_session.return_value = MagicMock(
            session_id="sess-saved", modes=None, models=None, config_options=None
        )
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)

        with _spawn_patch(conn, proc):
            if isinstance(failure, asyncio.CancelledError):
                with pytest.raises(asyncio.CancelledError):
                    await session.run_restored("sess-saved")
            else:
                await session.run_restored("sess-saved")
                assert any(
                    isinstance(e, BrokerError) and "settle blew up" in e.message
                    for e in _drain_all(broker)
                )

        assert session._suppress_history_replay is False

    async def test_suppress_flag_clears_when_load_session_raises(
        self, tmp_path: Path
    ) -> None:
        """Ordinary-path invariant: a failed restore must not leave the session
        permanently suppressed."""
        broker = _loss_broker(tmp_path, maxsize=2000)
        session = _replay_session(broker)
        conn = _handshake_conn()
        conn.load_session.side_effect = RuntimeError("no such session")
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)

        with _spawn_patch(conn, proc):
            await session.run_restored("sess-saved")

        assert session._suppress_history_replay is False


class TestJournalFidelityAgainstAccumulator:
    """The bypass must write byte-identical journal rows.

    Silent failure: an update type that used to produce an event stops producing
    one, so restored sessions are missing content and nothing crashes.
    """

    FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)

    EXPECTED_TYPES: ClassVar[tuple[str, ...]] = (
        "MessageChunkReceived",
        "ToolCallUpdated",
        "AgentThoughtReceived",
        "PlanReceived",
        "MessageChunkReceived",
        "TurnComplete",
    )

    @staticmethod
    def _stream() -> list[Any]:
        def chunk(text: str) -> AgentMessageChunk:
            return AgentMessageChunk(
                content=TextContentBlock(type="text", text=text),
                message_id="m1",
                session_update="agent_message_chunk",
            )

        return [
            chunk("a"),
            chunk("b"),
            ToolCallStart(
                tool_call_id="tc-1",
                title="Read",
                kind="read",
                status="pending",
                content=None,
                locations=None,
                raw_input=None,
                raw_output=None,
                session_update="tool_call",
            ),
            ToolCallProgress(
                tool_call_id="tc-1",
                title="Read",
                kind="read",
                status="completed",
                content=None,
                locations=None,
                raw_input=None,
                raw_output=None,
                session_update="tool_call_update",
            ),
            AgentThoughtChunk(
                content=TextContentBlock(type="text", text="t"),
                message_id="m1",
                session_update="agent_thought_chunk",
            ),
            AgentPlanUpdate(
                entries=[PlanEntry(content="step", priority="medium", status="pending")],
                session_update="plan",
            ),
            chunk("c"),
        ]

    async def _rows(self, tmp_path: Path, session: ACPSession, broker: ACPBroker) -> list[tuple]:
        lifecycle = await broker._ensure_lifecycle()
        await lifecycle._db_op(ensure_schema_sync)
        for update in self._stream():
            await session.session_update("sess-1", update)
        await session._drain_pending_emissions()
        await broker._sink(TurnComplete(agent_id="agent-1", stop_reason="end_turn"))
        await asyncio.gather(*list(broker._pending_flushes), return_exceptions=True)

        def _query(conn: sqlite3.Connection) -> list[tuple]:
            return conn.execute(
                "SELECT seq, event_type, payload FROM ui_events ORDER BY seq"
            ).fetchall()

        return await lifecycle._db_op(_query)

    async def test_journal_rows_identical_to_accumulator_path(self, tmp_path: Path) -> None:
        class _LegacyAccumulatorSession(ACPSession):
            """The pre-phase session_update: feed the accumulator, emit from its
            subscriber callback."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._accumulator = SessionAccumulator()
                self._accumulator.subscribe(self._legacy_on_snapshot)

            def _legacy_on_snapshot(self, snapshot: Any, notification: Any) -> None:
                if self._suppress_history_replay:
                    return
                task = asyncio.create_task(
                    self._emit_from_notification(notification.update)
                )
                self._pending_emissions.add(task)
                task.add_done_callback(self._pending_emissions.discard)

            async def session_update(
                self, session_id: str, update: Any, **kwargs: Any
            ) -> None:
                if session_id != self._session_id:
                    return
                if isinstance(update, UsageUpdate):
                    return
                self._accumulator.apply(
                    SessionNotification(session_id=session_id, update=update)
                )

        fixed = MagicMock()
        fixed.now.return_value = self.FIXED_NOW

        with patch("synth_acp.models.events.datetime", fixed):
            bypass_broker = _loss_broker(tmp_path / "bypass", maxsize=2000)
            bypass = ACPSession(
                agent_id="agent-1",
                binary="echo",
                args=[],
                cwd=".",
                event_sink=bypass_broker._sink,
            )
            bypass._session_id = "sess-1"
            bypass_rows = await self._rows(tmp_path / "bypass", bypass, bypass_broker)

            legacy_broker = _loss_broker(tmp_path / "legacy", maxsize=2000)
            legacy = _LegacyAccumulatorSession(
                agent_id="agent-1",
                binary="echo",
                args=[],
                cwd=".",
                event_sink=legacy_broker._sink,
            )
            legacy._session_id = "sess-1"
            legacy_rows = await self._rows(tmp_path / "legacy", legacy, legacy_broker)

        # Non-vacuity first: neither side may be silently empty.
        assert [row[0] for row in bypass_rows] == [0, 1, 2, 3, 4, 5]
        assert tuple(row[1] for row in bypass_rows) == self.EXPECTED_TYPES
        # Timestamps are pinned, so complete payloads are comparable.
        assert legacy_rows == bypass_rows


# ── Agent handoff: broker-owned state ──


def _seed_predecessor(broker: ACPBroker, agent_id: str = "worker") -> None:
    """Insert the agents rows a handoff reads, plus a child pointing at the agent."""
    import sqlite3 as _sq

    from synth_acp.db import ensure_schema_sync

    conn = _sq.connect(str(broker._db_path))
    try:
        ensure_schema_sync(conn)
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent, task, "
            "harness, agent_mode, cwd, acp_session_id) "
            "VALUES (?, ?, 'active', 100, 'boss', 'the task', 'kiro', NULL, '.', 'acp-1')",
            (agent_id, broker.session_id),
        )
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent) "
            "VALUES ('kid', ?, 'active', 101, ?)",
            (broker.session_id, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def _fake_session(broker: ACPBroker, agent_id: str, state: AgentState = AgentState.IDLE) -> ACPSession:
    session = ACPSession(
        agent_id=agent_id, binary="echo", args=[], cwd=".", event_sink=broker._sink
    )
    session._sm._state = state
    session.force_kill = MagicMock()  # type: ignore[method-assign]
    session.prompt = AsyncMock()  # type: ignore[method-assign]
    session.steer = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return session


async def _handoff_harness(broker: ACPBroker, agent_id: str = "worker") -> tuple[Any, list[ACPSession]]:
    """Wire a broker so handoff() runs for real without spawning a subprocess.

    Returns the lifecycle and the list successors are appended to.  Only two seams are
    replaced: session construction (so no binary is needed) and run-task creation (so
    nothing is spawned).  Everything the handoff actually decides stays real.
    """
    _seed_predecessor(broker, agent_id)
    lifecycle = await broker._ensure_lifecycle()
    lifecycle.set_handoff_state(broker)
    predecessor = _fake_session(broker, agent_id)
    broker._registry.register(agent_id, predecessor)
    broker._registry.set_parent(agent_id, "boss")
    broker._registry.set_parent("kid", agent_id)
    broker._registry.set_harness(agent_id, "kiro")

    successors: list[ACPSession] = []

    def _build(agent_cfg: Any, _entry: Any) -> ACPSession:
        s = _fake_session(broker, agent_cfg.agent_id, state=AgentState.INITIALIZING)
        s._session_id = "acp-successor"
        successors.append(s)
        return s

    lifecycle._build_session = _build  # type: ignore[method-assign]
    def _run_task(_aid: str, _session: Any) -> asyncio.Task:
        return asyncio.create_task(asyncio.sleep(0))

    lifecycle._make_run_task = _run_task  # type: ignore[method-assign]
    return lifecycle, successors


async def _reach_idle(broker: ACPBroker, agent_id: str, session: ACPSession) -> None:
    """Drive the successor to IDLE through the real sink, which triggers a drain."""
    session._sm._state = AgentState.IDLE
    await broker._sink(
        AgentStateChanged(
            agent_id=agent_id, old_state=AgentState.INITIALIZING, new_state=AgentState.IDLE
        )
    )
    await asyncio.sleep(0)


class TestHandoffRekey:
    async def test_apply_handoff_rekey_moves_owned_state_and_keeps_recipients(
        self, tmp_path: Path
    ) -> None:
        """The in-memory classification, all three rules at once.

        Every wrong bucket here fails without an exception: a moved queue key strands an
        already-delivered item, a moved _journal_seq starts the successor mid-sequence,
        an unmoved diagnostic credits the predecessor's dropped chunks to the successor,
        and a cleared-versus-moved permission mistake leaves a dead agent's prompt
        pending forever.
        """
        from synth_acp.db import AgentRenameResult

        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        session = _fake_session(broker, "worker")
        broker._registry.register("worker", session)
        broker._registry.set_parent("worker", "boss")
        broker._registry.set_parent("kid", "worker")
        broker._registry.set_harness("worker", "kiro")
        broker._journal_seq["worker"] = 7
        broker._chunk_drops["worker"] = 3
        broker._drop_armed["worker"] = True
        broker._warned_agents.add("worker")
        broker._prompt_queue.enqueue(
            "worker", QueuedItem(id="inbound", text="for the successor", from_agent="kid")
        )
        broker._prompt_queue.enqueue(
            "kid", QueuedItem(id="authored", text="from the predecessor", from_agent="worker")
        )
        broker._active_permission["worker"] = "req-1"
        broker._permission_queue["worker"] = []
        broker._permission_counter["worker"] = (1, 1)
        lock = broker._registry.agent_lock("worker")

        result = AgentRenameResult(
            old_agent_id="worker",
            new_agent_id="worker.h0000dead",
            parent="boss",
            task="the task",
            harness="kiro",
            agent_mode=None,
            cwd=".",
            retired_acp_session_id="acp-1",
            rows_moved={},
        )
        broker.apply_handoff_rekey(result)

        # Predecessor-owned state moved, and the session object now stamps its events
        # with the retired id.
        assert broker._registry.get_session("worker.h0000dead") is session
        assert session.agent_id == "worker.h0000dead"
        assert broker._journal_seq == {"worker.h0000dead": 7}
        assert broker._chunk_drops == {"worker.h0000dead": 3}
        assert broker._drop_armed == {"worker.h0000dead": True}
        assert broker._warned_agents == {"worker.h0000dead"}
        assert "worker" not in broker._turn_buffer
        assert "worker" not in broker._turn_buffer_tool_index

        # POINTER: the child still points at the original id, and the successor inherits
        # the predecessor's own parent and harness AT that id.
        assert broker._registry.get_parent("kid") == "worker"
        assert broker._registry.get_parent("worker") == "boss"
        assert broker._registry.get_harness("worker") == "kiro"

        # RECIPIENT: the queue key stays so the successor receives the inherited item,
        # while the predecessor's AUTHORED value follows it.
        assert [i.text for i in broker._prompt_queue.items("worker")] == ["for the successor"]
        assert broker._prompt_queue.items("kid")[0].from_agent == "worker.h0000dead"

        # The lock belongs to the ID and is inherited by the successor, unchanged.
        assert broker._registry.agent_lock("worker") is lock

        # CLEARED, not moved: a dead predecessor can never answer these.
        assert "worker" not in broker._active_permission
        assert "worker" not in broker._permission_queue
        assert "worker" not in broker._permission_counter
        assert lifecycle is not None

    async def test_successor_first_journal_event_lands_at_seq_zero(self, tmp_path: Path) -> None:
        """The successor's journal must start empty, or its transcript interleaves with
        the predecessor's rows on restore."""
        from synth_acp.db import AgentRenameResult

        broker = _make_broker("worker", tmp_path=tmp_path)
        broker._journal_seq["worker"] = 42
        broker.apply_handoff_rekey(
            AgentRenameResult(
                old_agent_id="worker",
                new_agent_id="worker.h0000dead",
                parent=None,
                task="",
                harness="kiro",
                agent_mode=None,
                cwd=".",
                retired_acp_session_id=None,
                rows_moved={},
            )
        )

        broker._buffer_journal_event(UserPromptSubmitted(agent_id="worker", text="first turn"))

        assert [seq for seq, _ in broker._turn_buffer["worker"]] == [0]
        assert broker._journal_seq["worker.h0000dead"] == 42

    async def test_agent_handed_off_is_not_journaled(self, tmp_path: Path) -> None:
        """Journaling it would replay a transition on restore for an agent that is not
        restored at all: the predecessor is 'inactive' and skipped entirely."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        broker._buffer_journal_event(
            AgentHandedOff(
                agent_id="worker", retired_agent_id="worker.h1", parent=None, task=""
            )
        )
        assert broker._turn_buffer == {}
        assert broker._journal_seq == {}


class TestDrainAgentJournal:
    async def test_drain_awaits_in_flight_flush_and_empties_buffer(self, tmp_path: Path) -> None:
        """A flush that lands after the rename writes rows under the OLD id that the
        UPDATE already passed, so those events silently belong to the wrong agent."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        await broker._ensure_lifecycle()
        release = asyncio.Event()
        landed: list[str] = []

        async def _slow() -> None:
            await release.wait()
            landed.append("in-flight")

        task = asyncio.create_task(_slow(), name="journal-flush-worker")
        broker._pending_flushes.add(task)
        broker._turn_buffer["worker"] = [
            (0, TurnComplete(agent_id="worker", stop_reason="end_turn"))
        ]
        broker._turn_buffer_tool_index["worker"] = {"tc-1": 0}

        drain = asyncio.create_task(broker.drain_agent_journal("worker"))
        await asyncio.sleep(0)
        assert not drain.done(), "drain returned without awaiting the in-flight flush"
        release.set()
        await drain

        assert landed == ["in-flight"]
        assert "worker" not in broker._turn_buffer
        assert "worker" not in broker._turn_buffer_tool_index
        broker._pending_flushes.discard(task)

    async def test_drain_ignores_another_agents_flush(self, tmp_path: Path) -> None:
        """Names are matched exactly: 'journal-flush-x' is a PREFIX of
        'journal-flush-x-2', so a prefix test would block a handoff behind an unrelated
        agent's flush -- or worse, await one that never completes."""
        broker = _make_broker("x", tmp_path=tmp_path)
        await broker._ensure_lifecycle()
        never = asyncio.Event()

        async def _blocked() -> None:
            await never.wait()

        other = asyncio.create_task(_blocked(), name="journal-flush-x-2")
        broker._pending_flushes.add(other)

        await asyncio.wait_for(broker.drain_agent_journal("x"), timeout=1.0)

        assert not other.done()
        other.cancel()
        broker._pending_flushes.discard(other)


class TestHandoffFirstPromptOrdering:
    """The handoff message must be the successor's first DELIVERED text.

    Text reaches a session through exactly two calls in this codebase --
    ``session.prompt`` and ``session.steer`` -- and a caller that was already parked on
    the inherited agent lock when the reservation appeared cannot be ordered by any check
    at an entry point. These tests drive those parked schedules deterministically, with no
    sleeps, because every one of them fails silently: the successor simply opens on the
    wrong text and every count-based assertion still passes.
    """

    @staticmethod
    def _texts(session: ACPSession) -> list[str]:
        return [c.args[0] for c in session.prompt.await_args_list]  # type: ignore[attr-defined]

    async def test_seed_goes_ahead_of_an_inherited_queue_item(self, tmp_path: Path) -> None:
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, successors = await _handoff_harness(broker)
        # Already sitting under the recipient key at handoff time: its row is delivered,
        # so nothing will ever retry it.
        broker._prompt_queue.enqueue(
            "worker", QueuedItem(text="inherited", from_agent="kid", source="mcp")
        )

        result = await lifecycle.handoff("worker", "HANDOFF BRIEF")

        assert result.successor_started is True
        queued = broker._prompt_queue.items("worker")
        assert queued[0].first_prompt is True
        assert queued[0].from_agent == result.retired_agent_id
        assert queued[0].steerable is False
        assert queued[1].text == "inherited"

        successor = successors[0]
        await _reach_idle(broker, "worker", successor)
        assert self._texts(successor)[0].endswith("HANDOFF BRIEF")
        await _reach_idle(broker, "worker", successor)
        assert self._texts(successor)[1] == "inherited"

    async def test_a_prompt_parked_on_the_inherited_lock_cannot_precede_the_brief(
        self, tmp_path: Path
    ) -> None:
        """A prompt that entered lifecycle.prompt before the reservation existed is parked
        on the lock and resumes past every outer check. asyncio.Lock is FIFO, so it
        acquires BEFORE the drain carrying the handoff message."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, successors = await _handoff_harness(broker)
        released = asyncio.Event()
        real_drain = broker.drain_agent_journal

        async def _blocked_drain(agent_id: str) -> None:
            await released.wait()
            await real_drain(agent_id)

        broker.drain_agent_journal = _blocked_drain  # type: ignore[method-assign]

        handoff = asyncio.create_task(lifecycle.handoff("worker", "HANDOFF BRIEF"))
        await asyncio.sleep(0)
        parked = asyncio.create_task(broker.submit_prompt("worker", "later", "mcp", "kid"))
        await asyncio.sleep(0)
        assert broker._registry.agent_lock("worker").locked()
        assert not parked.done(), "the prompt should be parked on the inherited lock"

        released.set()
        result = await handoff
        await parked

        successor = successors[0]
        await _reach_idle(broker, "worker", successor)

        assert result.successor_started is True
        assert self._texts(successor) == [self._texts(successor)[0]]
        assert self._texts(successor)[0].endswith("HANDOFF BRIEF")
        # Refused, not lost: it is queued behind the reserved item.
        assert "later" in [i.text for i in broker._prompt_queue.items("worker")]

    async def test_a_steer_parked_before_the_reservation_cannot_precede_the_brief(
        self, tmp_path: Path
    ) -> None:
        """session.steer needs only an ACP session id, which a successor has while it is
        still INITIALIZING, and _try_steer bails only when the state IS idle. So a parked
        steer would inject text ahead of the reserved prompt -- bypassing the prompt seam
        entirely, leaving every prompt-level assertion green."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, successors = await _handoff_harness(broker)
        broker._registry.get_session("worker")._sm._state = AgentState.BUSY
        broker._registry.get_session("worker")._steer_protocol = "kiro"
        released = asyncio.Event()
        real_drain = broker.drain_agent_journal

        async def _blocked_drain(agent_id: str) -> None:
            await released.wait()
            await real_drain(agent_id)

        broker.drain_agent_journal = _blocked_drain  # type: ignore[method-assign]

        handoff = asyncio.create_task(lifecycle.handoff("worker", "HANDOFF BRIEF"))
        await asyncio.sleep(0)
        parked = asyncio.create_task(
            broker.submit_prompt("worker", "steer me", "mcp", "kid", steerable=True)
        )
        await asyncio.sleep(0)
        assert not parked.done(), "the steer should be parked on the inherited lock"

        released.set()
        await handoff
        await parked

        successor = successors[0]
        successor.steer.assert_not_awaited()  # type: ignore[attr-defined]
        queued = broker._prompt_queue.items("worker")
        assert queued[0].first_prompt is True
        assert "steer me" in [i.text for i in queued]

    async def test_a_refused_prompt_is_announced_under_neither_agent(
        self, tmp_path: Path
    ) -> None:
        """UserPromptSubmitted is journaled by event.agent_id, so announcing before
        acceptance filed a refused prompt in the PREDECESSOR's transcript while the
        successor that eventually received the text had no prompt event at all -- output
        with no causal prompt on restore."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, successors = await _handoff_harness(broker)
        released = asyncio.Event()
        real_drain = broker.drain_agent_journal

        async def _blocked_drain(agent_id: str) -> None:
            await released.wait()
            await real_drain(agent_id)

        broker.drain_agent_journal = _blocked_drain  # type: ignore[method-assign]

        handoff = asyncio.create_task(lifecycle.handoff("worker", "HANDOFF BRIEF"))
        await asyncio.sleep(0)
        parked = asyncio.create_task(broker.submit_prompt("worker", "later", "mcp", "kid"))
        await asyncio.sleep(0)
        released.set()
        result = await handoff
        await parked

        def _announced(agent_id: str) -> list[str]:
            return [
                e.text
                for seq_and_event in broker._turn_buffer.get(agent_id, [])
                for e in [seq_and_event[1]]
                if isinstance(e, UserPromptSubmitted)
            ]

        # Refused text is announced nowhere at all.
        assert _announced(result.retired_agent_id) == []
        assert _announced("worker") == []

        successor = successors[0]
        await _reach_idle(broker, "worker", successor)
        # Once accepted it is announced exactly once, under the id that accepted it, and
        # the successor's journal starts at seq 0 (its first entry is the startup-hook
        # line, mirroring a launch).
        buffered = broker._turn_buffer["worker"]
        assert next(seq for seq, _ in buffered) == 0
        prompts = [e.text for _, e in buffered if isinstance(e, UserPromptSubmitted)]
        assert len(prompts) == 1
        assert prompts[0].endswith("HANDOFF BRIEF")
        assert _announced(result.retired_agent_id) == []

    async def test_an_already_popped_inherited_item_is_requeued_behind_the_reservation(
        self, tmp_path: Path
    ) -> None:
        """Requeueing a refused non-first item at the FRONT livelocks: it sits ahead of
        the reserved prompt, is popped and refused on every drain, and the handoff message
        is never delivered at all -- with no exception ever raised."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, _ = await _handoff_harness(broker)
        item = QueuedItem(text="inherited", from_agent="kid", source="mcp")
        lifecycle._reserved_first_prompt["worker"] = "worker.h0000dead"
        broker.seed_first_prompt("worker", "BRIEF", "worker.h0000dead")

        refused = await lifecycle.prompt("worker", item.text, first_prompt=item.first_prompt)
        broker._restore_refused_item("worker", item)

        assert refused is False
        texts = [i.text for i in broker._prompt_queue.items("worker")]
        assert texts == ["BRIEF", "inherited"], "the refused item must land BEHIND the reserved one"

    async def test_a_refused_reserved_item_returns_to_the_front(self, tmp_path: Path) -> None:
        """If the successor leaves IDLE between the drain's check and prompt's, the
        reserved item is refused by the state check. Losing it would leave the successor
        running with no context while the reservation still refuses everything else."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, successors = await _handoff_harness(broker)
        result = await lifecycle.handoff("worker", "HANDOFF BRIEF")
        successor = successors[0]
        successor._sm._state = AgentState.BUSY  # not IDLE when prompt re-checks

        drained = await broker._force_drain("worker")

        assert drained is False
        queued = broker._prompt_queue.items("worker")
        assert queued[0].first_prompt is True
        assert queued[0].text.endswith("HANDOFF BRIEF")
        assert lifecycle._reserved_first_prompt["worker"] == result.retired_agent_id

        await _reach_idle(broker, "worker", successor)
        assert self._texts(successor)[0].endswith("HANDOFF BRIEF")
        assert "worker" not in lifecycle._reserved_first_prompt


class TestHandoffBehavioral:
    @staticmethod
    def _successor(broker: ACPBroker, agent_id: str = "worker") -> Any:
        return broker._registry.get_session(agent_id)

    async def test_child_can_address_and_reach_the_successor_in_local_mode(
        self, tmp_path: Path
    ) -> None:
        """The failure this whole pointer rule exists to prevent.

        A severed parent pointer makes LOCAL visibility quietly stop resolving the id, so
        the child's messages simply vanish -- no exception, no log line, nothing for a
        test of the rename alone to catch. Nothing changes on the child's side: it keeps
        addressing the same literal id it was told at startup.
        """
        import sqlite3 as _sq

        from synth_acp.models.visibility import get_visible_agents

        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, successors = await _handoff_harness(broker)

        def _child_parent() -> str | None:
            conn = _sq.connect(str(broker._db_path))
            try:
                return conn.execute(
                    "SELECT parent FROM agents WHERE agent_id = 'kid' AND session_id = ?",
                    (broker.session_id,),
                ).fetchone()[0]
            finally:
                conn.close()

        before = _child_parent()
        result = await lifecycle.handoff("worker", "HANDOFF BRIEF")
        assert _child_parent() == before == "worker"

        conn = _sq.connect(str(broker._db_path))
        try:
            visible = get_visible_agents(conn, "kid", broker.session_id, "LOCAL")
        finally:
            conn.close()
        assert "worker" in visible, "the child can no longer address its own parent"

        successor = successors[0]
        await _reach_idle(broker, "worker", successor)  # drains the reserved brief
        await broker._on_mcp_message("worker", "child says hi", "kid", "chat")
        await _reach_idle(broker, "worker", successor)

        delivered = [c.args[0] for c in successor.prompt.await_args_list]  # type: ignore[attr-defined]
        assert delivered[0].endswith("HANDOFF BRIEF")
        assert len([t for t in delivered if "child says hi" in t]) == 1
        assert result.successor_started is True

    async def test_a_pending_message_row_still_targets_the_original_id(
        self, tmp_path: Path
    ) -> None:
        """A blanket UPDATE of messages.to_agent would strand a pending row on a retired
        agent: message_bus._poll_messages does not filter on recipient status, so a row
        left at the original id is delivered to the successor with no forwarding code."""
        import sqlite3 as _sq

        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle, _ = await _handoff_harness(broker)
        conn = _sq.connect(str(broker._db_path))
        try:
            conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
                "VALUES (?, 'kid', 'worker', 'in flight', 'pending', 100)",
                (broker.session_id,),
            )
            conn.commit()
        finally:
            conn.close()

        result = await lifecycle.handoff("worker", "HANDOFF BRIEF")

        # The row still targets the ORIGINAL id, so the poll loop -- which does not filter
        # on recipient status -- hands it to whoever occupies that id now.
        conn = _sq.connect(str(broker._db_path))
        try:
            assert conn.execute(
                "SELECT to_agent FROM messages WHERE body = 'in flight'"
            ).fetchone() == ("worker",)
        finally:
            conn.close()

        # Deliver it for real, then assert exactly-once against the successor AND the
        # terminal row status. Stopping at the to_agent value would still pass if delivery
        # were broken, which is the silent failure criterion 38 exists to catch.
        from synth_acp.broker.message_bus import MessageBus

        successor = self._successor(broker)
        await _reach_idle(broker, "worker", successor)  # drains the reserved brief first
        bus = MessageBus(broker._db_path, broker.session_id, broker._on_mcp_message, None)
        await bus._poll_messages()
        await _reach_idle(broker, "worker", successor)
        await bus._poll_messages()  # a second poll must find nothing to redeliver

        delivered = [c.args[0] for c in successor.prompt.await_args_list]
        assert len([t for t in delivered if "in flight" in t]) == 1
        conn = _sq.connect(str(broker._db_path))
        try:
            assert conn.execute(
                "SELECT status FROM messages WHERE body = 'in flight'"
            ).fetchone() == ("delivered",)
        finally:
            conn.close()
        assert result.successor_started is True


class TestSerializedCommandTail:
    """FIFO across the whole session, and a tail that is never cancelled mid-command."""

    @staticmethod
    def _insert(broker: ACPBroker, command: str, from_agent: str = "worker") -> int:
        import sqlite3 as _sq

        from synth_acp.db import ensure_schema_sync

        conn = _sq.connect(str(broker._db_path))
        try:
            ensure_schema_sync(conn)
            cur = conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, "
                "created_at) VALUES (?, ?, ?, '{}', 'processing', 100)",
                (broker.session_id, from_agent, command),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    @staticmethod
    def _status(broker: ACPBroker, cmd_id: int) -> str:
        import sqlite3 as _sq

        conn = _sq.connect(str(broker._db_path))
        try:
            return conn.execute(
                "SELECT status FROM agent_commands WHERE id = ?", (cmd_id,)
            ).fetchone()[0]
        finally:
            conn.close()

    async def test_a_later_command_cannot_overtake_an_active_handoff(
        self, tmp_path: Path
    ) -> None:
        """A claimed batch of [handoff, launch] must run in that order. A bare
        create_task would let the launch see a half-transitioned agent -- and would strand
        the already-claimed remainder of the batch in 'processing' when _process_commands
        returned."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        order: list[str] = []
        released = asyncio.Event()
        started = asyncio.Event()

        async def _handoff_cmd(cmd_id: int, from_agent: str, data: dict) -> None:
            order.append("handoff-start")
            started.set()
            await released.wait()
            order.append("handoff-end")
            await lifecycle.update_command_status(cmd_id, "processed")

        async def _launch_cmd(cmd_id: int, from_agent: str, data: dict) -> None:
            order.append("launch-start")
            await lifecycle.update_command_status(cmd_id, "processed")

        lifecycle.handle_handoff_command = _handoff_cmd  # type: ignore[method-assign]
        lifecycle.handle_launch_command = _launch_cmd  # type: ignore[method-assign]
        h_id = self._insert(broker, "handoff")
        l_id = self._insert(broker, "launch")

        batch = asyncio.create_task(
            broker._process_commands([(h_id, "worker", "handoff", "{}"), (l_id, "worker", "launch", "{}")])
        )
        await started.wait()
        assert order == ["handoff-start"]
        assert not batch.done()

        released.set()
        await batch

        assert order == ["handoff-start", "handoff-end", "launch-start"]
        # Callers still observe SETTLED rows when _process_commands returns.
        assert self._status(broker, h_id) == "processed"
        assert self._status(broker, l_id) == "processed"
        await broker._stop_command_tail()

    async def test_a_new_batch_cannot_overtake_an_active_handoff(self, tmp_path: Path) -> None:
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        order: list[str] = []
        released = asyncio.Event()
        started = asyncio.Event()

        async def _handoff_cmd(cmd_id: int, from_agent: str, data: dict) -> None:
            order.append("handoff-start")
            started.set()
            await released.wait()
            order.append("handoff-end")
            await lifecycle.update_command_status(cmd_id, "processed")

        async def _terminate_cmd(cmd_id: int, from_agent: str, data: dict) -> None:
            order.append("terminate-start")
            await lifecycle.update_command_status(cmd_id, "processed")

        lifecycle.handle_handoff_command = _handoff_cmd  # type: ignore[method-assign]
        lifecycle.handle_terminate_command = _terminate_cmd  # type: ignore[method-assign]
        first = asyncio.create_task(
            broker._process_commands([(self._insert(broker, "handoff"), "worker", "handoff", "{}")])
        )
        await started.wait()
        second = asyncio.create_task(
            broker._process_commands([(self._insert(broker, "terminate"), "worker", "terminate", "{}")])
        )
        await asyncio.sleep(0)
        assert order == ["handoff-start"], "a newly claimed batch overtook the handoff"

        released.set()
        await first
        await second
        # The exact order proves it: an overtake would put terminate-start BEFORE
        # handoff-end, which no yield-count assumption is needed to detect.
        assert order == ["handoff-start", "handoff-end", "terminate-start"]
        await broker._stop_command_tail()

    async def test_every_row_settles_when_a_handler_raises(self, tmp_path: Path) -> None:
        """A stranded 'processing' row is invisible until the next process start requeues
        it, so a raising handler must not take the rest of its batch down with it."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()

        async def _boom(cmd_id: int, from_agent: str, data: dict) -> None:
            raise RuntimeError("handler exploded")

        async def _ok(cmd_id: int, from_agent: str, data: dict) -> None:
            await lifecycle.update_command_status(cmd_id, "processed")

        lifecycle.handle_handoff_command = _boom  # type: ignore[method-assign]
        lifecycle.handle_terminate_command = _ok  # type: ignore[method-assign]
        bad = self._insert(broker, "handoff")
        good = self._insert(broker, "terminate")

        await broker._process_commands(
            [(bad, "worker", "handoff", "{}"), (good, "worker", "terminate", "{}")]
        )

        assert self._status(broker, bad) == "rejected"
        assert self._status(broker, good) == "processed"
        await broker._stop_command_tail()

    async def test_stopping_the_tail_lets_an_active_command_finish(self, tmp_path: Path) -> None:
        """Cancelling the tail mid-dispatch would tear a handoff apart between its
        committed rename and its in-memory re-key -- the exact partial state that
        AgentLifecycle.shutdown deliberately refuses to create by not cancelling."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        released = asyncio.Event()
        dispatching = asyncio.Event()
        finished: list[str] = []

        async def _handoff_cmd(cmd_id: int, from_agent: str, data: dict) -> None:
            dispatching.set()
            await released.wait()
            finished.append("handoff")
            await lifecycle.update_command_status(cmd_id, "processed")

        lifecycle.handle_handoff_command = _handoff_cmd  # type: ignore[method-assign]
        cmd_id = self._insert(broker, "handoff")
        batch = asyncio.create_task(
            broker._process_commands([(cmd_id, "worker", "handoff", "{}")])
        )
        await dispatching.wait()

        stop = asyncio.create_task(broker._stop_command_tail())
        await asyncio.sleep(0)
        released.set()
        await batch
        await stop

        assert finished == ["handoff"], "the in-flight command was not allowed to finish"
        assert broker._command_tail_task is not None
        assert not broker._command_tail_task.cancelled()
        assert self._status(broker, cmd_id) == "processed"

    async def test_a_processing_row_is_not_reclaimed_by_a_later_poll(self, tmp_path: Path) -> None:
        """Leaving a handoff's row at 'processing' while the owned task runs is correct
        BECAUSE later claims select only 'pending'. If they did not, returning early would
        double-process the handoff and produce a second successor."""
        import sqlite3 as _sq

        from synth_acp.broker.message_bus import MessageBus

        broker = _make_broker("worker", tmp_path=tmp_path)
        cmd_id = self._insert(broker, "handoff")  # inserted already 'processing'
        claimed: list[list] = []

        async def _capture(rows: list) -> None:
            claimed.append(rows)

        bus = MessageBus(broker._db_path, broker.session_id, AsyncMock(), _capture)
        await bus._process_pending_commands()

        assert claimed == [], "a 'processing' row was reclaimed and would be run twice"
        conn = _sq.connect(str(broker._db_path))
        try:
            assert conn.execute(
                "SELECT status FROM agent_commands WHERE id = ?", (cmd_id,)
            ).fetchone() == ("processing",)
        finally:
            conn.close()


class TestRefusedItemPreservesQueueOrder:
    """A refusal is reachable with no handoff anywhere: both drains check IDLE, then
    await, and ``prompt`` re-checks under the agent lock."""

    async def test_refused_item_returns_to_the_front_when_nothing_is_reserved(
        self, tmp_path: Path
    ) -> None:
        """Sending it to the back reorders the user's queue on every refusal, silently.
        The back is only correct while a reservation is pending, where the front would
        livelock ahead of the reserved prompt."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        first = QueuedItem(text="one", source="user")
        second = QueuedItem(text="two", source="user")
        broker._prompt_queue.enqueue("worker", second)

        broker._restore_refused_item("worker", first)

        texts = [i.text for i in broker._prompt_queue.items("worker")]
        assert texts == ["one", "two"], "a refused item must keep its place in the queue"

    async def test_handoff_in_flight_is_false_with_no_handoff(self, tmp_path: Path) -> None:
        """The UI arms its selection-restore marker on this predicate.  Armed on every
        termination instead, a later INITIALIZING for the same id -- an ordinary resurrect
        -- would steal the user's selection."""
        broker = _make_broker("worker", tmp_path=tmp_path)
        lifecycle = await broker._ensure_lifecycle()
        assert broker.handoff_in_flight("worker") is False

        lifecycle._active_handoffs["worker"] = asyncio.Event()
        assert broker.handoff_in_flight("worker") is True


def _nudge_rows(db_path: Path) -> list[tuple[str, str, str, str]]:
    """Return (from_agent, to_agent, kind, body) for every queued message row."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT from_agent, to_agent, kind, body FROM messages ORDER BY id"
        ).fetchall()


class TestHandoffNudge:
    """One nudge per agent, only above the threshold, only when enabled.

    Every failure in this area is silent: an agent is either nudged repeatedly
    (noise it must answer) or never nudged at all.
    """

    @staticmethod
    def _broker(tmp_path: Path, **settings: Any) -> ACPBroker:
        config = SessionConfig(project="test-session", settings=settings)
        broker = ACPBroker(
            config=config,
            initial_agent=AgentConfig(agent_id="agent-1", harness="kiro"),
            db_path=tmp_path / "synth.db",
        )
        ensure_schema_sync(sqlite3.connect(tmp_path / "synth.db"))
        return broker

    async def test_crossing_threshold_queues_one_notification(self, tmp_path: Path) -> None:
        """The nudge is written from 'synth' with kind 'notification'.

        The kind is load-bearing: 'system' would be blocked from in-turn steering,
        and a kind with no normalize_message_kind branch degrades to 'chat'.
        """
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=62))

        rows = _nudge_rows(tmp_path / "synth.db")
        assert len(rows) == 1
        from_agent, to_agent, kind, body = rows[0]
        assert (from_agent, to_agent, kind) == ("synth", "agent-1", "notification")
        assert "62% full" in body
        assert "50% nudge threshold" in body
        assert "handoff" in body

    async def test_second_report_above_threshold_queues_nothing(self, tmp_path: Path) -> None:
        """Kiro reports usage several times per turn; only the first may nudge."""
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=62))
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=71))
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=99))

        assert len(_nudge_rows(tmp_path / "synth.db")) == 1

    async def test_below_threshold_queues_nothing(self, tmp_path: Path) -> None:
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=49))

        assert _nudge_rows(tmp_path / "synth.db") == []

    async def test_disabled_queues_nothing_even_when_full(self, tmp_path: Path) -> None:
        broker = self._broker(tmp_path, handoff_nudge=False)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=100))

        assert _nudge_rows(tmp_path / "synth.db") == []

    async def test_unknown_context_size_queues_nothing(self, tmp_path: Path) -> None:
        """size=0 means the harness reported no usable figure, not a full window."""
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=0, used=0))

        assert _nudge_rows(tmp_path / "synth.db") == []

    async def test_real_token_counts_use_the_same_ratio(self, tmp_path: Path) -> None:
        """Claude reports genuine token counts; the nudge is harness-agnostic."""
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=200_000, used=150_000))

        rows = _nudge_rows(tmp_path / "synth.db")
        assert len(rows) == 1
        assert "75% full" in rows[0][3]

    async def test_rekey_moves_the_flag_so_the_successor_is_nudgeable(
        self, tmp_path: Path
    ) -> None:
        """A handoff hands the id to a successor that starts with an empty context.

        The flag is predecessor-owned. Leaving it at the original id would mute the
        successor for the whole life of the process, which is the exact failure the
        feature exists to prevent.
        """
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=80))
        assert broker._nudged_agents == {"agent-1"}

        from synth_acp.db import AgentRenameResult

        broker.apply_handoff_rekey(
            AgentRenameResult(
                old_agent_id="agent-1",
                new_agent_id="agent-1.h0badc0de",
                parent=None,
                task="the task",
                harness="kiro",
                agent_mode=None,
                cwd=".",
                retired_acp_session_id="acp-1",
                rows_moved={},
            )
        )
        assert broker._nudged_agents == {"agent-1.h0badc0de"}

        # The successor, at the original id, fills up in its turn and IS nudged.
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=80))
        rows = _nudge_rows(tmp_path / "synth.db")
        assert [r[1] for r in rows] == ["agent-1", "agent-1"]

    async def test_notification_is_steer_eligible_unlike_system(self, tmp_path: Path) -> None:
        """A nudge must reach a mid-turn agent, so it must not be blocked like 'system'.

        _steer_eligible hard-blocks kind == "system" because join/exit notices are
        "no action required". The nudge is the opposite: it wants to be acted on.
        Without this the nudge silently waits for IDLE, which on a long turn is
        exactly when a context-full agent least benefits from it.
        """
        broker = self._broker(tmp_path)
        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
            steer_protocol="kiro",
        )
        broker._registry.register("agent-1", session)

        assert broker._steer_eligible("agent-1", "notification") is True
        assert broker._steer_eligible("agent-1", "system") is False

    async def test_real_kiro_payload_reaches_the_message_table(self, tmp_path: Path) -> None:
        """Drive a verbatim kiro-cli 2.18.1 payload through a real ACPSession.

        The unit tests above cover the adaptor and the nudge separately. This one
        joins them, so a mismatch at the seam — a wrong notification name, a
        percentage that never becomes a ratio — cannot pass while both halves do.
        Payload copied from a live probe against kiro-cli 2.18.1.
        """
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        session = ACPSession(
            agent_id="agent-1",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=broker._sink,
        )
        session._session_id = "sess-1"

        await session.ext_notification(
            "kiro.dev/metadata",
            {
                "sessionId": "sess-1",
                "contextUsagePercentage": 82.4,
                "meteringUsage": [
                    {"value": 0.7706246474958541, "unit": "credit", "unitPlural": "credits"}
                ],
                "turnDurationMs": 4372,
            },
        )

        rows = _nudge_rows(tmp_path / "synth.db")
        assert len(rows) == 1
        assert (rows[0][0], rows[0][1], rows[0][2]) == ("synth", "agent-1", "notification")
        assert "82% full" in rows[0][3]

        usage = broker.get_usage("agent-1")
        assert usage is not None
        assert (usage.used, usage.size) == (82, 100)
        assert usage.cost_currency == "credits"

    async def test_simultaneous_reports_queue_exactly_one(self, tmp_path: Path) -> None:
        """Concurrency is the whole reason the claim precedes the first await.

        Kiro reports usage several times per turn, so these arrive close together.
        The claim is a membership test plus set.add with no await between them, so
        it is atomic against other coroutines.

        The boundary that matters is the DB write, which suspends. Verified by
        mutation: moving the claim to after send_notification makes this test
        write 20 rows instead of 1, while the sequential test above still passes.
        Moving it merely below _ensure_lifecycle changes nothing, because awaiting
        a coroutine that never suspends does not yield.
        """
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        await asyncio.gather(
            *(broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=80)) for _ in range(20))
        )

        assert len(_nudge_rows(tmp_path / "synth.db")) == 1
        assert broker._nudged_agents == {"agent-1"}

    async def test_failed_write_releases_the_claim_for_a_later_retry(
        self, tmp_path: Path
    ) -> None:
        """A claim held for a write that never happened would mute the agent forever.

        The claim is taken before the write, so the write failing must release it.
        Without the release the agent is silently never nudged: the flag says it
        was, and no row exists.
        """
        broker = self._broker(tmp_path, handoff_nudge_threshold=0.5)
        lifecycle = await broker._ensure_lifecycle()

        with patch.object(
            lifecycle, "send_notification", side_effect=sqlite3.OperationalError("disk I/O error")
        ):
            await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=80))

        assert _nudge_rows(tmp_path / "synth.db") == []
        assert broker._nudged_agents == set()

        # A later report retries and succeeds.
        await broker._sink(UsageUpdated(agent_id="agent-1", size=100, used=85))
        rows = _nudge_rows(tmp_path / "synth.db")
        assert len(rows) == 1
        assert "85% full" in rows[0][3]
