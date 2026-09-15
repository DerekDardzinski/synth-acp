"""Tests for ACPSession state machine enforcement."""

from __future__ import annotations

import ast
import asyncio
import gc
import inspect
import logging
import math
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.contrib.session_state import SessionAccumulator
from acp.exceptions import RequestError
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    Cost,
    CurrentModeUpdate,
    FileEditToolCallContent,
    McpServerStdio,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionNotification,
    TextContentBlock,
    ToolCallLocation as AcpToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)

from synth_acp.acp.session import (
    DRAIN_PASSES,
    ACPSession,
    HandshakeTimeoutError,
    _spawn_isolated_agent,
    _StderrTail,
)
from synth_acp.diagnostics import LoopLagSampler
from synth_acp.models.agent import AgentState
from synth_acp.models.events import (
    AgentModeChanged,
    AgentModelChanged,
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerError,
    BrokerEvent,
    ConfigOptionChanged,
    ConfigOptionsReceived,
    MessageChunkReceived,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
)


def _msg_chunk(text: str, message_id: str = "m1") -> AgentMessageChunk:
    return AgentMessageChunk(
        content=TextContentBlock(type="text", text=text),
        message_id=message_id,
        session_update="agent_message_chunk",
    )


def _thought_chunk(text: str, message_id: str = "m1") -> AgentThoughtChunk:
    return AgentThoughtChunk(
        content=TextContentBlock(type="text", text=text),
        message_id=message_id,
        session_update="agent_thought_chunk",
    )


class TestSessionUpdate:
    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
        )
        s._session_id = "sess-1"
        return s

    async def test_session_update_when_thought_chunk_emits_event(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Thought chunks must emit AgentThoughtReceived — otherwise agent reasoning is invisible."""
        await session.session_update("sess-1", _thought_chunk("reasoning"))
        await asyncio.sleep(0)
        assert len(events) == 1
        assert isinstance(events[0], AgentThoughtReceived)
        assert events[0].chunk == "reasoning"
        assert events[0].agent_id == "test"

    async def test_session_update_when_usage_update_emits_event(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Usage updates must emit UsageUpdated — otherwise cost/context data is lost."""
        update = UsageUpdate(
            size=128000,
            used=32000,
            cost=Cost(amount=0.14, currency="USD"),
            session_update="usage_update",
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
        assert isinstance(events[0], UsageUpdated)
        assert events[0].size == 128000
        assert events[0].used == 32000
        assert events[0].cost_amount == 0.14
        assert events[0].cost_currency == "USD"

    async def test_session_update_when_tool_call_branch_has_diff_extracts_diffs(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Diffs on initial tool_call must be extracted — otherwise file edits are silently lost."""
        diff_item = FileEditToolCallContent(
            type="diff",
            path="src/main.py",
            old_text="old",
            new_text="new",
        )
        update = ToolCallStart(
            tool_call_id="tc-1",
            title="Edit file",
            kind="edit",
            status="pending",
            content=[diff_item],
            locations=None,
            raw_input=None,
            raw_output=None,
            session_update="tool_call",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert len(evt.diffs) == 1
        assert evt.diffs[0] == ToolCallDiff(path="src/main.py", old_text="old", new_text="new")

    async def test_session_update_when_tool_call_has_text_content_extracts_text(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Text content must be extracted — otherwise command output is silently dropped."""
        text_block = TextContentBlock(type="text", text="hello world")
        content_item = ContentToolCallContent(type="content", content=text_block)
        update = ToolCallStart(
            tool_call_id="tc-3",
            title="Run command",
            kind="execute",
            status="pending",
            content=[content_item],
            locations=None,
            raw_input=None,
            raw_output=None,
            session_update="tool_call",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert evt.text_content == "hello world"

    async def test_session_update_when_tool_call_has_locations_extracts_locations(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Locations must be extracted — otherwise file path context is silently lost."""
        loc = AcpToolCallLocation(path="/abs/path.py", line=42)
        update = ToolCallStart(
            tool_call_id="tc-4",
            title="Read file",
            kind="read",
            status="pending",
            content=None,
            locations=[loc],
            raw_input=None,
            raw_output=None,
            session_update="tool_call",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert len(evt.locations) == 1
        assert evt.locations[0] == ToolCallLocation(path="/abs/path.py", line=42)

    async def test_session_update_from_wrong_session_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Updates from a different session must be silently dropped.
        Guards against probe/throwaway session bleed-through on shared connections."""
        await session.session_update("other-session-id", _msg_chunk("should not appear"))
        await asyncio.sleep(0)
        assert len(events) == 0


    async def test_emit_extracts_parent_tool_call_id_from_field_meta(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """field_meta.claudeCode.parentToolUseId must propagate to ToolCallUpdated —
        otherwise nesting relationship is silently lost."""
        update = ToolCallStart(
            tool_call_id="tc-child",
            title="SubAgent",
            kind="other",
            status="pending",
            content=None,
            locations=None,
            raw_input=None,
            raw_output=None,
            session_update="tool_call",
            field_meta={"claudeCode": {"parentToolUseId": "tc-parent"}},
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert evt.parent_tool_call_id == "tc-parent"

    async def test_emit_parent_tool_call_id_none_when_no_field_meta(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Missing field_meta must not crash — parent_tool_call_id stays None."""
        update = ToolCallStart(
            tool_call_id="tc-1",
            title="Edit",
            kind="edit",
            status="pending",
            content=None,
            locations=None,
            raw_input=None,
            raw_output=None,
            session_update="tool_call",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert evt.parent_tool_call_id is None


class TestSessionUpdateAccumulator:
    """Tests for direct dispatch in session_update."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
        )
        s._session_id = "sess-1"
        return s

    async def test_session_update_when_tool_call_progress_emits_tool_call_updated_with_correct_status(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Progress path must use 'in_progress' default — not 'pending'."""
        update = ToolCallProgress(
            tool_call_id="tc-2",
            title="Run",
            kind="execute",
            content=None,
            locations=None,
            raw_input=None,
            raw_output=None,
            session_update="tool_call_update",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert evt.status == "in_progress"


class TestSessionModes:
    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
        )
        s._session_id = "sess-1"
        return s

    async def test_current_mode_update_emits_agent_mode_changed(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """current_mode_update must emit AgentModeChanged — otherwise mode switches are invisible."""
        update = CurrentModeUpdate(
            current_mode_id="architect",
            session_update="current_mode_update",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        assert isinstance(events[0], AgentModeChanged)
        assert events[0].mode_id == "architect"
        assert events[0].agent_id == "test"


    async def test_set_mode_transitions_through_configuring(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """set_mode must enter CONFIGURING before any RPC and return to IDLE after.
        If state never enters CONFIGURING, concurrent prompts can race the switch."""
        observed_states: list[AgentState] = []

        original_sink = session._event_sink

        async def tracking_sink(event: BrokerEvent) -> None:
            if isinstance(event, AgentStateChanged):
                observed_states.append(event.new_state)
            await original_sink(event)

        session._event_sink = tracking_sink

        class StubConn:
            async def set_session_mode(self, **kwargs: Any) -> None:
                pass

            async def set_session_model(self, **kwargs: Any) -> None:
                pass

            async def load_session(self, **kwargs: Any) -> None:
                pass

        session._conn = StubConn()
        session._mcp_servers = []
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()
        observed_states.clear()

        await session.set_mode("kiro_planner")

        assert AgentState.CONFIGURING in observed_states, (
            "set_mode must transition to CONFIGURING — without it concurrent "
            "prompts can race the mode switch"
        )
        assert session.state == AgentState.IDLE, "set_mode must return to IDLE when complete"
        configuring_idx = observed_states.index(AgentState.CONFIGURING)
        idle_idx = len(observed_states) - 1 - observed_states[::-1].index(AgentState.IDLE)
        assert configuring_idx < idle_idx

    async def test_set_mode_returns_to_idle_on_rpc_failure(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """set_mode must return to IDLE even if an RPC raises.
        If it stays in CONFIGURING the agent is permanently unusable."""

        class FailingConn:
            async def set_session_mode(self, **kwargs: Any) -> None:
                raise RuntimeError("RPC failed")

        session._conn = FailingConn()
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        with pytest.raises(RuntimeError, match="RPC failed"):
            await session.set_mode("kiro_planner")

        assert session.state == AgentState.IDLE, (
            "set_mode must restore IDLE after an RPC failure — "
            "the finally block must not skip the transition"
        )


class TestMcpRestore:
    """Tests for _suppress_history_replay and _restore_mcp_servers behaviour."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
        )
        s._session_id = "sess-1"
        return s


    async def test_restore_mcp_servers_skips_when_no_mcp_servers(self, session: ACPSession) -> None:
        """_restore_mcp_servers must be a no-op when _mcp_servers is empty.
        Prevents a spurious load_session call for agents launched without MCP servers."""
        session._mcp_servers = []
        # No conn set — would raise AttributeError if it tried to call load_session
        await session._restore_mcp_servers()
        # Reaching here without error confirms early return fired

    async def test_restore_mcp_servers_clears_flag_on_exception(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """_suppress_history_replay must be False after _restore_mcp_servers even if
        load_session raises. Guards against the flag being permanently stuck True."""
        session._mcp_servers = [McpServerStdio(name="test-mcp", command="true", args=[], env=[])]

        class FailingConn:
            async def load_session(self, **kwargs: Any) -> None:
                raise RuntimeError("load_session failed")

        session._conn = FailingConn()
        assert session._suppress_history_replay is False
        await session._restore_mcp_servers()
        assert session._suppress_history_replay is False

        # session_update must still work normally after the failed restore
        await session.session_update("sess-1", _msg_chunk("still works"))
        await asyncio.sleep(0)
        assert len(events) == 1


class TestTerminalCallbacks:
    """Tests for ACPSession terminal RPC callbacks."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd="/tmp",
            event_sink=sink,
        )
        s._session_id = "sess-1"
        return s

    async def test_create_terminal_when_called_returns_terminal_id_and_emits_event(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Terminal creation must return an ID and emit TerminalCreated — otherwise
        agent gets no terminal_id and all subsequent RPCs fail."""
        from synth_acp.models.events import TerminalCreated

        resp = await session.create_terminal(command="echo", session_id="s1", args=["hi"])
        assert resp.terminal_id
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, TerminalCreated)
        assert evt.terminal_id == resp.terminal_id
        assert evt.agent_id == "test"

    async def test_terminal_output_when_process_exited_includes_exit_status(
        self, session: ACPSession
    ) -> None:
        """terminal_output must include exit_status when process has exited — otherwise
        agent can't tell if command succeeded."""
        from unittest.mock import MagicMock

        from synth_acp.terminal.manager import ToolState

        mock_terminal = MagicMock()
        mock_terminal.tool_state = ToolState(output="hi", truncated=False, return_code=0)
        session._terminals["t-1"] = mock_terminal

        resp = await session.terminal_output(session_id="s1", terminal_id="t-1")
        assert resp.output == "hi"
        assert resp.exit_status is not None
        assert resp.exit_status.exit_code == 0

    async def test_terminal_output_when_unknown_terminal_id_raises_key_error(
        self, session: ACPSession
    ) -> None:
        """Unknown terminal_id must raise KeyError — SDK propagates as JSON-RPC error."""
        with pytest.raises(KeyError):
            await session.terminal_output(session_id="s1", terminal_id="nonexistent")

    async def test_emit_from_notification_when_terminal_content_extracts_terminal_id(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """TerminalToolCallContent must extract terminal_id into ToolCallUpdated —
        otherwise UI can't associate terminal widget with tool call block."""
        from acp.schema import TerminalToolCallContent

        terminal_item = TerminalToolCallContent(type="terminal", terminal_id="t-1")
        update = ToolCallStart(
            tool_call_id="tc-1",
            title="Run command",
            kind="execute",
            status="pending",
            content=[terminal_item],
            locations=None,
            raw_input=None,
            raw_output=None,
            session_update="tool_call",
        )
        await session.session_update("sess-1", update)
        await asyncio.sleep(0)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert evt.terminal_id == "t-1"


class TestSessionRunLifecycle:
    """Tests for run() exception handling and cleanup (Phase 2)."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        return ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
        )

    async def test_cancelled_error_not_emitted_as_broker_error(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Cancelling run() must re-raise CancelledError, not emit BrokerError.
        A BrokerError on cancel would show a spurious error notification in the TUI."""

        async def fake_spawn(*_args: Any, **_kwargs: Any) -> Any:
            """Context manager that simulates a long-running agent."""

            class _Ctx:
                async def __aenter__(self) -> tuple:
                    await asyncio.sleep(100)
                    return (None, None)  # unreachable

                async def __aexit__(self, *args: Any) -> None:
                    pass

            return _Ctx()

        with patch("synth_acp.acp.session._spawn_isolated_agent", side_effect=asyncio.CancelledError):
            task = asyncio.create_task(session.run())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        from synth_acp.models.events import BrokerError as BErr

        assert not any(isinstance(e, BErr) for e in events)

    async def test_finally_uses_force_terminal(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """The finally block must use force_terminal() so that reaching TERMINATED
        from any state (including already-TERMINATED) never raises."""
        # Simulate an error path that leaves session in TERMINATED
        await session._sm.force_terminal()
        assert session.state == AgentState.TERMINATED
        events.clear()

        # Calling force_terminal again (as the finally block does) must not raise
        await session._sm.force_terminal()
        assert session.state == AgentState.TERMINATED
        # No duplicate AgentStateChanged emitted
        assert len(events) == 0


class TestAgentModeTarget:
    """Tests for agent_mode_target passthrough in run() and run_restored()."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def mock_conn(self) -> Any:
        from unittest.mock import MagicMock

        conn = AsyncMock()
        init_resp = MagicMock()
        init_resp.agent_capabilities = None
        conn.initialize.return_value = init_resp

        session_resp = MagicMock()
        session_resp.session_id = "sess-new"
        session_resp.modes = None
        session_resp.models = None
        conn.new_session.return_value = session_resp
        conn.load_session.return_value = session_resp
        return conn

    @pytest.fixture()
    def mock_proc(self) -> Any:
        from unittest.mock import MagicMock

        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.stdin = MagicMock()
        proc.stdin.is_closing.return_value = True
        return proc

    async def test_run_meta_agent_passes_meta_and_skips_set_session_mode(
        self, events: list[BrokerEvent], mock_conn: Any, mock_proc: Any
    ) -> None:
        """When agent_mode_target='meta_agent', new_session must receive _meta kwarg
        and set_session_mode must NOT be called. Silent failure: agent launches
        without its config or gets double-applied mode."""
        from contextlib import asynccontextmanager

        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        session = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
            agent_mode="plan",
            agent_mode_target="meta_agent",
        )

        @asynccontextmanager
        async def fake_spawn(*_args: Any, **_kwargs: Any):
            # Third element is the drained stderr tail; a real one, so the handshake
            # wrapper under test behaves as it does in production.
            yield mock_conn, mock_proc, _StderrTail()

        with patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await session.run()

        mock_conn.new_session.assert_called_once()
        call_kwargs = mock_conn.new_session.call_args[1]
        assert call_kwargs["claudeCode"] == {"options": {"agent": "plan"}}
        mock_conn.set_session_mode.assert_not_called()

    async def test_run_restored_fallback_passes_meta(
        self, events: list[BrokerEvent], mock_conn: Any, mock_proc: Any
    ) -> None:
        """When run_restored() falls back to new_session and agent_mode_target='meta_agent',
        the fallback new_session must also receive _meta. Silent failure: restored agent
        that fails load launches without agent config."""
        from contextlib import asynccontextmanager

        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        session = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
            agent_mode="plan",
            agent_mode_target="meta_agent",
        )

        mock_conn.load_session.side_effect = RuntimeError("session not found")
        # new_session is the fallback
        from unittest.mock import MagicMock

        fallback_resp = MagicMock()
        fallback_resp.session_id = "sess-fallback"
        fallback_resp.modes = None
        fallback_resp.models = None
        mock_conn.new_session.return_value = fallback_resp

        @asynccontextmanager
        async def fake_spawn(*_args: Any, **_kwargs: Any):
            # Third element is the drained stderr tail; a real one, so the handshake
            # wrapper under test behaves as it does in production.
            yield mock_conn, mock_proc, _StderrTail()

        with patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await session.run_restored("old-session-id")

        mock_conn.new_session.assert_called_once()
        call_kwargs = mock_conn.new_session.call_args[1]
        assert call_kwargs["claudeCode"] == {"options": {"agent": "plan"}}
        mock_conn.set_session_mode.assert_not_called()

    async def test_run_default_target_calls_set_session_mode(
        self, events: list[BrokerEvent], mock_conn: Any, mock_proc: Any
    ) -> None:
        """When agent_mode_target is None (default/Kiro path), set_session_mode must
        be called and _meta must NOT be passed. Silent failure: Kiro mode switching
        regresses."""
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        session = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
            agent_mode="code",
            agent_mode_target=None,
        )

        # Set up modes so set_session_mode path is triggered
        mode_obj = MagicMock()
        mode_obj.id = "code"
        mode_obj.name = "Code"
        mode_obj.description = None
        modes_resp = MagicMock()
        modes_resp.available_modes = [mode_obj]
        modes_resp.current_mode_id = "chat"

        session_resp = MagicMock()
        session_resp.session_id = "sess-new"
        session_resp.modes = modes_resp
        session_resp.models = None
        mock_conn.new_session.return_value = session_resp

        # load_session for model re-read after mode switch
        loaded_resp = MagicMock()
        loaded_resp.models = None
        mock_conn.load_session.return_value = loaded_resp

        @asynccontextmanager
        async def fake_spawn(*_args: Any, **_kwargs: Any):
            # Third element is the drained stderr tail; a real one, so the handshake
            # wrapper under test behaves as it does in production.
            yield mock_conn, mock_proc, _StderrTail()

        with patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await session.run()

        # _meta should NOT be in new_session call
        call_kwargs = mock_conn.new_session.call_args[1]
        assert "_meta" not in call_kwargs
        # set_session_mode should be called
        mock_conn.set_session_mode.assert_called_once_with(mode_id="code", session_id="sess-new")


class TestConfigOptionsCapture:
    """Tests for config_options capture and synthesis in run()/run_restored()."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def mock_conn(self) -> Any:
        from unittest.mock import MagicMock

        conn = AsyncMock()
        init_resp = MagicMock()
        init_resp.agent_capabilities = None
        conn.initialize.return_value = init_resp
        return conn

    @pytest.fixture()
    def mock_proc(self) -> Any:
        from unittest.mock import MagicMock

        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.stdin = MagicMock()
        proc.stdin.is_closing.return_value = True
        return proc

    async def test_run_captures_native_config_options(
        self, events: list[BrokerEvent], mock_conn: Any, mock_proc: Any
    ) -> None:
        """Native config_options from session response must be stored and emitted.
        Silent failure: config_options silently dropped, UI shows no pickers."""
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        from synth_acp.models.events import ConfigOptionsReceived

        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        session = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )

        native_options = [
            SessionConfigOptionSelect(
                id="mode", name="Mode", category="mode", type="select",
                current_value="normal",
                options=[SessionConfigSelectOption(name="Normal", value="normal")],
            )
        ]
        session_resp = MagicMock()
        session_resp.session_id = "sess-1"
        session_resp.modes = None
        session_resp.models = None
        session_resp.config_options = native_options
        mock_conn.new_session.return_value = session_resp

        @asynccontextmanager
        async def fake_spawn(*_args: Any, **_kwargs: Any):
            # Third element is the drained stderr tail; a real one, so the handshake
            # wrapper under test behaves as it does in production.
            yield mock_conn, mock_proc, _StderrTail()

        with patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await session.run()

        assert session._has_native_config_options is True
        assert session._config_options == native_options
        config_events = [e for e in events if isinstance(e, ConfigOptionsReceived)]
        assert len(config_events) == 1
        assert config_events[0].config_options == native_options

    async def test_run_synthesizes_config_options_from_modes_and_models(
        self, events: list[BrokerEvent], mock_conn: Any, mock_proc: Any
    ) -> None:
        """When config_options is None but modes/models present, synthesis must produce
        SessionConfigOptionSelect entries. Silent failure: Kiro sessions get no pickers."""
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from synth_acp.models.events import ConfigOptionsReceived

        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        session = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )

        mode_obj = MagicMock()
        mode_obj.id = "code"
        mode_obj.name = "Code"
        mode_obj.description = None
        modes_resp = MagicMock()
        modes_resp.available_modes = [mode_obj]
        modes_resp.current_mode_id = "code"

        model_obj = MagicMock()
        model_obj.model_id = "gpt-4"
        model_obj.name = "GPT-4"
        model_obj.description = None
        models_resp = MagicMock()
        models_resp.available_models = [model_obj]
        models_resp.current_model_id = "gpt-4"

        session_resp = MagicMock()
        session_resp.session_id = "sess-1"
        session_resp.modes = modes_resp
        session_resp.models = models_resp
        session_resp.config_options = None
        mock_conn.new_session.return_value = session_resp

        @asynccontextmanager
        async def fake_spawn(*_args: Any, **_kwargs: Any):
            # Third element is the drained stderr tail; a real one, so the handshake
            # wrapper under test behaves as it does in production.
            yield mock_conn, mock_proc, _StderrTail()

        with patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await session.run()

        assert session._has_native_config_options is False
        assert len(session._config_options) == 2
        mode_opt = session._config_options[0]
        assert mode_opt.id == "mode"
        assert mode_opt.category == "mode"
        assert mode_opt.current_value == "code"
        assert len(mode_opt.options) == 1
        assert mode_opt.options[0].name == "Code"
        assert mode_opt.options[0].value == "code"
        model_opt = session._config_options[1]
        assert model_opt.id == "model"
        assert model_opt.current_value == "gpt-4"
        config_events = [e for e in events if isinstance(e, ConfigOptionsReceived)]
        assert len(config_events) == 1

    async def test_run_restored_captures_config_options(
        self, events: list[BrokerEvent], mock_conn: Any, mock_proc: Any
    ) -> None:
        """run_restored must also capture/emit config_options.
        Silent failure: restored sessions have no pickers."""
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        from synth_acp.models.events import ConfigOptionsReceived

        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        session = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )

        native_options = [
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="high",
                options=[SessionConfigSelectOption(name="High", value="high")],
            )
        ]
        session_resp = MagicMock()
        session_resp.modes = None
        session_resp.models = None
        session_resp.config_options = native_options
        mock_conn.load_session.return_value = session_resp

        @asynccontextmanager
        async def fake_spawn(*_args: Any, **_kwargs: Any):
            # Third element is the drained stderr tail; a real one, so the handshake
            # wrapper under test behaves as it does in production.
            yield mock_conn, mock_proc, _StderrTail()

        with patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await session.run_restored("saved-sess-id")

        assert session._has_native_config_options is True
        config_events = [e for e in events if isinstance(e, ConfigOptionsReceived)]
        assert len(config_events) == 1
        assert config_events[0].config_options == native_options


class TestConfigOptionUpdateNotification:
    """Tests for ConfigOptionUpdate handling in _emit_from_notification."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )
        s._session_id = "sess-1"
        return s

    async def test_config_option_update_notification_emits_changed(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """ConfigOptionUpdate must update stored options and emit ConfigOptionChanged.
        Silent failure: UI never updates when harness pushes config changes."""
        from acp.schema import (
            ConfigOptionUpdate,
            SessionConfigOptionSelect,
            SessionConfigSelectOption,
        )

        from synth_acp.models.events import ConfigOptionChanged

        # Set initial config_options
        session._config_options = [
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="low",
                options=[
                    SessionConfigSelectOption(name="Low", value="low"),
                    SessionConfigSelectOption(name="High", value="high"),
                ],
            )
        ]

        # Push a ConfigOptionUpdate with changed value
        updated_options = [
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="high",
                options=[
                    SessionConfigSelectOption(name="Low", value="low"),
                    SessionConfigSelectOption(name="High", value="high"),
                ],
            )
        ]
        update = ConfigOptionUpdate(
            config_options=updated_options,
            session_update="config_option_update",
        )
        await session._emit_from_notification(update)

        assert session._config_options == updated_options
        assert len(events) == 1
        assert isinstance(events[0], ConfigOptionChanged)
        assert events[0].config_id == "effort"
        assert events[0].value == "high"

    async def test_config_option_update_notification_only_emits_for_changed(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Unchanged options must not emit spurious events.
        Silent failure: UI re-renders all pickers on every notification."""
        from acp.schema import (
            ConfigOptionUpdate,
            SessionConfigOptionSelect,
            SessionConfigSelectOption,
        )

        from synth_acp.models.events import ConfigOptionChanged

        # Set initial config_options with two options
        session._config_options = [
            SessionConfigOptionSelect(
                id="mode", name="Mode", category="mode", type="select",
                current_value="normal",
                options=[SessionConfigSelectOption(name="Normal", value="normal")],
            ),
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="low",
                options=[SessionConfigSelectOption(name="Low", value="low")],
            ),
        ]

        # Push update where only effort changed
        updated_options = [
            SessionConfigOptionSelect(
                id="mode", name="Mode", category="mode", type="select",
                current_value="normal",
                options=[SessionConfigSelectOption(name="Normal", value="normal")],
            ),
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="high",
                options=[SessionConfigSelectOption(name="High", value="high")],
            ),
        ]
        update = ConfigOptionUpdate(
            config_options=updated_options,
            session_update="config_option_update",
        )
        await session._emit_from_notification(update)

        changed_events = [e for e in events if isinstance(e, ConfigOptionChanged)]
        assert len(changed_events) == 1
        assert changed_events[0].config_id == "effort"
        assert changed_events[0].value == "high"


class TestSetConfigOption:
    """Tests for set_config_option() method."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )
        s._session_id = "sess-1"
        return s

    async def test_set_config_option_native_path(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Native path must call conn.set_config_option and emit ConfigOptionsReceived.
        Silent failure: Claude effort changes silently fail."""
        from unittest.mock import MagicMock

        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        from synth_acp.models.events import ConfigOptionsReceived

        session._has_native_config_options = True
        session._config_options = [
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="low",
                options=[
                    SessionConfigSelectOption(name="Low", value="low"),
                    SessionConfigSelectOption(name="High", value="high"),
                ],
            )
        ]

        updated_options = [
            SessionConfigOptionSelect(
                id="effort", name="Effort", category="thought_level", type="select",
                current_value="high",
                options=[
                    SessionConfigSelectOption(name="Low", value="low"),
                    SessionConfigSelectOption(name="High", value="high"),
                ],
            )
        ]
        resp = MagicMock()
        resp.config_options = updated_options

        conn = AsyncMock()
        conn.set_config_option.return_value = resp
        session._conn = conn

        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        await session.set_config_option("effort", "high")

        conn.set_config_option.assert_called_once_with("effort", "sess-1", "high")
        assert session._config_options == updated_options
        received_events = [e for e in events if isinstance(e, ConfigOptionsReceived)]
        assert len(received_events) == 1
        assert received_events[0].config_options == updated_options
        assert session.state == AgentState.IDLE

    async def test_set_config_option_synthesized_mode(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Synthesized mode path must call set_session_mode + model preserve + MCP restore.
        Silent failure: Kiro mode switching breaks via new API."""

        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        from synth_acp.models.events import ConfigOptionChanged

        session._has_native_config_options = False
        session._current_model_id = "gpt-4"
        session._config_options = [
            SessionConfigOptionSelect(
                id="mode", name="Mode", category="mode", type="select",
                current_value="chat",
                options=[
                    SessionConfigSelectOption(name="Chat", value="chat"),
                    SessionConfigSelectOption(name="Code", value="code"),
                ],
            )
        ]

        conn = AsyncMock()
        conn.load_session.return_value = None  # _restore_mcp_servers
        session._conn = conn
        session._mcp_servers = []

        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        await session.set_config_option("mode", "code")

        conn.set_session_mode.assert_called_once_with(mode_id="code", session_id="sess-1")
        conn.set_session_model.assert_called_once_with(model_id="gpt-4", session_id="sess-1")
        assert session._current_mode_id == "code"
        changed_events = [e for e in events if isinstance(e, ConfigOptionChanged)]
        assert len(changed_events) == 1
        assert changed_events[0].config_id == "mode"
        assert changed_events[0].value == "code"
        assert session.state == AgentState.IDLE

    async def test_set_config_option_synthesized_model(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Synthesized model path must call set_session_model.
        Silent failure: Kiro model switching breaks."""

        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        from synth_acp.models.events import ConfigOptionChanged

        session._has_native_config_options = False
        session._config_options = [
            SessionConfigOptionSelect(
                id="model", name="Model", category="model", type="select",
                current_value="gpt-4",
                options=[
                    SessionConfigSelectOption(name="GPT-4", value="gpt-4"),
                    SessionConfigSelectOption(name="GPT-3.5", value="gpt-3.5"),
                ],
            )
        ]

        conn = AsyncMock()
        session._conn = conn

        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        await session.set_config_option("model", "gpt-3.5")

        conn.set_session_model.assert_called_once_with(model_id="gpt-3.5", session_id="sess-1")
        assert session._current_model_id == "gpt-3.5"
        changed_events = [e for e in events if isinstance(e, ConfigOptionChanged)]
        assert len(changed_events) == 1
        assert changed_events[0].config_id == "model"
        assert changed_events[0].value == "gpt-3.5"
        assert session.state == AgentState.IDLE

    async def test_set_config_option_not_idle_emits_error(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Precondition: must emit BrokerError when not IDLE.
        Silent failure: concurrent config changes corrupt state."""

        from synth_acp.models.events import BrokerError as BErr

        conn = AsyncMock()
        session._conn = conn

        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        await session._sm.transition(AgentState.BUSY)
        events.clear()

        await session.set_config_option("effort", "high")

        conn.set_config_option.assert_not_called()
        conn.set_session_mode.assert_not_called()
        conn.set_session_model.assert_not_called()
        error_events = [e for e in events if isinstance(e, BErr)]
        assert len(error_events) == 1
        assert error_events[0].severity == "warning"
        assert session.state == AgentState.BUSY


class TestForkWithAgent:
    """Tests for ACPSession.fork_with_agent."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd="/project",
            event_sink=sink,
            agent_mode="old-agent",
            agent_mode_target="meta_agent",
        )
        s._session_id = "sess-1"
        return s

    async def test_fork_with_agent_calls_fork_session_with_correct_meta(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """fork_session must receive claudeCode options with the agent name.
        Silent failure: wrong agent loaded if _meta is malformed."""


        conn = AsyncMock()
        conn.fork_session.return_value = AsyncMock(
            session_id="sess-2", config_options=None, modes=None, models=None
        )
        session._conn = conn
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        await session.fork_with_agent("new-agent")

        conn.fork_session.assert_called_once_with(
            cwd="/project",
            session_id="sess-1",
            mcp_servers=None,
            claudeCode={"options": {"agent": "new-agent"}},
        )

    async def test_fork_with_agent_swaps_session_id_and_agent_mode(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Session ID and agent_mode must update on success.
        Silent failure: stale session_id means prompts go to old session."""

        conn = AsyncMock()
        conn.fork_session.return_value = AsyncMock(
            session_id="sess-new", config_options=None, modes=None, models=None
        )
        session._conn = conn
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        result = await session.fork_with_agent("new-agent")

        assert result == "sess-new"
        assert session._session_id == "sess-new"
        assert session._agent_mode == "new-agent"

    async def test_fork_with_agent_returns_none_on_failure(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Must return None and emit BrokerError on exception.
        Silent failure: exception propagates and crashes caller."""

        from synth_acp.models.events import BrokerError as BErr

        conn = AsyncMock()
        conn.fork_session.side_effect = RuntimeError("connection lost")
        session._conn = conn
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        result = await session.fork_with_agent("new-agent")

        assert result is None
        assert session._session_id == "sess-1"  # unchanged
        error_events = [e for e in events if isinstance(e, BErr)]
        assert len(error_events) == 1
        assert "new-agent" in error_events[0].message

    async def test_fork_with_agent_transitions_through_configuring(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Must enter CONFIGURING before fork and return to IDLE after.
        Silent failure: concurrent prompts race the fork."""

        observed_states: list[AgentState] = []
        original_sink = session._event_sink

        async def tracking_sink(event: BrokerEvent) -> None:
            if isinstance(event, AgentStateChanged):
                observed_states.append(event.new_state)
            await original_sink(event)

        session._event_sink = tracking_sink

        conn = AsyncMock()
        conn.fork_session.return_value = AsyncMock(
            session_id="sess-2", config_options=None, modes=None, models=None
        )
        session._conn = conn
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        observed_states.clear()

        await session.fork_with_agent("new-agent")

        assert AgentState.CONFIGURING in observed_states
        assert session.state == AgentState.IDLE

    async def test_fork_with_agent_falls_back_to_new_session_on_resource_not_found(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """When fork fails with resource-not-found (empty session), must fall back to new_session.
        Silent failure: agent switch fails on empty sessions."""

        from acp.exceptions import RequestError

        conn = AsyncMock()
        conn.fork_session.side_effect = RequestError(-32002, "Resource not found")
        conn.new_session.return_value = AsyncMock(
            session_id="sess-new", config_options=None, modes=None, models=None
        )
        session._conn = conn
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        result = await session.fork_with_agent("new-agent")

        assert result == "sess-new"
        assert session._session_id == "sess-new"
        assert session._agent_mode == "new-agent"
        conn.new_session.assert_called_once_with(
            cwd="/project",
            mcp_servers=[],
            claudeCode={"options": {"agent": "new-agent"}},
        )


# ---------------------------------------------------------------------------
# Accumulator bypass: direct dispatch, chunk ordering, and the finite drain
# ---------------------------------------------------------------------------


def _bare_session(events: list[BrokerEvent]) -> ACPSession:
    """Build a session whose sink appends to ``events``."""

    async def sink(event: BrokerEvent) -> None:
        events.append(event)

    session = ACPSession(
        agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
    )
    session._session_id = "sess-1"
    return session


async def _pump_until(predicate: Callable[[], bool], limit: int = 2000) -> None:
    """Yield to the loop until ``predicate`` holds, or fail after ``limit`` yields.

    Bounded so a regression fails by assertion instead of hanging the suite.
    """
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"predicate never became true within {limit} yields")


def _fn_body(fn: Callable[..., object]) -> list[ast.stmt]:
    """Return the top-level statements of ``fn``'s definition."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node.body


def _has_await(nodes: Sequence[ast.AST]) -> bool:
    return any(isinstance(n, ast.Await) for node in nodes for n in ast.walk(node))


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


class TestSessionUpdateDispatch:
    """Behavioural coverage replacing the deleted accumulator tests."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        return _bare_session(events)

    async def test_usage_update_emits_usage_updated(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """UsageUpdate must reach the UI, or usage and cost display freeze."""
        await session.session_update(
            "sess-1",
            UsageUpdate(
                size=100,
                used=50,
                cost=Cost(amount=0.01, currency="USD"),
                session_update="usage_update",
            ),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, UsageUpdated)
        assert (evt.size, evt.used, evt.cost_amount, evt.cost_currency) == (
            100,
            50,
            0.01,
            "USD",
        )

    async def test_usage_update_suppressed_during_replay_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Replayed usage must not re-emit, or restore double-counts cost."""
        session._suppress_history_replay = True
        await session.session_update(
            "sess-1", UsageUpdate(size=1, used=1, cost=None, session_update="usage_update")
        )
        assert events == []

    async def test_suppressed_update_emits_nothing_and_registers_no_task(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """The relocated guard must short-circuit BEFORE task creation.

        Without it, load_session history replay emits into the live UI and the
        broker journals it, duplicating the journal on every restore.
        """
        session._suppress_history_replay = True
        await session.session_update("sess-1", _msg_chunk("replayed"))
        await asyncio.sleep(0)
        assert events == []
        assert session._pending_emissions == set()

    async def test_dispatch_failure_is_logged_and_does_not_wedge_the_session(
        self,
        session: ACPSession,
        events: list[BrokerEvent],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A dispatch failure must be logged and swallowed.

        Propagating would kill the SDK notification runner; swallowing silently
        would make the loss undiagnosable.
        """

        def boom(coro: Any, **kwargs: Any) -> Any:
            coro.close()  # the task is never created, so close it here
            raise ValueError("no loop for you")

        with (
            caplog.at_level(logging.WARNING, logger="synth_acp.acp.session"),
            patch("synth_acp.acp.session.asyncio.create_task", side_effect=boom),
        ):
            await session.session_update("sess-1", _msg_chunk("doomed"))

        assert events == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Failed to dispatch" in warnings[0].getMessage()

        # The session still works afterwards.
        await session.session_update("sess-1", _msg_chunk("recovery"))
        await asyncio.sleep(0)
        assert len(events) == 1
        assert isinstance(events[0], MessageChunkReceived)


class TestChunkOrdering:
    async def test_chunk_order_is_preserved_across_200_notifications(self) -> None:
        """Chunk-to-chunk order must survive concurrent notification tasks.

        The SDK dispatches one task per notification, so a reordering here
        scrambles the rendered text AND the journal, permanently.
        """
        arrivals: list[str] = []
        emitted: list[str] = []
        gate = asyncio.Event()

        async def sink(event: BrokerEvent) -> None:
            assert isinstance(event, MessageChunkReceived)
            arrivals.append(event.chunk)
            await gate.wait()
            emitted.append(event.chunk)

        session = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )
        session._session_id = "sess-1"

        expected = [f"c{i}" for i in range(200)]
        tasks = [
            asyncio.create_task(session.session_update("sess-1", _msg_chunk(text)))
            for text in expected
        ]

        # Pump rather than gather: gathering would deadlock an implementation
        # that awaited emission inline, hiding the very defect under test.
        await _pump_until(lambda: len(arrivals) == 200)
        assert len(session._pending_emissions) == 200

        gate.set()
        await asyncio.gather(*tasks)
        await session._drain_pending_emissions()

        assert arrivals == expected
        assert emitted == expected


class TestSessionUpdateStructure:
    """AST guards for properties no behavioural test can pin down."""

    def _session_update_parts(self) -> tuple[list[ast.stmt], int, int, int]:
        body = _fn_body(ACPSession.session_update)
        usage = next(
            i
            for i, stmt in enumerate(body)
            if isinstance(stmt, ast.If) and "UsageUpdate" in _names(stmt.test)
        )
        suppress = next(
            i
            for i, stmt in enumerate(body)
            if isinstance(stmt, ast.If) and "_suppress_history_replay" in _names(stmt.test)
        )
        create = next(
            i
            for i, stmt in enumerate(body)
            if "create_task" in {n.attr for n in ast.walk(stmt) if isinstance(n, ast.Attribute)}
        )
        return body, usage, suppress, create

    def test_suppress_guard_lies_between_usage_branch_and_task_creation(self) -> None:
        """A guard after create_task still emits; one before the UsageUpdate
        branch silently kills live usage updates."""
        body, usage, suppress, create = self._session_update_parts()
        assert usage < suppress <= create

    def test_no_await_between_usage_branch_and_task_creation(self) -> None:
        """An await here reorders chunks under load with nothing else failing."""
        body, usage, suppress, create = self._session_update_parts()
        assert not _has_await(body[suppress : create + 1])
        # Non-vacuity: the excluded UsageUpdate branch really does await.
        assert _has_await([body[usage]])

    def test_session_update_constructs_no_session_notification(self) -> None:
        """SessionNotification existed only to feed the accumulator."""
        body, *_ = self._session_update_parts()
        assert not any("SessionNotification" in _names(stmt) for stmt in body)

    def test_src_never_references_session_accumulator(self) -> None:
        """The O(history) deep copy must not creep back in.

        AST-scoped to imports and code references, so the docstrings that
        explain WHY the accumulator is not used are not false positives.
        """
        src = Path(__file__).resolve().parents[2] / "src"
        offenders: list[str] = []
        for path in src.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                imported = (
                    isinstance(node, (ast.Import, ast.ImportFrom))
                    and any(a.name == "SessionAccumulator" for a in node.names)
                )
                referenced = (
                    isinstance(node, ast.Name) and node.id == "SessionAccumulator"
                ) or (isinstance(node, ast.Attribute) and node.attr == "SessionAccumulator")
                if imported or referenced:
                    offenders.append(str(path.relative_to(src)))
                    break
        assert offenders == []

    def test_drain_is_bounded_by_drain_passes(self) -> None:
        """An unbounded drain leaves the agent BUSY forever and wedges queued
        prompts."""
        assert DRAIN_PASSES == 4
        body = _fn_body(ACPSession._drain_pending_emissions)
        assert not any(
            isinstance(node, ast.While) for stmt in body for node in ast.walk(stmt)
        )


class TestDrainPendingEmissions:
    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        return _bare_session(events)

    async def test_drain_yields_before_deciding_quiescence(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """A runner created but not started registers nothing yet.

        Breaking on an empty set at the top of a pass would emit TurnComplete
        first, misattributing that chunk to the next turn.
        """

        runners: list[asyncio.Task[None]] = []

        async def fake_prompt(**kwargs: Any) -> Any:
            # Created, not started: no await between here and the drain.
            assert session._pending_emissions == set()
            runners.append(
                asyncio.create_task(session.session_update("sess-1", _msg_chunk("late")))
            )
            return MagicMock(stop_reason="end_turn")

        session._conn = MagicMock(prompt=fake_prompt)
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)
        events.clear()

        await session.prompt("go")

        streamed = [
            e for e in events if isinstance(e, (MessageChunkReceived, TurnComplete))
        ]
        assert [type(e).__name__ for e in streamed] == [
            "MessageChunkReceived",
            "TurnComplete",
        ]
        assert isinstance(streamed[0], MessageChunkReceived)
        assert streamed[0].chunk == "late"

    async def test_prompt_returns_under_continuous_emission_registration(self) -> None:
        """Arrival that never stops must not wedge the agent BUSY.

        An unbounded drain never returns, so prompt() never reaches its finally
        block, IDLE never fires, and queued prompts stall forever.
        """
        events: list[BrokerEvent] = []
        stop = asyncio.Event()
        registrations: list[int] = []
        runners: list[asyncio.Task[None]] = []
        session: ACPSession

        async def sink(event: BrokerEvent) -> None:
            events.append(event)
            if isinstance(event, MessageChunkReceived) and not stop.is_set():
                registrations.append(1)
                runners.append(
                    asyncio.create_task(session.session_update("sess-1", _msg_chunk("more")))
                )

        session = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )
        session._session_id = "sess-1"

        async def fake_prompt(**kwargs: Any) -> Any:
            runners.append(
                asyncio.create_task(session.session_update("sess-1", _msg_chunk("seed")))
            )
            return MagicMock(stop_reason="end_turn")

        session._conn = MagicMock(prompt=fake_prompt)
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        # The bounded PUMP is the falsifier, not the wait_for: the drain swallows
        # CancelledError by contract, so a wait_for timeout cancels the drain,
        # lets prompt() finish anyway, and reports SUCCESS against an unbounded
        # implementation. Verified by mutation. The contracted 5 s wait_for is
        # kept as the outer bound so the test can never hang the suite.
        turn = asyncio.create_task(session.prompt("go"))
        await _pump_until(turn.done, limit=500)
        await asyncio.wait_for(turn, timeout=5.0)

        stop.set()
        assert len(registrations) > DRAIN_PASSES
        assert any(isinstance(e, TurnComplete) for e in events)
        assert session.state == AgentState.IDLE

        for task in list(session._pending_emissions):
            task.cancel()
        await asyncio.gather(*session._pending_emissions, return_exceptions=True)

    async def test_drain_returns_cleanly_when_cancelled(self) -> None:
        """Cancellation during the drain must not wedge shutdown."""
        blocked = asyncio.Event()

        async def sink(event: BrokerEvent) -> None:
            await blocked.wait()

        session = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink
        )
        session._session_id = "sess-1"
        await session.session_update("sess-1", _msg_chunk("stuck"))

        drain = asyncio.create_task(session._drain_pending_emissions())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        drain.cancel()

        assert await drain is None
        assert not drain.cancelled()

        blocked.set()
        for task in list(session._pending_emissions):
            task.cancel()
        await asyncio.gather(*session._pending_emissions, return_exceptions=True)


# ---------------------------------------------------------------------------
# Mechanism gate: session_update cost, measured at the ACP layer
#
# perf_replay cannot see this: it drives BrokerEvents into a headless App and
# never constructs an ACPSession.  No App is built here at all.
# ---------------------------------------------------------------------------


class _BenchResult(NamedTuple):
    wall_ms: float
    worst_loop_block_ms: float
    worst_single_call_ms: float
    lag_samples: int


class _LegacyAccumulatorSession(ACPSession):
    """The pre-phase session_update: feed the accumulator on every notification.

    Reproduces the cost being removed so the gate below is a before/after on the
    same host in the same run, rather than a comparison against a constant
    measured elsewhere.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._accumulator = SessionAccumulator()
        self._accumulator.subscribe(self._legacy_on_snapshot)

    def _legacy_on_snapshot(self, snapshot: Any, notification: Any) -> None:
        task = asyncio.create_task(self._emit_from_notification(notification.update))
        self._pending_emissions.add(task)
        task.add_done_callback(self._pending_emissions.discard)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if session_id != self._session_id:
            return
        self._accumulator.apply(SessionNotification(session_id=session_id, update=update))


async def _drive_session_updates(
    session: ACPSession, n: int, *, legacy: bool
) -> _BenchResult:
    """Drive ``n`` notifications through ``session`` and measure the loop cost.

    Args:
        session: The session under measurement.
        n: Number of notifications to drive.
        legacy: True when ``session`` is the accumulator-backed baseline.
            Asserted against the session type so a mislabelled measurement
            cannot be compared against the wrong path.

    Returns:
        Wall time, worst sampled loop block, worst single call, and the sampler's
        sample count.
    """
    assert isinstance(session, _LegacyAccumulatorSession) is legacy

    # 0.5 ms so the bypass window — a handful of milliseconds — still yields
    # several samples; the assertions are ceilings, so extra sampling noise is
    # harmless while an unsampled window would be a false pass.
    sampler = LoopLagSampler(interval=0.0005)
    sampler_task = asyncio.create_task(sampler.run())

    # Hand off to the sampler until it is genuinely sampling: resetting before
    # run() has armed its first sleep would leave the window unmeasured and make
    # the non-vacuity assertion flaky.
    for _ in range(200):
        if sampler.stats().samples > 0:
            break
        await asyncio.sleep(0.005)
    sampler.reset()

    update = _msg_chunk("x" * 64)
    worst_call = 0.0
    started = time.perf_counter()
    for i in range(n):
        call_start = time.perf_counter()
        await session.session_update("sess-1", update)
        worst_call = max(worst_call, (time.perf_counter() - call_start) * 1000)
        if i % 50 == 0:
            await asyncio.sleep(0)
    wall_ms = (time.perf_counter() - started) * 1000

    stats = sampler.stats()
    sampler_task.cancel()
    await asyncio.gather(sampler_task, return_exceptions=True)
    await session._drain_pending_emissions()
    for task in list(session._pending_emissions):
        task.cancel()
    await asyncio.gather(*session._pending_emissions, return_exceptions=True)
    # Reclaim the run's garbage here rather than leaving a large collection to
    # land inside a later test's wall-clock assertion.
    gc.collect()

    return _BenchResult(wall_ms, stats.max_ms, worst_call, stats.samples)


def _bench_session(cls: type[ACPSession]) -> ACPSession:
    async def sink(event: BrokerEvent) -> None:
        return None

    session = cls(agent_id="bench", binary="echo", args=[], cwd=".", event_sink=sink)
    session._session_id = "sess-1"
    return session


class TestSessionUpdateMechanismGate:
    """The only tests in the suite that measure this phase's mechanism.

    Silent failure they guard: the accumulator — or any equivalent O(history)
    deep copy — is reintroduced, every correctness test still passes, and the
    user's 166 ms/chunk freeze comes back.
    """

    async def test_bypass_is_at_least_10x_cheaper_than_the_accumulator_path(
        self,
    ) -> None:
        fixed = await _drive_session_updates(_bench_session(ACPSession), 2000, legacy=False)
        legacy = await _drive_session_updates(
            _bench_session(_LegacyAccumulatorSession), 2000, legacy=True
        )

        assert fixed.lag_samples > 0  # non-vacuity: the window really was sampled
        assert 0.0 < fixed.worst_loop_block_ms < 200.0
        assert legacy.wall_ms / fixed.wall_ms >= 10.0

    async def test_worst_single_notification_block_at_20000_updates(self) -> None:
        """Per-notification cost must stay O(1); anything proportional to
        accumulated history only shows up on the user's real 21k-chunk session."""
        result = await _drive_session_updates(
            _bench_session(ACPSession), 20000, legacy=False
        )

        assert result.lag_samples > 0
        assert result.worst_single_call_ms < 20.0


class TestSteer:
    """Tests for ACPSession.steer."""

    def _session(
        self,
        *,
        steer_protocol: str | None = "kiro",
        events: list[BrokerEvent] | None = None,
    ) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            if events is not None:
                events.append(event)

        s = ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
            steer_protocol=steer_protocol,
        )
        s._session_id = "sess-1"
        return s

    async def test_steer_sends_unprefixed_method_and_logs_acceptance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An accepted Kiro response stays True and records its safe outcome.

        The body sentinel proves request diagnostics cannot silently copy the
        inter-agent message into logs.
        """
        calls: list[tuple[str, dict[str, Any]]] = []
        sentinel = "PRIVATE_STEER_BODY_ACCEPTED"

        async def ext_method(method: str, params: dict[str, Any]) -> dict[str, Any]:
            calls.append((method, params))
            return {"queued": True}

        session = self._session()
        session._conn = MagicMock(ext_method=ext_method)

        with caplog.at_level(logging.DEBUG, logger="synth_acp.acp.session"):
            assert await session.steer(sentinel) is True

        assert calls == [
            ("session/steer", {"sessionId": "sess-1", "message": sentinel})
        ]
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Steer acceptance started agent=test session=sess-1 timeout_s=10.0"
            in message
            for message in messages
        )
        accepted = next(message for message in messages if "outcome=accepted" in message)
        assert "agent=test" in accepted
        assert "session=sess-1" in accepted
        assert "elapsed_ms=" in accepted
        assert "queued=True" in accepted
        assert all(sentinel not in message for message in messages)

    async def test_steer_timeout_reports_warning_without_message_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing acceptance response must release its caller with a safe warning.

        This catches the unbounded wait and the separate privacy failure where
        diagnostics expose an inter-agent message body.
        """
        events: list[BrokerEvent] = []
        sentinel = "PRIVATE_STEER_BODY_TIMEOUT"

        async def ext_method(method: str, params: dict[str, Any]) -> dict[str, Any]:
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable")

        session = self._session(events=events)
        session._conn = MagicMock(ext_method=ext_method)

        with (
            patch("synth_acp.acp.session._STEER_ACCEPTANCE_TIMEOUT", 0.01),
            caplog.at_level(logging.DEBUG, logger="synth_acp.acp.session"),
        ):
            result = await asyncio.wait_for(session.steer(sentinel), timeout=1.0)

        assert result is False
        warnings = [event for event in events if isinstance(event, BrokerError)]
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert "existing queue fallback" in warnings[0].message
        assert "may arrive twice" in warnings[0].message
        messages = [record.getMessage() for record in caplog.records]
        timed_out = next(message for message in messages if "outcome=timed_out" in message)
        assert "agent=test" in timed_out
        assert "session=sess-1" in timed_out
        assert "elapsed_ms=" in timed_out
        assert "timeout_s=0.01" in timed_out
        assert all(sentinel not in message for message in messages)
        assert sentinel not in warnings[0].message

    async def test_steer_timeout_warning_failure_still_returns_false(self) -> None:
        """A broken warning sink must not escape and retain the broker's agent lock."""

        async def failing_sink(event: BrokerEvent) -> None:
            raise RuntimeError("event sink unavailable")

        async def ext_method(method: str, params: dict[str, Any]) -> dict[str, Any]:
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable")

        session = self._session()
        session._event_sink = failing_sink
        session._conn = MagicMock(ext_method=ext_method)

        with patch("synth_acp.acp.session._STEER_ACCEPTANCE_TIMEOUT", 0.01):
            assert await asyncio.wait_for(session.steer("private"), timeout=1.0) is False

    @pytest.mark.parametrize(
        ("exc", "exception_type"),
        [
            (RequestError(code=-32601, message="Method not found"), "RequestError"),
            (ConnectionError("broken pipe"), "ConnectionError"),
            (OSError("closed"), "OSError"),
        ],
    )
    async def test_steer_transport_failure_logs_safe_diagnostics_and_returns_false(
        self,
        exc: Exception,
        exception_type: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Transport failures remain non-raising and classify without logging text."""
        sentinel = "PRIVATE_STEER_BODY_TRANSPORT"

        async def ext_method(method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise exc

        session = self._session()
        session._conn = MagicMock(ext_method=ext_method)

        with caplog.at_level(logging.DEBUG, logger="synth_acp.acp.session"):
            assert await session.steer(sentinel) is False

        assert session.state == AgentState.UNSTARTED
        messages = [record.getMessage() for record in caplog.records]
        failed = next(message for message in messages if "outcome=transport_failed" in message)
        assert "agent=test" in failed
        assert "session=sess-1" in failed
        assert "elapsed_ms=" in failed
        assert f"exception_type={exception_type}" in failed
        assert all(sentinel not in message for message in messages)

    async def test_steer_returns_false_without_steer_protocol(self) -> None:
        """A harness that declares no protocol must not be called at all: Claude's
        payload shape differs and would fail with -32602."""
        called = False

        async def ext_method(method: str, params: dict[str, Any]) -> dict[str, Any]:
            nonlocal called
            called = True
            return {}

        session = self._session(steer_protocol=None)
        session._conn = MagicMock(ext_method=ext_method)

        assert await session.steer("hello") is False
        assert called is False


class TestRename:
    def test_rename_updates_both_id_copies(self) -> None:
        """The id is stored twice -- on the session, which stamps every outgoing event,
        and on the state machine, which names the agent in transition errors. Updating
        only one leaves the session attributing its output to two different agents.
        """

        async def sink(event: object) -> None:
            return None

        s = ACPSession(agent_id="worker", binary="echo", args=[], cwd=".", event_sink=sink)

        s.rename("worker.h0000dead")

        assert s.agent_id == "worker.h0000dead"
        assert s._sm._agent_id == "worker.h0000dead"


class TestKiroMetadataAdaptor:
    """kiro.dev/metadata is the ONLY context-usage source Kiro provides.

    Kiro 2.18.1 sends no standard ACP usage_update at all, so a defect here means
    Kiro agents report no usage and the handoff nudge never fires — with no error,
    because the SDK drops an unhandled ext-notification silently.
    """

    @pytest.fixture
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture
    def session(self, events: list[BrokerEvent]) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(agent_id="kiro-1", binary="echo", args=[], cwd=".", event_sink=sink)
        s._session_id = "sess-1"
        return s

    async def test_percentage_becomes_a_usage_ratio(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """A mid-turn percentage maps onto UsageUpdated as used/size."""
        await session.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": "sess-1", "contextUsagePercentage": 18.79960060119629},
        )
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, UsageUpdated)
        assert (event.agent_id, event.used, event.size) == ("kiro-1", 19, 100)
        assert event.cost_amount is None
        assert event.cost_currency is None

    async def test_turn_end_payload_carries_metering_cost(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """The turn-final payload adds credits, which must reach the cost fields."""
        await session.ext_notification(
            "kiro.dev/metadata",
            {
                "sessionId": "sess-1",
                "contextUsagePercentage": 62.0,
                "meteringUsage": [
                    {"value": 0.7706246474958541, "unit": "credit", "unitPlural": "credits"}
                ],
                "turnDurationMs": 4372,
            },
        )
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, UsageUpdated)
        assert event.used == 62
        assert event.cost_amount == 0.7706246474958541
        assert event.cost_currency == "credits"

    async def test_history_replay_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """A replayed usage figure is stale and would fire the nudge on restore."""
        session._suppress_history_replay = True
        await session.ext_notification(
            "kiro.dev/metadata", {"sessionId": "sess-1", "contextUsagePercentage": 99.0}
        )
        assert events == []

    @pytest.mark.parametrize(
        "params",
        [
            {"sessionId": "sess-1"},  # no percentage at all
            {"sessionId": "sess-1", "contextUsagePercentage": "62"},  # string
            {"sessionId": "sess-1", "contextUsagePercentage": True},  # bool is not a reading
        ],
    )
    async def test_unusable_percentage_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent], params: dict[str, Any]
    ) -> None:
        """A malformed payload must not emit a bogus usage figure."""
        await session.ext_notification("kiro.dev/metadata", params)
        assert events == []

    async def test_other_ext_notifications_ignored(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Kiro sends several other ext-notifications on every turn.

        Each payload carries a VALID percentage, so only the method guard can
        suppress it. Without that the test passes even with the guard deleted,
        because the later guards reject a payload that has no percentage.
        """
        await session.ext_notification(
            "kiro.dev/commands/available",
            {"sessionId": "sess-1", "contextUsagePercentage": 62.0, "commands": []},
        )
        await session.ext_notification(
            "kiro.dev/subagent/list_update",
            {"sessionId": "sess-1", "contextUsagePercentage": 62.0, "subagents": []},
        )
        assert events == []

    async def test_other_session_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Usage for a session this agent has left is not this agent's usage.

        session_update drops a non-matching session_id silently; this path must
        too, or a delayed notification stores a figure for the wrong conversation
        and can consume the agent's single handoff nudge.
        """
        await session.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": "some-other-session", "contextUsagePercentage": 72.0},
        )
        assert events == []

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    async def test_non_finite_percentage_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent], bad: float
    ) -> None:
        """NaN and the infinities survive json.loads and defeat the clamp.

        Every comparison against NaN is False, so min(100.0, nan) returns 100.0
        rather than propagating NaN. Without an explicit finiteness check a NaN
        reading therefore became a false 100% event and nudged an agent whose real
        usage is unknown. Measured: nan -> 100.0, inf -> 100.0, -inf -> 0.0.
        """
        await session.ext_notification(
            "kiro.dev/metadata", {"sessionId": "sess-1", "contextUsagePercentage": bad}
        )
        assert events == []

    async def test_shutting_down_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """A notification racing shutdown must not emit, as in session_update."""
        session._shutting_down = True
        await session.ext_notification(
            "kiro.dev/metadata", {"sessionId": "sess-1", "contextUsagePercentage": 80.0}
        )
        assert events == []

    async def test_percentage_clamped_to_scale(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """used must never exceed size, or the UI bar overflows its width."""
        await session.ext_notification(
            "kiro.dev/metadata", {"sessionId": "sess-1", "contextUsagePercentage": 140.0}
        )
        assert isinstance(events[0], UsageUpdated)
        assert events[0].used == 100

    async def test_no_session_yet_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Usage reported before a session exists is not a measurement.

        A bare equality check against _session_id treats a payload that omits
        sessionId as matching while _session_id is still None, so usage was
        emitted for an agent with no ACP session.
        """
        session._session_id = None
        await session.ext_notification(
            "kiro.dev/metadata", {"contextUsagePercentage": 55.0}
        )
        await session.ext_notification(
            "kiro.dev/metadata", {"sessionId": None, "contextUsagePercentage": 56.0}
        )
        assert events == []


_STALL_AGENT = """\
import sys
sys.stderr.write({stderr!r})
sys.stderr.flush()
# Consume stdin so the client's request is accepted, then never answer it.
for _line in sys.stdin:
    pass
"""


def _stall_agent(tmp_path: Path, stderr: str = "") -> list[str]:
    """Write a fake agent that reads stdin, optionally talks, and never replies.

    Returned as a script FILE rather than ``python -c <source>``, because the failure
    message under test echoes the spawned command. With the source inline, a test
    asserting that the child's stderr reached the message passes on the command echo
    alone -- which is exactly how the first version of these tests passed with the drain
    disabled.
    """
    script = tmp_path / "stall_agent.py"
    script.write_text(_STALL_AGENT.format(stderr=stderr))
    return [str(script)]


class TestHandshakeTimeout:
    """A child that never answers must be reported, not waited on forever.

    The silent failure: the ACP SDK has no internal timeout, so before this an
    unanswered handshake left the session in INITIALIZING for the lifetime of the synth
    process with nothing emitted. That was reachable and observed -- a package-manager
    wrapper facing an unreachable registry wrote nothing to either stream for 30s.

    Driven against a real stalling subprocess rather than mocks, because the contract
    under test is the interaction with a process that accepts input and never replies.
    """

    def _session(self, events: list, args: list[str]) -> ACPSession:
        async def sink(e: object) -> None:
            events.append(e)

        return ACPSession(
            agent_id="stalled",
            binary=sys.executable,
            args=args,
            cwd=".",
            event_sink=sink,
        )

    async def test_initialize_that_never_answers_is_reported_and_terminated(
        self, tmp_path: Path
    ) -> None:
        events: list = []
        session = self._session(events, _stall_agent(tmp_path))

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 1.0):
            await asyncio.wait_for(session.run(), timeout=20)

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "initialize" in errors[0].message
        assert "did not respond within 1s" in errors[0].message
        # The whole point: not left INITIALIZING.
        assert session.state == AgentState.TERMINATED

    async def test_empty_stderr_is_reported_as_itself_diagnostic(self, tmp_path: Path) -> None:
        """A silent stall is what a stalled process manager in the spawn path looks like
        from here, so the message must distinguish it from a crashing agent."""
        events: list = []
        session = self._session(events, _stall_agent(tmp_path))

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 1.0):
            await asyncio.wait_for(session.run(), timeout=20)

        message = next(e for e in events if isinstance(e, BrokerError)).message
        assert "wrote nothing to stderr" in message

    async def test_child_stderr_reaches_the_error(self, tmp_path: Path) -> None:
        """Before this, nothing in synth ever read the child's stderr, so the one piece of
        evidence about a failed launch was discarded.

        The marker deliberately does not appear in the spawned command: see _stall_agent.
        """
        events: list = []
        session = self._session(
            events, _stall_agent(tmp_path, "MODULE_RESOLUTION_FAILED_MARKER\n")
        )

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 1.0):
            await asyncio.wait_for(session.run(), timeout=20)

        message = next(e for e in events if isinstance(e, BrokerError)).message
        assert "stderr tail: MODULE_RESOLUTION_FAILED_MARKER" in message

    async def test_a_timeout_is_reported_once_and_not_retried(self, tmp_path: Path) -> None:
        """A retry would burn the whole budget again before saying anything, so the
        terminal decision is asserted rather than left to the reader."""
        events: list = []
        session = self._session(events, _stall_agent(tmp_path))

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 1.0):
            started = time.monotonic()
            await asyncio.wait_for(session.run(), timeout=20)
            elapsed = time.monotonic() - started

        assert len([e for e in events if isinstance(e, BrokerError)]) == 1
        # One budget, not two: a retry or fallback would put this above 2s.
        assert elapsed < 2.0, f"took {elapsed:.1f}s, which is more than one timeout budget"


class TestStderrTail:
    """The read loop that keeps the pipe drained and the last bytes available."""

    async def test_retains_the_tail_and_discards_the_head(self) -> None:
        tail = _StderrTail(limit=16)
        reader = asyncio.StreamReader()
        reader.feed_data(b"0123456789" * 4)
        reader.feed_eof()
        await tail.drain(reader)
        assert tail.text() == "4567890123456789"

    async def test_a_chatty_agents_exit_stays_visible(self, tmp_path: Path) -> None:
        """Regression for the unread pipe, asserted at the consequence rather than the
        cause.

        asyncio drains a subprocess pipe on its own until roughly twice the `limit` passed
        to create_subprocess_exec, so a small volume of unread stderr is harmless and a
        test at that scale proves nothing. Past the threshold the transport pauses and
        `proc.wait()` stops returning -- which is what ACPSession.run() spends the whole
        session awaiting. MEASURED at synth's 8 MiB limit, child writes stderr then exits:
        50 KB returns in 0.02s, 20 MB never returns, 20 MB with a drain returns in 0.05s.

        So the silent failure is not a stalled launch. It is an agent that exits and is
        never seen to exit: the session stays IDLE instead of reaching TERMINATED, with no
        error and no dead-agent tile.
        """
        script = tmp_path / "chatty_agent.py"
        script.write_text(
            "import sys\n"
            "sys.stderr.write('x' * 20_000_000)\n"
            "sys.stderr.flush()\n"
            "raise SystemExit(0)\n"
        )

        async def sink(_e: object) -> None:
            return None

        session = ACPSession(
            agent_id="chatty",
            binary=sys.executable,
            args=[str(script)],
            cwd=".",
            event_sink=sink,
        )

        async with _spawn_isolated_agent(
            session, sys.executable, str(script), cwd="."
        ) as (_conn, proc, tail):
            returncode = await asyncio.wait_for(proc.wait(), timeout=15)

        assert returncode == 0
        # The tail is bounded, not the whole 20 MB.
        assert len(tail.text()) <= 8192


class TestLaterHandshakeTimeouts:
    """initialize is not the only request that can hang, and a stall there is the easy case.

    The code critic proved these were uncovered: making session/new and session/load
    unbounded left every relevant test passing, and deleting the restore path's
    `except HandshakeTimeoutError: raise` left six passing. A stalling child cannot reach
    these -- it never gets past initialize -- so the agent is faked at the connection,
    which is what lets a request be answered and the NEXT one hang.
    """

    def _never_answers(self) -> Any:
        """An awaitable that is accepted and never completes."""
        return asyncio.get_running_loop().create_future()

    def _session(self, events: list, conn: Any) -> tuple[ACPSession, Any]:
        async def sink(e: object) -> None:
            events.append(e)

        session = ACPSession(
            agent_id="agent-1", binary="fake", args=[], cwd=".", event_sink=sink
        )
        proc = MagicMock()
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        proc.stdin = MagicMock(is_closing=MagicMock(return_value=True))

        @asynccontextmanager
        async def fake_spawn(*_a: Any, **_kw: Any) -> Any:
            tail = _StderrTail()
            yield conn, proc, tail

        return session, fake_spawn

    async def test_session_new_that_never_answers_is_reported(self) -> None:
        events: list = []
        conn = MagicMock()
        conn.initialize = AsyncMock(return_value=MagicMock(agent_capabilities=None))
        conn.new_session = MagicMock(side_effect=lambda **_kw: self._never_answers())
        session, fake_spawn = self._session(events, conn)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3), \
             patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await asyncio.wait_for(session.run(), timeout=10)

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/new' did not respond" in errors[0].message
        assert session.state == AgentState.TERMINATED

    async def test_restore_load_timeout_does_not_reach_the_new_session_fallback(self) -> None:
        """The branch its predecessor test only claimed to cover.

        An ordinary load_session failure falls back to new_session; a TIMEOUT must not,
        because the child is not answering at all and the fallback would spend a second
        full budget before anything was reported.
        """
        events: list = []
        conn = MagicMock()
        conn.initialize = AsyncMock(return_value=MagicMock(agent_capabilities=None))
        conn.load_session = MagicMock(side_effect=lambda **_kw: self._never_answers())
        conn.new_session = AsyncMock(return_value=MagicMock(session_id="should-not-happen"))
        session, fake_spawn = self._session(events, conn)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3), \
             patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await asyncio.wait_for(session.run_restored("acp-1"), timeout=10)

        conn.new_session.assert_not_called()
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/load' did not respond" in errors[0].message
        assert session.session_id == "acp-1"

    async def test_an_ordinary_load_failure_still_reaches_the_fallback(self) -> None:
        """The other half of that branch: narrowing the except must not disable the
        fallback that non-timeout failures depend on."""
        events: list = []
        conn = MagicMock()
        conn.initialize = AsyncMock(return_value=MagicMock(agent_capabilities=None))
        conn.load_session = AsyncMock(side_effect=RuntimeError("session not found"))
        fresh = MagicMock(session_id="fresh-session", modes=None, models=None)
        fresh.config_options = None
        conn.new_session = AsyncMock(return_value=fresh)
        session, fake_spawn = self._session(events, conn)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 5.0), \
             patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await asyncio.wait_for(session.run_restored("acp-1"), timeout=10)

        conn.new_session.assert_called_once()
        assert session.session_id == "fresh-session"

    async def test_a_timeout_never_suppresses_the_replay_guard_permanently(self) -> None:
        """A flag left set would silence the agent for the rest of the session, which is
        worse than the hang it replaced."""
        events: list = []
        conn = MagicMock()
        conn.initialize = AsyncMock(return_value=MagicMock(agent_capabilities=None))
        conn.load_session = MagicMock(side_effect=lambda **_kw: self._never_answers())
        session, fake_spawn = self._session(events, conn)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3), \
             patch("synth_acp.acp.session._spawn_isolated_agent", fake_spawn):
            await asyncio.wait_for(session.run_restored("acp-1"), timeout=10)

        assert session._suppress_history_replay is False


class TestHandshakeTimeoutMessage:
    """What the operator reads. Each case sends them somewhere different."""

    def _raise(self, stderr: str) -> str:
        return str(HandshakeTimeoutError("initialize", 60.0, "agent --acp", stderr))

    def test_whitespace_only_stderr_is_reported_with_its_content(self) -> None:
        """Silent failure guarded: branching on strip() rather than emptiness. A child that
        wrote a newline HAS produced output, and calling that silence sends an operator
        looking for a process that never started.

        Asserts the tail is actually CARRIED, not merely that the message avoids the wrong
        word: a branch that announced whitespace-only stderr and then dropped it would pass
        a not-in assertion.
        """
        message = self._raise("\n\t ")
        assert "wrote nothing to stderr" not in message
        assert repr("\n\t ") in message

    def test_empty_stderr_says_so_and_claims_nothing_about_stdout(self) -> None:
        message = self._raise("")
        assert "wrote nothing to stderr" in message
        # stdout is consumed by the JSON-RPC reader and never observed here.
        assert "any output" not in message

    def test_the_command_and_step_are_both_named(self) -> None:
        message = self._raise("boom")
        assert "initialize" in message
        assert "agent --acp" in message
        assert "stderr tail: boom" in message


class TestTimeoutIsTerminalEverywhere:
    """A bounded stall must not be traded for an unbounded one on the next prompt.

    asyncio.wait_for cancels only synth's local future; it sends no ACP cancellation, so
    after a timeout the agent is still processing the request. set_mode's own docstring
    records that a prompt concurrent with a live load_session hangs Kiro indefinitely, so
    any handler that reports a timeout and returns the session to IDLE has made the
    failure worse rather than better. Every timeout therefore ends the session.
    """

    def _mocks(self) -> tuple[Any, Any]:
        conn = MagicMock()
        conn.initialize = AsyncMock(return_value=MagicMock(agent_capabilities=None))
        proc = MagicMock()
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        proc.stdin = MagicMock(is_closing=MagicMock(return_value=True))
        return conn, proc

    def _spawn(self, conn: Any, proc: Any) -> Any:
        @asynccontextmanager
        async def fake_spawn(*_a: Any, **_kw: Any) -> Any:
            yield conn, proc, _StderrTail()

        return fake_spawn

    def _never(self) -> Any:
        return asyncio.get_running_loop().create_future()

    def _session(self, events: list, **kw: Any) -> ACPSession:
        async def sink(e: object) -> None:
            events.append(e)

        return ACPSession(
            agent_id="agent-1", binary="fake", args=[], cwd=".", event_sink=sink, **kw
        )

    async def test_pre_idle_set_mode_that_never_answers_is_reported(self) -> None:
        """Silent failure guarded: this request runs before the agent is usable, so leaving
        it unbounded keeps the permanent-INITIALIZING failure this work exists to remove
        reachable by a second route."""
        events: list = []
        conn, proc = self._mocks()
        conn.new_session = AsyncMock(
            return_value=MagicMock(
                session_id="s1",
                modes=MagicMock(
                    available_modes=[SimpleNamespace(id="plan", name="Plan", description=None)],
                    current_mode_id="default",
                ),
                models=None,
                config_options=None,
            )
        )
        conn.set_session_mode = MagicMock(side_effect=lambda **_kw: self._never())
        session = self._session(events, agent_mode="plan")

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3), \
             patch("synth_acp.acp.session._spawn_isolated_agent", self._spawn(conn, proc)):
            await asyncio.wait_for(session.run(), timeout=10)

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/set_mode' did not respond" in errors[0].message
        assert session.state == AgentState.TERMINATED

    async def test_launch_time_model_reread_timeout_ends_the_session(self) -> None:
        """The re-read is best-effort, but its timeout is not: an earlier version reported
        it and carried on to IDLE, which permits the prompt/load overlap that hangs Kiro."""
        events: list = []
        conn, proc = self._mocks()
        conn.new_session = AsyncMock(
            return_value=MagicMock(
                session_id="s1",
                modes=MagicMock(
                    available_modes=[SimpleNamespace(id="plan", name="Plan", description=None)],
                    current_mode_id="default",
                ),
                models=None,
                config_options=None,
            )
        )
        conn.set_session_mode = AsyncMock()
        conn.load_session = MagicMock(side_effect=lambda **_kw: self._never())
        session = self._session(events, agent_mode="plan")

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3), \
             patch("synth_acp.acp.session._spawn_isolated_agent", self._spawn(conn, proc)):
            await asyncio.wait_for(session.run(), timeout=10)

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/load' did not respond" in errors[0].message
        assert session.state == AgentState.TERMINATED

    async def test_mcp_restore_timeout_terminates_rather_than_returning_to_idle(self) -> None:
        """_restore_mcp_servers is post-launch and outside the spawn context, so nothing
        else kills the process. Returning to IDLE here is the specific defect: the agent
        would present as usable while still processing a load."""
        events: list = []
        session = self._session(events)
        conn = MagicMock()
        conn.set_session_mode = AsyncMock()
        conn.set_session_model = AsyncMock()
        conn.load_session = MagicMock(side_effect=lambda **_kw: self._never())
        session._conn = conn
        session._session_id = "acp-1"
        session._mcp_servers = [
            McpServerStdio(name="synth-mcp", command="synth-mcp", args=[], env=[])
        ]
        session._proc = MagicMock(returncode=None, pid=-1)
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_mode("plan"), timeout=10)

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/load' did not respond" in errors[0].message
        assert "was terminated" in errors[0].message
        # NOT IDLE: the caller's finally is guarded on CONFIGURING, so TERMINATED sticks.
        assert session.state == AgentState.TERMINATED

    async def test_fork_session_that_never_answers_is_reported(self) -> None:
        events: list = []
        session = self._session(events)
        conn = MagicMock()
        conn.fork_session = MagicMock(side_effect=lambda **_kw: self._never())
        session._conn = conn
        session._session_id = "acp-1"
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            result = await asyncio.wait_for(session.fork_with_agent("some-agent"), timeout=10)

        assert result is None
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/fork' did not respond" in errors[0].message
        assert session.state == AgentState.TERMINATED

    async def test_fork_fallback_new_session_that_never_answers_is_reported(self) -> None:
        """The -32002 fallback is a second session-creating request and needs its own bound."""
        events: list = []
        session = self._session(events)
        conn = MagicMock()
        conn.fork_session = AsyncMock(side_effect=RequestError(code=-32002, message="empty"))
        conn.new_session = MagicMock(side_effect=lambda **_kw: self._never())
        session._conn = conn
        session._session_id = "acp-1"
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            result = await asyncio.wait_for(session.fork_with_agent("some-agent"), timeout=10)

        assert result is None
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "'session/new' did not respond" in errors[0].message
        assert session.state == AgentState.TERMINATED


class TestTerminatedRestoreEmitsNoSuccess:
    """A terminated agent must not be announced as having switched.

    _restore_mcp_servers terminates on a timeout but returns normally, so both callers'
    success paths run unless guarded. The silent failure is a WRONG EVENT rather than a
    missing one: subscribers, the UI and the journal all recorded AgentModeChanged -- and
    through set_config_option, ConfigOptionChanged too -- for a session the same call had
    just killed. Asserting the error and the terminal state is not enough, because the
    false success passes that.
    """

    def _stalled_session(self, events: list) -> ACPSession:
        async def sink(e: object) -> None:
            events.append(e)

        session = ACPSession(
            agent_id="agent-1", binary="fake", args=[], cwd=".", event_sink=sink
        )
        conn = MagicMock()
        conn.set_session_mode = AsyncMock()
        conn.set_session_model = AsyncMock()
        conn.load_session = MagicMock(
            side_effect=lambda **_kw: asyncio.get_running_loop().create_future()
        )
        session._conn = conn
        session._session_id = "acp-1"
        session._mcp_servers = [
            McpServerStdio(name="synth-mcp", command="synth-mcp", args=[], env=[])
        ]
        session._proc = MagicMock(returncode=None, pid=-1)
        session._current_mode_id = "default"
        return session

    async def test_set_mode_emits_no_mode_change_after_terminating(self) -> None:
        events: list = []
        session = self._stalled_session(events)
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_mode("plan"), timeout=10)

        assert session.state == AgentState.TERMINATED
        assert [e for e in events if isinstance(e, AgentModeChanged)] == []
        # The recorded mode must not move either, or a later read reports the switch.
        assert session.current_mode_id == "default"

    async def test_set_config_option_emits_no_change_after_terminating(self) -> None:
        events: list = []
        session = self._stalled_session(events)
        # Forces the SYNTHESIZED path. The native branch routes through
        # conn.set_config_option and never reaches _restore_mcp_servers, so it is
        # not the path this guard protects.
        session._has_native_config_options = False
        session._config_options = [
            SessionConfigOptionSelect(
                id="mode",
                name="Mode",
                type="select",
                current_value="default",
                options=[
                    SessionConfigSelectOption(value="default", name="Manual"),
                    SessionConfigSelectOption(value="plan", name="Plan"),
                ],
            )
        ]
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_config_option("mode", "plan"), timeout=10)

        assert session.state == AgentState.TERMINATED
        assert [e for e in events if isinstance(e, AgentModeChanged)] == []
        assert [e for e in events if isinstance(e, ConfigOptionChanged)] == []
        assert session._config_options[0].current_value == "default"

    async def test_a_healthy_switch_still_emits_its_mode_change(self) -> None:
        """The guard must not suppress the success it is meant to gate."""
        events: list = []
        session = self._stalled_session(events)
        session._conn.load_session = AsyncMock(return_value=MagicMock(models=None))
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

        await asyncio.wait_for(session.set_mode("plan"), timeout=10)

        assert session.state == AgentState.IDLE
        assert [e.mode_id for e in events if isinstance(e, AgentModeChanged)] == ["plan"]
        assert session.current_mode_id == "plan"


class TestConfigRequestsAreBounded:
    """Post-launch config requests hold CONFIGURING, a state the user cannot leave.

    Unbounded, an agent that stops answering set_session_mode stranded the session there
    with no error, exactly as an unanswered session/load did before it was bounded. Each
    test drives one request to its timeout through the PUBLIC method a user's click
    reaches, and asserts the session left CONFIGURING rather than that a helper was called.
    """

    def _session(self, events: list) -> ACPSession:
        async def sink(e: object) -> None:
            events.append(e)

        session = ACPSession(
            agent_id="agent-1", binary="fake", args=[], cwd=".", event_sink=sink
        )
        conn = MagicMock()
        conn.set_session_mode = AsyncMock()
        conn.set_session_model = AsyncMock()
        conn.load_session = AsyncMock(return_value=MagicMock(models=None))
        session._conn = conn
        session._session_id = "acp-1"
        session._proc = MagicMock(returncode=None, pid=-1)
        session._current_mode_id = "default"
        session._current_model_id = "old-model"
        return session

    def _never(self) -> Any:
        return asyncio.get_running_loop().create_future()

    async def _idle(self, session: ACPSession) -> None:
        await session._sm.transition(AgentState.INITIALIZING)
        await session._sm.transition(AgentState.IDLE)

    def _assert_terminated_with(self, events: list, session: ACPSession, step: str) -> None:
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1, [e.message for e in errors]
        assert f"'{step}' did not respond" in errors[0].message
        assert "was terminated" in errors[0].message
        assert session.state == AgentState.TERMINATED

    async def test_set_mode_stalling_on_set_session_mode(self) -> None:
        events: list = []
        session = self._session(events)
        session._conn.set_session_mode = MagicMock(side_effect=lambda **_kw: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_mode("plan"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_mode")
        assert [e for e in events if isinstance(e, AgentModeChanged)] == []
        assert session.current_mode_id == "default"

    async def test_set_mode_stalling_on_the_model_preserving_call(self) -> None:
        """set_mode preserves the current model with a second request. A timeout there is
        the one a reader is most likely to miss, since the mode call already succeeded."""
        events: list = []
        session = self._session(events)
        session._conn.set_session_model = MagicMock(side_effect=lambda **_kw: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_mode("plan"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_model")
        assert [e for e in events if isinstance(e, AgentModeChanged)] == []

    async def test_set_model_stalling(self) -> None:
        events: list = []
        session = self._session(events)
        session._conn.set_session_model = MagicMock(side_effect=lambda **_kw: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_model("new-model"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_model")
        assert [e for e in events if isinstance(e, AgentModelChanged)] == []
        assert session.current_model_id == "old-model"

    async def test_native_set_config_option_stalling(self) -> None:
        """The native branch has its own except Exception that reports a warning and
        returns. Without an explicit re-raise it swallows the timeout, leaving a live
        request on an agent the user is told is fine."""
        events: list = []
        session = self._session(events)
        session._has_native_config_options = True
        session._conn.set_config_option = MagicMock(side_effect=lambda *_a: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_config_option("model", "m2"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_config_option")
        assert [e for e in events if isinstance(e, ConfigOptionsReceived)] == []

    async def test_synthesized_config_model_branch_stalling(self) -> None:
        events: list = []
        session = self._session(events)
        session._has_native_config_options = False
        session._conn.set_session_model = MagicMock(side_effect=lambda **_kw: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_config_option("model", "m2"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_model")
        assert [e for e in events if isinstance(e, ConfigOptionChanged)] == []

    async def test_synthesized_config_mode_branch_stalling_on_set_mode(self) -> None:
        """The synthesized mode branch of set_config_option makes TWO bounded requests, and
        both were untested: unwrapping either one passed the whole suite while production
        went back to stranding CONFIGURING forever. This covers the first."""
        events: list = []
        session = self._session(events)
        session._has_native_config_options = False
        session._conn.set_session_mode = MagicMock(side_effect=lambda **_kw: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_config_option("mode", "plan"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_mode")
        assert [e for e in events if isinstance(e, AgentModeChanged)] == []
        assert [e for e in events if isinstance(e, ConfigOptionChanged)] == []
        assert session.current_mode_id == "default"

    async def test_synthesized_config_mode_branch_stalling_on_the_model_preserving_call(
        self,
    ) -> None:
        """The second request in that branch, which preserves the current model. Hardest to
        notice because the mode request already succeeded."""
        events: list = []
        session = self._session(events)
        session._has_native_config_options = False
        session._conn.set_session_model = MagicMock(side_effect=lambda **_kw: self._never())
        await self._idle(session)

        with patch("synth_acp.acp.session._HANDSHAKE_TIMEOUT", 0.3):
            await asyncio.wait_for(session.set_config_option("mode", "plan"), timeout=10)

        self._assert_terminated_with(events, session, "session/set_model")
        assert [e for e in events if isinstance(e, AgentModeChanged)] == []
        assert [e for e in events if isinstance(e, ConfigOptionChanged)] == []
        assert session.current_mode_id == "default"

    async def test_a_healthy_mode_switch_is_unaffected(self) -> None:
        """The bound must not change the path that answers."""
        events: list = []
        session = self._session(events)
        await self._idle(session)

        await asyncio.wait_for(session.set_mode("plan"), timeout=10)

        assert session.state == AgentState.IDLE
        assert [e.mode_id for e in events if isinstance(e, AgentModeChanged)] == ["plan"]
        assert [e for e in events if isinstance(e, BrokerError)] == []

    async def test_a_healthy_model_switch_is_unaffected(self) -> None:
        events: list = []
        session = self._session(events)
        await self._idle(session)

        await asyncio.wait_for(session.set_model("new-model"), timeout=10)

        assert session.state == AgentState.IDLE
        assert [e.model_id for e in events if isinstance(e, AgentModelChanged)] == ["new-model"]
        assert [e for e in events if isinstance(e, BrokerError)] == []


class TestTurnRequestsStayUnbounded:
    """prompt, cancel and ext_method must NEVER be wrapped.

    A turn legitimately runs for many minutes -- one tool-heavy turn was measured at 95
    seconds -- so bounding one at the handshake timeout would abort healthy work. The
    silent failure is the opposite direction from everything else here: a later reader
    "completes the set" and turns start dying at 60 seconds.
    """

    def test_no_turn_or_control_request_is_wrapped_in_the_handshake_bound(self) -> None:
        source = Path("src/synth_acp/acp/session.py").read_text()
        tree = ast.parse(source)

        wrapped: set[str] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "_handshake":
                continue
            for arg in node.args:
                for inner in ast.walk(arg):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                        wrapped.add(inner.func.attr)

        assert "prompt" not in wrapped
        assert "cancel" not in wrapped
        assert "ext_method" not in wrapped
        # Guards against the assertion passing because _handshake vanished entirely.
        assert {"initialize", "new_session"} <= wrapped
