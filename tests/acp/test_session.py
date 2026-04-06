"""Tests for ACPSession state machine enforcement."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    Cost,
    CurrentModeUpdate,
    FileEditToolCallContent,
    McpServerStdio,
    TextContentBlock,
    ToolCallLocation as AcpToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState
from synth_acp.models.events import (
    AgentModeChanged,
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerEvent,
    MessageChunkReceived,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
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


class TestSessionUpdateAccumulator:
    """Tests for SessionAccumulator integration in session_update."""

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

    async def test_session_update_when_usage_update_bypasses_accumulator(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """UsageUpdate must emit UsageUpdated without calling accumulator.apply()."""
        update = UsageUpdate(
            size=100,
            used=50,
            cost=Cost(amount=0.01, currency="USD"),
            session_update="usage_update",
        )
        with patch.object(
            session._accumulator, "apply", side_effect=AssertionError("should not be called")
        ) as mock_apply:
            await session.session_update("sess-1", update)
        assert len(events) == 1
        assert isinstance(events[0], UsageUpdated)
        mock_apply.assert_not_called()

    async def test_session_update_when_suppress_active_accumulator_fed_but_no_events(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Suppress flag must prevent events but still feed accumulator state."""
        session._suppress_history_replay = True
        with patch.object(
            session._accumulator, "apply", wraps=session._accumulator.apply
        ) as mock_apply:
            await session.session_update("sess-1", _msg_chunk("replayed"))
            await asyncio.sleep(0)
        mock_apply.assert_called_once()
        assert len(events) == 0

    async def test_session_update_when_apply_raises_logs_and_continues(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Malformed update must not crash the session — log and skip."""
        with patch.object(session._accumulator, "apply", side_effect=ValueError("bad update")):
            await session.session_update("sess-1", _msg_chunk("test"))
            await asyncio.sleep(0)
        assert len(events) == 0
        # Session still works after the failure
        with patch.object(session._accumulator, "apply", wraps=session._accumulator.apply):
            await session.session_update("sess-1", _msg_chunk("recovery"))
            await asyncio.sleep(0)
        assert len(events) == 1
        assert isinstance(events[0], MessageChunkReceived)


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

    async def test_current_mode_update_with_no_mode_id_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """SDK requires current_mode_id to be a non-None str, so this scenario
        is unreachable through normal ACP dispatch. Verify the defensive guard
        in _emit_from_notification by calling it directly with a patched update."""
        from unittest.mock import MagicMock

        from acp.schema import SessionNotification

        fake_update = MagicMock(spec=CurrentModeUpdate)
        fake_update.current_mode_id = None
        fake_notification = MagicMock(spec=SessionNotification)
        fake_notification.update = fake_update
        await session._emit_from_notification(fake_notification)
        assert len(events) == 0

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

    async def test_suppress_flag_cleared_restores_updates(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """session_update must resume normal emission once flag is cleared.
        Guards against the flag being left True after a failed load_session."""
        session._suppress_history_replay = True
        session._suppress_history_replay = False
        await session.session_update("sess-1", _msg_chunk("live message"))
        await asyncio.sleep(0)
        assert len(events) == 1

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

    async def test_unsubscribe_called_on_session_exit(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """The accumulator subscription must be cleaned up when run() exits.
        Without this, the accumulator holds a reference preventing GC."""
        from unittest.mock import MagicMock

        mock_unsub = MagicMock()
        session._unsubscribe = mock_unsub

        # Trigger run() with a spawn that immediately fails
        with patch(
            "synth_acp.acp.session._spawn_isolated_agent",
            side_effect=RuntimeError("spawn failed"),
        ):
            await session.run()

        mock_unsub.assert_called_once()
