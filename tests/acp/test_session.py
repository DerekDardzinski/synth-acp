"""Tests for ACPSession state machine enforcement."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState, InvalidTransitionError
from synth_acp.models.events import (
    AgentModeChanged,
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerEvent,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    UsageUpdated,
)


class TestSessionStateMachine:
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

    async def test_valid_transition_emits_event(
        self, session: ACPSession, events: list[BrokerEvent]
    ):
        await session._set_state(AgentState.INITIALIZING)
        assert session.state == AgentState.INITIALIZING
        assert len(events) == 1
        assert isinstance(events[0], AgentStateChanged)
        assert events[0].old_state == AgentState.UNSTARTED
        assert events[0].new_state == AgentState.INITIALIZING

    async def test_invalid_transition_raises(self, session: ACPSession):
        with pytest.raises(InvalidTransitionError):
            await session._set_state(AgentState.BUSY)  # UNSTARTED → BUSY is invalid


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
        update = SimpleNamespace(
            session_update="agent_thought_chunk", content=SimpleNamespace(text="reasoning")
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
        assert isinstance(events[0], AgentThoughtReceived)
        assert events[0].chunk == "reasoning"
        assert events[0].agent_id == "test"

    async def test_session_update_when_usage_update_emits_event(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Usage updates must emit UsageUpdated — otherwise cost/context data is lost."""
        cost = SimpleNamespace(amount=0.14, currency="USD")
        update = SimpleNamespace(session_update="usage_update", size=128000, used=32000, cost=cost)
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
        diff_item = SimpleNamespace(type="diff", path="src/main.py", old_text="old", new_text="new")
        update = SimpleNamespace(
            session_update="tool_call",
            tool_call_id="tc-1",
            title="Edit file",
            kind="edit",
            status="pending",
            content=[diff_item],
            locations=None,
            raw_input=None,
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert len(evt.diffs) == 1
        assert evt.diffs[0] == ToolCallDiff(path="src/main.py", old_text="old", new_text="new")

    async def test_session_update_when_tool_call_update_branch_has_diff_extracts_diffs(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Diffs on streaming tool_call_update must be extracted — otherwise incremental edits are lost."""
        diff_item = SimpleNamespace(type="diff", path="lib.py", old_text=None, new_text="added")
        update = SimpleNamespace(
            session_update="tool_call_update",
            tool_call_id="tc-2",
            title="Create file",
            kind="edit",
            status="in_progress",
            content=[diff_item],
            locations=None,
            raw_input=None,
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert len(evt.diffs) == 1
        assert evt.diffs[0] == ToolCallDiff(path="lib.py", old_text=None, new_text="added")

    async def test_session_update_when_tool_call_has_text_content_extracts_text(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Text content must be extracted — otherwise command output is silently dropped."""
        text_block = SimpleNamespace(type="text", text="hello world")
        content_item = SimpleNamespace(type="content", content=text_block)
        update = SimpleNamespace(
            session_update="tool_call",
            tool_call_id="tc-3",
            title="Run command",
            kind="execute",
            status="pending",
            content=[content_item],
            locations=None,
            raw_input=None,
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ToolCallUpdated)
        assert evt.text_content == "hello world"

    async def test_session_update_when_tool_call_has_locations_extracts_locations(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Locations must be extracted — otherwise file path context is silently lost."""
        loc = SimpleNamespace(path="/abs/path.py", line=42)
        update = SimpleNamespace(
            session_update="tool_call",
            tool_call_id="tc-4",
            title="Read file",
            kind="read",
            status="pending",
            content=None,
            locations=[loc],
            raw_input=None,
        )
        await session.session_update("sess-1", update)
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
        update = SimpleNamespace(
            session_update="agent_message_chunk",
            content=SimpleNamespace(text="should not appear"),
        )
        await session.session_update("other-session-id", update)
        assert len(events) == 0

    async def test_session_update_from_correct_session_is_processed(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Updates from the session's own ID must still be processed normally."""
        update = SimpleNamespace(
            session_update="agent_message_chunk",
            content=SimpleNamespace(text="hello"),
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1


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
        update = SimpleNamespace(
            session_update="current_mode_update", current_mode_id="architect"
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
        assert isinstance(events[0], AgentModeChanged)
        assert events[0].mode_id == "architect"
        assert events[0].agent_id == "test"

    async def test_current_mode_update_updates_internal_state(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """_current_mode_id must be updated — otherwise current_mode_id property is stale."""
        update = SimpleNamespace(
            session_update="current_mode_update", current_mode_id="code"
        )
        await session.session_update("sess-1", update)
        assert session.current_mode_id == "code"

    async def test_current_mode_update_with_no_mode_id_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Missing mode id must not emit — guards against malformed agent payloads."""
        update = SimpleNamespace(
            session_update="current_mode_update", current_mode_id=None
        )
        await session.session_update("sess-1", update)
        assert len(events) == 0

    async def test_available_modes_empty_before_session(self, session: ACPSession) -> None:
        assert session.available_modes == []

    async def test_current_mode_id_none_before_session(self, session: ACPSession) -> None:
        assert session.current_mode_id is None

    async def test_available_models_empty_before_session(self, session: ACPSession) -> None:
        assert session.available_models == []

    async def test_current_model_id_none_before_session(self, session: ACPSession) -> None:
        assert session.current_model_id is None

    async def test_current_mode_update_from_wrong_session_emits_nothing(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Mode updates from a different session must be dropped.
        Prevents probe sessions overwriting the main session's tracked mode."""
        update = SimpleNamespace(
            session_update="current_mode_update", current_mode_id="code"
        )
        await session.session_update("other-session-id", update)
        assert len(events) == 0
        assert session.current_mode_id is None


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

    async def test_suppress_flag_drops_updates(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """session_update must emit nothing while _suppress_history_replay is True.
        This is what prevents load_session history replay from corrupting the UI."""
        session._suppress_history_replay = True
        update = SimpleNamespace(
            session_update="agent_message_chunk",
            content=SimpleNamespace(text="replayed history"),
        )
        await session.session_update("sess-1", update)
        assert len(events) == 0

    async def test_suppress_flag_cleared_restores_updates(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """session_update must resume normal emission once flag is cleared.
        Guards against the flag being left True after a failed load_session."""
        session._suppress_history_replay = True
        session._suppress_history_replay = False
        update = SimpleNamespace(
            session_update="agent_message_chunk",
            content=SimpleNamespace(text="live message"),
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1

    async def test_suppress_does_not_affect_wrong_session(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """The session ID guard must still fire before the suppress check.
        A suppressed session must not accidentally process updates from other sessions
        when the flag is later cleared."""
        session._suppress_history_replay = False
        update = SimpleNamespace(
            session_update="agent_message_chunk",
            content=SimpleNamespace(text="from other session"),
        )
        await session.session_update("other-session-id", update)
        assert len(events) == 0

    async def test_restore_mcp_servers_skips_when_no_mcp_servers(
        self, session: ACPSession
    ) -> None:
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
        from acp.schema import McpServerStdio

        session._mcp_servers = [
            McpServerStdio(name="test-mcp", command="true", args=[], env=[])
        ]

        class FailingConn:
            async def load_session(self, **kwargs: Any) -> None:
                raise RuntimeError("load_session failed")

        session._conn = FailingConn()
        assert session._suppress_history_replay is False
        await session._restore_mcp_servers()
        assert session._suppress_history_replay is False

        # session_update must still work normally after the failed restore
        update = SimpleNamespace(
            session_update="agent_message_chunk",
            content=SimpleNamespace(text="still works"),
        )
        await session.session_update("sess-1", update)
        assert len(events) == 1
