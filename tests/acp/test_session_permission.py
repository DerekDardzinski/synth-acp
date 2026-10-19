"""Tests for ACPSession permission flow and TurnComplete emission."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    PermissionOption,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
)

from synth_acp.acp.extensions import AcpExtension
from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState
from synth_acp.models.events import BrokerEvent, PermissionRequested
from synth_acp.models.extensions import PermissionHints


class TestSessionPermissionFlow:
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

    @pytest.fixture()
    def tool_call(self) -> ToolCallUpdate:
        return ToolCallUpdate(tool_call_id="tc-1", title="Write file", kind="edit")

    @pytest.fixture()
    def options(self) -> list[PermissionOption]:
        return [PermissionOption(kind="allow_once", name="Allow", option_id="opt-allow")]

    async def test_request_permission_when_resolved_returns_allowed(
        self,
        session: ACPSession,
        events: list[BrokerEvent],
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> None:
        # Get session into BUSY state (bypass state machine — no real subprocess)
        session._sm._state = AgentState.BUSY

        async def resolve_later() -> None:
            while "tc-1" not in session._permission_futures:
                await asyncio.sleep(0.01)
            session.resolve_permission("tc-1", "opt-allow")

        task = asyncio.create_task(resolve_later())
        response = await session.request_permission(options, "sess-1", tool_call)
        await task

        assert isinstance(response.outcome, AllowedOutcome)
        assert response.outcome.option_id == "opt-allow"
        assert response.outcome.outcome == "selected"
        assert session.state == AgentState.BUSY

    async def test_request_permission_when_cancelled_returns_denied(
        self,
        session: ACPSession,
        events: list[BrokerEvent],
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> None:
        session._sm._state = AgentState.BUSY

        async def cancel_later() -> None:
            while "tc-1" not in session._permission_futures:
                await asyncio.sleep(0.01)
            session._permission_futures["tc-1"].cancel()

        task = asyncio.create_task(cancel_later())
        response = await session.request_permission(options, "sess-1", tool_call)
        await task

        assert isinstance(response.outcome, DeniedOutcome)
        assert response.outcome.outcome == "cancelled"
        assert "tc-1" not in session._permission_futures

    async def test_request_permission_when_called_transitions_and_emits_event(
        self,
        session: ACPSession,
        events: list[BrokerEvent],
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> None:
        session._sm._state = AgentState.BUSY

        async def resolve_later() -> None:
            while "tc-1" not in session._permission_futures:
                await asyncio.sleep(0.01)
            # Check state before resolving
            assert session.state == AgentState.AWAITING_PERMISSION
            session.resolve_permission("tc-1", "opt-allow")

        task = asyncio.create_task(resolve_later())
        await session.request_permission(options, "sess-1", tool_call)
        await task

        perm_events = [e for e in events if isinstance(e, PermissionRequested)]
        assert len(perm_events) == 1
        pe = perm_events[0]
        assert pe.request_id == "tc-1"
        assert pe.title == "Write file"
        assert pe.kind == "edit"
        assert len(pe.options) == 1
        assert pe.options[0].option_id == "opt-allow"


class _IdentityExtension(AcpExtension):
    """Returns a fixed identity and records the meta it was handed."""

    name = "test-identity"

    def __init__(self) -> None:
        self.seen_meta: list[dict[str, Any]] = []

    def permission_hints(self, tool_call: Any, meta: dict[str, Any]) -> PermissionHints:
        self.seen_meta.append(meta)
        return PermissionHints(tool_identity="probe-echo/echo")


class TestPermissionToolIdentity:
    """Kind recovery by toolCallId and tool identity from the extension."""

    @pytest.fixture()
    def events(self) -> list[BrokerEvent]:
        return []

    @pytest.fixture()
    def ext(self) -> _IdentityExtension:
        return _IdentityExtension()

    @pytest.fixture()
    def session(self, events: list[BrokerEvent], ext: _IdentityExtension) -> ACPSession:
        async def sink(event: BrokerEvent) -> None:
            events.append(event)

        s = ACPSession(
            agent_id="test", binary="echo", args=[], cwd=".", event_sink=sink, extension=ext
        )
        s._session_id = "sess-1"
        s._sm._state = AgentState.BUSY
        return s

    @staticmethod
    async def _request(
        session: ACPSession, events: list[BrokerEvent], tool_call: ToolCallUpdate, **meta: Any
    ) -> PermissionRequested:
        options = [PermissionOption(kind="allow_once", name="Allow", option_id="opt-allow")]

        async def resolve_later() -> None:
            while tool_call.tool_call_id not in session._permission_futures:
                await asyncio.sleep(0.01)
            session.resolve_permission(tool_call.tool_call_id, "opt-allow")

        task = asyncio.create_task(resolve_later())
        await session.request_permission(options, "sess-1", tool_call, **meta)
        await task
        return [e for e in events if isinstance(e, PermissionRequested)][-1]

    async def test_kind_recovered_from_earlier_tool_call(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Silent failure: every kind-less request keys its rule on 'other'."""
        await session.session_update(
            "sess-1",
            ToolCallStart(
                tool_call_id="tc-1", title="Run", kind="execute", session_update="tool_call"
            ),
        )
        pe = await self._request(session, events, ToolCallUpdate(tool_call_id="tc-1", title="ls"))
        assert pe.kind == "execute"

    async def test_kind_falls_back_to_other_and_request_kind_wins(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Silent failure: a recorded kind overrides the kind the request carries."""
        await session.session_update(
            "sess-1",
            ToolCallStart(
                tool_call_id="tc-2", title="Run", kind="execute", session_update="tool_call"
            ),
        )
        unknown = await self._request(session, events, ToolCallUpdate(tool_call_id="tc-9"))
        own = await self._request(session, events, ToolCallUpdate(tool_call_id="tc-2", kind="edit"))
        assert (unknown.kind, own.kind) == ("other", "edit")

    async def test_recorded_kind_dropped_on_terminal_status(
        self, session: ACPSession, events: list[BrokerEvent]
    ) -> None:
        """Silent failure: the kind map grows for the agent's whole lifetime."""
        for tc_id, status in (("tc-3", "completed"), ("tc-4", "failed")):
            await session.session_update(
                "sess-1",
                ToolCallStart(
                    tool_call_id=tc_id, title="Run", kind="execute", session_update="tool_call"
                ),
            )
            await session.session_update(
                "sess-1",
                ToolCallProgress(
                    tool_call_id=tc_id, status=status, session_update="tool_call_update"
                ),
            )
        pe = await self._request(session, events, ToolCallUpdate(tool_call_id="tc-3"))
        assert pe.kind == "other"
        assert session._tool_kinds == {}

    async def test_tool_identity_from_extension_with_request_meta(
        self, session: ACPSession, events: list[BrokerEvent], ext: _IdentityExtension
    ) -> None:
        """Silent failure: identity never reaches the broker, so auto-approve falls
        back to matching the free-text title."""
        pe = await self._request(
            session, events, ToolCallUpdate(tool_call_id="tc-5"), kiro={"toolId": "x"}
        )
        assert pe.tool_identity == "probe-echo/echo"
        assert ext.seen_meta == [{"kiro": {"toolId": "x"}}]
