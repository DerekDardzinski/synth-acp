"""Tests for ACPSession permission flow and TurnComplete emission."""

from __future__ import annotations

import asyncio

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome, PermissionOption, ToolCallUpdate

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState
from synth_acp.models.events import BrokerEvent, PermissionRequested


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
        session.state = AgentState.BUSY

        async def resolve_later() -> None:
            # Wait for the future to be created
            while session._permission_future is None:
                await asyncio.sleep(0.01)
            session.resolve_permission("opt-allow")

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
        session.state = AgentState.BUSY

        async def cancel_later() -> None:
            while session._permission_future is None:
                await asyncio.sleep(0.01)
            session._permission_future.cancel()

        task = asyncio.create_task(cancel_later())
        response = await session.request_permission(options, "sess-1", tool_call)
        await task

        assert isinstance(response.outcome, DeniedOutcome)
        assert response.outcome.outcome == "cancelled"
        assert session._permission_future is None

    async def test_request_permission_when_called_transitions_and_emits_event(
        self,
        session: ACPSession,
        events: list[BrokerEvent],
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> None:
        session.state = AgentState.BUSY

        async def resolve_later() -> None:
            while session._permission_future is None:
                await asyncio.sleep(0.01)
            # Check state before resolving
            assert session.state == AgentState.AWAITING_PERMISSION
            session.resolve_permission("opt-allow")

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
