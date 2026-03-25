"""Tests for ACPSession state machine enforcement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState, InvalidTransitionError
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerEvent,
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

        return ACPSession(
            agent_id="test",
            binary="echo",
            args=[],
            cwd=".",
            event_sink=sink,
        )

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
