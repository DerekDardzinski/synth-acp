"""Tests for ACPSession state machine enforcement."""

from __future__ import annotations

import pytest

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState, InvalidTransitionError
from synth_acp.models.events import AgentStateChanged, BrokerEvent


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
