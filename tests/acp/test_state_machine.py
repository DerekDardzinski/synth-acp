"""Tests for AgentStateMachine."""

from __future__ import annotations

import pytest

from synth_acp.acp.state_machine import AgentStateMachine
from synth_acp.models.agent import AgentState, InvalidTransitionError


class TestTransitions:
    async def test_valid_transition_updates_state(self):
        calls: list[tuple[AgentState, AgentState]] = []

        async def cb(old: AgentState, new: AgentState) -> None:
            calls.append((old, new))

        sm = AgentStateMachine("a1", cb)
        await sm.transition(AgentState.INITIALIZING)
        assert sm.state == AgentState.INITIALIZING
        assert calls == [(AgentState.UNSTARTED, AgentState.INITIALIZING)]

    async def test_invalid_transition_raises(self):
        sm = AgentStateMachine("a1", _noop_cb)
        with pytest.raises(InvalidTransitionError, match="a1"):
            await sm.transition(AgentState.IDLE)

    async def test_callback_receives_old_and_new(self):
        captured: list[tuple[AgentState, AgentState]] = []

        async def cb(old: AgentState, new: AgentState) -> None:
            captured.append((old, new))

        sm = AgentStateMachine("a1", cb)
        await sm.transition(AgentState.INITIALIZING)
        assert captured == [(AgentState.UNSTARTED, AgentState.INITIALIZING)]


class TestForceTerminal:
    async def test_force_terminal_from_any_state(self):
        for start in AgentState:
            sm = AgentStateMachine("a1", _noop_cb)
            sm._state = start
            await sm.force_terminal()
            assert sm.state == AgentState.TERMINATED

    async def test_force_terminal_is_idempotent(self):
        calls: list[tuple[AgentState, AgentState]] = []

        async def cb(old: AgentState, new: AgentState) -> None:
            calls.append((old, new))

        sm = AgentStateMachine("a1", cb)
        sm._state = AgentState.BUSY
        await sm.force_terminal()
        await sm.force_terminal()
        assert len(calls) == 1


async def _noop_cb(old: AgentState, new: AgentState) -> None:
    pass
