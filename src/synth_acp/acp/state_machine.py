"""Agent state machine — single source of truth for lifecycle state."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from synth_acp.models.agent import TRANSITIONS, AgentState, InvalidTransitionError

log = logging.getLogger(__name__)

type TransitionCallback = Callable[[AgentState, AgentState], Awaitable[None]]


class AgentStateMachine:
    """Encapsulates validated state transitions with an async notification hook.

    Args:
        agent_id: Identifier used in log messages and errors.
        on_transition: Async callback invoked after every state change with (old, new).
    """

    def __init__(self, agent_id: str, on_transition: TransitionCallback) -> None:
        self._agent_id = agent_id
        self._state = AgentState.UNSTARTED
        self._on_transition = on_transition

    @property
    def state(self) -> AgentState:
        return self._state

    async def transition(self, new_state: AgentState) -> None:
        """Validated transition. Raises InvalidTransitionError if disallowed."""
        if new_state not in TRANSITIONS[self._state]:
            raise InvalidTransitionError(f"{self._agent_id}: {self._state} → {new_state}")
        old = self._state
        self._state = new_state
        await self._on_transition(old, new_state)

    async def force_terminal(self) -> None:
        """Unconditional transition to TERMINATED. Idempotent.

        Use for cleanup paths (finally blocks, voluntary exit) where
        raising InvalidTransitionError would be harmful.
        """
        if self._state == AgentState.TERMINATED:
            return
        old = self._state
        self._state = AgentState.TERMINATED
        await self._on_transition(old, AgentState.TERMINATED)
