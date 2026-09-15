"""AgentRegistry — owns agent sessions and metadata."""

from __future__ import annotations

import asyncio
import logging

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentMode, AgentModel, AgentState
from synth_acp.models.events import UsageUpdated

log = logging.getLogger(__name__)


class AgentRegistry:
    """Central store for agent sessions, parentage, harness info, and usage.

    Pure data object — no I/O, no async, no tasks.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ACPSession] = {}
        self._parents: dict[str, str | None] = {}
        self._harnesses: dict[str, str] = {}
        self._initial_messages: dict[str, str] = {}
        self._usage: dict[str, UsageUpdated] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register(self, agent_id: str, session: ACPSession) -> None:
        self._sessions[agent_id] = session

    def unregister(self, agent_id: str) -> ACPSession | None:
        self._locks.pop(agent_id, None)
        return self._sessions.pop(agent_id, None)

    def rename(self, old_agent_id: str, new_agent_id: str) -> None:
        """Move the predecessor's own per-agent entries from old to new.

        SYNCHRONOUS BY CONTRACT.  Contains no await and must never gain one: being
        synchronous is what makes the re-key atomic under asyncio, because without a
        yield point no other coroutine can observe a partially re-keyed registry.

        Moves the KEY old_agent_id to new_agent_id in _sessions, _parents, _harnesses,
        _initial_messages and _usage.  The _usage value is a frozen event and is rebuilt
        with model_copy(update=...), not mutated.

        Does NOT rewrite any _parents VALUE.  A value is a live parent pointer: a child
        whose parent is old_agent_id must keep pointing at old_agent_id, because that id
        now denotes the successor and agents.parent in SQLite still says so.  Rewriting
        it would desynchronize the two stores and break terminate authorization.

        Does NOT move _locks.  The lock belongs to the ID, since every call site keys
        agent_lock() on an id.  Leaving _locks[old_agent_id] in place means the successor
        inherits it and any coroutine already parked on it wakes up holding the correct
        lock for whatever that id now denotes.  Moving or popping it would silently
        destroy mutual exclusion, which is also why this method must never call
        unregister().

        Leaves _sessions[old_agent_id] ABSENT.  The caller binds the successor's session
        there in the same synchronous block.

        Args:
            old_agent_id: Current id.  Entries are moved, not copied.
            new_agent_id: Destination id.  Must not already be present.
        """
        if old_agent_id in self._sessions:
            self._sessions[new_agent_id] = self._sessions.pop(old_agent_id)
        if old_agent_id in self._parents:
            self._parents[new_agent_id] = self._parents.pop(old_agent_id)
        if old_agent_id in self._harnesses:
            self._harnesses[new_agent_id] = self._harnesses.pop(old_agent_id)
        if old_agent_id in self._initial_messages:
            self._initial_messages[new_agent_id] = self._initial_messages.pop(old_agent_id)
        if old_agent_id in self._usage:
            usage = self._usage.pop(old_agent_id)
            self._usage[new_agent_id] = usage.model_copy(update={"agent_id": new_agent_id})

    def agent_lock(self, agent_id: str) -> asyncio.Lock:
        """Return the per-agent serialization lock. Creates on first access.

        The lock serializes all operations that transition an agent out of IDLE.
        Acquired internally by lifecycle.prompt(), lifecycle.set_mode(), lifecycle.set_model().
        External callers should use lock.locked() as a non-blocking guard only.
        """
        if agent_id not in self._locks:
            self._locks[agent_id] = asyncio.Lock()
        return self._locks[agent_id]

    def get_session(self, agent_id: str) -> ACPSession | None:
        return self._sessions.get(agent_id)

    def has_session(self, agent_id: str) -> bool:
        return agent_id in self._sessions

    def all_sessions(self) -> dict[str, ACPSession]:
        return dict(self._sessions)

    def set_parent(self, agent_id: str, parent: str | None) -> None:
        self._parents[agent_id] = parent

    def get_parent(self, agent_id: str) -> str | None:
        return self._parents.get(agent_id)

    def set_harness(self, agent_id: str, harness: str) -> None:
        self._harnesses[agent_id] = harness

    def get_harness(self, agent_id: str) -> str:
        return self._harnesses.get(agent_id, "")

    def get_cwd(self, agent_id: str) -> str:
        s = self._sessions.get(agent_id)
        return s._cwd if s else ""

    def orphan_children(self, parent_id: str) -> None:
        for aid, p in self._parents.items():
            if p == parent_id:
                self._parents[aid] = None

    def set_initial_message(self, agent_id: str, message: str) -> None:
        self._initial_messages[agent_id] = message

    def pop_initial_message(self, agent_id: str) -> str | None:
        return self._initial_messages.pop(agent_id, None)

    def update_usage(self, event: UsageUpdated) -> None:
        prev = self._usage.get(event.agent_id)
        if prev is not None and (
            event.cost_currency is not None
            and prev.cost_currency is not None
            and event.cost_currency != prev.cost_currency
        ):
            log.warning(
                "cost_currency changed for %s: %s → %s",
                event.agent_id, prev.cost_currency, event.cost_currency,
            )
        self._usage[event.agent_id] = event

    def get_usage(self, agent_id: str) -> UsageUpdated | None:
        return self._usage.get(agent_id)

    def get_states(self) -> dict[str, AgentState]:
        return {aid: s.state for aid, s in self._sessions.items()}

    def get_modes(self, agent_id: str) -> list[AgentMode]:
        s = self._sessions.get(agent_id)
        return s.available_modes if s else []

    def get_current_mode(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        return s.current_mode_id if s else None

    def get_models(self, agent_id: str) -> list[AgentModel]:
        s = self._sessions.get(agent_id)
        return s.available_models if s else []

    def get_current_model(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        return s.current_model_id if s else None

    def get_agent_mode(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        return s.agent_mode if s else None

    def get_agent_mode_target(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        return s.agent_mode_target if s else None

    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.state != AgentState.TERMINATED)
