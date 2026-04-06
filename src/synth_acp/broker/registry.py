"""AgentRegistry — owns agent sessions and metadata."""

from __future__ import annotations

import logging

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentConfig, AgentMode, AgentModel, AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import UsageUpdated

log = logging.getLogger(__name__)


class AgentRegistry:
    """Central store for agent sessions, parentage, harness info, and usage.

    Pure data object — no I/O, no async, no tasks.
    """

    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._sessions: dict[str, ACPSession] = {}
        self._parents: dict[str, str | None] = {a.agent_id: None for a in config.agents}
        self._harnesses: dict[str, str] = {a.agent_id: a.harness for a in config.agents}
        self._usage: dict[str, UsageUpdated] = {}

    def register(self, agent_id: str, session: ACPSession) -> None:
        self._sessions[agent_id] = session

    def unregister(self, agent_id: str) -> ACPSession | None:
        return self._sessions.pop(agent_id, None)

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

    def orphan_children(self, parent_id: str) -> None:
        for aid, p in self._parents.items():
            if p == parent_id:
                self._parents[aid] = None

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

    def get_configs(self) -> list[AgentConfig]:
        return list(self._config.agents)

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

    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.state != AgentState.TERMINATED)
