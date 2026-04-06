"""Tests for AgentRegistry."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

from synth_acp.broker.registry import AgentRegistry
from synth_acp.models.agent import AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import UsageUpdated


def _config(*ids: str) -> SessionConfig:
    return SessionConfig(
        project="test",
        agents=[{"agent_id": aid, "harness": "kiro"} for aid in ids],
    )


def _mock_session(state: AgentState = AgentState.IDLE) -> AsyncMock:
    s = AsyncMock()
    s.state = state
    return s


class TestParentage:
    def test_orphan_children(self) -> None:
        reg = AgentRegistry(_config("a", "b", "c"))
        reg.set_parent("b", "a")
        reg.set_parent("c", "a")
        reg.orphan_children("a")
        assert reg.get_parent("b") is None
        assert reg.get_parent("c") is None


class TestUsage:
    def test_usage_warns_on_currency_change(self, caplog) -> None:
        reg = AgentRegistry(_config("a"))
        e1 = UsageUpdated(agent_id="a", size=100, used=50, cost_amount=1.0, cost_currency="USD")
        e2 = UsageUpdated(agent_id="a", size=200, used=100, cost_amount=2.0, cost_currency="EUR")
        reg.update_usage(e1)
        with caplog.at_level(logging.WARNING):
            reg.update_usage(e2)
        assert "cost_currency changed" in caplog.text


class TestActiveCount:
    def test_active_count(self) -> None:
        reg = AgentRegistry(_config("a", "b"))
        reg.register("a", _mock_session(AgentState.IDLE))
        reg.register("b", _mock_session(AgentState.TERMINATED))
        assert reg.active_count() == 1
