"""Tests for AgentRegistry."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

from synth_acp.broker.registry import AgentRegistry
from synth_acp.models.agent import AgentState
from synth_acp.models.events import UsageUpdated


def _mock_session(state: AgentState = AgentState.IDLE) -> AsyncMock:
    s = AsyncMock()
    s.state = state
    return s


class TestParentage:
    def test_orphan_children(self) -> None:
        reg = AgentRegistry()
        reg.set_parent("b", "a")
        reg.set_parent("c", "a")
        reg.orphan_children("a")
        assert reg.get_parent("b") is None
        assert reg.get_parent("c") is None


class TestUsage:
    def test_usage_warns_on_currency_change(self, caplog) -> None:
        reg = AgentRegistry()
        e1 = UsageUpdated(agent_id="a", size=100, used=50, cost_amount=1.0, cost_currency="USD")
        e2 = UsageUpdated(agent_id="a", size=200, used=100, cost_amount=2.0, cost_currency="EUR")
        reg.update_usage(e1)
        with caplog.at_level(logging.WARNING):
            reg.update_usage(e2)
        assert "cost_currency changed" in caplog.text


class TestActiveCount:
    def test_active_count(self) -> None:
        reg = AgentRegistry()
        reg.register("a", _mock_session(AgentState.IDLE))
        reg.register("b", _mock_session(AgentState.TERMINATED))
        assert reg.active_count() == 1


class TestAgentLock:
    def test_unregister_removes_lock(self) -> None:
        reg = AgentRegistry()
        reg.register("a", _mock_session())
        lock_before = reg.agent_lock("a")
        reg.unregister("a")
        lock_after = reg.agent_lock("a")
        assert lock_before is not lock_after


class TestRename:
    async def test_rename_moves_owned_keys_and_keeps_lock(self) -> None:
        """The in-memory half of the handoff classification, where both halves fail
        silently.

        Rewriting a child's _parents VALUE makes a later command from the successor fail
        "Not authorized" while agents.parent correctly still says the original id, so the
        two stores disagree with no error anywhere.  Replacing the lock destroys mutual
        exclusion just as quietly, because agent_lock() lazily creates a DIFFERENT object.
        """
        reg = AgentRegistry()
        session = _mock_session()
        reg.register("worker", session)
        reg.set_parent("worker", "boss")
        reg.set_parent("kid", "worker")
        reg.set_harness("worker", "kiro")
        reg.set_initial_message("worker", "go")
        reg.update_usage(UsageUpdated(agent_id="worker", size=100, used=50))
        lock = reg.agent_lock("worker")
        await lock.acquire()

        reg.rename("worker", "worker.h0000dead")

        # The predecessor's own entries moved; the original key is free for the successor.
        assert reg.get_session("worker.h0000dead") is session
        assert reg.has_session("worker") is False
        assert reg.get_harness("worker.h0000dead") == "kiro"
        assert reg.pop_initial_message("worker.h0000dead") == "go"

        # POINTER: the child still points at the original id, which now denotes the
        # successor.  The predecessor's own parent moved with it, unchanged in value.
        assert reg.get_parent("kid") == "worker"
        assert reg.get_parent("worker.h0000dead") == "boss"

        # The frozen usage event is rebuilt, not mutated, and re-attributed.
        usage = reg.get_usage("worker.h0000dead")
        assert usage is not None
        assert (usage.agent_id, usage.size, usage.used) == ("worker.h0000dead", 100, 50)
        assert reg.get_usage("worker") is None

        # The lock belongs to the ID: the successor inherits the very same object, still
        # held, so anything parked on it wakes up correctly retargeted.
        assert reg.agent_lock("worker") is lock
        assert lock.locked() is True
        lock.release()

    def test_rename_contains_no_await(self) -> None:
        """Synchronous by contract: a yield point inside the re-key would let another
        coroutine observe a half-renamed registry, which is the one thing the design
        relies on being impossible."""
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(AgentRegistry.rename)))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Await)]
