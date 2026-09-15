"""Tests for the broker's PromptQueue drain primitives."""

from __future__ import annotations

from synth_acp.broker.prompt_queue import PromptQueue, QueuedItem


class TestPopSteerable:
    def test_pop_steerable_stops_at_editing_item(self) -> None:
        """An item under edit and everything behind it must survive. Consuming past
        it would send half-typed text and silently drop the user's edit."""
        q = PromptQueue()
        for text in ("a", "b", "c", "d"):
            q.enqueue("agent-1", QueuedItem(id=text, text=text, steerable=True))
        q.mark_editing("agent-1", "c")

        popped = q.pop_steerable("agent-1")

        assert [i.text for i in popped] == ["a", "b"]
        assert [i.text for i in q.items("agent-1")] == ["c", "d"]

    def test_pop_steerable_returns_empty_when_front_is_editing(self) -> None:
        """Same rule at position 0: nothing is drainable, and nothing is consumed."""
        q = PromptQueue()
        q.enqueue("agent-1", QueuedItem(id="a", text="a", steerable=True))
        q.enqueue("agent-1", QueuedItem(id="b", text="b", steerable=True))
        q.mark_editing("agent-1", "a")

        assert q.pop_steerable("agent-1") == []
        assert [i.text for i in q.items("agent-1")] == ["a", "b"]

    def test_pop_steerable_stops_at_ineligible_item(self) -> None:
        """A queued user prompt or system notification must never ride along in a
        steer payload — being queued does not make a prohibited class eligible."""
        q = PromptQueue()
        q.enqueue("agent-1", QueuedItem(id="a", text="a", steerable=True))
        q.enqueue("agent-1", QueuedItem(id="user", text="typed by user", steerable=False))
        q.enqueue("agent-1", QueuedItem(id="c", text="c", steerable=True))

        popped = q.pop_steerable("agent-1")

        assert [i.text for i in popped] == ["a"]
        assert [i.text for i in q.items("agent-1")] == ["typed by user", "c"]


class TestRequeueFront:
    def test_requeue_front_restores_order_ahead_of_new_arrivals(self) -> None:
        """Undoing a failed steer must put the popped bodies back ahead of anything
        that arrived while the attempt was in flight, or delivery order inverts."""
        q = PromptQueue()
        q.enqueue("agent-1", QueuedItem(id="a", text="a", steerable=True))
        q.enqueue("agent-1", QueuedItem(id="b", text="b", steerable=True))
        popped = q.pop_steerable("agent-1")
        q.enqueue("agent-1", QueuedItem(id="c", text="c"))

        q.requeue_front("agent-1", popped)

        assert [i.text for i in q.items("agent-1")] == ["a", "b", "c"]


class TestRenameAuthor:
    def test_rename_author_rewrites_values_not_keys(self) -> None:
        """A handoff re-attributes what the predecessor WROTE without moving where
        anything is DELIVERED.

        The keys are recipient ids. Moving the renamed agent's key would strand an item
        whose database row is already marked delivered, and that row never retries -- a
        silent exactly-once violation with no error anywhere. The scan covers every
        queue because the predecessor may have authored an item still waiting in
        somebody else's queue.
        """
        q = PromptQueue()
        q.enqueue("worker", QueuedItem(id="inbound", text="for worker", from_agent="kid"))
        q.enqueue("kid", QueuedItem(id="authored", text="from worker", from_agent="worker"))
        q.enqueue("kid", QueuedItem(id="other", text="from boss", from_agent="boss"))

        moved = q.rename_author("worker", "worker.h0000dead")

        assert moved == 1
        assert sorted(q._queues) == ["kid", "worker"]
        assert q.items("worker")[0].from_agent == "kid"
        authored, other = q.items("kid")
        assert authored.from_agent == "worker.h0000dead"
        assert other.from_agent == "boss"
