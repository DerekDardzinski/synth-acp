"""PromptQueue — per-agent ordered prompt buffer with drain conditions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4


@dataclass
class QueuedItem:
    """A prompt waiting in the queue.

    ``steerable`` records whether THIS item may be injected into a running turn.
    It is decided by the broker when the item is queued and cannot be recovered
    later: the queue stores rendered text, not the message kind, so a user prompt
    and a system notification are indistinguishable from an eligible chat body
    once queued. Both are prohibited from being steered.

    ``first_prompt`` marks the one item a handoff reserves as its successor's opening
    turn. It is what lets a drain identify the reserved item at the moment it hands the
    text to ``AgentLifecycle.prompt``, so that item is admitted while every other caller
    is refused until it has been delivered.
    """

    id: str = field(default_factory=lambda: f"q-{uuid4().hex[:8]}")
    text: str = ""
    source: Literal["user", "mcp"] = "user"
    from_agent: str | None = None
    editing: bool = False
    steerable: bool = False
    first_prompt: bool = False
    timestamp: float = field(default_factory=time.time)


class PromptQueue:
    """Per-agent ordered prompt buffer with drain conditions.

    Pure synchronous state — no DB, no async. The broker owns drain
    logic and calls pop() when conditions are met.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[QueuedItem]] = {}

    def enqueue(self, agent_id: str, item: QueuedItem) -> None:
        """Append an item to the agent's queue."""
        self._queues.setdefault(agent_id, []).append(item)

    def pop(self, agent_id: str) -> QueuedItem | None:
        """Pop and return the front item if drainable.

        Returns None if queue is empty or front item is being edited.
        """
        queue = self._queues.get(agent_id)
        if not queue:
            return None
        if queue[0].editing:
            return None
        item = queue.pop(0)
        if not queue:
            del self._queues[agent_id]
        return item

    def pop_steerable(self, agent_id: str) -> list[QueuedItem]:
        """Pop every LEADING item that may be steered, in queue order.

        Stops at the first item that is flagged ``editing`` or is not
        ``steerable``, leaving it and everything behind it queued. The editing
        rule matches what ``can_drain``/``pop`` already apply to the front item.
        The steerable rule keeps user prompts and system notifications out of a
        steer payload: a message class that is prohibited from being steered on
        arrival is still prohibited once it is sitting in the queue.

        Returns ``[]`` when the front item is either being edited or not
        steerable.
        """
        queue = self._queues.get(agent_id)
        if not queue:
            return []
        stop = len(queue)
        for i, item in enumerate(queue):
            if item.editing or not item.steerable:
                stop = i
                break
        if stop == 0:
            return []
        popped = queue[:stop]
        del queue[:stop]
        if not queue:
            del self._queues[agent_id]
        return popped

    def requeue_front(self, agent_id: str, items: list[QueuedItem]) -> None:
        """Put previously popped items back at the front, preserving their order.

        Used to undo a ``pop_steerable`` whose consumer failed. Popping before
        the attempt rather than after is what keeps delivery exactly-once:
        anything enqueued while the attempt was in flight stays queued instead of
        being consumed as though it had been sent.

        Also used by an agent handoff to seed the successor's reserved first prompt
        ahead of anything it inherited, and to restore that item if the drain
        carrying it is refused.
        """
        if not items:
            return
        self._queues.setdefault(agent_id, [])[:0] = items

    def rename_author(self, old_agent_id: str, new_agent_id: str) -> int:
        """Re-attribute queued items authored by *old_agent_id*, across every queue.

        Synchronous, and deliberately touches VALUES only.  ``from_agent`` records who
        wrote an item, so when that author is renamed by a handoff the record follows it.

        The queue KEYS are left alone: a key is a RECIPIENT id, and an item sitting under
        the renamed agent's key belongs to whoever now occupies that id.  Moving the key
        would strand an item whose database row is already marked delivered, which never
        retries -- a silent exactly-once violation.

        Scans every queue, not just the renamed agent's own, because the predecessor may
        have authored an item that is still waiting in another agent's queue.

        Args:
            old_agent_id: The author being renamed.
            new_agent_id: Its new id.

        Returns:
            How many items were re-attributed.
        """
        moved = 0
        for queue in self._queues.values():
            for item in queue:
                if item.from_agent == old_agent_id:
                    item.from_agent = new_agent_id
                    moved += 1
        return moved

    def mark_editing(self, agent_id: str, item_id: str) -> None:
        """Mark a queued item as being edited (blocks drain at that item)."""
        item = self._find(agent_id, item_id)
        if item:
            item.editing = True

    def commit_edit(self, agent_id: str, item_id: str, text: str) -> None:
        """Save an edit and clear the editing flag."""
        item = self._find(agent_id, item_id)
        if item:
            item.text = text
            item.editing = False

    def delete(self, agent_id: str, item_id: str) -> None:
        """Remove an item from the queue."""
        queue = self._queues.get(agent_id)
        if not queue:
            return
        self._queues[agent_id] = [i for i in queue if i.id != item_id]
        if not self._queues[agent_id]:
            del self._queues[agent_id]

    def can_drain(self, agent_id: str) -> bool:
        """Whether the queue has a drainable item (non-empty, front not editing)."""
        queue = self._queues.get(agent_id)
        if not queue:
            return False
        return not queue[0].editing

    def items(self, agent_id: str) -> list[QueuedItem]:
        """Return a snapshot of the queue for UI reconciliation."""
        return list(self._queues.get(agent_id, []))

    def is_empty(self, agent_id: str) -> bool:
        """Whether the queue has no items."""
        return not self._queues.get(agent_id)

    def _find(self, agent_id: str, item_id: str) -> QueuedItem | None:
        """Find an item by id in an agent's queue."""
        for item in self._queues.get(agent_id, []):
            if item.id == item_id:
                return item
        return None
