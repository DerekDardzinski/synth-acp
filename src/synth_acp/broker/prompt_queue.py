"""PromptQueue — per-agent ordered prompt buffer with drain conditions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4


@dataclass
class QueuedItem:
    """A prompt waiting in the queue."""

    id: str = field(default_factory=lambda: f"q-{uuid4().hex[:8]}")
    text: str = ""
    source: Literal["user", "mcp"] = "user"
    from_agent: str | None = None
    editing: bool = False
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
