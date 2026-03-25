"""Permission engine with persistent rule storage."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from synth_acp.models.permissions import PermissionDecision, PermissionRule

logger = logging.getLogger(__name__)


class PermissionEngine:
    """Loads, caches, and persists per-agent permission rules.

    Rules are keyed on ``(agent_id, tool_kind, session_id)`` and stored in a
    SQLite ``rules`` table.  The in-memory cache starts empty each session —
    no pre-loading from SQLite.
    """

    def __init__(self, db_path: Path, session_id: str) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._cache: dict[tuple[str, str, str], PermissionDecision] = {}
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rules ("
            "agent_id TEXT, tool_kind TEXT, session_id TEXT, decision TEXT, "
            "PRIMARY KEY (agent_id, tool_kind, session_id))"
        )
        conn.commit()
        conn.close()

    def check(self, agent_id: str, tool_kind: str, session_id: str) -> PermissionDecision | None:
        """Return the cached decision for *(agent_id, tool_kind, session_id)*, or ``None``.

        Args:
            agent_id: The agent that requested permission.
            tool_kind: The kind of tool call (e.g. ``"execute"``).
            session_id: The per-run session UUID.

        Returns:
            The stored decision as-is, or ``None`` if no rule is cached.
        """
        return self._cache.get((agent_id, tool_kind, session_id))

    def persist(self, rule: PermissionRule) -> None:
        """Write a rule to both the in-memory cache and SQLite.

        Args:
            rule: The permission rule to persist.
        """
        self._cache[(rule.agent_id, rule.tool_kind, rule.session_id)] = rule.decision
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "INSERT OR REPLACE INTO rules (agent_id, tool_kind, session_id, decision) "
            "VALUES (?, ?, ?, ?)",
            (rule.agent_id, rule.tool_kind, rule.session_id, rule.decision.value),
        )
        conn.commit()
        conn.close()
