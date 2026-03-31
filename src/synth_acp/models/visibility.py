"""Shared visibility logic for inter-agent communication."""

from __future__ import annotations

import sqlite3


def get_visible_agents(
    conn: sqlite3.Connection,
    agent_id: str,
    session_id: str,
    communication_mode: str,
) -> list[str]:
    """Return agent_ids visible to *agent_id* based on communication mode.

    MESH: all active agents except self.
    LOCAL: parent, children, and siblings of self.

    Args:
        conn: Open SQLite connection.
        agent_id: The agent to compute visibility for.
        session_id: Session to scope the query.
        communication_mode: ``"MESH"`` or ``"LOCAL"``.

    Returns:
        List of visible agent_ids.
    """
    if communication_mode != "LOCAL":
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE session_id = ? AND status = 'active' AND agent_id != ?",
            (session_id, agent_id),
        ).fetchall()
        return [r[0] for r in rows]

    # LOCAL mode
    row = conn.execute(
        "SELECT parent FROM agents WHERE agent_id = ? AND session_id = ?",
        (agent_id, session_id),
    ).fetchone()
    parent = row[0] if row else None

    visible: set[str] = set()
    if parent:
        visible.add(parent)
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE parent = ? AND status = 'active' AND agent_id != ? AND session_id = ?",
            (parent, agent_id, session_id),
        ).fetchall()
        visible.update(r[0] for r in rows)
    rows = conn.execute(
        "SELECT agent_id FROM agents WHERE parent = ? AND status = 'active' AND session_id = ?",
        (agent_id, session_id),
    ).fetchall()
    visible.update(r[0] for r in rows)
    return list(visible)
