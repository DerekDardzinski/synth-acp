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
    LOCAL: parent, children, siblings, and live handoff-lineage relatives of self.

    Only ACTIVE agents are returned, so a retired predecessor is invisible until somebody
    resurrects it.  Use ``get_resurrectable_agents`` to enumerate the inactive agents a
    caller may bring back.

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

    # Handoff lineage.  A resurrected predecessor is a child of nobody and a sibling of
    # nobody, so without this branch the successor that just resurrected it could not
    # message it and it could not reply.
    #
    # Deliberately narrow: it joins a predecessor to the id it handed off TO and nothing
    # else.  Two separately resurrected predecessors of the same lineage do NOT become
    # peers, because no requirement asks for it and every added edge is one more agent a
    # broadcast reaches.
    row = conn.execute(
        "SELECT retired_from FROM agents WHERE agent_id = ? AND session_id = ?",
        (agent_id, session_id),
    ).fetchone()
    retired_from = row[0] if row else None
    if retired_from:
        # Self is a predecessor: it sees whoever currently holds the id it retired from.
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE session_id = ? AND status = 'active' "
            "AND agent_id = ?",
            (session_id, retired_from),
        ).fetchall()
    else:
        # Self holds a lineage id: it sees its own live predecessors.
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE session_id = ? AND status = 'active' "
            "AND retired_from = ? AND agent_id != ?",
            (session_id, agent_id, agent_id),
        ).fetchall()
    visible.update(r[0] for r in rows)
    return list(visible)


def get_resurrectable_agents(
    conn: sqlite3.Connection,
    agent_id: str,
    session_id: str,
) -> list[tuple[str, str | None, str | None, str | None, int | None]]:
    """Return the inactive agents *agent_id* is authorized to resurrect.

    Deliberately independent of communication mode: authorization to resurrect is a
    lineage-and-parentage fact, not a visibility one, and it must match
    ``AgentLifecycle.handle_resurrect_command`` exactly or an agent would be shown a
    revival it cannot perform.  Two halves have to agree:

    - The STATUS predicate is ``= 'inactive'``, not ``!= 'active'``.  ``resurrect``
      refuses anything else, and ``restorable`` is a real status in this schema
      (``list_restorable_sessions``), so the looser form advertises a row whose
      resurrection reports success while nothing starts.
    - The RELATION predicate is the same pair: the caller is the row's ``parent``, or the
      caller holds the id its ``retired_from`` names.

    Returns:
        ``(agent_id, parent, task, retired_from, retired_at)`` per row, ordered most
        recently retired first with rows carrying no retirement time last.
    """
    rows = conn.execute(
        "SELECT agent_id, parent, task, retired_from, retired_at FROM agents "
        "WHERE session_id = ? AND status = 'inactive' AND agent_id != ? "
        "AND (parent = ? OR retired_from = ?) "
        "ORDER BY retired_at IS NULL, retired_at DESC, agent_id",
        (session_id, agent_id, agent_id, agent_id),
    ).fetchall()
    return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
