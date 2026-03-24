"""synth-mcp — FastMCP server for inter-agent messaging via SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("synth-mcp")

SESSION_ID = os.environ.get("SYNTH_SESSION_ID", "")
DB_PATH = os.environ.get("SYNTH_DB_PATH", "")
AGENT_ID = os.environ.get("SYNTH_AGENT_ID", "")

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  INTEGER NOT NULL,
    claimed_at  INTEGER
);
"""


def _get_db() -> sqlite3.Connection:
    """Open a WAL-mode connection and ensure schema exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _ensure_registered(conn: sqlite3.Connection) -> None:
    """Auto-register this agent if not already present."""
    conn.execute(
        "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, 'active', ?)",
        (AGENT_ID, SESSION_ID, int(time.time() * 1000)),
    )
    conn.commit()


@mcp.tool()
def send_message(to_agent: str, body: str) -> str:
    """Send a message to another agent, or '*' to broadcast to all active agents.

    Args:
        to_agent: Target agent ID, or '*' for broadcast.
        body: Message body text.

    Returns:
        JSON with inserted message ID(s).
    """
    conn = _get_db()
    _ensure_registered(conn)
    now = int(time.time() * 1000)
    if to_agent == "*":
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE session_id = ? AND status = 'active' AND agent_id != ?",
            (SESSION_ID, AGENT_ID),
        ).fetchall()
        ids = []
        for (aid,) in rows:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (SESSION_ID, AGENT_ID, aid, body, now),
            )
            ids.append(cursor.lastrowid)
        conn.commit()
        conn.close()
        return json.dumps({"message_ids": ids})
    cursor = conn.execute(
        "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
        (SESSION_ID, AGENT_ID, to_agent, body, now),
    )
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return json.dumps({"message_id": msg_id})


@mcp.tool()
def check_delivery(message_id: int) -> str:
    """Check the delivery status of a message.

    Args:
        message_id: The message ID to check.

    Returns:
        JSON with the message status.
    """
    conn = _get_db()
    row = conn.execute("SELECT status FROM messages WHERE id = ?", (message_id,)).fetchone()
    conn.close()
    if row:
        return json.dumps({"message_id": message_id, "status": row[0]})
    return json.dumps({"message_id": message_id, "status": "not_found"})


@mcp.tool()
def list_agents() -> str:
    """List all agents in this session.

    Returns:
        JSON array of agents with status.
    """
    conn = _get_db()
    _ensure_registered(conn)
    rows = conn.execute(
        "SELECT agent_id, status, registered FROM agents WHERE session_id = ?",
        (SESSION_ID,),
    ).fetchall()
    conn.close()
    agents = [{"agent_id": r[0], "status": r[1], "registered": r[2]} for r in rows]
    return json.dumps(agents)


@mcp.tool()
def deregister_agent() -> str:
    """Mark this agent as inactive.

    Returns:
        Confirmation message.
    """
    conn = _get_db()
    conn.execute("UPDATE agents SET status = 'inactive' WHERE agent_id = ?", (AGENT_ID,))
    conn.commit()
    conn.close()
    return json.dumps({"status": "inactive", "agent_id": AGENT_ID})


def main() -> None:
    """Entry point for the synth-mcp CLI."""
    mcp.run(transport="stdio")
