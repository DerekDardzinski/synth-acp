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
COMMUNICATION_MODE = os.environ.get("SYNTH_COMMUNICATION_MODE", "MESH")

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL,
    parent      TEXT,
    task        TEXT
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
CREATE TABLE IF NOT EXISTS agent_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    command     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    created_at  INTEGER NOT NULL
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


def _get_visible_agents(conn: sqlite3.Connection) -> list[str]:
    """Return agent_ids visible to AGENT_ID based on COMMUNICATION_MODE.

    MESH: all active agents except self.
    LOCAL: parent, children, and siblings of self.

    Args:
        conn: Open SQLite connection.

    Returns:
        List of visible agent_ids.
    """
    if COMMUNICATION_MODE != "LOCAL":
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE session_id = ? AND status = 'active' AND agent_id != ?",
            (SESSION_ID, AGENT_ID),
        ).fetchall()
        return [r[0] for r in rows]

    # LOCAL mode
    row = conn.execute(
        "SELECT parent FROM agents WHERE agent_id = ? AND session_id = ?",
        (AGENT_ID, SESSION_ID),
    ).fetchone()
    parent = row[0] if row else None

    visible: set[str] = set()
    if parent:
        visible.add(parent)
        # Siblings: same parent, excluding self
        rows = conn.execute(
            "SELECT agent_id FROM agents WHERE parent = ? AND status = 'active' AND agent_id != ? AND session_id = ?",
            (parent, AGENT_ID, SESSION_ID),
        ).fetchall()
        visible.update(r[0] for r in rows)
    # Children
    rows = conn.execute(
        "SELECT agent_id FROM agents WHERE parent = ? AND status = 'active' AND session_id = ?",
        (AGENT_ID, SESSION_ID),
    ).fetchall()
    visible.update(r[0] for r in rows)
    return list(visible)


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
        visible = _get_visible_agents(conn)
        ids = []
        for aid in visible:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (SESSION_ID, AGENT_ID, aid, body, now),
            )
            ids.append(cursor.lastrowid)
        conn.commit()
        conn.close()
        return json.dumps({"message_ids": ids})
    # Single target: validate visibility
    visible = _get_visible_agents(conn)
    if to_agent not in visible:
        conn.close()
        return json.dumps({"error": f"Agent not visible: {to_agent}"})
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
def launch_agent(
    agent_id: str,
    agent_name: str,
    harness: str,
    cwd: str = ".",
    task: str = "",
    message: str = "",
) -> str:
    """Request the broker to launch a new agent. You become its parent.

    Args:
        agent_id: Unique identifier for the new agent.
        agent_name: Name/profile for the agent.
        harness: Harness to use (e.g. 'kiro', 'claude').
        cwd: Working directory for the agent.
        task: Description of the task for the agent.
        message: Initial message to send after the agent is idle.

    Returns:
        JSON with ok status and agent_id. Includes queued=true if at capacity.
    """
    conn = _get_db()
    _ensure_registered(conn)
    now = int(time.time() * 1000)
    payload = json.dumps(
        {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "harness": harness,
            "cwd": cwd,
            "task": task,
            "message": message,
        }
    )
    conn.execute(
        "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'launch', ?, 'pending', ?)",
        (SESSION_ID, AGENT_ID, payload, now),
    )
    conn.commit()

    # Advisory pre-check: hint if at capacity
    max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
    row = conn.execute(
        "SELECT COUNT(*) FROM agents WHERE session_id = ? AND status = 'active'",
        (SESSION_ID,),
    ).fetchone()
    conn.close()
    active = row[0] if row else 0
    if active >= max_agents:
        return json.dumps({"ok": True, "agent_id": agent_id, "queued": True})
    return json.dumps({"ok": True, "agent_id": agent_id})


@mcp.tool()
def terminate_agent(agent_id: str) -> str:
    """Request the broker to terminate an agent you launched.

    Args:
        agent_id: The agent to terminate.

    Returns:
        JSON with ok status.
    """
    conn = _get_db()
    _ensure_registered(conn)
    now = int(time.time() * 1000)
    payload = json.dumps({"agent_id": agent_id})
    conn.execute(
        "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'terminate', ?, 'pending', ?)",
        (SESSION_ID, AGENT_ID, payload, now),
    )
    conn.commit()
    conn.close()
    return json.dumps({"ok": True})


@mcp.tool()
def list_agents() -> str:
    """List all agents in this session visible to the current agent.

    Returns:
        JSON array of agents with status, parent, and task.
    """
    conn = _get_db()
    _ensure_registered(conn)
    visible = _get_visible_agents(conn)
    rows = (
        conn.execute(
            "SELECT agent_id, status, parent, task FROM agents WHERE session_id = ? AND agent_id IN ({})".format(
                ",".join("?" * len(visible))
            ),
            (SESSION_ID, *visible),
        ).fetchall()
        if visible
        else []
    )
    conn.close()
    agents = [{"agent_id": r[0], "status": r[1], "parent": r[2], "task": r[3]} for r in rows]
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
