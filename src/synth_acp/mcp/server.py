"""synth-mcp — FastMCP server for inter-agent messaging via SQLite."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time

import aiosqlite
from mcp.server.fastmcp import FastMCP

from synth_acp.db import ensure_schema_async

mcp = FastMCP("synth-mcp")

SESSION_ID = os.environ.get("SYNTH_SESSION_ID", "")
DB_PATH = os.environ.get("SYNTH_DB_PATH", "")
AGENT_ID = os.environ.get("SYNTH_AGENT_ID", "")
COMMUNICATION_MODE = os.environ.get("SYNTH_COMMUNICATION_MODE", "MESH")


async def _get_db() -> aiosqlite.Connection:
    """Open a WAL-mode async connection and ensure schema exists."""
    conn = await aiosqlite.connect(DB_PATH)
    await conn.execute("PRAGMA journal_mode=WAL")
    await ensure_schema_async(conn)
    return conn


async def _ensure_registered(conn: aiosqlite.Connection) -> None:
    """Auto-register this agent if not already present."""
    await conn.execute(
        "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, 'active', ?)",
        (AGENT_ID, SESSION_ID, int(time.time() * 1000)),
    )
    await conn.commit()


async def _get_visible_agents_async(db_path: str) -> list[str]:
    """Return agent_ids visible to AGENT_ID using a sync connection in a thread."""
    from synth_acp.models.visibility import get_visible_agents

    def _query() -> list[str]:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            return get_visible_agents(conn, AGENT_ID, SESSION_ID, COMMUNICATION_MODE)
        finally:
            conn.close()

    return await asyncio.to_thread(_query)


@mcp.tool()
async def send_message(to_agent: str, body: str, kind: str = "chat", reply_to: int | None = None) -> str:
    """Send a message to another agent, or '*' to broadcast to all active agents.

    Args:
        to_agent: Target agent ID, or '*' for broadcast.
        body: Message body text.
        kind: Message kind: chat, system, request, or response.
        reply_to: Optional message ID this is replying to.

    Returns:
        JSON with inserted message ID(s).
    """
    valid_kinds = {"chat", "system", "request", "response"}
    if kind not in valid_kinds:
        return json.dumps({"error": f"Invalid kind: {kind}. Must be one of: {', '.join(sorted(valid_kinds))}"})

    conn = await _get_db()
    await _ensure_registered(conn)

    if reply_to is not None:
        cursor = await conn.execute(
            "SELECT id FROM messages WHERE id = ? AND session_id = ?",
            (reply_to, SESSION_ID),
        )
        row = await cursor.fetchone()
        if not row:
            await conn.close()
            return json.dumps({"error": f"reply_to message not found: {reply_to}"})

    now = int(time.time() * 1000)
    if to_agent == "*":
        visible = await _get_visible_agents_async(DB_PATH)
        ids = []
        for aid in visible:
            cursor = await conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind, reply_to) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                (SESSION_ID, AGENT_ID, aid, body, now, kind, reply_to),
            )
            ids.append(cursor.lastrowid)
        await conn.commit()
        await conn.close()
        return json.dumps({"message_ids": ids})

    visible = await _get_visible_agents_async(DB_PATH)
    if to_agent not in visible:
        await conn.close()
        return json.dumps({"error": f"Agent not visible: {to_agent}"})
    cursor = await conn.execute(
        "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind, reply_to) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
        (SESSION_ID, AGENT_ID, to_agent, body, now, kind, reply_to),
    )
    msg_id = cursor.lastrowid
    await conn.commit()
    await conn.close()
    return json.dumps({"message_id": msg_id})


@mcp.tool()
async def check_delivery(message_id: int) -> str:
    """Check the delivery status of a message.

    Args:
        message_id: The message ID to check.

    Returns:
        JSON with the message status.
    """
    conn = await _get_db()
    cursor = await conn.execute("SELECT status FROM messages WHERE id = ?", (message_id,))
    row = await cursor.fetchone()
    await conn.close()
    if row:
        return json.dumps({"message_id": message_id, "status": row[0]})
    return json.dumps({"message_id": message_id, "status": "not_found"})


@mcp.tool()
async def launch_agent(
    agent_id: str,
    harness: str,
    cwd: str = ".",
    agent_mode: str = "",
    task: str = "",
    message: str = "",
) -> str:
    """Request the broker to launch a new agent. You become its parent.

    Args:
        agent_id: Unique identifier for the new agent.
        harness: Harness to use (e.g. 'kiro', 'claude', 'opencode').
        cwd: Working directory for the agent.
        agent_mode: ACP mode id to apply after session creation (optional).
        task: Description of the task for the agent.
        message: Initial message to send after the agent is idle.

    Returns:
        JSON with ok status and agent_id. Includes queued=true if at capacity.
    """
    conn = await _get_db()
    await _ensure_registered(conn)
    now = int(time.time() * 1000)
    payload = json.dumps(
        {
            "agent_id": agent_id,
            "harness": harness,
            "agent_mode": agent_mode,
            "cwd": cwd,
            "task": task,
            "message": message,
        }
    )
    await conn.execute(
        "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'launch', ?, 'pending', ?)",
        (SESSION_ID, AGENT_ID, payload, now),
    )
    await conn.commit()

    # Advisory pre-check: hint if at capacity
    max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM agents WHERE session_id = ? AND status = 'active'",
        (SESSION_ID,),
    )
    row = await cursor.fetchone()
    await conn.close()
    active = row[0] if row else 0
    if active >= max_agents:
        return json.dumps({"ok": True, "agent_id": agent_id, "queued": True})
    return json.dumps({"ok": True, "agent_id": agent_id})


@mcp.tool()
async def terminate_agent(agent_id: str) -> str:
    """Request the broker to terminate an agent you launched.

    Args:
        agent_id: The agent to terminate.

    Returns:
        JSON with ok status.
    """
    conn = await _get_db()
    await _ensure_registered(conn)
    now = int(time.time() * 1000)
    payload = json.dumps({"agent_id": agent_id})
    await conn.execute(
        "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'terminate', ?, 'pending', ?)",
        (SESSION_ID, AGENT_ID, payload, now),
    )
    await conn.commit()
    await conn.close()
    return json.dumps({"ok": True})


@mcp.tool()
async def list_agents() -> str:
    """List all agents in this session visible to you, plus yourself.

    Each entry includes is_self (true for your own entry), parent (who launched
    the agent), and task. In LOCAL mode only your family is visible.

    Returns:
        JSON array of {agent_id, status, parent, task, is_self}.
    """
    conn = await _get_db()
    await _ensure_registered(conn)
    visible = await _get_visible_agents_async(DB_PATH)
    # Include self so the caller can see its own parent/task
    all_ids = [*visible, AGENT_ID]
    cursor = await conn.execute(
        "SELECT agent_id, status, parent, task FROM agents WHERE session_id = ? AND agent_id IN ({})".format(
            ",".join("?" * len(all_ids))
        ),
        (SESSION_ID, *all_ids),
    )
    rows = await cursor.fetchall()
    await conn.close()
    agents = [
        {
            "agent_id": r[0],
            "status": r[1],
            "parent": r[2],
            "task": r[3],
            "is_self": r[0] == AGENT_ID,
        }
        for r in rows
    ]
    return json.dumps(agents)


@mcp.tool()
async def deregister_agent() -> str:
    """Mark this agent as inactive.

    Returns:
        Confirmation message.
    """
    conn = await _get_db()
    await conn.execute("UPDATE agents SET status = 'inactive' WHERE agent_id = ?", (AGENT_ID,))
    await conn.commit()
    await conn.close()
    return json.dumps({"status": "inactive", "agent_id": AGENT_ID})


def main() -> None:
    """Entry point for the synth-mcp CLI."""
    mcp.run(transport="stdio")
