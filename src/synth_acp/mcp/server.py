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
    """Send a message to another agent. Call list_agents first to discover valid agent IDs.

    Use '*' as to_agent to broadcast to all visible agents. Messages are
    asynchronous — the recipient processes them on their next poll cycle.
    Use check_delivery to confirm receipt.

    Args:
        to_agent: Agent ID from list_agents, or '*' to broadcast.
        body: Message content. Be specific — the recipient has no shared context with you.
        kind: 'chat' for conversation (default), 'request' to ask for work,
            'response' to answer a request, 'system' for coordination signals.
        reply_to: Message ID from a previous send_message result to create a thread.

    Returns:
        {"message_id": int} for single sends, {"message_ids": [int, ...]} for broadcasts.
        {"error": str} if the target agent is not visible or reply_to is invalid.
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
    """Poll whether a previously sent message has been delivered.

    Use after send_message when you need confirmation before proceeding.
    Status transitions: pending → delivered. Poll periodically if still pending.

    Args:
        message_id: ID returned by send_message.

    Returns:
        {"message_id": int, "status": "pending"|"delivered"|"not_found"}.
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
    """Launch a new child agent. You become its parent and can terminate it later.

    The agent starts asynchronously — this returns immediately. Use list_agents
    to check when the new agent appears as 'active'. The child inherits your
    session and can message any agent visible to it.

    Args:
        agent_id: Unique name for the new agent (e.g. 'researcher', 'test-runner').
            Must not collide with existing agent IDs in the session.
        harness: Runtime to use: 'kiro', 'claude', 'opencode', etc.
        cwd: Working directory. Defaults to current directory.
        agent_mode: Optional ACP mode ID applied after session creation.
        task: Short description of what this agent should do. Shown in list_agents
            so other agents understand its purpose.
        message: Initial message sent to the agent once it becomes idle.
            Use this to give it its first instruction.

    Returns:
        {"ok": true, "agent_id": str}. Includes "queued": true if the session
        is at max capacity — the agent will launch when a slot opens.
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
    """Terminate a child agent you previously launched. Only works on agents you are the parent of.

    The agent is stopped asynchronously. Use list_agents to confirm it becomes inactive.

    Args:
        agent_id: ID of the child agent to terminate (from list_agents).

    Returns:
        {"ok": true}.
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
    """List all agents visible to you in this session. Call this before send_message
    to discover valid agent IDs, or to check the status of agents you launched.

    Visibility depends on communication mode: in MESH mode all session agents are
    visible; in LOCAL mode only your parent, children, and siblings are visible.
    Your own entry is always included (is_self: true).

    Returns:
        JSON array of {"agent_id": str, "status": "active"|"inactive",
        "parent": str|null, "task": str|null, "is_self": bool}.
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
    """Permanently mark yourself as inactive and leave the session. This is irreversible —
    call only when your task is fully complete and you have no more work to do.
    Other agents will no longer see you in list_agents.

    Returns:
        {"status": "inactive", "agent_id": str}.
    """
    conn = await _get_db()
    await conn.execute("UPDATE agents SET status = 'inactive' WHERE agent_id = ?", (AGENT_ID,))
    await conn.commit()
    await conn.close()
    return json.dumps({"status": "inactive", "agent_id": AGENT_ID})


def main() -> None:
    """Entry point for the synth-mcp CLI."""
    mcp.run(transport="stdio")
