"""synth-mcp — FastMCP server for inter-agent messaging via SQLite."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import aiosqlite
from mcp.server.fastmcp import FastMCP

from synth_acp.db import ensure_schema_async

type NotifyFn = Callable[[], Awaitable[None]]


async def _noop_notify() -> None:
    """Default no-op notifier used until the notification channel is wired."""


def create_mcp_server(
    db_path: str,
    session_id: str,
    agent_id: str,
    communication_mode: str = "MESH",
    notify: NotifyFn = _noop_notify,
) -> FastMCP:
    """Create a configured synth-mcp server instance.

    All tool functions close over the provided parameters instead of
    reading module-level globals.
    """
    mcp = FastMCP("synth-mcp")
    _schema_ensured = False

    @asynccontextmanager
    async def _db_conn() -> AsyncIterator[aiosqlite.Connection]:
        nonlocal _schema_ensured
        if not db_path:
            raise RuntimeError("SYNTH_DB_PATH is not set — synth-mcp must be launched by synth")
        conn = await aiosqlite.connect(db_path)
        try:
            await conn.execute("PRAGMA journal_mode=WAL")
            if not _schema_ensured:
                await ensure_schema_async(conn)
                _schema_ensured = True
            yield conn
        finally:
            await conn.close()

    async def _ensure_registered(conn: aiosqlite.Connection) -> None:
        await conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, 'active', ?)",
            (agent_id, session_id, int(time.time() * 1000)),
        )
        await conn.commit()

    async def _get_visible_agents_async() -> list[str]:
        from synth_acp.models.visibility import get_visible_agents

        def _query() -> list[str]:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                return get_visible_agents(conn, agent_id, session_id, communication_mode)
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

        async with _db_conn() as conn:
            await _ensure_registered(conn)

            if reply_to is not None:
                cursor = await conn.execute(
                    "SELECT id FROM messages WHERE id = ? AND session_id = ?",
                    (reply_to, session_id),
                )
                row = await cursor.fetchone()
                if not row:
                    return json.dumps({"error": f"reply_to message not found: {reply_to}"})

            now = int(time.time() * 1000)
            if to_agent == "*":
                visible = await _get_visible_agents_async()
                ids = []
                for aid in visible:
                    cursor = await conn.execute(
                        "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind, reply_to) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                        (session_id, agent_id, aid, body, now, kind, reply_to),
                    )
                    ids.append(cursor.lastrowid)
                await conn.commit()
                await notify()
                return json.dumps({"message_ids": ids})

            visible = await _get_visible_agents_async()
            if to_agent not in visible:
                return json.dumps({"error": f"Agent not visible: {to_agent}"})
            cursor = await conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind, reply_to) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                (session_id, agent_id, to_agent, body, now, kind, reply_to),
            )
            msg_id = cursor.lastrowid
            await conn.commit()
        await notify()
        return json.dumps({"message_id": msg_id})

    @mcp.tool()
    async def check_delivery(message_id: int) -> str:
        """Poll whether a previously sent message has been delivered.

        Args:
            message_id: ID returned by send_message.

        Returns:
            {"message_id": int, "status": "pending"|"delivered"|"not_found"}.
        """
        async with _db_conn() as conn:
            cursor = await conn.execute("SELECT status FROM messages WHERE id = ?", (message_id,))
            row = await cursor.fetchone()
        if row:
            return json.dumps({"message_id": message_id, "status": row[0]})
        return json.dumps({"message_id": message_id, "status": "not_found"})

    @mcp.tool()
    async def launch_agent(
        agent_id_param: str,
        harness: str,
        cwd: str = ".",
        agent_mode: str = "",
        task: str = "",
        message: str = "",
    ) -> str:
        """Launch a new child agent.

        Args:
            agent_id_param: Unique name for the new agent.
            harness: Runtime to use: 'kiro', 'claude', 'opencode', etc.
            cwd: Working directory.
            agent_mode: Optional ACP mode ID.
            task: Short description of what this agent should do.
            message: Initial message sent to the agent once it becomes idle.

        Returns:
            {"ok": true, "agent_id": str}.
        """
        async with _db_conn() as conn:
            await _ensure_registered(conn)
            now = int(time.time() * 1000)
            payload = json.dumps(
                {
                    "agent_id": agent_id_param,
                    "harness": harness,
                    "agent_mode": agent_mode,
                    "cwd": cwd,
                    "task": task,
                    "message": message,
                }
            )
            await conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'launch', ?, 'pending', ?)",
                (session_id, agent_id, payload, now),
            )
            await conn.commit()

            max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM agents WHERE session_id = ? AND status = 'active'",
                (session_id,),
            )
            row = await cursor.fetchone()
        await notify()
        active = row[0] if row else 0
        if active >= max_agents:
            return json.dumps({"ok": True, "agent_id": agent_id_param, "queued": True})
        return json.dumps({"ok": True, "agent_id": agent_id_param})

    @mcp.tool()
    async def terminate_agent(target_agent_id: str) -> str:
        """Terminate a child agent you previously launched.

        Args:
            target_agent_id: ID of the child agent to terminate.

        Returns:
            {"ok": true}.
        """
        async with _db_conn() as conn:
            await _ensure_registered(conn)
            now = int(time.time() * 1000)
            payload = json.dumps({"agent_id": target_agent_id})
            await conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'terminate', ?, 'pending', ?)",
                (session_id, agent_id, payload, now),
            )
            await conn.commit()
        await notify()
        return json.dumps({"ok": True})

    @mcp.tool()
    async def list_agents() -> str:
        """List all agents visible to you in this session.

        Returns:
            JSON array of agent info objects.
        """
        async with _db_conn() as conn:
            await _ensure_registered(conn)
            visible = await _get_visible_agents_async()
            all_ids = [*visible, agent_id]
            cursor = await conn.execute(
                "SELECT agent_id, status, parent, task FROM agents WHERE session_id = ? AND agent_id IN ({})".format(
                    ",".join("?" * len(all_ids))
                ),
                (session_id, *all_ids),
            )
            rows = await cursor.fetchall()
        agents = [
            {
                "agent_id": r[0],
                "status": r[1],
                "parent": r[2],
                "task": r[3],
                "is_self": r[0] == agent_id,
            }
            for r in rows
        ]
        return json.dumps(agents)

    @mcp.tool()
    async def deregister_agent() -> str:
        """Permanently mark yourself as inactive and leave the session.

        Returns:
            {"status": "inactive", "agent_id": str}.
        """
        async with _db_conn() as conn:
            await conn.execute(
                "UPDATE agents SET status = 'inactive' WHERE agent_id = ? AND session_id = ?",
                (agent_id, session_id),
            )
            now = int(time.time() * 1000)
            await conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) "
                "VALUES (?, ?, 'self_terminate', '{}', 'pending', ?)",
                (session_id, agent_id, now),
            )
            await conn.commit()
        await notify()
        return json.dumps({"status": "inactive", "agent_id": agent_id})

    return mcp


def main() -> None:
    """Entry point for the synth-mcp CLI."""
    db_path = os.environ.get("SYNTH_DB_PATH", "")
    session_id = os.environ.get("SYNTH_SESSION_ID", "")
    agent_id = os.environ.get("SYNTH_AGENT_ID", "")
    communication_mode = os.environ.get("SYNTH_COMMUNICATION_MODE", "MESH")

    missing = [
        name for name, val in [
            ("SYNTH_SESSION_ID", session_id),
            ("SYNTH_DB_PATH", db_path),
            ("SYNTH_AGENT_ID", agent_id),
        ]
        if not val
    ]
    if missing:
        print(
            f"synth-mcp: missing required environment variables: {', '.join(missing)}\n"
            "This tool is launched automatically by synth. Do not run it directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    notify_socket = os.environ.get("SYNTH_NOTIFY_SOCKET", "")
    notify: NotifyFn = _noop_notify
    if notify_socket:
        from synth_acp.mcp.notifier import BrokerNotifier
        notifier = BrokerNotifier(notify_socket)
        notify = notifier.notify

    server = create_mcp_server(db_path, session_id, agent_id, communication_mode, notify=notify)
    server.run(transport="stdio")
