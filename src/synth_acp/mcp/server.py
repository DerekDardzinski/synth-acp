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
        """The ONLY mechanism for inter-agent communication. Your text responses are
        streamed to the orchestration UI — they are NOT delivered to other agents.

        Call this whenever you need to share results, request work, or report completion.
        Use '*' as to_agent to broadcast to all visible agents.

        Args:
            to_agent: Agent ID from list_agents, or '*' to broadcast.
            body: Full, self-contained message. The recipient cannot see your
                previous text output or tool call history.
            kind: 'chat' for conversation (default), 'request' to ask for work,
                'response' to return results to the requesting agent.
            reply_to: Message ID from a previous send_message result to create a thread.

        Returns:
            {"message_id": int} for single sends, {"message_ids": [int, ...]} for broadcasts.
            {"error": str} if the target agent is not visible or reply_to is invalid.
        """
        valid_kinds = {"chat", "request", "response"}
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

    _caller_id = agent_id  # capture closure before parameter shadows it

    @mcp.tool()
    async def launch_agent(
        agent_id: str,
        harness: str,
        message: str,
        cwd: str = ".",
        agent_mode: str = "",
        task: str = "",
    ) -> str:
        """Launch a new child agent.

        Args:
            agent_id: Unique name for the new agent.
            harness: Runtime to use: 'kiro', 'claude', 'opencode', etc.
            message: Initial prompt sent to the agent once it becomes idle. Include
                explicit instructions to report back using send_message. Example:
                "...When complete, call send_message(to_agent='YOUR_ID', kind='response')
                with your findings."
            cwd: Working directory.
            agent_mode: Optional ACP mode ID.
            task: Short description shown in list_agents.

        Returns:
            {"ok": true, "agent_id": str}.
        """
        async with _db_conn() as conn:
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
                (session_id, _caller_id, payload, now),
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
            return json.dumps({"ok": True, "agent_id": agent_id, "queued": True})
        return json.dumps({"ok": True, "agent_id": agent_id})

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
    async def get_my_context() -> str:
        """Get your identity and communication rules for this session.

        Call this at the start of a task or whenever you need to confirm who you are,
        who launched you, and how to send results back.

        To see all active agents and their tasks, use list_agents instead.

        Returns:
            {"agent_id": str, "parent_agent": str|null, "task": str|null,
             "communication_rules": [str]}
        """
        async with _db_conn() as conn:
            await _ensure_registered(conn)
            cursor = await conn.execute(
                "SELECT parent, task FROM agents WHERE agent_id = ? AND session_id = ?",
                (agent_id, session_id),
            )
            row = await cursor.fetchone()
        parent = row[0] if row else None
        task = row[1] if row else None
        rules = [
            "Your text output is visible in the UI only — not to other agents.",
            "Use send_message() to communicate. Use kind='response' for results to your parent.",
            "Use list_agents() to discover other agents and their tasks.",
        ]
        return json.dumps({
            "agent_id": agent_id,
            "parent_agent": parent,
            "task": task,
            "communication_rules": rules,
        })

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
