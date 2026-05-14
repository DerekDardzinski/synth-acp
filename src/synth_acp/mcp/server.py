"""synth-mcp — FastMCP server for inter-agent messaging via SQLite."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import closing
from typing import Any

from mcp.server.fastmcp import FastMCP

from synth_acp.db import ensure_schema_sync
from synth_acp.models.visibility import get_visible_agents

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

    async def _db_op(fn: Callable[[sqlite3.Connection], Any]) -> Any:
        nonlocal _schema_ensured
        do_init = not _schema_ensured
        if do_init:
            _schema_ensured = True

        def _run() -> Any:
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                if do_init:
                    ensure_schema_sync(conn)
                return fn(conn)

        return await asyncio.to_thread(_run)

    def _ensure_registered(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, 'active', ?)",
            (agent_id, session_id, int(time.time() * 1000)),
        )
        conn.commit()

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

        now = int(time.time() * 1000)

        def _sync(conn: sqlite3.Connection) -> str:
            _ensure_registered(conn)

            if reply_to is not None:
                row = conn.execute(
                    "SELECT id FROM messages WHERE id = ? AND session_id = ?",
                    (reply_to, session_id),
                ).fetchone()
                if not row:
                    return json.dumps({"error": f"reply_to message not found: {reply_to}"})

            if to_agent == "*":
                visible = get_visible_agents(conn, agent_id, session_id, communication_mode)
                ids = []
                for aid in visible:
                    cursor = conn.execute(
                        "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind, reply_to) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                        (session_id, agent_id, aid, body, now, kind, reply_to),
                    )
                    ids.append(cursor.lastrowid)
                conn.commit()
                return json.dumps({"message_ids": ids})

            visible = get_visible_agents(conn, agent_id, session_id, communication_mode)
            if to_agent not in visible:
                return json.dumps({"error": f"Agent not visible: {to_agent}"})
            cursor = conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind, reply_to) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                (session_id, agent_id, to_agent, body, now, kind, reply_to),
            )
            msg_id = cursor.lastrowid
            conn.commit()
            return json.dumps({"message_id": msg_id})

        result = await _db_op(_sync)
        await notify()
        return result

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
        def _sync(conn: sqlite3.Connection) -> tuple[int, str | None]:
            _ensure_registered(conn)

            max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
            row = conn.execute(
                "SELECT COUNT(*) FROM agents WHERE session_id = ? AND status = 'active'",
                (session_id,),
            ).fetchone()
            active = row[0] if row else 0
            if active >= max_agents:
                return -1, json.dumps({"error": f"Max agents ({max_agents}) reached"})

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
            cursor = conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'launch', ?, 'pending', ?)",
                (session_id, _caller_id, payload, now),
            )
            conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid, None

        cmd_id, error = await _db_op(_sync)
        if error:
            return error
        await notify()

        for _ in range(10):
            await asyncio.sleep(0.3)

            def _poll(conn: sqlite3.Connection, cid: int = cmd_id) -> tuple | None:
                return conn.execute(
                    "SELECT status, error FROM agent_commands WHERE id = ?",
                    (cid,),
                ).fetchone()

            row = await _db_op(_poll)
            if row and row[0] != "pending":
                if row[0] == "rejected":
                    return json.dumps({"error": row[1] or "Launch rejected"})
                return json.dumps({"ok": True, "agent_id": agent_id})

        return json.dumps({"ok": True, "agent_id": agent_id})

    @mcp.tool()
    async def terminate_agent(target_agent_id: str) -> str:
        """Terminate a child agent you previously launched.

        Args:
            target_agent_id: ID of the child agent to terminate.

        Returns:
            {"ok": true}.
        """
        def _sync(conn: sqlite3.Connection) -> None:
            _ensure_registered(conn)
            now = int(time.time() * 1000)
            payload = json.dumps({"agent_id": target_agent_id})
            conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'terminate', ?, 'pending', ?)",
                (session_id, agent_id, payload, now),
            )
            conn.commit()

        await _db_op(_sync)
        await notify()
        return json.dumps({"ok": True})

    @mcp.tool()
    async def resurrect_agent(target_agent_id: str) -> str:
        """Resurrect a previously terminated agent, restoring its conversation history.

        Args:
            target_agent_id: ID of the terminated agent to resurrect.

        Returns:
            {"ok": true, "agent_id": str} on success, {"error": str} on failure.
        """
        def _sync(conn: sqlite3.Connection) -> int:
            _ensure_registered(conn)
            now = int(time.time() * 1000)
            payload = json.dumps({"agent_id": target_agent_id})
            cursor = conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'resurrect', ?, 'pending', ?)",
                (session_id, _caller_id, payload, now),
            )
            conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

        cmd_id = await _db_op(_sync)
        await notify()

        for _ in range(10):
            await asyncio.sleep(0.3)

            def _poll(conn: sqlite3.Connection, cid: int = cmd_id) -> tuple | None:
                return conn.execute(
                    "SELECT status, error FROM agent_commands WHERE id = ?",
                    (cid,),
                ).fetchone()

            row = await _db_op(_poll)
            if row and row[0] != "pending":
                if row[0] == "rejected":
                    return json.dumps({"error": row[1] or "Resurrect rejected"})
                return json.dumps({"ok": True, "agent_id": target_agent_id})

        return json.dumps({"ok": True, "agent_id": target_agent_id})

    @mcp.tool()
    async def list_agents() -> str:
        """List all agents visible to you in this session.

        Returns:
            JSON array of agent info objects.
        """
        def _sync(conn: sqlite3.Connection) -> str:
            _ensure_registered(conn)
            visible = get_visible_agents(conn, agent_id, session_id, communication_mode)
            all_ids = [*visible, agent_id]
            rows = conn.execute(
                "SELECT agent_id, status, parent, task FROM agents WHERE session_id = ? AND agent_id IN ({})".format(
                    ",".join("?" * len(all_ids))
                ),
                (session_id, *all_ids),
            ).fetchall()
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

        return await _db_op(_sync)

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
        def _sync(conn: sqlite3.Connection) -> tuple | None:
            _ensure_registered(conn)
            return conn.execute(
                "SELECT parent, task FROM agents WHERE agent_id = ? AND session_id = ?",
                (agent_id, session_id),
            ).fetchone()

        row = await _db_op(_sync)
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
