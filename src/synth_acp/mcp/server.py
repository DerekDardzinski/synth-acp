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

from synth_acp.db import configure_connection, ensure_schema_sync
from synth_acp.models.visibility import get_resurrectable_agents, get_visible_agents

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
                configure_connection(conn)
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
            agent_id: Name for the new agent. Must be unique within the session:
                launching with the id of any existing agent — including a terminated
                one — is rejected ("Agent already exists"). To bring a terminated
                agent back, use resurrect_agent. An id is also reused when an agent
                hands itself off, but only through the handoff tool, which the agent
                calls on itself; launch_agent can never take over an existing id.
            harness: Runtime to use: 'kiro', 'claude', 'opencode', etc.
            message: Initial prompt sent to the agent once it becomes idle. Include
                explicit instructions to report back using send_message. Example:
                "...When complete, call send_message(to_agent='YOUR_ID', kind='response')
                with your findings."
            cwd: Working directory.
            agent_mode: Launch the child as a specific pre-defined agent
                configuration — a persona with its own system prompt, tools, and
                model that the harness already knows. This is the same named-agent
                concept the harness uses for its native subagents (Kiro agent
                configs, Claude Code agents). The value must match an agent
                configuration that already exists in this harness. It is not
                a task description or a free-form label — pass the task itself in
                `message` and a short summary in `task`.
            task: Short description shown in list_agents.

        Returns:
            {"ok": true, "agent_id": str} on success, or {"error": str} if the
            launch is rejected (e.g. the agent_id already exists, or the
            max-agents limit is reached).
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
        """Terminate a child agent you previously launched. Its id stays reserved
        for the session; bring the agent back later with resurrect_agent
        (launch_agent cannot reuse the id). The one other way an id changes hands is
        the handoff tool, which an agent calls on itself.

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

        Use this for an agent that was terminated — launch_agent rejects a reused id.
        Resurrection restores the agent with its prior conversation history intact.

        You may resurrect two kinds of agent: one you LAUNCHED, and any past self that
        handed off to the id you now hold. Call list_agents to see both — every entry
        whose status is not "active" is one you are authorized to bring back.

        To wake the past self that handed off to you, pass the "retired" entry whose
        lineage_of is your own id and whose generations_back is 1. Larger
        generations_back values are earlier selves, and you can wake those too. Do not
        pass your own bare id; a retired self is always the suffixed
        ``<original-id>.h<8 hex>`` form.

        A resurrected predecessor runs alongside you and reclaims nothing — see the
        handoff tool for what that means.

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
    async def handoff(handoff_message: str) -> str:
        """Hand your work to a fresh instance of YOURSELF, keeping your agent id.

        Call this when your context window is filling up. Synth stops you, starts a brand
        new session that takes over your agent_id, and gives it your handoff message as
        its opening prompt. From every other agent's point of view nothing happened: your
        parent and your children keep addressing the same id and are NOT notified.

        THE HANDOFF MESSAGE IS THE ONLY THING YOUR SUCCESSOR INHERITS. It starts with an
        empty transcript and cannot see your conversation, your tool output, or anything
        you were told. Write it as a complete briefing: what the task is, what you have
        already done, what you learned that is not obvious from the code, what is left,
        and anything you were about to do next.

        You are retired under a suffixed id (``<your-id>.h<8 hex>``) which keeps your full
        transcript. YOUR SUCCESSOR CAN BRING YOU BACK: whoever holds your id may call
        resurrect_agent on you, so the fresh session can wake you to ask what you meant.
        It finds you in list_agents as a "retired" entry with generations_back 1.

        A resurrected predecessor runs ALONGSIDE the successor that replaced it. It keeps
        its transcript and its own id, and it reclaims nothing: the children it launched
        stay with the successor, and the successor is neither paused nor notified. The two
        can message each other. Wake a predecessor to ask it questions, not to resume its
        work — two live agents prompted on one task duplicate it.

        Args:
            handoff_message: The complete briefing for your successor.

        Returns:
            {"accepted": true, "command_id": int} as soon as the request is durably
            recorded. This reports ACCEPTANCE, not completion: carrying out the handoff
            kills the process group this tool call is running in, so no completion
            response could reach you. Do not wait for one, and do not call this twice.
        """
        def _sync(conn: sqlite3.Connection) -> int:
            _ensure_registered(conn)
            now = int(time.time() * 1000)
            payload = json.dumps(
                {"agent_id": _caller_id, "handoff_message": handoff_message}
            )
            cursor = conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) VALUES (?, ?, 'handoff', ?, 'pending', ?)",
                (session_id, _caller_id, payload, now),
            )
            conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

        cmd_id = await _db_op(_sync)
        await notify()
        return json.dumps({"accepted": True, "command_id": cmd_id})

    @mcp.tool()
    async def list_agents() -> str:
        """List the agents visible to you, plus the ones you could bring back.

        Returns:
            JSON array of agent objects, each with: agent_id (str), status (str),
            parent (str|null — the agent that launched it), task (str|null),
            is_self (bool), resurrectable (bool), lineage_of (str|null) and
            generations_back (int|null).

            status is "active" for a live agent, "retired" for a past self that handed
            off, and "terminated" for an agent that was terminated.

            resurrectable says whether YOU may pass this agent to resurrect_agent. It is
            not implied by status: an inactive agent can be visible to you because it is
            your parent while still being outside your authority to revive, so check this
            field rather than assuming every non-active entry is revivable.

            lineage_of names the id whose lineage an entry belongs to, and is set on any
            agent that once handed off — including one you have already resurrected, which
            is active again but still a past self of that id. It is not always you: you
            also see the retired predecessors of agents you launched.

            generations_back is set only on "retired" entries, because it orders revival
            candidates. Within one lineage_of, 1 is the direct predecessor and 2 is the one
            before it. A resurrected predecessor keeps its lineage_of but reports
            generations_back null, since it is no longer a candidate.

            Entries are ordered: yourself, then live agents, then inactive ones most
            recently retired first.
        """
        def _sync(conn: sqlite3.Connection) -> str:
            _ensure_registered(conn)
            visible = get_visible_agents(conn, agent_id, session_id, communication_mode)
            all_ids = [*visible, agent_id]
            rows = conn.execute(
                "SELECT agent_id, status, parent, task, retired_from FROM agents "
                "WHERE session_id = ? AND agent_id IN ({})".format(
                    ",".join("?" * len(all_ids))
                ),
                (session_id, *all_ids),
            ).fetchall()

            def entry(
                aid: str,
                status: str,
                parent: str | None,
                task: str | None,
                retired_from: str | None,
            ) -> dict:
                # LOCAL visibility adds `parent` with no status filter, so a row from the
                # visible set is not necessarily active and gets the same three-way
                # mapping as a resurrectable one.
                if status == "active":
                    shown = "active"
                else:
                    shown = "retired" if retired_from else "terminated"
                return {
                    "agent_id": aid,
                    "status": shown,
                    "parent": parent,
                    "task": task,
                    "is_self": aid == agent_id,
                    # False by default.  A row arriving through the visible set has not
                    # been authorization-checked, and LOCAL adds `parent` with no status
                    # filter, so an inactive parent outside this caller's authority lands
                    # here.  Only the resurrectable pass below may set this true.
                    "resurrectable": False,
                    "lineage_of": retired_from,
                    "generations_back": None,
                }

            by_id: dict[str, dict] = {
                r[0]: entry(r[0], r[1], r[2], r[3], r[4]) for r in rows
            }
            live = sorted(by_id.values(), key=lambda a: (not a["is_self"], a["agent_id"]))

            # get_resurrectable_agents already returns newest-retired first, so the
            # counter below assigns generations_back in retirement order.  It is scoped
            # per lineage because one caller can see predecessors of several lineages:
            # its own, and those of the agents it launched.
            seen_per_lineage: dict[str, int] = {}
            dead = []
            for aid, parent, task, retired_from, _retired_at in get_resurrectable_agents(
                conn, agent_id, session_id
            ):
                row = entry(aid, "inactive", parent, task, retired_from)
                row["resurrectable"] = True
                if retired_from:
                    n = seen_per_lineage.get(retired_from, 0) + 1
                    seen_per_lineage[retired_from] = n
                    row["generations_back"] = n
                # An inactive agent reachable BOTH as a visible parent and as a
                # resurrectable row is one agent, not two.  Updating the existing dict in
                # place keeps its position in `live`, which holds the same objects.
                existing = by_id.get(aid)
                if existing is not None:
                    existing.update(row)
                    continue
                dead.append(row)
            return json.dumps([*live, *dead])

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
