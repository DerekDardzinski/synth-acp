"""AgentLifecycle — agent launch, termination, prompting, and task management."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path

import aiosqlite
from acp.schema import EnvVariable, McpServerStdio

from synth_acp.acp.session import ACPSession
from synth_acp.broker.registry import AgentRegistry
from synth_acp.db import ensure_schema_async, expire_old_sessions_async
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.config import MessageHook, SessionConfig, render_template
from synth_acp.models.events import BrokerError, BrokerEvent, HookFired
from synth_acp.models.visibility import get_visible_agents

log = logging.getLogger(__name__)

type EventSink = Callable[[BrokerEvent], Awaitable[None]]
type EnqueuePendingFn = Callable[[str, str, str], None]
type EnqueueRawFn = Callable[[str, str], None]


def _natural_list(items: list[str]) -> str:
    """Render a list as natural language: 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class AgentLifecycle:
    """Manages agent launch, termination, prompting, and background tasks.

    Every asyncio.Task created by this class has a done-callback that
    removes it from _tasks on completion, preventing accumulation.
    """

    def __init__(
        self,
        config: SessionConfig,
        registry: AgentRegistry,
        event_sink: EventSink,
        db_path: Path,
        session_id: str,
    ) -> None:
        self._config = config
        self._registry = registry
        self._sink = event_sink
        self._db_path = db_path
        self._session_id = session_id
        self._notify_socket_path: str = ""
        self._enqueue_pending: EnqueuePendingFn | None = None
        self._enqueue_raw: EnqueueRawFn | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._first_prompted: set[str] = set()
        self._harness_registry = load_harness_registry()
        self._db: aiosqlite.Connection | None = None
        self._terminate_timeout: float = 5.0

    def set_message_bus(self, socket_path: str, enqueue: EnqueuePendingFn, enqueue_raw: EnqueueRawFn) -> None:
        """Wire the message bus after construction. Must be called before launching agents."""
        self._notify_socket_path = socket_path
        self._enqueue_pending = enqueue
        self._enqueue_raw = enqueue_raw

    def _make_run_task(self, agent_id: str, session: ACPSession) -> asyncio.Task[None]:
        task = asyncio.create_task(session.run(), name=f"run-{agent_id}")

        def _on_done(t: asyncio.Task[None]) -> None:
            self._tasks.pop(agent_id, None)
            if not t.cancelled() and (exc := t.exception()):
                log.error("session.run() for %s raised", agent_id, exc_info=exc)

        task.add_done_callback(_on_done)
        return task

    def _make_prompt_task(self, agent_id: str, coro: Coroutine[object, object, None]) -> asyncio.Task[None]:
        key = f"prompt-{agent_id}"
        task = asyncio.create_task(coro, name=key)
        task.add_done_callback(lambda _: self._tasks.pop(key, None))
        return task

    async def launch(self, agent_id: str, *, adhoc_config: AgentConfig | None = None) -> None:
        """Launch an agent by ID from the config, or from an ad-hoc config."""
        if adhoc_config is not None:
            agent_cfg = adhoc_config
        else:
            agent_cfg = next((a for a in self._config.agents if a.agent_id == agent_id), None)
        if not agent_cfg:
            await self._sink(
                BrokerError(agent_id=agent_id, message=f"No config for agent '{agent_id}'")
            )
            return

        entry = next(
            (e for e in self._harness_registry if e.short_name == agent_cfg.harness), None
        )
        if not entry:
            await self._sink(
                BrokerError(
                    agent_id=agent_id,
                    message=f"Unknown harness '{agent_cfg.harness}'. "
                    f"Known: {', '.join(sorted(e.short_name for e in self._harness_registry))}",
                )
            )
            return

        cmd = entry.run_cmd.split()
        mcp_servers = [
            McpServerStdio(
                name="synth-mcp",
                command="synth-mcp",
                args=[],
                env=self._build_mcp_env(agent_id, agent_cfg.env),
            )
        ]

        if self._registry.has_session(agent_id):
            old = self._registry.get_session(agent_id)
            if old and old.state != AgentState.TERMINATED:
                await self._sink(
                    BrokerError(agent_id=agent_id, message=f"Agent '{agent_id}' is still running")
                )
                return
            self._registry.unregister(agent_id)
            task = self._tasks.pop(agent_id, None)
            if task and not task.done():
                task.cancel()

        session = ACPSession(
            agent_id=agent_cfg.agent_id,
            binary=cmd[0],
            args=cmd[1:],
            cwd=agent_cfg.cwd,
            event_sink=self._sink,
            mcp_servers=mcp_servers,
            agent_mode=agent_cfg.agent_mode,
        )
        session.set_session_created_callback(self._on_acp_session_created)
        self._registry.register(agent_id, session)
        self._registry.set_harness(agent_id, agent_cfg.harness)
        self._tasks[agent_id] = self._make_run_task(agent_id, session)

        if adhoc_config is not None:
            db = await self._ensure_db()
            await ensure_schema_async(db)
            now = int(time.time() * 1000)
            await db.execute(
                "INSERT OR REPLACE INTO agents "
                "(agent_id, session_id, status, registered, harness, agent_mode, cwd) "
                "VALUES (?, ?, 'active', ?, ?, ?, ?)",
                (agent_id, self._session_id, now, agent_cfg.harness, agent_cfg.agent_mode, agent_cfg.cwd),
            )
            await db.commit()

    async def terminate(self, agent_id: str) -> None:
        """Terminate a running agent session and clean up SQLite state."""
        session = self._registry.get_session(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        if session.state != AgentState.TERMINATED:
            try:
                await asyncio.wait_for(session.terminate(), timeout=self._terminate_timeout)
            except TimeoutError:
                log.warning("session.terminate() timed out for %s", agent_id)
            for key in (agent_id, f"prompt-{agent_id}"):
                task = self._tasks.get(key)
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, ConnectionError, OSError, RuntimeError):
                        pass

        db = await self._ensure_db()
        await db.execute(
            "UPDATE agents SET status = 'inactive' WHERE agent_id = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        await db.execute(
            "UPDATE agents SET parent = NULL WHERE parent = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        await db.execute(
            "UPDATE messages SET status = 'expired' WHERE to_agent = ? AND session_id = ? AND status = 'pending'",
            (agent_id, self._session_id),
        )
        await db.commit()
        task = await self._get_agent_task(agent_id)
        parent = self._registry.get_parent(agent_id)
        await self._fire_message_hook(
            self._config.settings.hooks.on_agent_exit, agent_id, task, parent, "on_agent_exit",
        )
        self._registry.orphan_children(agent_id)
        self._first_prompted.discard(agent_id)

    async def prompt(self, agent_id: str, text: str) -> None:
        """Send a prompt to a running agent."""
        session = self._registry.get_session(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        if session.state != AgentState.IDLE:
            await self._sink(
                BrokerError(
                    agent_id=agent_id,
                    message=f"Agent '{agent_id}' is {session.state}, cannot prompt",
                    severity="warning",
                )
            )
            return
        if agent_id not in self._first_prompted:
            self._first_prompted.add(agent_id)
            hook = self._config.settings.hooks.on_agent_startup
            if hook.prepend:
                rendered = render_template(hook.prepend, {"agent_id": agent_id})
                log.debug("on_agent_startup hook fired for %s:\n%s", agent_id, rendered)
                text = rendered + text
                await self._sink(HookFired(agent_id=agent_id, hook_name="on_agent_startup"))
        self._tasks[f"prompt-{agent_id}"] = self._make_prompt_task(agent_id, session.prompt(text))

    async def cancel(self, agent_id: str) -> None:
        """Cancel the active prompt on an agent."""
        session = self._registry.get_session(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        await session.cancel()

    async def set_mode(self, agent_id: str, mode_id: str) -> None:
        """Forward a mode-switch request to the agent session."""
        session = self._registry.get_session(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        if session.state != AgentState.IDLE:
            await self._sink(
                BrokerError(
                    agent_id=agent_id,
                    message=f"Agent '{agent_id}' is {session.state}, cannot switch mode",
                    severity="warning",
                )
            )
            return
        await session.set_mode(mode_id)

    async def set_model(self, agent_id: str, model_id: str) -> None:
        """Forward a model-switch request to the agent session."""
        session = self._registry.get_session(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        if session.state != AgentState.IDLE:
            await self._sink(
                BrokerError(
                    agent_id=agent_id,
                    message=f"Agent '{agent_id}' is {session.state}, cannot switch model",
                    severity="warning",
                )
            )
            return
        await session.set_model(model_id)

    async def handle_launch_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a launch command from an agent."""
        agent_id = data["agent_id"]
        harness = data["harness"]
        agent_mode = data.get("agent_mode") or None
        cwd = data.get("cwd", ".")
        task = data.get("task", "")
        message = data.get("message", "")

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", agent_id):
            await self.update_command_status(cmd_id, "rejected", "Invalid agent_id")
            return

        if self._registry.has_session(agent_id):
            await self.update_command_status(cmd_id, "rejected", f"Agent already exists: {agent_id}")
            return

        max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
        if self._registry.active_count() >= max_agents:
            return

        entry = next((e for e in self._harness_registry if e.short_name == harness), None)
        if not entry:
            await self.update_command_status(cmd_id, "rejected", f"Unknown harness: {harness}")
            return

        cmd = entry.run_cmd.split()
        agent_cfg = AgentConfig(agent_id=agent_id, harness=harness, agent_mode=agent_mode, cwd=cwd)

        db = await self._ensure_db()
        now = int(time.time() * 1000)
        await db.execute(
            "INSERT OR REPLACE INTO agents "
            "(agent_id, session_id, status, registered, parent, task, harness, agent_mode, cwd) "
            "VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?)",
            (agent_id, self._session_id, now, from_agent, task, harness, agent_mode, cwd),
        )
        await db.commit()

        self._registry.set_parent(agent_id, from_agent)
        self._registry.set_harness(agent_id, harness)

        mcp_servers = [
            McpServerStdio(
                name="synth-mcp",
                command="synth-mcp",
                args=[],
                env=self._build_mcp_env(agent_id, agent_cfg.env),
            )
        ]
        session = ACPSession(
            agent_id=agent_cfg.agent_id,
            binary=cmd[0],
            args=cmd[1:],
            cwd=agent_cfg.cwd,
            event_sink=self._sink,
            mcp_servers=mcp_servers,
            agent_mode=agent_cfg.agent_mode,
        )
        session.set_session_created_callback(self._on_acp_session_created)
        self._registry.register(agent_id, session)
        self._tasks[agent_id] = self._make_run_task(agent_id, session)
        self._first_prompted.add(agent_id)

        if message and self._enqueue_pending:
            self._registry.set_initial_message(agent_id, message)
            prompt_hook = self._config.settings.hooks.on_agent_prompt
            if prompt_hook.prepend:
                slots = {
                    "agent_id": agent_id,
                    "task": task,
                    "parent_id": from_agent,
                    "message": message,
                }
                rendered = render_template(prompt_hook.prepend, slots)
                log.debug("on_agent_prompt hook fired for %s:\n%s", agent_id, rendered)
                message = rendered + message
            if self._enqueue_raw:
                self._enqueue_raw(agent_id, message)
            else:
                self._enqueue_pending(agent_id, from_agent, message)

        await self.update_command_status(cmd_id, "processed")
        await self._fire_message_hook(
            self._config.settings.hooks.on_agent_join, agent_id, task, from_agent, "on_agent_join",
        )

    async def handle_terminate_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a terminate command from an agent."""
        agent_id = data["agent_id"]
        parent = self._registry.get_parent(agent_id)
        if parent != from_agent:
            await self.update_command_status(
                cmd_id, "rejected",
                f"Not authorized: {from_agent} is not parent of {agent_id}",
            )
            return
        await self.terminate(agent_id)
        await self.update_command_status(cmd_id, "processed")

    async def shutdown(self) -> None:
        """Shutdown all agents."""
        for session in self._registry.all_sessions().values():
            if session.state == AgentState.BUSY:
                await session.cancel()
            elif session.state == AgentState.AWAITING_PERMISSION:
                try:
                    await asyncio.wait_for(session.terminate(), timeout=self._terminate_timeout)
                except TimeoutError:
                    log.warning("session.terminate() timed out for %s", session.agent_id)

        for session in self._registry.all_sessions().values():
            if session.state != AgentState.TERMINATED:
                try:
                    await asyncio.wait_for(session.terminate(), timeout=self._terminate_timeout)
                except TimeoutError:
                    log.warning("session.terminate() timed out for %s", session.agent_id)

        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.wait(self._tasks.values(), timeout=2.0)

    async def _on_acp_session_created(self, agent_id: str, acp_session_id: str) -> None:
        """Write back the ACP session ID after the agent process creates it.

        Also transitions status from 'restorable' to 'active' at this point,
        so only fully-connected agents are considered active.
        """
        db = await self._ensure_db()
        await db.execute(
            "UPDATE agents SET acp_session_id = ?, status = 'active' "
            "WHERE agent_id = ? AND session_id = ?",
            (acp_session_id, agent_id, self._session_id),
        )
        await db.commit()

    async def mark_agents_restorable(self) -> None:
        """Mark all active agents in this session as restorable.

        Called during broker shutdown before close_db(). Uses the DB directly
        rather than the in-memory registry, which may be incomplete if agents
        were still initialising when shutdown was triggered.
        """
        db = await self._ensure_db()
        await db.execute(
            "UPDATE agents SET status = 'restorable' "
            "WHERE session_id = ? AND status = 'active'",
            (self._session_id,),
        )
        await db.commit()

    async def restore(
        self,
        agent_id: str,
        acp_session_id: str | None,
        harness: str,
        agent_mode: str | None,
        cwd: str,
        parent: str | None,
    ) -> None:
        """Restore a previously-running agent from saved state (no SQLite insert)."""
        entry = next((e for e in self._harness_registry if e.short_name == harness), None)
        if not entry:
            await self._sink(
                BrokerError(agent_id=agent_id, message=f"Cannot restore: unknown harness '{harness}'")
            )
            return

        cmd = entry.run_cmd.split()
        agent_cfg = AgentConfig(agent_id=agent_id, harness=harness, agent_mode=agent_mode, cwd=cwd)
        mcp_servers = [
            McpServerStdio(
                name="synth-mcp",
                command="synth-mcp",
                args=[],
                env=self._build_mcp_env(agent_id, agent_cfg.env),
            )
        ]

        session = ACPSession(
            agent_id=agent_id,
            binary=cmd[0],
            args=cmd[1:],
            cwd=cwd,
            event_sink=self._sink,
            mcp_servers=mcp_servers,
            agent_mode=agent_mode,
        )
        self._registry.register(agent_id, session)
        if parent:
            self._registry.set_parent(agent_id, parent)
        self._registry.set_harness(agent_id, harness)

        # Always register the session-created callback on both branches.
        session.set_session_created_callback(self._on_acp_session_created)

        # Suppress on_agent_startup hook — restored agents already have
        # orchestration context from their prior conversation history.
        self._first_prompted.add(agent_id)

        # No ACP session ID means the agent never had a conversation — launch fresh.
        if acp_session_id:
            coro = session.run_restored(acp_session_id)
        else:
            coro = session.run()

        task = asyncio.create_task(coro, name=f"run-{agent_id}")

        def _on_done(t: asyncio.Task[None]) -> None:
            self._tasks.pop(agent_id, None)
            if not t.cancelled() and (exc := t.exception()):
                log.error("session.run_restored() for %s raised", agent_id, exc_info=exc)

        task.add_done_callback(_on_done)
        self._tasks[agent_id] = task

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
        return self._db

    async def close_db(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def expire_old_sessions(self) -> None:
        """Remove restorable sessions older than 30 days."""
        db = await self._ensure_db()
        await expire_old_sessions_async(db)

    async def register_agents(self) -> None:
        """Pre-register all config agents in SQLite."""
        db = await self._ensure_db()
        await ensure_schema_async(db)
        now = int(time.time() * 1000)
        for agent in self._config.agents:
            await db.execute(
                "INSERT OR REPLACE INTO agents "
                "(agent_id, session_id, status, registered, harness, agent_mode, cwd) "
                "VALUES (?, ?, 'active', ?, ?, ?, ?)",
                (agent.agent_id, self._session_id, now, agent.harness, agent.agent_mode, agent.cwd),
            )
        await db.commit()

    async def update_command_status(self, cmd_id: int, status: str, error: str | None = None) -> None:
        db = await self._ensure_db()
        await db.execute(
            "UPDATE agent_commands SET status = ?, error = ? WHERE id = ?",
            (status, error, cmd_id),
        )
        await db.commit()

    def _build_mcp_env(self, agent_id: str, extra_env: dict[str, str] | None = None) -> list[EnvVariable]:
        env = [
            EnvVariable(name="SYNTH_SESSION_ID", value=self._session_id),
            EnvVariable(name="SYNTH_DB_PATH", value=str(self._db_path)),
            EnvVariable(name="SYNTH_AGENT_ID", value=agent_id),
            EnvVariable(name="SYNTH_COMMUNICATION_MODE", value=self._config.settings.communication_mode.value),
            EnvVariable(name="SYNTH_MAX_AGENTS", value=os.environ.get("SYNTH_MAX_AGENTS", "10")),
            EnvVariable(name="SYNTH_NOTIFY_SOCKET", value=self._notify_socket_path),
        ]
        if extra_env:
            env.extend(EnvVariable(name=k, value=v) for k, v in extra_env.items())
        return env

    async def _get_visible_agents_for(self, agent_id: str) -> list[str]:
        def _query() -> list[str]:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                return get_visible_agents(
                    conn, agent_id, self._session_id,
                    self._config.settings.communication_mode.value,
                )
            finally:
                conn.close()
        return await asyncio.to_thread(_query)

    async def _fire_message_hook(
        self,
        hook: MessageHook,
        agent_id: str,
        task: str,
        parent_id: str | None,
        hook_name: str,
    ) -> None:
        """Send a templated message to the configured recipients."""
        if hook.recipients == "none" or not hook.template:
            return
        recipients = await self._resolve_recipients(hook.recipients, agent_id, parent_id)
        if not recipients:
            return
        siblings = await self._get_siblings(agent_id, parent_id)
        slots = {
            "agent_id": agent_id,
            "task": task or "",
            "parent_id": parent_id or "",
            "sibling_ids": _natural_list(siblings),
        }
        body = render_template(hook.template, slots)
        log.debug("%s hook fired for %s → %s:\n%s", hook_name, agent_id, recipients, body)
        db = await self._ensure_db()
        now = int(time.time() * 1000)
        for recipient in recipients:
            await db.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
                "VALUES (?, 'system', ?, ?, 'pending', ?, ?)",
                (self._session_id, recipient, body, now, hook.kind),
            )
        await db.commit()
        for recipient in recipients:
            await self._sink(HookFired(agent_id=recipient, hook_name=hook_name))

    async def _resolve_recipients(
        self, mode: str, agent_id: str, parent_id: str | None,
    ) -> list[str]:
        """Resolve recipient list based on the configured mode."""
        if mode == "parent":
            return [parent_id] if parent_id else []
        if mode == "family":
            family = await self._get_siblings(agent_id, parent_id)
            if parent_id:
                family = [parent_id, *family]
            return family
        if mode == "mesh":
            return await self._get_visible_agents_for(agent_id)
        return []

    async def _get_siblings(self, agent_id: str, parent_id: str | None) -> list[str]:
        """Get sibling agent IDs (agents sharing the same parent, excluding self)."""
        if not parent_id:
            return []
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT agent_id FROM agents WHERE session_id = ? AND parent = ? AND agent_id != ? AND status = 'active'",
            (self._session_id, parent_id, agent_id),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def _get_agent_task(self, agent_id: str) -> str:
        """Look up the task description for an agent from SQLite."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT task FROM agents WHERE agent_id = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        row = await cursor.fetchone()
        return row[0] or "" if row else ""
