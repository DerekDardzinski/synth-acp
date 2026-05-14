"""AgentLifecycle — agent launch, termination, prompting, and task management."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import closing
from pathlib import Path
from typing import Any

from acp.schema import EnvVariable, McpServerStdio

from synth_acp.acp.session import ACPSession
from synth_acp.broker.registry import AgentRegistry
from synth_acp.db import ensure_schema_sync, expire_old_sessions_sync
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.config import (
    HarnessEntry,
    MessageHook,
    SessionConfig,
    load_startup_context,
    render_template,
)
from synth_acp.models.events import BrokerError, BrokerEvent, HookFired, InitialPromptDelivered
from synth_acp.models.visibility import get_visible_agents

log = logging.getLogger(__name__)

RE_FILE_REF = re.compile(r"(?:^|(?<=\s))@(\S+)")
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

        def _on_done(t: asyncio.Task[None]) -> None:
            self._tasks.pop(key, None)
            if not t.cancelled() and (exc := t.exception()):
                log.error("prompt task for %s raised", agent_id, exc_info=exc)

        task.add_done_callback(_on_done)
        return task

    async def launch(self, agent_id: str, *, adhoc_config: AgentConfig) -> None:
        """Launch an agent from an ad-hoc config."""
        agent_cfg = adhoc_config

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
        if agent_cfg.agent_mode and entry.mode_arg:
            cmd += [entry.mode_arg, agent_cfg.agent_mode]
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
            env=self._resolve_harness_env(entry),
            agent_mode_target=entry.agent_mode_target,
        )
        session.set_session_created_callback(self._on_acp_session_created)
        self._registry.register(agent_id, session)
        self._registry.set_harness(agent_id, agent_cfg.harness)
        self._tasks[agent_id] = self._make_run_task(agent_id, session)

        if adhoc_config is not None:
            now = int(time.time() * 1000)
            session_id = self._session_id

            def _sync(conn: sqlite3.Connection) -> None:
                ensure_schema_sync(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO agents "
                    "(agent_id, session_id, status, registered, harness, agent_mode, cwd) "
                    "VALUES (?, ?, 'active', ?, ?, ?, ?)",
                    (agent_id, session_id, now, agent_cfg.harness, agent_cfg.agent_mode, agent_cfg.cwd),
                )
                conn.commit()

            await self._db_op(_sync)

    async def terminate(self, agent_id: str) -> None:
        """Terminate a running agent session and clean up SQLite state."""
        async with self._registry.agent_lock(agent_id):
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

            session_id = self._session_id

            def _sync(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "UPDATE agents SET status = 'inactive' WHERE agent_id = ? AND session_id = ?",
                    (agent_id, session_id),
                )
                conn.execute(
                    "UPDATE agents SET parent = NULL WHERE parent = ? AND session_id = ?",
                    (agent_id, session_id),
                )
                conn.execute(
                    "UPDATE messages SET status = 'expired' WHERE to_agent = ? AND session_id = ? AND status = 'pending'",
                    (agent_id, session_id),
                )
                conn.commit()

            await self._db_op(_sync)
        # Keep these OUTSIDE the lock — they don't need serialization against
        # prompt/set_mode/set_model and shouldn't block other operations.
        task = await self._get_agent_task(agent_id)
        parent = self._registry.get_parent(agent_id)
        await self._fire_message_hook(
            self._config.settings.hooks.on_agent_exit, agent_id, task, parent, "on_agent_exit",
        )
        self._registry.orphan_children(agent_id)
        self._first_prompted.discard(agent_id)

    async def prompt(self, agent_id: str, text: str) -> bool:
        """Send a prompt to a running agent. Returns True if dispatched."""
        async with self._registry.agent_lock(agent_id):
            session = self._registry.get_session(agent_id)
            if not session:
                await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
                return False
            if session.state != AgentState.IDLE:
                await self._sink(
                    BrokerError(
                        agent_id=agent_id,
                        message=f"Agent '{agent_id}' is {session.state}, cannot prompt",
                        severity="warning",
                    )
                )
                return False
            if agent_id not in self._first_prompted:
                self._first_prompted.add(agent_id)
                hook = self._config.settings.hooks.on_agent_startup
                if hook.active:
                    context = load_startup_context()
                    rendered = render_template(context, {"agent_id": agent_id, "parent_id": "", "task": ""})
                    log.debug("on_agent_startup hook fired for %s:\n%s", agent_id, rendered)
                    text = rendered + text
                    await self._sink(HookFired(agent_id=agent_id, hook_name="on_agent_startup"))
            # Inject file contents for @path references
            text = self._inject_file_refs(agent_id, text)
            self._tasks[f"prompt-{agent_id}"] = self._make_prompt_task(agent_id, session.prompt(text))
            return True

    def _inject_file_refs(self, agent_id: str, text: str) -> str:
        """Parse @path references and prepend file contents as XML blocks."""
        cwd = self._registry.get_cwd(agent_id)
        if not cwd:
            return text
        refs = RE_FILE_REF.findall(text)
        if not refs:
            return text
        blocks: list[str] = []
        for rel_path in dict.fromkeys(refs):  # deduplicate, preserve order
            try:
                contents = (Path(cwd) / rel_path).read_text()
                blocks.append(f'<file path="{rel_path}">\n{contents}\n</file>')
            except (OSError, UnicodeDecodeError):
                log.debug("Could not read file ref: %s", rel_path)
        if blocks:
            return "\n".join(blocks) + "\n\n" + text
        return text

    async def cancel(self, agent_id: str) -> None:
        """Cancel the active prompt on an agent."""
        session = self._registry.get_session(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        await session.cancel()

    async def set_mode(self, agent_id: str, mode_id: str) -> None:
        """Forward a mode-switch request to the agent session."""
        async with self._registry.agent_lock(agent_id):
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
        async with self._registry.agent_lock(agent_id):
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

    async def set_config_option(self, agent_id: str, config_id: str, value: str | bool) -> None:
        """Forward a config option change to the agent session."""
        async with self._registry.agent_lock(agent_id):
            session = self._registry.get_session(agent_id)
            if not session:
                await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
                return
            if session.state != AgentState.IDLE:
                await self._sink(
                    BrokerError(
                        agent_id=agent_id,
                        message=f"Agent '{agent_id}' is {session.state}, cannot change config option",
                        severity="warning",
                    )
                )
                return
            await session.set_config_option(config_id, value)

    async def handle_launch_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a launch command from an agent."""
        agent_id = data["agent_id"]
        harness = data["harness"]
        agent_mode = data.get("agent_mode") or None
        cwd = str(Path(data.get("cwd", ".")).resolve())
        task = data.get("task", "")
        message = data.get("message", "")

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$", agent_id):
            await self.update_command_status(cmd_id, "rejected", "Invalid agent_id")
            return

        async with self._registry.agent_lock(agent_id):
            if self._registry.has_session(agent_id):
                await self.update_command_status(cmd_id, "rejected", f"Agent already exists: {agent_id}")
                return

            max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
            if self._registry.active_count() >= max_agents:
                await self.update_command_status(cmd_id, "rejected", f"Max agents ({max_agents}) reached")
                return

            entry = next((e for e in self._harness_registry if e.short_name == harness), None)
            if not entry:
                await self.update_command_status(cmd_id, "rejected", f"Unknown harness: {harness}")
                return

            cmd = entry.run_cmd.split()
            agent_cfg = AgentConfig(agent_id=agent_id, harness=harness, agent_mode=agent_mode, cwd=cwd)
            if agent_cfg.agent_mode and entry.mode_arg:
                cmd += [entry.mode_arg, agent_cfg.agent_mode]

            now = int(time.time() * 1000)
            session_id = self._session_id

            def _sync(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "INSERT OR REPLACE INTO agents "
                    "(agent_id, session_id, status, registered, parent, task, harness, agent_mode, cwd) "
                    "VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?)",
                    (agent_id, session_id, now, from_agent, task, harness, agent_mode, cwd),
                )
                conn.commit()

            await self._db_op(_sync)

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
                env=self._resolve_harness_env(entry),
                agent_mode_target=entry.agent_mode_target,
            )
            session.set_session_created_callback(self._on_acp_session_created)
            self._registry.register(agent_id, session)
            self._tasks[agent_id] = self._make_run_task(agent_id, session)

        if message and self._enqueue_pending:
            self._first_prompted.add(agent_id)
            self._registry.set_initial_message(agent_id, message)
            hook = self._config.settings.hooks.on_agent_startup
            if hook.active:
                context = load_startup_context()
                slots = {
                    "agent_id": agent_id,
                    "parent_id": from_agent,
                    "task": task,
                }
                rendered = render_template(context, slots)
                log.debug("on_agent_startup hook fired for %s:\n%s", agent_id, rendered)
                message = rendered + message
            if self._enqueue_raw:
                self._enqueue_raw(agent_id, message)
            else:
                self._enqueue_pending(agent_id, from_agent, message)
            await self._sink(
                InitialPromptDelivered(agent_id=agent_id, from_agent=from_agent, text=data.get("message", ""))
            )
            if hook.active:
                await self._sink(HookFired(agent_id=agent_id, hook_name="on_agent_startup"))

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

    async def resurrect(self, agent_id: str) -> None:
        """Re-launch a terminated agent by reconnecting to its previous ACP session."""
        async with self._registry.agent_lock(agent_id):
            session_id = self._session_id

            def _fetch(conn: sqlite3.Connection) -> tuple | None:
                return conn.execute(
                    "SELECT acp_session_id, harness, agent_mode, cwd, parent, task, status "
                    "FROM agents WHERE agent_id = ? AND session_id = ?",
                    (agent_id, session_id),
                ).fetchone()

            row = await self._db_op(_fetch)
            if not row:
                await self._sink(BrokerError(agent_id=agent_id, message=f"Agent '{agent_id}' not found"))
                return
            acp_session_id, harness, agent_mode, cwd, parent, task, status = row
            if status != "inactive":
                await self._sink(
                    BrokerError(agent_id=agent_id, message=f"Agent '{agent_id}' is {status}, not inactive")
                )
                return

            # Clean up old terminated session from registry
            old = self._registry.get_session(agent_id)
            if old:
                self._registry.unregister(agent_id)
                t = self._tasks.pop(agent_id, None)
                if t and not t.done():
                    t.cancel()

            await self.restore(
                agent_id=agent_id,
                acp_session_id=acp_session_id,
                harness=harness,
                agent_mode=agent_mode,
                cwd=cwd or ".",
                parent=parent,
            )

            def _update(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "UPDATE agents SET status = 'active' WHERE agent_id = ? AND session_id = ?",
                    (agent_id, session_id),
                )
                conn.commit()

            await self._db_op(_update)
        # Fire message hook OUTSIDE the lock — it doesn't mutate registry state
        # and shouldn't block other operations on this agent.
        await self._fire_message_hook(
            self._config.settings.hooks.on_agent_join, agent_id, task or "", parent, "on_agent_join",
        )

    async def handle_resurrect_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a resurrect command from an agent."""
        agent_id = data["agent_id"]
        session_id = self._session_id

        def _fetch_parent(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT parent FROM agents WHERE agent_id = ? AND session_id = ?",
                (agent_id, session_id),
            ).fetchone()
            return row[0] if row else None

        parent = await self._db_op(_fetch_parent)
        if parent != from_agent:
            await self.update_command_status(
                cmd_id, "rejected",
                f"Not authorized: {from_agent} is not parent of {agent_id}",
            )
            return
        await self.resurrect(agent_id)
        await self.update_command_status(cmd_id, "processed")

    async def shutdown(self) -> None:
        """Shutdown all agents: SIGKILL all process groups, then cancel tasks."""
        for session in self._registry.all_sessions().values():
            session.force_kill()

        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.wait(list(self._tasks.values()), timeout=3.0)

    async def _on_acp_session_created(self, agent_id: str, acp_session_id: str) -> None:
        """Write back the ACP session ID after the agent process creates it.

        Sets status to 'active' so only fully-connected agents are considered active.

        Uses asyncio.to_thread + sync sqlite3 because this callback fires
        from a background agent task and can race with shutdown.  A sync
        write on a daemon pool thread cannot keep the process alive.
        """
        def _write() -> None:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "UPDATE agents SET acp_session_id = ?, status = 'active' "
                    "WHERE agent_id = ? AND session_id = ?",
                    (acp_session_id, agent_id, self._session_id),
                )
                conn.commit()
            finally:
                conn.close()

        try:
            await asyncio.to_thread(_write)
        except Exception:
            log.debug("Failed to persist acp_session_id for %s", agent_id, exc_info=True)

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
        if agent_cfg.agent_mode and entry.mode_arg:
            cmd += [entry.mode_arg, agent_cfg.agent_mode]
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
            env=self._resolve_harness_env(entry),
            agent_mode_target=entry.agent_mode_target,
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

    async def _db_op(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run *fn* on a fresh sync sqlite3 connection in a thread-pool thread.

        Each call opens, uses, and closes its own connection via
        contextlib.closing.  WAL mode is set on every connection so
        concurrent access from MCP subprocesses is safe.
        """
        db_path = str(self._db_path)

        def _run() -> Any:
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                return fn(conn)

        return await asyncio.to_thread(_run)

    async def journal_ui_events(
        self, rows: list[tuple[str, str, int, str, str, int]]
    ) -> None:
        """Persist a batch of UI event rows to the journal table.

        Each row is (session_id, agent_id, seq, event_type, payload, created_at).
        """
        if not rows:
            return

        def _sync(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO ui_events "
                "(session_id, agent_id, seq, event_type, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        await self._db_op(_sync)

    async def expire_old_sessions(self) -> None:
        """Remove restorable sessions older than 30 days."""
        await self._db_op(expire_old_sessions_sync)

    async def update_command_status(self, cmd_id: int, status: str, error: str | None = None) -> None:
        def _sync(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE agent_commands SET status = ?, error = ? WHERE id = ?",
                (status, error, cmd_id),
            )
            conn.commit()

        await self._db_op(_sync)

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

    def _resolve_harness_env(self, entry: HarnessEntry) -> dict[str, str] | None:
        """Build environment overrides for a harness subprocess."""
        overrides: dict[str, str] = {}

        if entry.executable_env_var:
            for name in entry.binary_names:
                path = shutil.which(name)
                if path:
                    overrides[entry.executable_env_var] = path
                    log.debug("Harness '%s': %s=%s", entry.short_name, entry.executable_env_var, path)
                    break
            else:
                log.warning(
                    "Harness '%s': executable_env_var '%s' set but none of %s found in PATH",
                    entry.short_name, entry.executable_env_var, entry.binary_names,
                )

        for var in entry.clear_env_vars:
            overrides[var] = ""

        return overrides if overrides else None

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
        if not hook.active or not hook.template:
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
        now = int(time.time() * 1000)
        session_id = self._session_id

        def _sync(conn: sqlite3.Connection) -> None:
            for recipient in recipients:
                conn.execute(
                    "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
                    "VALUES (?, 'system', ?, ?, 'pending', ?, ?)",
                    (session_id, recipient, body, now, hook.kind),
                )
            conn.commit()

        await self._db_op(_sync)
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
        session_id = self._session_id

        def _sync(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute(
                "SELECT agent_id FROM agents WHERE session_id = ? AND parent = ? AND agent_id != ? AND status = 'active'",
                (session_id, parent_id, agent_id),
            ).fetchall()
            return [r[0] for r in rows]

        return await self._db_op(_sync)

    async def _get_agent_task(self, agent_id: str) -> str:
        """Look up the task description for an agent from SQLite."""
        session_id = self._session_id

        def _sync(conn: sqlite3.Connection) -> str:
            row = conn.execute(
                "SELECT task FROM agents WHERE agent_id = ? AND session_id = ?",
                (agent_id, session_id),
            ).fetchone()
            return row[0] or "" if row else ""

        return await self._db_op(_sync)
