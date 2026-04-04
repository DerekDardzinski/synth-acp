"""ACPBroker — owns all agent sessions and routes events."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
from acp.schema import EnvVariable, McpServerStdio

from synth_acp.acp.session import ACPSession
from synth_acp.broker.permissions import PermissionEngine
from synth_acp.broker.poller import MessagePoller
from synth_acp.db import ensure_schema_async
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig, AgentMode, AgentModel, AgentState
from synth_acp.models.commands import (
    BrokerCommand,
    CancelTurn,
    LaunchAgent,
    RespondPermission,
    SendPrompt,
    SetAgentMode,
    SetAgentModel,
    TerminateAgent,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    BrokerError,
    BrokerEvent,
    McpMessageDelivered,
    PermissionAutoResolved,
    PermissionRequested,
    UsageUpdated,
)
from synth_acp.models.permissions import PermissionDecision, PermissionRule
from synth_acp.models.visibility import get_visible_agents

log = logging.getLogger(__name__)


class ACPBroker:
    """Central orchestration service for agent sessions."""

    def __init__(
        self,
        config: SessionConfig,
        db_path: Path | None = None,
    ) -> None:
        self._config = config
        self._db_path = db_path or Path.home() / ".synth" / "synth.db"
        self._session_id = f"{config.project}-{uuid.uuid4().hex[:8]}"
        self._sessions: dict[str, ACPSession] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._event_queue: asyncio.Queue[BrokerEvent] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self._shutting_down = False
        self._permission_engine = PermissionEngine(
            db_path=self._db_path,
            session_id=self._session_id,
        )
        self._pending_permissions: dict[str, PermissionRequested] = {}
        self._usage: dict[str, UsageUpdated] = {}
        self._poller: MessagePoller | None = None
        self._pending_initial_prompts: dict[str, str] = {}
        self._agent_parents: dict[str, str | None] = {a.agent_id: None for a in config.agents}
        self._agent_harnesses: dict[str, str] = {a.agent_id: a.harness for a in config.agents}
        self._db: aiosqlite.Connection | None = None
        self._harness_registry: list = load_harness_registry()

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def handle(self, command: BrokerCommand) -> None:
        """Dispatch a command to the appropriate handler.

        Args:
            command: The broker command to dispatch.
        """
        match command:
            case LaunchAgent(agent_id=aid, config=cfg):
                await self._launch(aid, adhoc_config=cfg)
            case TerminateAgent(agent_id=aid):
                await self._terminate(aid)
            case SendPrompt(agent_id=aid, text=text):
                await self._prompt(aid, text)
            case RespondPermission(agent_id=aid, option_id=oid):
                self._resolve_permission(aid, oid)
            case CancelTurn(agent_id=aid):
                await self._cancel(aid)
            case SetAgentMode(agent_id=aid, mode_id=mid):
                await self._set_mode(aid, mid)
            case SetAgentModel(agent_id=aid, model_id=mid):
                await self._set_model(aid, mid)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_agent_states(self) -> dict[str, AgentState]:
        """Return current state of all launched agents."""
        return {aid: s.state for aid, s in self._sessions.items()}

    def get_agent_configs(self) -> list[AgentConfig]:
        """Return all agent configs from the session config."""
        return list(self._config.agents)

    def get_usage(self, agent_id: str) -> UsageUpdated | None:
        """Return the latest usage snapshot for an agent.

        Values come directly from the ACP SDK's ``usage_update``
        events, which already carry cumulative session cost.

        Args:
            agent_id: The agent to query.

        Returns:
            Latest usage snapshot, or ``None`` if no usage reported.
        """
        return self._usage.get(agent_id)

    def get_agent_parent(self, agent_id: str) -> str | None:
        """Return the parent agent ID, or None if no parent."""
        return self._agent_parents.get(agent_id)

    def get_agent_harness(self, agent_id: str) -> str:
        """Return the harness short_name for an agent, or empty string."""
        return self._agent_harnesses.get(agent_id, "")

    def get_agent_modes(self, agent_id: str) -> list[AgentMode]:
        """Return available modes for an agent, or [] if not yet received."""
        session = self._sessions.get(agent_id)
        return session.available_modes if session else []

    def get_current_mode(self, agent_id: str) -> str | None:
        """Return the current mode id for an agent, or None."""
        session = self._sessions.get(agent_id)
        return session.current_mode_id if session else None

    def get_agent_models(self, agent_id: str) -> list[AgentModel]:
        """Return available models for an agent, or [] if not yet received.

        May always return [] for agents that do not support the UNSTABLE
        models capability.
        """
        session = self._sessions.get(agent_id)
        return session.available_models if session else []

    def get_current_model(self, agent_id: str) -> str | None:
        """Return the current model id for an agent, or None."""
        session = self._sessions.get(agent_id)
        return session.current_model_id if session else None

    def is_permission_pending(self, agent_id: str) -> bool:
        """Return True if the agent has an unresolved permission request."""
        return agent_id in self._pending_permissions

    def _accumulate_usage(self, event: UsageUpdated) -> None:
        """Store the latest usage snapshot for an agent.

        The ACP SDK emits cumulative session cost in each
        ``usage_update``, so no summation is needed — the latest event
        is the authoritative value.  Logs a warning if
        ``cost_currency`` changes between updates.

        Args:
            event: The usage snapshot from the session.
        """
        prev = self._usage.get(event.agent_id)
        if prev is not None and (
            event.cost_currency is not None
            and prev.cost_currency is not None
            and event.cost_currency != prev.cost_currency
        ):
            log.warning(
                "cost_currency changed for %s: %s → %s",
                event.agent_id,
                prev.cost_currency,
                event.cost_currency,
            )
        self._usage[event.agent_id] = event

    # ------------------------------------------------------------------
    # Event sink with permission interception
    # ------------------------------------------------------------------

    async def _sink(self, event: BrokerEvent) -> None:
        """Event sink passed to sessions. Intercepts PermissionRequested for auto-resolve."""
        if isinstance(event, PermissionRequested):
            self._pending_permissions[event.agent_id] = event
            decision = self._permission_engine.check(event.agent_id, event.kind, self._session_id)
            if decision is not None:
                session = self._sessions.get(event.agent_id)
                if session:
                    option_id = self._find_option_id(event.options, decision)
                    if option_id:
                        session.resolve_permission(option_id)
                        await self._event_queue.put(
                            PermissionAutoResolved(
                                agent_id=event.agent_id,
                                request_id=event.request_id,
                                decision=decision,
                            )
                        )
                        return
        elif isinstance(event, UsageUpdated):
            self._accumulate_usage(event)
        await self._event_queue.put(event)
        if (
            isinstance(event, AgentStateChanged)
            and event.new_state == AgentState.IDLE
            and event.agent_id in self._pending_initial_prompts
        ):
            msg = self._pending_initial_prompts.pop(event.agent_id)
            session = self._sessions.get(event.agent_id)
            if session:
                self._tasks[f"prompt-{event.agent_id}"] = asyncio.create_task(session.prompt(msg))

    @staticmethod
    def _find_option_id(options: list, decision: PermissionDecision) -> str | None:
        """Map a PermissionDecision to the matching option_id.

        Args:
            options: List of PermissionOption from the SDK.
            decision: The persisted decision.

        Returns:
            The option_id string, or None if no match found.
        """
        for opt in options:
            if opt.kind == decision.value:
                return opt.option_id
        return None

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    async def _launch(self, agent_id: str, *, adhoc_config: AgentConfig | None = None) -> None:
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

        if agent_id in self._sessions:
            old = self._sessions[agent_id]
            if old.state != AgentState.TERMINATED:
                await self._sink(
                    BrokerError(agent_id=agent_id, message=f"Agent '{agent_id}' is still running")
                )
                return
            del self._sessions[agent_id]
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
        self._sessions[agent_id] = session
        self._agent_harnesses[agent_id] = agent_cfg.harness
        self._tasks[agent_id] = asyncio.create_task(session.run())

        if adhoc_config is not None:
            db = await self._ensure_db()
            await ensure_schema_async(db)
            now = int(time.time() * 1000)
            await db.execute(
                "INSERT OR REPLACE INTO agents (agent_id, session_id, status, registered) "
                "VALUES (?, ?, 'active', ?)",
                (agent_id, self._session_id, now),
            )
            await db.commit()

        await self._start_poller()

    async def _prompt(self, agent_id: str, text: str) -> None:
        """Send a prompt to a running agent."""
        session = self._sessions.get(agent_id)
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
        self._tasks[f"prompt-{agent_id}"] = asyncio.create_task(session.prompt(text))

    async def _terminate(self, agent_id: str) -> None:
        """Terminate a running agent session."""
        session = self._sessions.get(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        if session.state != AgentState.TERMINATED:
            await session.terminate()
            # Cancel the run() task and prompt task to kill the subprocess
            for key in (agent_id, f"prompt-{agent_id}"):
                task = self._tasks.get(key)
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, ConnectionError, OSError, RuntimeError):
                        pass

    async def _cancel(self, agent_id: str) -> None:
        """Cancel the active prompt on an agent."""
        session = self._sessions.get(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        await session.cancel()

    async def _set_mode(self, agent_id: str, mode_id: str) -> None:
        """Forward a mode-switch request to the agent session."""
        session = self._sessions.get(agent_id)
        if not session:
            await self._sink(
                BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'")
            )
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

    async def _set_model(self, agent_id: str, model_id: str) -> None:
        """Forward a model-switch request to the agent session."""
        session = self._sessions.get(agent_id)
        if not session:
            await self._sink(
                BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'")
            )
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

    def _resolve_permission(self, agent_id: str, option_id: str) -> None:
        """Resolve a pending permission Future on a session.

        Looks up the pending ``PermissionRequested`` event to determine
        ``tool_kind``.  If the selected option is an *always* variant,
        persists the rule for current-session auto-resolve.

        Args:
            agent_id: The agent whose permission is being resolved.
            option_id: The selected option ID.
        """
        session = self._sessions.get(agent_id)
        if session:
            session.resolve_permission(option_id)

        pending = self._pending_permissions.pop(agent_id, None)
        if not pending:
            return

        # Find the kind of the selected option
        selected_kind: str | None = None
        for opt in pending.options:
            if opt.option_id == option_id:
                selected_kind = opt.kind
                break

        if selected_kind is None:
            log.warning(
                "option_id %r not found in pending options for agent %r — skipping persist",
                option_id,
                agent_id,
            )
            return

        if selected_kind in ("allow_always", "reject_always"):
            self._permission_engine.persist(
                PermissionRule(
                    agent_id=agent_id,
                    tool_kind=pending.kind,
                    session_id=self._session_id,
                    decision=PermissionDecision(selected_kind),
                )
            )

    # ------------------------------------------------------------------
    # Message poller
    # ------------------------------------------------------------------

    async def _ensure_db(self) -> aiosqlite.Connection:
        """Lazily open the shared aiosqlite connection."""
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
        return self._db

    async def _start_poller(self) -> None:
        """Start the message poller if not already running."""
        if self._poller is None:
            await self._register_agents()
            self._poller = MessagePoller(
                self._db_path, self._deliver_message, self._session_id, self._process_commands
            )
            await self._poller.start()

    async def _register_agents(self) -> None:
        """Pre-register all config agents in SQLite so list_agents works immediately."""
        db = await self._ensure_db()
        await ensure_schema_async(db)
        now = int(time.time() * 1000)
        for agent in self._config.agents:
            await db.execute(
                "INSERT OR REPLACE INTO agents (agent_id, session_id, status, registered) "
                "VALUES (?, ?, 'active', ?)",
                (agent.agent_id, self._session_id, now),
            )
        await db.commit()

    def _build_mcp_env(
        self, agent_id: str, extra_env: dict[str, str] | None = None
    ) -> list[EnvVariable]:
        """Build the env var list for an MCP server instance.

        Args:
            agent_id: The agent this MCP server belongs to.
            extra_env: Additional env vars from AgentConfig.env.

        Returns:
            List of EnvVariable entries.
        """
        env = [
            EnvVariable(name="SYNTH_SESSION_ID", value=self._session_id),
            EnvVariable(name="SYNTH_DB_PATH", value=str(self._db_path)),
            EnvVariable(name="SYNTH_AGENT_ID", value=agent_id),
            EnvVariable(
                name="SYNTH_COMMUNICATION_MODE",
                value=self._config.settings.communication_mode.value,
            ),
            EnvVariable(
                name="SYNTH_MAX_AGENTS",
                value=os.environ.get("SYNTH_MAX_AGENTS", "10"),
            ),
        ]
        if extra_env:
            env.extend(EnvVariable(name=k, value=v) for k, v in extra_env.items())
        return env

    # ------------------------------------------------------------------
    # Command processing (CommandFn implementation)
    # ------------------------------------------------------------------

    async def _process_commands(self, commands: list[tuple[int, str, str, str]]) -> None:
        """Process pending agent commands from the poller.

        Args:
            commands: List of (cmd_id, from_agent, command, payload) tuples.
        """
        for cmd_id, from_agent, command, payload in commands:
            try:
                data = json.loads(payload)
                if command == "launch":
                    await self._handle_launch_command(cmd_id, from_agent, data)
                elif command == "terminate":
                    await self._handle_terminate_command(cmd_id, from_agent, data)
                else:
                    await self._update_command_status(cmd_id, "rejected", f"Unknown command: {command}")
            except Exception as exc:
                await self._update_command_status(cmd_id, "rejected", str(exc))

    async def _handle_launch_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a launch command from an agent.

        Args:
            cmd_id: The command row ID.
            from_agent: The agent requesting the launch.
            data: Parsed JSON payload with agent_id, harness, agent_mode, cwd, task, message.
        """
        agent_id = data["agent_id"]
        harness = data["harness"]
        agent_mode = data.get("agent_mode") or None
        cwd = data.get("cwd", ".")
        task = data.get("task", "")
        message = data.get("message", "")

        # Validate agent_id format
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", agent_id):
            await self._update_command_status(
                cmd_id,
                "rejected",
                "Invalid agent_id: must match [a-zA-Z0-9][a-zA-Z0-9_-]*",
            )
            return

        # Check uniqueness
        if agent_id in self._sessions:
            await self._update_command_status(cmd_id, "rejected", f"Agent already exists: {agent_id}")
            return

        # Check global agent limit
        max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
        active = len([s for s in self._sessions.values() if s.state != AgentState.TERMINATED])
        if active >= max_agents:
            return  # Leave as pending

        # Resolve harness
        entry = next((e for e in self._harness_registry if e.short_name == harness), None)
        if not entry:
            await self._update_command_status(cmd_id, "rejected", f"Unknown harness: {harness}")
            return

        cmd = entry.run_cmd.split()
        agent_cfg = AgentConfig(agent_id=agent_id, harness=harness, agent_mode=agent_mode, cwd=cwd)

        # Register in SQLite
        db = await self._ensure_db()
        now = int(time.time() * 1000)
        await db.execute(
            "INSERT OR REPLACE INTO agents (agent_id, session_id, status, registered, parent, task) "
            "VALUES (?, ?, 'active', ?, ?, ?)",
            (agent_id, self._session_id, now, from_agent, task),
        )
        await db.commit()

        # Track parentage
        self._agent_parents[agent_id] = from_agent
        self._agent_harnesses[agent_id] = harness

        # Spawn session
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
        self._sessions[agent_id] = session
        self._tasks[agent_id] = asyncio.create_task(session.run())

        if message:
            self._pending_initial_prompts[agent_id] = message

        await self._update_command_status(cmd_id, "processed")
        await self._send_join_broadcast(agent_id, task)

    async def _handle_terminate_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a terminate command from an agent.

        Args:
            cmd_id: The command row ID.
            from_agent: The agent requesting termination.
            data: Parsed JSON payload with agent_id.
        """
        agent_id = data["agent_id"]

        # Check parentage
        parent = self._agent_parents.get(agent_id)
        if parent != from_agent:
            await self._update_command_status(
                cmd_id,
                "rejected",
                f"Not authorized: {from_agent} is not parent of {agent_id}",
            )
            return

        await self._terminate(agent_id)

        # Mark agent as inactive in SQLite
        db = await self._ensure_db()
        await db.execute(
            "UPDATE agents SET status = 'inactive' WHERE agent_id = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        # Orphan handling: set children's parent to NULL
        await db.execute(
            "UPDATE agents SET parent = NULL WHERE parent = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        # Expire undelivered messages
        await db.execute(
            "UPDATE messages SET status = 'expired' WHERE to_agent = ? AND session_id = ? AND status = 'pending'",
            (agent_id, self._session_id),
        )
        await db.commit()
        for aid, p in self._agent_parents.items():
            if p == agent_id:
                self._agent_parents[aid] = None

        await self._update_command_status(cmd_id, "processed")

    async def _update_command_status(self, cmd_id: int, status: str, error: str | None = None) -> None:
        """Update a command's status in SQLite.

        Args:
            cmd_id: The command row ID.
            status: New status ('processed' or 'rejected').
            error: Error message if rejected.
        """
        db = await self._ensure_db()
        await db.execute(
            "UPDATE agent_commands SET status = ?, error = ? WHERE id = ?",
            (status, error, cmd_id),
        )
        await db.commit()

    async def _get_visible_agents_for(self, agent_id: str) -> list[str]:
        """Compute visible agents for a given agent using SQLite.

        Args:
            agent_id: The agent to compute visibility for.

        Returns:
            List of agent_ids visible to the given agent.
        """
        def _query() -> list[str]:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                return get_visible_agents(
                    conn,
                    agent_id,
                    self._session_id,
                    self._config.settings.communication_mode.value,
                )
            finally:
                conn.close()

        return await asyncio.to_thread(_query)

    async def _send_join_broadcast(self, agent_id: str, task: str) -> None:
        """Insert system join messages for visible agents.

        Args:
            agent_id: The newly joined agent.
            task: The agent's task description.
        """
        recipients = await self._get_visible_agents_for(agent_id)
        if not recipients:
            return

        if task:
            body = f'[System] Agent "{agent_id}" has joined. Task: {task}.'
        else:
            body = f'[System] Agent "{agent_id}" has joined.'

        db = await self._ensure_db()
        now = int(time.time() * 1000)
        for recipient in recipients:
            await db.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
                "VALUES (?, 'system', ?, ?, 'pending', ?, 'system')",
                (self._session_id, recipient, body, now),
            )
        await db.commit()

    async def _deliver_message(self, agent_id: str, text: str, from_agents: list[str]) -> bool:
        """Deliver combined message text to an idle agent.

        Args:
            agent_id: Target agent ID.
            text: Combined message text.
            from_agents: List of unique sender agent IDs.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        session = self._sessions.get(agent_id)
        if not session or session.state != AgentState.IDLE:
            return False
        try:
            for sender in from_agents:
                await self._event_queue.put(
                    McpMessageDelivered(
                        agent_id=agent_id,
                        from_agent=sender,
                        to_agent=agent_id,
                        preview=text,
                    )
                )
            await session.prompt(text)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Event stream
    # ------------------------------------------------------------------

    async def events(self) -> AsyncIterator[BrokerEvent]:
        """Yield events until shutdown."""
        while not self._shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except TimeoutError:
                continue
        # Drain remaining events
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel prompts, stop poller, persist, terminate."""
        self._shutting_down = True

        # 1. Cancel active prompts and pending permissions
        for session in self._sessions.values():
            if session.state == AgentState.BUSY:
                await session.cancel()
            elif session.state == AgentState.AWAITING_PERMISSION:
                await session.terminate()

        # 2. Stop poller (await current cycle)
        if self._poller:
            await self._poller.stop()

        # 3. Close shared DB connection
        if self._db is not None:
            await self._db.close()
            self._db = None

        # 3. Persist session IDs
        sessions_path = Path.home() / ".synth" / "sessions.json"
        sessions_path.parent.mkdir(parents=True, exist_ok=True)
        session_ids = {
            aid: s._session_id
            for aid, s in self._sessions.items()
            if s._session_id and s.state != AgentState.TERMINATED
        }
        sessions_path.write_text(json.dumps(session_ids))

        # 4. Terminate all sessions
        for session in self._sessions.values():
            if session.state != AgentState.TERMINATED:
                await session.terminate()

        for task in self._tasks.values():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, ConnectionError, OSError, TimeoutError):
                pass

        self._shutdown_event.set()
