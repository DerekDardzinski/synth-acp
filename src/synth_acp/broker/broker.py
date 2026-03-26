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

from acp.schema import EnvVariable, McpServerStdio

from synth_acp.acp.session import ACPSession
from synth_acp.broker.permissions import PermissionEngine
from synth_acp.broker.poller import MessagePoller
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.commands import (
    BrokerCommand,
    CancelTurn,
    LaunchAgent,
    RespondPermission,
    SendPrompt,
    TerminateAgent,
)
from synth_acp.models.config import CommunicationMode, SessionConfig
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
        self._agent_parents: dict[str, str | None] = {a.id: None for a in config.agents}

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def handle(self, command: BrokerCommand) -> None:
        """Dispatch a command to the appropriate handler.

        Args:
            command: The broker command to dispatch.
        """
        match command:
            case LaunchAgent(agent_id=aid):
                await self._launch(aid)
            case TerminateAgent(agent_id=aid):
                await self._terminate(aid)
            case SendPrompt(agent_id=aid, text=text):
                await self._prompt(aid, text)
            case RespondPermission(agent_id=aid, option_id=oid):
                self._resolve_permission(aid, oid)
            case CancelTurn(agent_id=aid):
                await self._cancel(aid)

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

    async def _launch(self, agent_id: str) -> None:
        """Launch an agent by ID from the config."""
        agent_cfg = next((a for a in self._config.agents if a.id == agent_id), None)
        if not agent_cfg:
            await self._sink(
                BrokerError(agent_id=agent_id, message=f"No config for agent '{agent_id}'")
            )
            return

        mcp_servers = [
            McpServerStdio(
                name="synth-mcp",
                command="synth-mcp",
                args=[],
                env=self._build_mcp_env(agent_id, agent_cfg.env),
            )
        ]

        session = ACPSession(
            agent_id=agent_cfg.id,
            binary=agent_cfg.binary,
            args=agent_cfg.args,
            cwd=agent_cfg.cwd,
            event_sink=self._sink,
            mcp_servers=mcp_servers,
        )
        self._sessions[agent_id] = session
        self._tasks[agent_id] = asyncio.create_task(session.run())
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
            # Cancel the run() task to kill the subprocess
            task = self._tasks.get(agent_id)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _cancel(self, agent_id: str) -> None:
        """Cancel the active prompt on an agent."""
        session = self._sessions.get(agent_id)
        if not session:
            await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
            return
        await session.cancel()

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

    async def _start_poller(self) -> None:
        """Start the message poller if not already running."""
        if self._poller is None:
            self._register_agents()
            self._poller = MessagePoller(
                self._db_path, self._deliver_message, self._session_id, self._process_commands
            )
            await self._poller.start()

    def _register_agents(self) -> None:
        """Pre-register all config agents in SQLite so list_agents works immediately."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agents ("
            "agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'active', registered INTEGER NOT NULL, "
            "parent TEXT, task TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_commands ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "from_agent TEXT NOT NULL, command TEXT NOT NULL, payload TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'pending', error TEXT, "
            "created_at INTEGER NOT NULL)"
        )
        now = int(time.time() * 1000)
        for agent in self._config.agents:
            conn.execute(
                "INSERT OR REPLACE INTO agents (agent_id, session_id, status, registered) "
                "VALUES (?, ?, 'active', ?)",
                (agent.id, self._session_id, now),
            )
        conn.commit()
        conn.close()

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
                    self._update_command_status(cmd_id, "rejected", f"Unknown command: {command}")
            except Exception as exc:
                self._update_command_status(cmd_id, "rejected", str(exc))

    async def _handle_launch_command(
        self, cmd_id: int, from_agent: str, data: dict[str, str]
    ) -> None:
        """Handle a launch command from an agent.

        Args:
            cmd_id: The command row ID.
            from_agent: The agent requesting the launch.
            data: Parsed JSON payload with agent_id, agent_name, harness, cwd, task, message.
        """
        agent_id = data["agent_id"]
        agent_name = data["agent_name"]
        harness = data["harness"]
        cwd = data.get("cwd", ".")
        task = data.get("task", "")
        message = data.get("message", "")

        # Validate agent_id format
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", agent_id):
            self._update_command_status(
                cmd_id,
                "rejected",
                "Invalid agent_id: must match [a-zA-Z0-9][a-zA-Z0-9_-]*",
            )
            return

        # Check uniqueness
        if agent_id in self._sessions:
            self._update_command_status(cmd_id, "rejected", f"Agent already exists: {agent_id}")
            return

        # Check global agent limit
        max_agents = int(os.environ.get("SYNTH_MAX_AGENTS", "10"))
        active = len([s for s in self._sessions.values() if s.state != AgentState.TERMINATED])
        if active >= max_agents:
            return  # Leave as pending

        # Resolve harness
        registry = load_harness_registry()
        entry = next((e for e in registry if e.short_name == harness), None)
        if not entry:
            self._update_command_status(cmd_id, "rejected", f"Unknown harness: {harness}")
            return

        cmd = entry.run_cmd_with_agent.format(agent=agent_name).split()
        agent_cfg = AgentConfig(id=agent_id, cmd=cmd, cwd=cwd)

        # Register in SQLite
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT OR REPLACE INTO agents (agent_id, session_id, status, registered, parent, task) "
            "VALUES (?, ?, 'active', ?, ?, ?)",
            (agent_id, self._session_id, now, from_agent, task),
        )
        conn.commit()
        conn.close()

        # Track parentage
        self._agent_parents[agent_id] = from_agent

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
            agent_id=agent_cfg.id,
            binary=agent_cfg.binary,
            args=agent_cfg.args,
            cwd=agent_cfg.cwd,
            event_sink=self._sink,
            mcp_servers=mcp_servers,
        )
        self._sessions[agent_id] = session
        self._tasks[agent_id] = asyncio.create_task(session.run())

        if message:
            self._pending_initial_prompts[agent_id] = message

        self._update_command_status(cmd_id, "processed")

        # Join broadcast
        self._send_join_broadcast(agent_id, task)

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
            self._update_command_status(
                cmd_id,
                "rejected",
                f"Not authorized: {from_agent} is not parent of {agent_id}",
            )
            return

        await self._terminate(agent_id)

        # Mark agent as inactive in SQLite
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "UPDATE agents SET status = 'inactive' WHERE agent_id = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        # Orphan handling: set children's parent to NULL
        conn.execute(
            "UPDATE agents SET parent = NULL WHERE parent = ? AND session_id = ?",
            (agent_id, self._session_id),
        )
        conn.commit()
        conn.close()
        for aid, p in self._agent_parents.items():
            if p == agent_id:
                self._agent_parents[aid] = None

        self._update_command_status(cmd_id, "processed")

    def _update_command_status(self, cmd_id: int, status: str, error: str | None = None) -> None:
        """Update a command's status in SQLite.

        Args:
            cmd_id: The command row ID.
            status: New status ('processed' or 'rejected').
            error: Error message if rejected.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "UPDATE agent_commands SET status = ?, error = ? WHERE id = ?",
            (status, error, cmd_id),
        )
        conn.commit()
        conn.close()

    def _get_visible_agents_for(self, agent_id: str) -> list[str]:
        """Compute visible agents for a given agent using in-memory parentage.

        Args:
            agent_id: The agent to compute visibility for.

        Returns:
            List of agent_ids visible to the given agent.
        """
        active = {
            aid
            for aid, s in self._sessions.items()
            if s.state != AgentState.TERMINATED and aid != agent_id
        }
        if self._config.settings.communication_mode == CommunicationMode.MESH:
            return list(active)

        # LOCAL mode: parent, children, siblings
        parent = self._agent_parents.get(agent_id)
        visible: set[str] = set()
        if parent and parent in active:
            visible.add(parent)
        # Children
        for aid, p in self._agent_parents.items():
            if p == agent_id and aid in active:
                visible.add(aid)
        # Siblings (same parent, excluding self)
        if parent:
            for aid, p in self._agent_parents.items():
                if p == parent and aid in active and aid != agent_id:
                    visible.add(aid)
        return list(visible)

    def _send_join_broadcast(self, agent_id: str, task: str) -> None:
        """Insert system join messages for visible agents.

        Args:
            agent_id: The newly joined agent.
            task: The agent's task description.
        """
        recipients = self._get_visible_agents_for(agent_id)
        if not recipients:
            return

        if task:
            body = f'[System] Agent "{agent_id}" has joined. Task: {task}.'
        else:
            body = f'[System] Agent "{agent_id}" has joined.'

        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        now = int(time.time() * 1000)
        for recipient in recipients:
            conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
                "VALUES (?, 'system', ?, ?, 'pending', ?)",
                (self._session_id, recipient, body, now),
            )
        conn.commit()
        conn.close()

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
            await session.prompt(text)
            for sender in from_agents:
                await self._event_queue.put(
                    McpMessageDelivered(
                        agent_id=agent_id,
                        from_agent=sender,
                        to_agent=agent_id,
                        preview=text,
                    )
                )
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
