"""ACPBroker — owns all agent sessions and routes events."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from acp.schema import EnvVariable, McpServerStdio

from synth_acp.acp.session import ACPSession
from synth_acp.broker.permissions import PermissionEngine
from synth_acp.broker.poller import MessagePoller
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.commands import (
    BrokerCommand,
    CancelTurn,
    LaunchAgent,
    RespondPermission,
    SendPrompt,
    TerminateAgent,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
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
                env=[
                    EnvVariable(name="SYNTH_SESSION_ID", value=self._session_id),
                    EnvVariable(name="SYNTH_DB_PATH", value=str(self._db_path)),
                    EnvVariable(name="SYNTH_AGENT_ID", value=agent_id),
                ],
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
            self._poller = MessagePoller(self._db_path, self._deliver_message, self._session_id)
            await self._poller.start()

    def _register_agents(self) -> None:
        """Pre-register all config agents in SQLite so list_agents works immediately."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agents ("
            "agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'active', registered INTEGER NOT NULL)"
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
