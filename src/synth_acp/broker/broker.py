"""ACPBroker — thin coordinator for agent sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from synth_acp.broker.lifecycle import AgentLifecycle
from synth_acp.broker.message_bus import MessageBus
from synth_acp.broker.permissions import PermissionEngine
from synth_acp.broker.registry import AgentRegistry
from synth_acp.db import ensure_schema_async
from synth_acp.models.agent import AgentConfig, AgentMode, AgentModel, AgentState
from synth_acp.models.commands import (
    BrokerCommand,
    CancelTurn,
    LaunchAgent,
    RespondPermission,
    RestoreSession,
    SendPrompt,
    SetAgentMode,
    SetAgentModel,
    TerminateAgent,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    BrokerEvent,
    HookFired,
    InitialPromptDelivered,
    McpMessageDelivered,
    MessageChunkReceived,
    PermissionAutoResolved,
    PermissionRequested,
    UsageUpdated,
    UserPromptSubmitted,
)
from synth_acp.models.permissions import PermissionDecision, PermissionRule

log = logging.getLogger(__name__)


class ACPBroker:
    """Central orchestration service for agent sessions."""

    def __init__(
        self,
        config: SessionConfig,
        db_path: Path | None = None,
        event_queue_maxsize: int = 2000,
    ) -> None:
        self._config = config
        self._db_path = db_path or Path.home() / ".synth" / "synth.db"
        self._session_id = f"{config.project}-{uuid.uuid4().hex[:8]}"
        self._event_queue: asyncio.Queue[BrokerEvent] = asyncio.Queue(maxsize=event_queue_maxsize)
        self._shutdown_event = asyncio.Event()
        self._shutting_down = False
        self._permission_engine = PermissionEngine(
            db_path=self._db_path,
            session_id=self._session_id,
        )
        self._pending_permissions: dict[str, PermissionRequested] = {}  # keyed by request_id
        self._active_permission: dict[str, str] = {}  # agent_id → active request_id
        self._permission_queue: dict[str, list[PermissionRequested]] = {}  # agent_id → queued events
        self._permission_counter: dict[str, tuple[int, int]] = {}  # agent_id → (current, total)
        self._registry = AgentRegistry(config)
        self._message_bus: MessageBus | None = None
        self._lifecycle: AgentLifecycle | None = None
        self._expired: bool = False
        self._journal_seq: dict[str, int] = {}  # agent_id → next sequence number

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def handle(self, command: BrokerCommand) -> None:
        """Dispatch a command to the appropriate handler."""
        lifecycle = await self._ensure_lifecycle()
        match command:
            case LaunchAgent(agent_id=aid, config=cfg):
                await self._start_message_bus()
                await lifecycle.launch(aid, adhoc_config=cfg)
            case TerminateAgent(agent_id=aid):
                await lifecycle.terminate(aid)
            case SendPrompt(agent_id=aid, text=text):
                await self._sink(UserPromptSubmitted(agent_id=aid, text=text))
                await lifecycle.prompt(aid, text)
            case RespondPermission(agent_id=aid, request_id=rid, option_id=oid):
                await self._resolve_permission(aid, rid, oid)
            case CancelTurn(agent_id=aid):
                await lifecycle.cancel(aid)
            case SetAgentMode(agent_id=aid, mode_id=mid):
                await lifecycle.set_mode(aid, mid)
            case SetAgentModel(agent_id=aid, model_id=mid):
                await lifecycle.set_model(aid, mid)
            case RestoreSession(broker_session_id=sid):
                await self.restore_session(sid)

    # ------------------------------------------------------------------
    # State queries (thin delegations to registry)
    # ------------------------------------------------------------------

    def get_agent_states(self) -> dict[str, AgentState]:
        return self._registry.get_states()

    def get_agent_configs(self) -> list[AgentConfig]:
        return self._registry.get_configs()

    def get_usage(self, agent_id: str) -> UsageUpdated | None:
        return self._registry.get_usage(agent_id)

    def get_agent_parent(self, agent_id: str) -> str | None:
        return self._registry.get_parent(agent_id)

    def get_agent_harness(self, agent_id: str) -> str:
        return self._registry.get_harness(agent_id)

    def get_agent_modes(self, agent_id: str) -> list[AgentMode]:
        return self._registry.get_modes(agent_id)

    def get_current_mode(self, agent_id: str) -> str | None:
        return self._registry.get_current_mode(agent_id)

    def get_agent_models(self, agent_id: str) -> list[AgentModel]:
        return self._registry.get_models(agent_id)

    def get_current_model(self, agent_id: str) -> str | None:
        return self._registry.get_current_model(agent_id)

    def is_permission_pending(self, agent_id: str) -> bool:
        return any(p.agent_id == agent_id for p in self._pending_permissions.values())

    def permission_position(self, agent_id: str) -> str:
        """Return a position string like '1 of 3' for the active permission, or ''."""
        counter = self._permission_counter.get(agent_id)
        if not counter or counter[1] <= 1:
            return ""
        return f"{counter[0]} of {counter[1]}"

    # ------------------------------------------------------------------
    # Session restore
    # ------------------------------------------------------------------

    async def restore_session(self, broker_session_id: str) -> None:
        """Restore agents from a previous session."""
        self._session_id = broker_session_id
        self._permission_engine._session_id = broker_session_id
        # Sync an already-constructed lifecycle so its DB writes use the restored
        # session_id rather than the ephemeral one captured at construction.
        if self._lifecycle is not None:
            self._lifecycle._session_id = broker_session_id

        def _query_restorable() -> list[sqlite3.Row]:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                return conn.execute(
                    "SELECT agent_id, acp_session_id, harness, agent_mode, cwd, parent "
                    "FROM agents WHERE session_id = ? AND status = 'restorable' "
                    "ORDER BY parent NULLS FIRST",
                    (broker_session_id,),
                ).fetchall()
            finally:
                conn.close()

        rows = await asyncio.to_thread(_query_restorable)

        # Agents with an acp_session_id but no journal events have no
        # conversation history — load_session will fail for them.  Clear
        # the id so restore() takes the fresh-launch path directly.
        def _agents_with_history() -> set[str]:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                return {
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT agent_id FROM ui_events WHERE session_id = ?",
                        (broker_session_id,),
                    ).fetchall()
                }
            finally:
                conn.close()

        has_history = await asyncio.to_thread(_agents_with_history)

        # Start message bus without register_agents — rows already exist in SQLite.
        await self._start_message_bus(skip_register=True)
        lifecycle = await self._ensure_lifecycle()

        for row in rows:
            aid = row["agent_id"]
            await lifecycle.restore(
                agent_id=aid,
                acp_session_id=row["acp_session_id"] if aid in has_history else None,
                harness=row["harness"],
                agent_mode=row["agent_mode"],
                cwd=row["cwd"],
                parent=row["parent"],
            )
            if row["parent"]:
                self._registry.set_parent(aid, row["parent"])

        # Initialize journal seq counters from existing DB state so new
        # events don't collide with the original session's journal entries.
        try:
            def _query_journal_seq() -> list[tuple]:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    return conn.execute(
                        "SELECT agent_id, MAX(seq) FROM ui_events "
                        "WHERE session_id = ? GROUP BY agent_id",
                        (broker_session_id,),
                    ).fetchall()
                finally:
                    conn.close()

            for aid, max_seq in await asyncio.to_thread(_query_journal_seq):
                self._journal_seq[aid] = max_seq + 1
        except Exception:
            log.debug("Failed to init journal seq counters", exc_info=True)

    @staticmethod
    async def list_restorable_sessions(db_path: Path) -> list[dict]:
        """Return restorable sessions grouped by session_id."""
        def _query() -> list[dict]:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                rows = conn.execute(
                    "SELECT session_id, GROUP_CONCAT(agent_id) as agents, "
                    "MAX(registered) as last_active, COUNT(*) as agent_count "
                    "FROM agents WHERE status = 'restorable' "
                    "GROUP BY session_id ORDER BY MAX(registered) DESC"
                ).fetchall()
                return [
                    {
                        "session_id": r[0],
                        "agents": r[1].split(","),
                        "last_active": r[2],
                        "agent_count": r[3],
                    }
                    for r in rows
                ]
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_query)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Event sink with permission interception + backpressure
    # ------------------------------------------------------------------

    async def _sink(self, event: BrokerEvent) -> None:
        """Event sink passed to sessions. Intercepts permissions, applies backpressure."""
        if isinstance(event, PermissionRequested):
            self._pending_permissions[event.request_id] = event
            # Auto-approve if the tool matches a configured pattern
            if self._should_auto_approve(event):
                session = self._registry.get_session(event.agent_id)
                if session:
                    option_id = self._find_allow_once(event.options)
                    if option_id:
                        session.resolve_permission(event.request_id, option_id)
                        self._pending_permissions.pop(event.request_id, None)
                        await self._event_queue.put(
                            PermissionAutoResolved(
                                agent_id=event.agent_id,
                                request_id=event.request_id,
                                decision=PermissionDecision.allow_once,
                            )
                        )
                        return
            # Auto-resolve if a persisted rule matches
            decision = self._permission_engine.check(event.agent_id, event.kind, self._session_id)
            if decision is not None:
                session = self._registry.get_session(event.agent_id)
                if session:
                    option_id = self._find_option_id(event.options, decision)
                    if option_id:
                        session.resolve_permission(event.request_id, option_id)
                        self._pending_permissions.pop(event.request_id, None)
                        await self._event_queue.put(
                            PermissionAutoResolved(
                                agent_id=event.agent_id,
                                request_id=event.request_id,
                                decision=decision,
                            )
                        )
                        return
            # Show one permission bar at a time per agent; queue the rest
            aid = event.agent_id
            cur, total = self._permission_counter.get(aid, (0, 0))
            if aid in self._active_permission:
                self._permission_queue.setdefault(aid, []).append(event)
                self._permission_counter[aid] = (cur, total + 1)
                return
            self._active_permission[aid] = event.request_id
            self._permission_counter[aid] = (1, total + 1)
        elif isinstance(event, UsageUpdated):
            self._registry.update_usage(event)

        if isinstance(event, AgentStateChanged) and event.new_state == AgentState.TERMINATED:
            self._cleanup_agent_state(event.agent_id)

        if isinstance(event, MessageChunkReceived):
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                log.debug("Event queue full, dropping chunk for %s", event.agent_id)
        elif self._shutting_down:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        else:
            await self._event_queue.put(event)

        # Journal UI-visible events for session restore.
        await self._journal_event(event)

        if isinstance(event, AgentStateChanged) and event.new_state == AgentState.IDLE:
            if self._message_bus and self._lifecycle:
                pending = self._message_bus.pop_pending(event.agent_id)
                if pending:
                    original = self._registry.pop_initial_message(event.agent_id)
                    if original:
                        parent = self._registry.get_parent(event.agent_id)
                        await self._sink(
                            InitialPromptDelivered(
                                agent_id=event.agent_id,
                                from_agent=parent or "system",
                                text=original,
                            )
                        )
                        await self._sink(
                            HookFired(agent_id=event.agent_id, hook_name="on_agent_prompt")
                        )
                    await self._lifecycle.prompt(event.agent_id, pending)

    # ------------------------------------------------------------------
    # Agent state cleanup
    # ------------------------------------------------------------------

    def _cleanup_agent_state(self, agent_id: str) -> None:
        """Remove accumulated per-agent state for a terminated agent."""
        self._pending_permissions = {
            k: v for k, v in self._pending_permissions.items() if v.agent_id != agent_id
        }
        self._active_permission.pop(agent_id, None)
        self._permission_queue.pop(agent_id, None)
        self._permission_counter.pop(agent_id, None)
        self._journal_seq.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Event journal for session restore
    # ------------------------------------------------------------------

    _JOURNALABLE = frozenset({
        "MessageChunkReceived",
        "AgentThoughtReceived",
        "ToolCallUpdated",
        "TurnComplete",
        "HookFired",
        "InitialPromptDelivered",
        "McpMessageDelivered",
        "PlanReceived",
        "UserPromptSubmitted",
    })

    async def _journal_event(self, event: BrokerEvent) -> None:
        """Persist a UI-visible event to the journal table."""
        event_type = type(event).__name__
        if event_type not in self._JOURNALABLE:
            return
        if self._lifecycle is None or self._lifecycle._db is None:
            return
        try:
            db = self._lifecycle._db
            seq = self._journal_seq.get(event.agent_id, 0)
            self._journal_seq[event.agent_id] = seq + 1
            now = int(time.time() * 1000)
            payload = event.model_dump_json()
            await db.execute(
                "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._session_id, event.agent_id, seq, event_type, payload, now),
            )
            await db.commit()
        except Exception:
            log.debug("Failed to journal event %s for %s", event_type, event.agent_id, exc_info=True)

    async def load_journal(self, agent_id: str, session_id: str) -> list[BrokerEvent]:
        """Load journaled events for an agent from SQLite.

        Returns deserialized BrokerEvent objects in sequence order.
        The caller decides how to deliver them (buffer, queue, etc.).
        """
        from synth_acp.models import events as ev

        result: list[BrokerEvent] = []
        try:
            def _query() -> list[tuple[str, str]]:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    return conn.execute(
                        "SELECT event_type, payload FROM ui_events "
                        "WHERE session_id = ? AND agent_id = ? ORDER BY seq",
                        (session_id, agent_id),
                    ).fetchall()
                finally:
                    conn.close()

            rows = await asyncio.to_thread(_query)

            for event_type, payload in rows:
                cls = getattr(ev, event_type, None)
                if cls is None:
                    continue
                try:
                    result.append(cls.model_validate_json(payload))
                except Exception:
                    log.debug("Failed to deserialize journal event %s", event_type, exc_info=True)
        except Exception:
            log.debug("Journal load failed for %s", agent_id, exc_info=True)
        return result

    @staticmethod
    def _find_option_id(options: list, decision: PermissionDecision) -> str | None:
        for opt in options:
            if opt.kind == decision.value:
                return opt.option_id
        return None

    def _should_auto_approve(self, event: PermissionRequested) -> bool:
        """Check if the tool in the permission title matches an auto-approve pattern."""
        patterns = self._config.settings.auto_approve_tools
        if not patterns:
            return False
        title = event.title
        return any(pattern in title for pattern in patterns)

    @staticmethod
    def _find_allow_once(options: list) -> str | None:
        for opt in options:
            if opt.kind == "allow_once":
                return opt.option_id
        return None

    # ------------------------------------------------------------------
    # Permission resolution
    # ------------------------------------------------------------------

    async def _resolve_permission(self, agent_id: str, request_id: str, option_id: str) -> None:
        """Resolve a pending permission Future on a session, then show the next queued one."""
        session = self._registry.get_session(agent_id)
        if session:
            session.resolve_permission(request_id, option_id)

        pending = self._pending_permissions.pop(request_id, None)
        if not pending:
            return

        selected_kind: str | None = None
        for opt in pending.options:
            if opt.option_id == option_id:
                selected_kind = opt.kind
                break

        if selected_kind is None:
            log.warning("option_id %r not found for agent %r", option_id, agent_id)
            self._active_permission.pop(agent_id, None)
            await self._flush_permission_queue(agent_id)
            return

        if selected_kind in ("allow_always", "reject_always"):
            await self._permission_engine.persist_async(
                PermissionRule(
                    agent_id=agent_id,
                    tool_kind=pending.kind,
                    session_id=self._session_id,
                    decision=PermissionDecision(selected_kind),
                )
            )

        # Release the active slot and show the next queued permission
        self._active_permission.pop(agent_id, None)
        await self._flush_permission_queue(agent_id)

    async def _flush_permission_queue(self, agent_id: str) -> None:
        """Forward the next queued permission for this agent to the UI.

        Auto-resolves queued permissions that match persisted rules,
        draining until one needs manual resolution or the queue is empty.
        """
        queue = self._permission_queue.get(agent_id)
        while queue:
            nxt = queue.pop(0)
            if not queue:
                del self._permission_queue[agent_id]
            # Try auto-approve by tool pattern
            if self._should_auto_approve(nxt):
                session = self._registry.get_session(nxt.agent_id)
                if session:
                    option_id = self._find_allow_once(nxt.options)
                    if option_id:
                        session.resolve_permission(nxt.request_id, option_id)
                        self._pending_permissions.pop(nxt.request_id, None)
                        await self._event_queue.put(
                            PermissionAutoResolved(
                                agent_id=nxt.agent_id,
                                request_id=nxt.request_id,
                                decision=PermissionDecision.allow_once,
                            )
                        )
                        cur, total = self._permission_counter.get(agent_id, (1, 1))
                        self._permission_counter[agent_id] = (cur + 1, total)
                        continue
            # Try auto-resolve by persisted rule
            decision = self._permission_engine.check(nxt.agent_id, nxt.kind, self._session_id)
            if decision is not None:
                session = self._registry.get_session(nxt.agent_id)
                if session:
                    option_id = self._find_option_id(nxt.options, decision)
                    if option_id:
                        session.resolve_permission(nxt.request_id, option_id)
                        self._pending_permissions.pop(nxt.request_id, None)
                        await self._event_queue.put(
                            PermissionAutoResolved(
                                agent_id=nxt.agent_id,
                                request_id=nxt.request_id,
                                decision=decision,
                            )
                        )
                        cur, total = self._permission_counter.get(agent_id, (1, 1))
                        self._permission_counter[agent_id] = (cur + 1, total)
                        continue
            # Needs manual resolution — forward to UI
            self._active_permission[agent_id] = nxt.request_id
            cur, total = self._permission_counter.get(agent_id, (1, 1))
            self._permission_counter[agent_id] = (cur + 1, total)
            await self._event_queue.put(nxt)
            return
        # Queue fully drained
        self._permission_counter.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Lifecycle + message bus wiring
    # ------------------------------------------------------------------

    async def _ensure_lifecycle(self) -> AgentLifecycle:
        """Return the lifecycle, creating it if needed."""
        if self._lifecycle is None:
            self._lifecycle = AgentLifecycle(
                config=self._config,
                registry=self._registry,
                event_sink=self._sink,
                db_path=self._db_path,
                session_id=self._session_id,
            )
        return self._lifecycle

    async def _start_message_bus(self, *, skip_register: bool = False) -> None:
        if self._message_bus is None:
            lifecycle = await self._ensure_lifecycle()
            if skip_register:
                # Schema must exist for the message bus even without registration.
                db = await lifecycle._ensure_db()
                await ensure_schema_async(db)
            else:
                await lifecycle.register_agents()
            if not self._expired:
                self._expired = True
                await lifecycle.expire_old_sessions()
            self._message_bus = MessageBus(
                self._db_path, self._session_id, self._deliver_message, self._process_commands
            )
            await self._message_bus.start()
            lifecycle.set_message_bus(self._message_bus.socket_path, self._message_bus.enqueue_pending, self._message_bus.enqueue_raw)

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------

    async def _process_commands(self, commands: list[tuple[int, str, str, str]]) -> None:
        lifecycle = await self._ensure_lifecycle()
        for cmd_id, from_agent, command, payload in commands:
            try:
                data = json.loads(payload)
                if command == "launch":
                    await lifecycle.handle_launch_command(cmd_id, from_agent, data)
                elif command == "terminate":
                    await lifecycle.handle_terminate_command(cmd_id, from_agent, data)
                else:
                    await lifecycle.update_command_status(cmd_id, "rejected", f"Unknown command: {command}")
            except Exception as exc:
                await lifecycle.update_command_status(cmd_id, "rejected", str(exc))

    async def _deliver_message(self, agent_id: str, text: str, from_agents: list[str]) -> bool:
        """Deliver a message to an agent. Non-blocking — dispatches prompt as a task."""
        session = self._registry.get_session(agent_id)
        if not session or session.state != AgentState.IDLE:
            return False
        try:
            for sender in from_agents:
                await self._sink(
                    McpMessageDelivered(agent_id=agent_id, from_agent=sender, to_agent=agent_id, preview=text)
                )
            if self._lifecycle:
                await self._lifecycle.prompt(agent_id, text)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Event stream
    # ------------------------------------------------------------------

    async def events(self) -> AsyncIterator[BrokerEvent]:
        while not self._shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except TimeoutError:
                continue
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        self._shutting_down = True

        try:
            if self._lifecycle:
                await self._lifecycle.shutdown()
        except Exception:
            log.debug("Lifecycle shutdown error", exc_info=True)

        try:
            if self._lifecycle:
                await self._lifecycle.mark_agents_restorable()
        except Exception:
            log.debug("mark_agents_restorable error", exc_info=True)

        try:
            if self._message_bus:
                await self._message_bus.stop()
        except Exception:
            log.debug("MessageBus stop error", exc_info=True)

        try:
            if self._lifecycle:
                await self._lifecycle.close_db()
        except Exception:
            log.debug("close_db error", exc_info=True)

        # Backward-compat sessions.json
        sessions_path = Path.home() / ".synth" / "sessions.json"
        sessions_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        session_ids = {
            aid: s.session_id
            for aid, s in self._registry.all_sessions().items()
            if s.session_id and s.state == AgentState.TERMINATED
        }
        fd, tmp = tempfile.mkstemp(dir=sessions_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(session_ids, f)
            Path(tmp).rename(sessions_path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)

        self._shutdown_event.set()
