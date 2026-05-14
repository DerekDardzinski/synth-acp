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
from synth_acp.db import ensure_schema_sync
from synth_acp.models.agent import AgentConfig, AgentMode, AgentModel, AgentState
from synth_acp.models.commands import (
    BrokerCommand,
    CancelTurn,
    HoldDelivery,
    LaunchAgent,
    ReleaseDelivery,
    RespondPermission,
    RestoreSession,
    ResurrectAgent,
    SendPrompt,
    SetAgentMode,
    SetAgentModel,
    SetConfigOption,
    TerminateAgent,
)
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerError,
    BrokerEvent,
    McpMessageDelivered,
    McpMessageHeld,
    MessageChunkReceived,
    PermissionAutoResolved,
    PermissionRequested,
    ToolCallUpdated,
    TurnComplete,
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
        initial_agent: AgentConfig,
        db_path: Path | None = None,
        event_queue_maxsize: int = 2000,
    ) -> None:
        self._config = config
        self._initial_agent = initial_agent
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
        self._registry = AgentRegistry()
        self._message_bus: MessageBus | None = None
        self._message_bus_starting: bool = False
        self._lifecycle: AgentLifecycle | None = None
        self._expired: bool = False
        self._journal_seq: dict[str, int] = {}  # agent_id → next sequence number
        self._turn_buffer: dict[str, list[tuple[int, BrokerEvent]]] = {}
        self._turn_buffer_tool_index: dict[str, dict[str, int]] = {}
        self._pending_flushes: set[asyncio.Task] = set()
        self._delivery_held: set[str] = set()

    @property
    def session_id(self) -> str:
        """The current broker session ID."""
        return self._session_id

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def handle(self, command: BrokerCommand) -> None:
        """Dispatch a command to the appropriate handler."""
        lifecycle = await self._ensure_lifecycle()
        match command:
            case LaunchAgent(agent_id=aid, config=cfg):
                if cfg is None:
                    await self._sink(BrokerError(agent_id=aid, message=f"LaunchAgent requires config for '{aid}'"))
                else:
                    await self._start_message_bus()
                    await lifecycle.launch(aid, adhoc_config=cfg)
            case TerminateAgent(agent_id=aid):
                await lifecycle.terminate(aid)
            case ResurrectAgent(agent_id=aid):
                await self._start_message_bus()
                await lifecycle.resurrect(aid)
            case SendPrompt(agent_id=aid, text=text):
                await self._sink(UserPromptSubmitted(agent_id=aid, text=text))
                await lifecycle.prompt(aid, text)
            case RespondPermission(agent_id=aid, request_id=rid, option_id=oid):
                await self._resolve_permission(aid, rid, oid)
            case CancelTurn(agent_id=aid):
                await lifecycle.cancel(aid)
            case SetAgentMode(agent_id=aid, mode_id=mid):
                await lifecycle.set_config_option(aid, "mode", mid)
            case SetAgentModel(agent_id=aid, model_id=mid):
                await lifecycle.set_config_option(aid, "model", mid)
            case SetConfigOption(agent_id=aid, config_id=cid, value=val):
                await lifecycle.set_config_option(aid, cid, val)
            case RestoreSession(broker_session_id=sid):
                await self.restore_session(sid)
            case HoldDelivery(agent_id=aid):
                self._delivery_held.add(aid)
            case ReleaseDelivery(agent_id=aid):
                self._delivery_held.discard(aid)

    # ------------------------------------------------------------------
    # State queries (thin delegations to registry)
    # ------------------------------------------------------------------

    def get_agent_states(self) -> dict[str, AgentState]:
        return self._registry.get_states()

    def get_usage(self, agent_id: str) -> UsageUpdated | None:
        return self._registry.get_usage(agent_id)

    def get_agent_parent(self, agent_id: str) -> str | None:
        return self._registry.get_parent(agent_id)

    def get_agent_harness(self, agent_id: str) -> str:
        return self._registry.get_harness(agent_id)

    def get_agent_cwd(self, agent_id: str) -> str:
        return self._registry.get_cwd(agent_id)

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
                    "FROM agents WHERE session_id = ? AND status IN ('restorable', 'active') "
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
        await self._start_message_bus()
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
        """Return restorable sessions grouped by session_id with enriched metadata."""
        def _query() -> list[dict]:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                # Get sessions that have at least one restorable/active agent
                sessions = conn.execute(
                    "SELECT session_id, MAX(registered) as last_active, "
                    "COUNT(*) as agent_count "
                    "FROM agents WHERE status IN ('restorable', 'active') "
                    "GROUP BY session_id ORDER BY MAX(registered) DESC"
                ).fetchall()

                if not sessions:
                    return []

                sids = [s["session_id"] for s in sessions]
                placeholders = ",".join("?" * len(sids))

                # Bulk: all agents for these sessions
                all_agents: dict[str, list[str]] = {sid: [] for sid in sids}
                for r in conn.execute(
                    f"SELECT session_id, agent_id FROM agents WHERE session_id IN ({placeholders})",
                    sids,
                ).fetchall():
                    all_agents[r["session_id"]].append(r["agent_id"])

                # Bulk: CWD of root agent per session
                all_cwds: dict[str, str | None] = dict.fromkeys(sids)
                for r in conn.execute(
                    f"SELECT a.session_id, a.cwd FROM agents a "
                    f"INNER JOIN (SELECT session_id, MIN(registered) as min_reg FROM agents "
                    f"WHERE session_id IN ({placeholders}) GROUP BY session_id) sub "
                    f"ON a.session_id = sub.session_id AND a.registered = sub.min_reg",
                    sids,
                ).fetchall():
                    all_cwds[r["session_id"]] = r["cwd"]

                # Bulk: tasks
                all_tasks: dict[str, list[str]] = {sid: [] for sid in sids}
                for r in conn.execute(
                    f"SELECT session_id, task FROM agents WHERE session_id IN ({placeholders}) AND task IS NOT NULL",
                    sids,
                ).fetchall():
                    all_tasks[r["session_id"]].append(r["task"])

                # Bulk: first messages (use window function to get top 3 per session)
                all_messages: dict[str, list[str]] = {sid: [] for sid in sids}
                msg_rows = conn.execute(
                    f"SELECT session_id, payload FROM ("
                    f"  SELECT session_id, payload, ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY seq) as rn"
                    f"  FROM ui_events WHERE session_id IN ({placeholders}) AND event_type = 'UserPromptSubmitted'"
                    f") WHERE rn <= 3",
                    sids,
                ).fetchall()
                for r in msg_rows:
                    try:
                        data = json.loads(r["payload"])
                        text = data.get("text", "")
                        if text:
                            all_messages[r["session_id"]].append(text)
                    except (json.JSONDecodeError, TypeError):
                        pass

                return [
                    {
                        "session_id": s["session_id"],
                        "agents": all_agents[s["session_id"]],
                        "last_active": s["last_active"],
                        "agent_count": s["agent_count"],
                        "cwd": all_cwds[s["session_id"]],
                        "tasks": all_tasks[s["session_id"]],
                        "first_messages": all_messages[s["session_id"]],
                    }
                    for s in sessions
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
        self._buffer_journal_event(event)

        if isinstance(event, AgentStateChanged) and event.new_state == AgentState.IDLE:
            if self._message_bus:
                self._message_bus.wake(event.agent_id)

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

    def _buffer_journal_event(self, event: BrokerEvent) -> None:
        """Accumulate a journalable event into the per-agent turn buffer.

        Seq numbers are allocated at buffer time so monotonic ordering is
        structural.  Merge-or-update operations preserve the original seq.
        """
        event_type = type(event).__name__
        if event_type not in self._JOURNALABLE:
            return

        aid = event.agent_id
        buf = self._turn_buffer.setdefault(aid, [])
        tool_idx = self._turn_buffer_tool_index.setdefault(aid, {})

        def _next_seq() -> int:
            seq = self._journal_seq.get(aid, 0)
            self._journal_seq[aid] = seq + 1
            return seq

        if isinstance(event, MessageChunkReceived):
            if buf and isinstance((prev := buf[-1][1]), MessageChunkReceived):
                buf[-1] = (
                    buf[-1][0],
                    prev.model_copy(
                        update={"chunk": prev.chunk + event.chunk}
                    ),
                )
            else:
                buf.append((_next_seq(), event))

        elif isinstance(event, AgentThoughtReceived):
            if buf and isinstance((prev := buf[-1][1]), AgentThoughtReceived):
                buf[-1] = (
                    buf[-1][0],
                    prev.model_copy(
                        update={"chunk": prev.chunk + event.chunk}
                    ),
                )
            else:
                buf.append((_next_seq(), event))

        elif isinstance(event, ToolCallUpdated):
            existing_pos = tool_idx.get(event.tool_call_id)
            if existing_pos is not None:
                prev_seq, _ = buf[existing_pos]
                buf[existing_pos] = (prev_seq, event)
            else:
                tool_idx[event.tool_call_id] = len(buf)
                buf.append((_next_seq(), event))

        elif isinstance(event, TurnComplete):
            buf.append((_next_seq(), event))
            rows_to_flush = list(buf)
            buf.clear()
            tool_idx.clear()
            task = asyncio.create_task(
                self._flush_turn_buffer(aid, rows_to_flush),
                name=f"journal-flush-{aid}",
            )
            self._pending_flushes.add(task)
            task.add_done_callback(self._pending_flushes.discard)

        else:
            buf.append((_next_seq(), event))

    async def _flush_turn_buffer(
        self, agent_id: str, events: list[tuple[int, BrokerEvent]]
    ) -> None:
        """Write a completed turn's events to SQLite in one executemany call."""
        if not events or self._lifecycle is None:
            return
        now = int(time.time() * 1000)
        rows: list[tuple[str, str, int, str, str, int]] = [
            (
                self._session_id,
                agent_id,
                seq,
                type(event).__name__,
                event.model_dump_json(),
                now,
            )
            for seq, event in events
        ]
        try:
            await self._lifecycle.journal_ui_events(rows)
        except Exception:
            log.debug(
                "Failed to flush journal for %s (%d events)",
                agent_id, len(events), exc_info=True,
            )

    async def _flush_turn_buffer_all(self) -> None:
        """Drain unflushed buffers and await all in-flight flush tasks."""
        if self._lifecycle is None:
            return

        for agent_id, buf in list(self._turn_buffer.items()):
            if buf:
                events = list(buf)
                buf.clear()
                self._turn_buffer_tool_index.get(agent_id, {}).clear()
                task = asyncio.create_task(
                    self._flush_turn_buffer(agent_id, events),
                    name=f"journal-flush-final-{agent_id}",
                )
                self._pending_flushes.add(task)
                task.add_done_callback(self._pending_flushes.discard)

        if self._pending_flushes:
            await asyncio.gather(
                *list(self._pending_flushes), return_exceptions=True
            )

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

        While processing an entry — especially across the await on
        ``self._event_queue.put`` for an auto-resolved emission — we hold
        the active-permission slot so that any concurrent
        ``PermissionRequested`` arriving via ``_sink`` gets queued instead
        of racing to claim the slot.
        """
        queue = self._permission_queue.get(agent_id)
        while queue:
            nxt = queue.pop(0)
            # Reserve the active slot for the duration of this iteration.
            self._active_permission[agent_id] = nxt.request_id
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
                        if self._active_permission.get(agent_id) == nxt.request_id:
                            self._active_permission.pop(agent_id, None)
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
                        if self._active_permission.get(agent_id) == nxt.request_id:
                            self._active_permission.pop(agent_id, None)
                        continue
            # Needs manual resolution — forward to UI. Slot is already set above.
            cur, total = self._permission_counter.get(agent_id, (1, 1))
            self._permission_counter[agent_id] = (cur + 1, total)
            await self._event_queue.put(nxt)
            return
        # Queue fully drained
        self._permission_queue.pop(agent_id, None)
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

    async def _start_message_bus(self) -> None:
        if self._message_bus is not None or self._message_bus_starting:
            return
        self._message_bus_starting = True
        try:
            lifecycle = await self._ensure_lifecycle()
            await lifecycle._db_op(ensure_schema_sync)
            if not self._expired:
                self._expired = True
                await lifecycle.expire_old_sessions()
            self._message_bus = MessageBus(
                self._db_path, self._session_id, self._deliver_message, self._process_commands
            )
            await self._message_bus.start()
            lifecycle.set_message_bus(self._message_bus.socket_path, self._message_bus.enqueue_pending, self._message_bus.enqueue_raw)
        finally:
            self._message_bus_starting = False

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
                elif command == "resurrect":
                    await lifecycle.handle_resurrect_command(cmd_id, from_agent, data)
                else:
                    await lifecycle.update_command_status(cmd_id, "rejected", f"Unknown command: {command}")
            except Exception as exc:
                await lifecycle.update_command_status(cmd_id, "rejected", str(exc))

    async def _deliver_message(self, agent_id: str, text: str, from_agents: list[str]) -> bool:
        """Deliver a message to an agent. Non-blocking — dispatches prompt as a task."""
        if agent_id in self._delivery_held:
            for sender in from_agents:
                await self._sink(
                    McpMessageHeld(agent_id=agent_id, from_agent=sender, preview=text)
                )
            return True
        session = self._registry.get_session(agent_id)
        if not session or session.state != AgentState.IDLE:
            return False
        try:
            if self._lifecycle:
                success = await self._lifecycle.prompt(agent_id, text)
                if not success:
                    return False
            for sender in from_agents:
                await self._sink(
                    McpMessageDelivered(agent_id=agent_id, from_agent=sender, to_agent=agent_id, preview=text)
                )
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
            try:
                if self._lifecycle:
                    await self._lifecycle.shutdown()
            except Exception:
                log.debug("Lifecycle shutdown error", exc_info=True)

            try:
                if self._message_bus:
                    await self._message_bus.stop()
            except Exception:
                log.debug("MessageBus stop error", exc_info=True)

            try:
                await self._flush_turn_buffer_all()
            except Exception:
                log.debug("Journal flush error", exc_info=True)
        finally:
            self._shutdown_event.set()

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
