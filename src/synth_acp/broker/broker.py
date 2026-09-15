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
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from synth_acp.broker.lifecycle import AgentLifecycle
from synth_acp.broker.message_bus import MessageBus
from synth_acp.broker.permissions import PermissionEngine
from synth_acp.broker.prompt_queue import PromptQueue, QueuedItem
from synth_acp.broker.registry import AgentRegistry
from synth_acp.db import AgentRenameResult, configure_connection, ensure_schema_sync
from synth_acp.discovery import DiscoveredAgent
from synth_acp.models.agent import AgentConfig, AgentMode, AgentModel, AgentState
from synth_acp.models.commands import (
    BrokerCommand,
    CancelTurn,
    CommitQueueEdit,
    DeleteQueueItem,
    DrainQueue,
    EditQueueItem,
    HoldQueue,
    LaunchAgent,
    ReleaseQueue,
    RespondPermission,
    RestoreSession,
    ResurrectAgent,
    SendPrompt,
    SetAgentMode,
    SetAgentModel,
    SetConfigOption,
    TerminateAgent,
)
from synth_acp.models.config import (
    MessageKind,
    SessionConfig,
    format_handoff_nudge,
    format_mcp_message,
)
from synth_acp.models.events import (
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerError,
    BrokerEvent,
    MessageChunkReceived,
    MessageSteered,
    PermissionAutoResolved,
    PermissionRequested,
    QueueItemSnapshot,
    QueueUpdated,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
)
from synth_acp.models.permissions import PermissionDecision, PermissionRule

log = logging.getLogger(__name__)


class ChunkLossStats(BaseModel):
    """Cumulative live-UI chunk drop counters for the session."""

    model_config = ConfigDict(frozen=True)

    dropped_total: int
    dropped_by_agent: dict[str, int]
    warned_agents: frozenset[str]
    armed_agents: frozenset[str]
    last_drop_at: float | None


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
        self._prompt_queue = PromptQueue()
        self._chunk_drops: dict[str, int] = {}  # agent_id → dropped chunk count
        self._drop_armed: dict[str, bool] = {}  # agent_id → warning pending
        self._warned_agents: set[str] = set()
        # Agents already nudged to consider handoff. One nudge per agent per
        # process: PREDECESSOR-OWNED, so apply_handoff_rekey moves it.
        self._nudged_agents: set[str] = set()
        self._last_drop_at: float | None = None
        self._is_composing: Callable[[str], bool] = lambda _: False
        # One serialized tail for every agent command, so nothing can overtake an
        # active handoff. Created lazily: __init__ may run off the event loop.
        self._command_queue: (
            asyncio.Queue[tuple[list[tuple[int, str, str, str]], asyncio.Event] | None] | None
        ) = None
        self._command_tail_task: asyncio.Task[None] | None = None

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
                await self.submit_prompt(aid, text, source="user")
            case RespondPermission(agent_id=aid, request_id=rid, option_id=oid):
                await self._resolve_permission(aid, rid, oid)
            case CancelTurn(agent_id=aid):
                await lifecycle.cancel(aid)
            case SetAgentMode(agent_id=aid, mode_id=mid):
                await lifecycle.set_config_option(aid, "mode", mid)
            case SetAgentModel(agent_id=aid, model_id=mid):
                await lifecycle.set_config_option(aid, "model", mid)
            case SetConfigOption(agent_id=aid, config_id=cid, value=val):
                if cid == "agent" and self._registry.get_agent_mode_target(aid) == "meta_agent":
                    await lifecycle.set_agent(aid, str(val))
                else:
                    await lifecycle.set_config_option(aid, cid, val)
            case RestoreSession(broker_session_id=sid):
                await self.restore_session(sid)
            case HoldQueue():
                pass  # Deprecated — composing state checked at delivery time
            case ReleaseQueue(agent_id=aid):
                # User stopped composing — try draining queued MCP messages
                await self._try_drain(aid)
            case DrainQueue(agent_id=aid):
                # User explicitly requested drain — force regardless of composing
                await self._force_drain(aid)
            case EditQueueItem(agent_id=aid, item_id=iid):
                self._prompt_queue.mark_editing(aid, iid)
                await self._emit_queue_state(aid)
            case CommitQueueEdit(agent_id=aid, item_id=iid, text=text):
                self._prompt_queue.commit_edit(aid, iid, text)
                await self._emit_queue_state(aid)
                await self._try_drain(aid)
            case DeleteQueueItem(agent_id=aid, item_id=iid):
                self._prompt_queue.delete(aid, iid)
                await self._emit_queue_state(aid)
                await self._try_drain(aid)

    # ------------------------------------------------------------------
    # Prompt queue — unified delivery for user and MCP messages
    # ------------------------------------------------------------------

    async def submit_prompt(
        self,
        agent_id: str,
        text: str,
        source: str = "user",
        from_agent: str | None = None,
        *,
        steerable: bool = False,
    ) -> None:
        """Single entry point for all message submission (user and MCP).

        Delivery rules checked at decision time (no stale flags):
        - User message: deliver directly if IDLE (queue items wait)
        - MCP message: deliver if IDLE + queue empty + user not composing
        - Steerable MCP message on a not-IDLE agent: inject into the running turn
        - Otherwise: enqueue, attempt drain

        ``steerable`` defaults False so every existing caller keeps today's
        behavior. Only ``_on_mcp_message`` passes True — the dynamic-child
        startup path calls ``submit_prompt(source="mcp")`` directly, bypassing
        that seam, so a child's launch prompt is never steered.
        """
        lifecycle = await self._ensure_lifecycle()
        session = self._registry.get_session(agent_id)
        idle = session is not None and session.state == AgentState.IDLE

        if source == "user" and idle:
            # User submit always delivers directly — their explicit action
            # takes priority. Queued MCP messages wait until next IDLE.
            # prompt() announces the prompt itself and returns False if it refused it,
            # in which case fall through and queue the text rather than lose it.
            if await lifecycle.prompt(agent_id, text):
                return

        elif source != "user":
            # MCP: deliver if IDLE + queue empty + not composing
            empty = self._prompt_queue.is_empty(agent_id)
            if idle and empty and not self._is_composing(agent_id) and (
                await lifecycle.prompt(agent_id, text)
            ):
                return

        if steerable and not idle and await self._try_steer(agent_id, text, from_agent):
            # An accepted steer is the delivery of record. Enqueueing the same
            # text as well would deliver it twice, because Kiro holds a steer
            # sent to an idle session and rides it along with the next prompt.
            return

        # Enqueue and attempt immediate drain
        item = QueuedItem(
            text=text,
            source=source,  # type: ignore[arg-type]
            from_agent=from_agent,
            steerable=steerable,
        )
        self._prompt_queue.enqueue(agent_id, item)
        drained = await self._try_drain(agent_id)
        if not drained:
            # Item is sitting in queue — notify UI
            await self._emit_queue_state(agent_id)

    def _steer_eligible(self, agent_id: str, kind: MessageKind) -> bool:
        """Whether this message may be steered into a running turn.

        False for ``kind == "system"`` unconditionally (join/exit notifications
        are explicitly "no action required"), false when the resolved
        ``messages_interrupt`` setting is false, and false when the agent's
        harness declares no steer protocol.
        """
        if kind == "system":
            return False
        if not self._config.settings.messages_interrupt:
            return False
        session = self._registry.get_session(agent_id)
        return session is not None and session.steer_protocol is not None

    async def _try_steer(self, agent_id: str, text: str, from_agent: str | None) -> bool:
        """Attempt in-turn delivery. Returns True if the message was delivered.

        Builds the payload from any already-queued items that are themselves
        steerable, PLUS ``text``, so a steer that follows a previously failed one
        carries the backlog. A queued user prompt or system notification is never
        included: those classes are prohibited from being steered, and being
        queued does not make them eligible. On success the included items are
        consumed and the steered event is emitted. On failure the queue is
        restored and the caller enqueues normally.

        Holds ``self._registry.agent_lock(agent_id)`` across the whole attempt.
        Without it an IDLE transition can drain the queue between building the
        payload and consuming it, delivering the same text twice — the repo's
        documented read-await-mutate hazard. The pending items are popped BEFORE
        the call rather than after, so anything enqueued while the steer is in
        flight stays queued instead of being consumed unsent.
        """
        async with self._registry.agent_lock(agent_id):
            session = self._registry.get_session(agent_id)
            if session is None or session.state == AgentState.IDLE:
                return False
            if self._first_prompt_reserved(agent_id):
                # Refused before anything is popped, so the caller's normal enqueue path
                # preserves the text. Rechecked HERE rather than at entry because a steer
                # that passed an entry check before the reservation existed is already
                # parked on this lock -- and session.steer needs only a session id, which
                # a successor has while it is still INITIALIZING, so it would inject text
                # ahead of the reserved first prompt.
                return False
            pending = self._prompt_queue.pop_steerable(agent_id)
            payload = self._format_steer_text([i.text for i in pending] + [text])
            if not await session.steer(payload):
                self._prompt_queue.requeue_front(agent_id, pending)
                return False
        await self._sink(MessageSteered(agent_id=agent_id, from_agent=from_agent, text=payload))
        if pending:
            await self._emit_queue_state(agent_id)
        return True

    def _format_steer_text(self, bodies: list[str]) -> str:
        """Join message bodies for one steer call.

        A single body is sent verbatim — a lone ``# Message 1`` header would be
        noise. Several are joined as ``# Message 1``, ``# Message 2`` sections in
        queue order. Bodies are already attribution-signed by
        ``format_mcp_message`` and are inserted unchanged: no preamble and no
        stop directive, so the default wording stays passive and a user who wants
        stop-on-message behavior edits the ``on_mcp_message`` template.
        """
        if len(bodies) == 1:
            return bodies[0]
        return "\n\n".join(f"# Message {n}\n\n{body}" for n, body in enumerate(bodies, start=1))

    async def _try_drain(self, agent_id: str) -> bool:
        """Attempt to drain the front of the queue to the agent.

        Returns True if an item was drained and delivered.
        Won't auto-drain while user is typing — show drain button instead.
        """
        session = self._registry.get_session(agent_id)
        if not session or session.state != AgentState.IDLE:
            return False
        if not self._prompt_queue.can_drain(agent_id):
            return False
        # Don't auto-drain while user is composing — they should see the
        # drain button and choose when to inject. Emit queue state so the
        # UI can show the button.
        if self._is_composing(agent_id):
            await self._emit_queue_state(agent_id)
            return False
        lifecycle = await self._ensure_lifecycle()
        item = self._prompt_queue.pop(agent_id)
        if not item:
            return False
        if not await lifecycle.prompt(agent_id, item.text, first_prompt=item.first_prompt):
            self._restore_refused_item(agent_id, item)
            return False
        await self._emit_queue_state(agent_id)
        return True

    async def _force_drain(self, agent_id: str) -> bool:
        """Force-drain the front queue item regardless of composing state.

        Used when the user explicitly clicks the drain button.
        """
        session = self._registry.get_session(agent_id)
        if not session or session.state != AgentState.IDLE:
            return False
        if not self._prompt_queue.can_drain(agent_id):
            return False
        lifecycle = await self._ensure_lifecycle()
        item = self._prompt_queue.pop(agent_id)
        if not item:
            return False
        if not await lifecycle.prompt(agent_id, item.text, first_prompt=item.first_prompt):
            self._restore_refused_item(agent_id, item)
            return False
        await self._emit_queue_state(agent_id)
        return True

    def _restore_refused_item(self, agent_id: str, item: QueuedItem) -> None:
        """Put a popped item back after ``prompt`` refused it, at the right position.

        Both drains check ``state == IDLE`` on entry and then await before ``prompt``
        re-checks under the lock, so a refusal is reachable and a popped item would
        otherwise be silently lost.

        WHERE it goes back matters, and it depends on whether a reservation is pending.

        A reserved first prompt always returns to the FRONT: it must stay first.

        Anything else returns to the front too -- preserving the user's queue order --
        EXCEPT while a reservation is pending, when it must go to the END instead. Ahead
        of a reserved prompt it would LIVELOCK: popped and refused on every drain, with
        the handoff message never delivered. Sending it to the back unconditionally was
        the earlier behavior and silently reordered the queue on every refusal, which is
        reachable without any handoff because both drains check IDLE, then await, and
        ``prompt`` re-checks under the lock.
        """
        if item.first_prompt or not self._first_prompt_reserved(agent_id):
            self._prompt_queue.requeue_front(agent_id, [item])
        else:
            self._prompt_queue.enqueue(agent_id, item)

    async def _emit_queue_state(self, agent_id: str) -> None:
        """Emit a QueueUpdated event with current queue snapshot."""
        items = self._prompt_queue.items(agent_id)
        snapshots = [
            QueueItemSnapshot(
                id=i.id, text=i.text, source=i.source,
                from_agent=i.from_agent, editing=i.editing,
            )
            for i in items
        ]
        await self._sink(QueueUpdated(agent_id=agent_id, items=snapshots))

    def set_composing_check(self, fn: Callable[[str], bool]) -> None:
        """Set the callback that checks if a user is composing for an agent.

        The UI provides this — it reads TextArea.text.strip() directly.
        Called synchronously from submit_prompt and _try_drain to decide
        whether MCP messages should queue or deliver.
        """
        self._is_composing = fn

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

    def get_agent_display_name(self, agent_id: str) -> str | None:
        """Return a short display name for the agent's configured mode, or None.

        For harnesses where agent_mode_target is 'meta_agent' (e.g. Claude Code),
        the config_options 'mode' category represents the permission mode, not the
        agent. This method returns the actual agent name (short form) in that case.
        """
        if self._registry.get_agent_mode_target(agent_id) == "meta_agent":
            mode = self._registry.get_agent_mode(agent_id)
            if mode:
                return mode.split(":")[-1]
        return None

    def get_discovered_agents(self, agent_id: str) -> list[DiscoveredAgent]:
        """Return cached discovery results for the agent's harness.

        Delegates to ``self._lifecycle.get_discovered_agents``; returns ``[]``
        when the lifecycle has not been initialized yet.
        """
        if self._lifecycle is None:
            return []
        return self._lifecycle.get_discovered_agents(agent_id)

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
                configure_connection(conn)
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
                configure_connection(conn)
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
                    configure_connection(conn)
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
                configure_connection(conn)
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

                # Bulk: per-agent initial prompts (first inbound message per agent)
                all_initial_prompts: dict[str, dict[str, str]] = {sid: {} for sid in sids}
                prompt_rows = conn.execute(
                    f"SELECT session_id, agent_id, payload FROM ("
                    f"  SELECT session_id, agent_id, payload,"
                    f"  ROW_NUMBER() OVER (PARTITION BY session_id, agent_id ORDER BY seq) as rn"
                    f"  FROM ui_events WHERE session_id IN ({placeholders})"
                    f"  AND event_type IN ('InitialPromptDelivered', 'UserPromptSubmitted')"
                    f") WHERE rn = 1",
                    sids,
                ).fetchall()
                for r in prompt_rows:
                    try:
                        data = json.loads(r["payload"])
                        text = data.get("text", "")
                        if text:
                            all_initial_prompts[r["session_id"]][r["agent_id"]] = text
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
                        "initial_prompts": all_initial_prompts[s["session_id"]],
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
            await self._maybe_nudge_handoff(event)

        if isinstance(event, AgentStateChanged) and event.new_state == AgentState.TERMINATED:
            self._cleanup_agent_state(event.agent_id)

        if isinstance(event, MessageChunkReceived):
            # NOTHING ON THIS PATH MAY AWAIT: Queue.put fast-paths to
            # put_nowait when not full, so an awaited put here could let a new
            # producer overtake one parked in _putters, and out-of-order output
            # is worse than dropping.  No backpressure, and no sequence
            # stamping — a counter allocated here is tautologically monotonic
            # and could not detect upstream reordering anyway.
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Live-UI loss only: _buffer_journal_event below still runs, so
                # the text survives and reappears on restore.  ACCEPTED
                # RESIDUAL: a drop that arms after the final awaited event for
                # this agent has no later trigger, so no warning is surfaced.
                # The counters are the authoritative mechanism.
                self.record_chunk_drop(event.agent_id)
            else:
                self.try_deliver_drop_warning(event.agent_id)
        elif self._shutting_down:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        else:
            await self._event_queue.put(event)
            await self.deliver_drop_warning_after_awaited_event(event.agent_id)

        # Journal UI-visible events for session restore.
        self._buffer_journal_event(event)

        if isinstance(event, AgentStateChanged) and event.new_state == AgentState.IDLE:
            if self._message_bus:
                self._message_bus.wake(event.agent_id)
            # Drain the prompt queue now that the agent is IDLE
            await self._try_drain(event.agent_id)

    # ------------------------------------------------------------------
    # Live-UI chunk loss
    # ------------------------------------------------------------------

    def record_chunk_drop(self, agent_id: str) -> None:
        """Count a chunk dropped from the live UI queue.

        Synchronous and never awaits — an await here would let a later chunk
        overtake an earlier one.  Attempts NO delivery: this runs only after
        the chunk's ``put_nowait`` already raised ``QueueFull``, so the queue
        is provably full.  Arms a per-agent warning flag via a plain dict
        mutation, which cannot yield, so concurrent ``_sink`` calls cannot both
        arm it.  The chunk is still journaled by the caller, so this is
        live-UI loss only.

        Args:
            agent_id: Agent whose chunk was dropped.
        """
        self._chunk_drops[agent_id] = self._chunk_drops.get(agent_id, 0) + 1
        self._last_drop_at = time.time()
        if agent_id not in self._warned_agents:
            self._drop_armed[agent_id] = True

    def try_deliver_drop_warning(self, agent_id: str) -> None:
        """Best-effort delivery immediately after a successful chunk enqueue.

        Synchronous.  Runs after the chunk's ``put_nowait`` — the first state
        in which capacity can exist — so the warning can never sit ahead of a
        chunk.  Attempts one ``put_nowait`` and re-arms on ``QueueFull``.

        Args:
            agent_id: Agent to warn about.
        """
        if not self._claim_drop_warning(agent_id):
            return
        try:
            self._event_queue.put_nowait(self._drop_warning_event(agent_id))
        except asyncio.QueueFull:
            self._drop_armed[agent_id] = True
        else:
            self._warned_agents.add(agent_id)

    async def deliver_drop_warning_after_awaited_event(self, agent_id: str) -> None:
        """Best-effort delivery of any armed warning after an awaited enqueue.

        Called after ANY non-chunk awaited enqueue for ``agent_id`` —
        ``TurnComplete``, ``AgentStateChanged``, ``UsageUpdated``,
        ``BrokerError`` — never on the chunk path.  Awaits ``put``, which is
        safe here: awaiting suspends only the producing coroutine and the loop
        keeps servicing the UI.

        Delivery is BEST-EFFORT because of CAUSALITY, not capacity: a drop that
        arms AFTER the final awaited event for an agent has no later trigger.
        Counters remain authoritative.  Shares one claim-and-clear with
        :meth:`try_deliver_drop_warning`, and marks the agent warned before the
        await, so at most one warning is delivered per agent per session.

        Args:
            agent_id: Agent to warn about.
        """
        if not self._claim_drop_warning(agent_id):
            return
        self._warned_agents.add(agent_id)
        await self._event_queue.put(self._drop_warning_event(agent_id))

    def _claim_drop_warning(self, agent_id: str) -> bool:
        """Claim and clear the armed flag for one agent.

        A dict ``pop`` cannot yield, so two concurrent ``_sink`` calls for the
        same agent cannot both win the claim.

        Args:
            agent_id: Agent whose flag to claim.

        Returns:
            True if this caller won the claim and should attempt delivery.
        """
        return self._drop_armed.pop(agent_id, False)

    def _drop_warning_event(self, agent_id: str) -> BrokerError:
        """Build the chunk-loss warning event.

        Args:
            agent_id: Agent whose chunks were dropped.

        Returns:
            A warning-severity BrokerError naming the drop count so far.
        """
        dropped = self._chunk_drops.get(agent_id, 0)
        return BrokerError(
            agent_id=agent_id,
            message=(
                f"Dropped {dropped} streamed chunk(s) from the live view for "
                f"{agent_id} — the text is preserved in the session journal."
            ),
            severity="warning",
        )

    def chunk_loss_stats(self) -> ChunkLossStats:
        """Return cumulative live-UI chunk drop counters.

        Returns:
            Counters for the whole session; they are not reset when an agent
            terminates, since they are the record a caller relies on.
        """
        return ChunkLossStats(
            dropped_total=sum(self._chunk_drops.values()),
            dropped_by_agent=dict(self._chunk_drops),
            warned_agents=frozenset(self._warned_agents),
            armed_agents=frozenset(self._drop_armed),
            last_drop_at=self._last_drop_at,
        )

    # ------------------------------------------------------------------
    # Agent handoff — broker-owned state
    # ------------------------------------------------------------------

    def handoff_in_flight(self, agent_id: str) -> bool:
        """Whether a handoff for *agent_id* has started and not yet finished.

        True from before the predecessor is killed until the successor is started or the
        attempt fails, so it is the only predicate available while the predecessor's
        terminal event is being handled.  ``_first_prompt_reserved`` is not usable then:
        the reservation is installed after the kill.

        Args:
            agent_id: The id being handed off, which the successor takes over.

        Returns:
            True while a handoff for that id is in flight.
        """
        return (
            self._lifecycle is not None and agent_id in self._lifecycle._active_handoffs
        )

    def _first_prompt_reserved(self, agent_id: str) -> bool:
        """Whether a successor's reserved opening prompt is still undelivered."""
        return (
            self._lifecycle is not None
            and agent_id in self._lifecycle._reserved_first_prompt
        )

    async def drain_agent_journal(self, agent_id: str) -> None:
        """Land every pending journal write for this agent before its rows are moved.

        Awaits any in-flight flush task for this agent, then flushes whatever is still
        sitting in its turn buffer.  A flush that lands after the rename's UPDATE writes
        rows under the OLD id that the UPDATE has already passed, so those events would
        silently belong to the wrong agent.

        Task names are matched EXACTLY, never by prefix: agents ``x`` and ``x-2`` both
        produce names starting with ``journal-flush-x``, so a prefix test would await an
        unrelated agent's flush.
        """
        names = {f"journal-flush-{agent_id}", f"journal-flush-final-{agent_id}"}
        inflight = [t for t in self._pending_flushes if t.get_name() in names]
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        buffered = self._turn_buffer.pop(agent_id, None)
        self._turn_buffer_tool_index.pop(agent_id, None)
        if buffered:
            await self._flush_turn_buffer(agent_id, buffered)

    def apply_handoff_rekey(self, result: AgentRenameResult) -> None:
        """Re-key every broker-owned per-agent container for a committed rename.

        SYNCHRONOUS BY CONTRACT and must never gain an await: with no yield point, no
        other coroutine can observe a partially re-keyed process.

        Applies the same three rules as the SQL side.  Predecessor-owned state MOVES to
        the retired id.  Recipient state STAYS at the original id so the successor
        receives it -- the prompt queue's keys and the registry's locks are both keyed by
        recipient id, and moving either strands work or destroys mutual exclusion.
        Authored VALUES move even where their key does not.  The predecessor's pending
        permissions are CLEARED rather than moved, because a dead agent can never answer
        them.
        """
        old, new = result.old_agent_id, result.new_agent_id

        session = self._registry.get_session(old)
        self._registry.rename(old, new)
        if session is not None:
            session.rename(new)
        # The successor inherits the predecessor's parent and harness at the ORIGINAL id.
        self._registry.set_parent(old, result.parent)
        self._registry.set_harness(old, result.harness)

        # The predecessor's ui_events rows moved with it, so its seq counter follows.
        # Leaving the original key ABSENT is what starts the successor's journal at 0.
        if old in self._journal_seq:
            self._journal_seq[new] = self._journal_seq.pop(old)

        # Drained in drain_agent_journal, so expected empty; popped regardless so nothing
        # is ever left under the original key for the successor to inherit.
        buffered = self._turn_buffer.pop(old, None)
        tool_index = self._turn_buffer_tool_index.pop(old, None)
        if buffered:
            self._turn_buffer[new] = buffered
        if tool_index:
            self._turn_buffer_tool_index[new] = tool_index

        # Predecessor-owned diagnostics.
        if old in self._chunk_drops:
            self._chunk_drops[new] = self._chunk_drops.pop(old)
        if old in self._drop_armed:
            self._drop_armed[new] = self._drop_armed.pop(old)
        if old in self._warned_agents:
            self._warned_agents.discard(old)
            self._warned_agents.add(new)
        # A nudge is a fact about the conversation that just filled up, so it
        # follows the predecessor.  Leaving it at the original id would mute the
        # successor -- which starts empty and will fill up in its turn -- for the
        # whole life of the process, silently.
        if old in self._nudged_agents:
            self._nudged_agents.discard(old)
            self._nudged_agents.add(new)

        # AUTHORED VALUE whose KEY must not move: the queue is keyed by recipient.
        self._prompt_queue.rename_author(old, new)

        # Already run once from _sink on the predecessor's TERMINATED event, and
        # idempotent. Reused rather than duplicated.
        self._cleanup_agent_state(old)

    def seed_first_prompt(self, agent_id: str, text: str, from_agent: str) -> None:
        """Place the successor's reserved opening prompt at the FRONT of its queue.

        Synchronous and lock-free by design, so a handoff can call it while still holding
        the agent lock: going through ``submit_prompt`` would reach ``lifecycle.prompt``
        or ``_try_steer``, and both acquire that same non-reentrant lock.

        FRONT, not append. The queue key stays at the original id, so anything the
        successor INHERITED is already sitting there and an appended item would be
        delivered second -- the successor would open on a message it has no context for.

        ``from_agent`` is the RETIRED id: the predecessor wrote this text and the
        predecessor is now that id. Attributing it to the original id would credit the
        successor with writing its own briefing.
        """
        self._prompt_queue.requeue_front(
            agent_id,
            [
                QueuedItem(
                    text=text,
                    source="mcp",
                    from_agent=from_agent,
                    steerable=False,
                    first_prompt=True,
                )
            ],
        )

    # ------------------------------------------------------------------
    # Agent state cleanup
    # ------------------------------------------------------------------

    async def _maybe_nudge_handoff(self, event: UsageUpdated) -> None:
        """Post a one-time handoff nudge if this agent's context crossed the threshold.

        THE CLAIM IS SYNCHRONOUS AND MUST STAY THAT WAY.  ``set.add`` cannot yield,
        so the membership test and the claim together are atomic against every
        other coroutine, and two ``_sink`` calls for the same agent -- Kiro reports
        usage several times per turn -- cannot both win.  Claiming after the DB
        write would let both through and deliver the nudge twice.

        ``event.size`` is a denominator, not a token budget: Kiro reports a
        percentage and ``ACPSession.ext_notification`` encodes it as
        ``size=100``, while Claude's own ``usage_update`` carries real token
        counts.  Comparing the ratio works for both without either harness being
        named here.  ``size == 0`` means the harness reported no usable figure.

        The nudge is deliberately NOT ``kind="system"``: system messages are
        blocked from in-turn steering and framed "no action required", and this
        one wants to reach an agent mid-turn and be acted on.
        """
        settings = self._config.settings
        if not settings.handoff_nudge or event.size <= 0:
            return
        if event.used / event.size < settings.handoff_nudge_threshold:
            return
        agent_id = event.agent_id
        if agent_id in self._nudged_agents:
            return
        self._nudged_agents.add(agent_id)  # claim, before any await

        body = format_handoff_nudge(
            settings.handoff_nudge_template,
            agent_id=agent_id,
            used_fraction=event.used / event.size,
            threshold=settings.handoff_nudge_threshold,
        )
        try:
            lifecycle = await self._ensure_lifecycle()
            await lifecycle.send_notification(agent_id, body)
        except Exception:
            # Release the claim so a later report can retry: a nudge that was
            # claimed but never written would silence the agent permanently.
            self._nudged_agents.discard(agent_id)
            log.warning("Failed to post handoff nudge for %s", agent_id, exc_info=True)
            return
        if self._message_bus:
            self._message_bus.wake(agent_id)

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
        "PlanReceived",
        "UserPromptSubmitted",
        "MessageSteered",
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
                prev_seq, prev_event = buf[existing_pos]
                assert isinstance(prev_event, ToolCallUpdated)
                # Read field values via getattr, not model_dump(): model_copy
                # does not validate, so dumped values would leave dataclass
                # fields (locations, diffs) holding plain dicts and trip
                # pydantic serializer warnings at journal-flush time.
                merged = prev_event.model_copy(update={
                    k: v
                    for k in type(event).model_fields
                    if k != "tool_call_id"
                    and (v := getattr(event, k)) not in (None, "", [])
                    and not (k == "kind" and v == "other")
                })
                buf[existing_pos] = (prev_seq, merged)
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
                    configure_connection(conn)
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
                self._db_path, self._session_id, self._on_mcp_message, self._process_commands
            )
            await self._message_bus.start()
            lifecycle.set_message_bus(self._message_bus.socket_path)
            lifecycle.set_submit_prompt(self.submit_prompt)
            lifecycle.set_handoff_state(self)
        finally:
            self._message_bus_starting = False

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------

    async def _process_commands(self, commands: list[tuple[int, str, str, str]]) -> None:
        """Hand a claimed batch to the serialized command tail and await its completion.

        FIFO across the whole session is the property being protected. A batch like
        [handoff(from A), launch-child(from A)] must run in that order, or the launch races
        the identity transition and sees a half-transitioned agent. Dispatching the handoff
        as a bare task and returning would also strand the already-claimed remainder of the
        batch in 'processing'.

        So every command type is appended, in claimed order, to ONE queue drained by ONE
        long-lived owned task that handles a single command at a time, and this method
        awaits a per-batch completion signal. Existing callers therefore still observe
        settled rows when it returns. If the poll task is cancelled this await raises, but
        the tail is a separate owned task and keeps going -- which is precisely the
        property being bought, since a handoff must not be torn apart between its
        committed rename and its in-memory re-key.
        """
        self._ensure_command_tail()
        assert self._command_queue is not None
        settled = asyncio.Event()
        self._command_queue.put_nowait((commands, settled))
        await settled.wait()

    def _ensure_command_tail(self) -> None:
        """Start the command tail, or restart it if it has exited.

        A STRONG reference is kept: the event loop holds only weak references to tasks, so
        an unreferenced task can be collected mid-flight. Restarting a task that is already
        done also means a tail that died cannot leave a later batch waiting forever on an
        event nobody will set.

        Created lazily rather than in _start_message_bus because broker tests drive
        _process_commands directly, without a bus.
        """
        if self._command_queue is None:
            self._command_queue = asyncio.Queue()
        if self._command_tail_task is None or self._command_tail_task.done():
            self._command_tail_task = asyncio.create_task(
                self._run_command_tail(), name="command-tail"
            )

            def _on_done(t: asyncio.Task[None]) -> None:
                if not t.cancelled() and (exc := t.exception()):
                    log.error("command tail raised", exc_info=exc)

            self._command_tail_task.add_done_callback(_on_done)

    async def _run_command_tail(self) -> None:
        """Drain the command queue, one command at a time, until the stop sentinel."""
        assert self._command_queue is not None
        while True:
            item = await self._command_queue.get()
            if item is None:
                return
            commands, settled = item
            try:
                for cmd_id, from_agent, command, payload in commands:
                    await self._dispatch_command(cmd_id, from_agent, command, payload)
            finally:
                # Always, even if the tail is cancelled mid-batch: a caller parked on this
                # event must never be left waiting.
                settled.set()

    async def _dispatch_command(
        self, cmd_id: int, from_agent: str, command: str, payload: str
    ) -> None:
        lifecycle = await self._ensure_lifecycle()
        try:
            data = json.loads(payload)
            if command == "launch":
                await lifecycle.handle_launch_command(cmd_id, from_agent, data)
            elif command == "terminate":
                await lifecycle.handle_terminate_command(cmd_id, from_agent, data)
            elif command == "resurrect":
                await lifecycle.handle_resurrect_command(cmd_id, from_agent, data)
            elif command == "handoff":
                await lifecycle.handle_handoff_command(cmd_id, from_agent, data)
            else:
                await lifecycle.update_command_status(cmd_id, "rejected", f"Unknown command: {command}")
        except Exception as exc:
            # Settling on failure is what keeps a row from being stranded in 'processing'
            # until the next process start requeues it.
            await lifecycle.update_command_status(cmd_id, "rejected", str(exc))

    async def _stop_command_tail(self) -> None:
        """Stop the tail WITHOUT cancelling the command it is running.

        A sentinel, never task.cancel(). AgentLifecycle.shutdown deliberately does not
        cancel a handoff whose bounded wait expired, so cancelling the tail dispatching it
        would reintroduce exactly the partial-state failure that decision avoids. The tail
        exits once the command in flight has finished; a command that never finishes
        outlives the process the same way the expired bounded wait does.
        """
        if self._command_tail_task is None or self._command_queue is None:
            return
        self._command_queue.put_nowait(None)
        _, unfinished = await asyncio.wait([self._command_tail_task], timeout=2.0)
        if unfinished:
            log.warning("Command tail still busy at shutdown; not cancelling it")

    async def _on_mcp_message(
        self, to_agent: str, body: str, from_agent: str, kind: MessageKind
    ) -> None:
        """Callback from message bus — an inter-agent message was found in DB.

        Formats the recipient envelope ONCE here (the sole MCP delivery seam) so
        direct-deliver and both drain paths inherit already-signed text. Formatting
        is bound to this callback, NOT to ``source`` — the dynamic-child startup
        path submits ``source="mcp"`` directly via ``submit_prompt`` and must stay
        untransformed.
        """
        hook = self._config.settings.hooks.on_mcp_message
        text = format_mcp_message(
            hook, from_agent=from_agent, to_agent=to_agent, body=body, kind=kind
        )
        await self.submit_prompt(
            to_agent,
            text,
            source="mcp",
            from_agent=from_agent,
            steerable=self._steer_eligible(to_agent, kind),
        )

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

            # After the bus, so no new batch can be claimed while the sentinel is in
            # flight, and before the final flush.
            try:
                await self._stop_command_tail()
            except Exception:
                log.debug("Command tail stop error", exc_info=True)

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
