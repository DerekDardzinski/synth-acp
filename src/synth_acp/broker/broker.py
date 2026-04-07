"""ACPBroker — thin coordinator for agent sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from synth_acp.broker.lifecycle import AgentLifecycle
from synth_acp.broker.message_bus import MessageBus
from synth_acp.broker.permissions import PermissionEngine
from synth_acp.broker.registry import AgentRegistry
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
    BrokerEvent,
    HookFired,
    InitialPromptDelivered,
    McpMessageDelivered,
    MessageChunkReceived,
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
                await lifecycle.prompt(aid, text)
            case RespondPermission(agent_id=aid, request_id=rid, option_id=oid):
                await self._resolve_permission(aid, rid, oid)
            case CancelTurn(agent_id=aid):
                await lifecycle.cancel(aid)
            case SetAgentMode(agent_id=aid, mode_id=mid):
                await lifecycle.set_mode(aid, mid)
            case SetAgentModel(agent_id=aid, model_id=mid):
                await lifecycle.set_model(aid, mid)

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

        if isinstance(event, MessageChunkReceived):
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                log.debug("Event queue full, dropping chunk for %s", event.agent_id)
        else:
            await self._event_queue.put(event)

        if isinstance(event, AgentStateChanged) and event.new_state == AgentState.IDLE:
            if self._message_bus and self._lifecycle:
                pending = self._message_bus.pop_pending(event.agent_id)
                if pending:
                    original = self._registry.pop_initial_message(event.agent_id)
                    if original:
                        parent = self._registry.get_parent(event.agent_id)
                        await self._event_queue.put(
                            InitialPromptDelivered(
                                agent_id=event.agent_id,
                                from_agent=parent or "system",
                                text=original,
                            )
                        )
                        await self._event_queue.put(
                            HookFired(agent_id=event.agent_id, hook_name="on_agent_prompt")
                        )
                    await self._lifecycle.prompt(event.agent_id, pending)

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
            self._permission_engine.persist(
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

    async def _start_message_bus(self) -> None:
        if self._message_bus is None:
            lifecycle = await self._ensure_lifecycle()
            await lifecycle.register_agents()
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
                await self._event_queue.put(
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

        if self._lifecycle:
            await self._lifecycle.shutdown()

        if self._message_bus:
            await self._message_bus.stop()

        if self._lifecycle:
            await self._lifecycle.close_db()

        sessions_path = Path.home() / ".synth" / "sessions.json"
        sessions_path.parent.mkdir(parents=True, exist_ok=True)
        session_ids = {
            aid: s.session_id
            for aid, s in self._registry.all_sessions().items()
            if s.session_id and s.state == AgentState.TERMINATED
        }
        sessions_path.write_text(json.dumps(session_ids))

        self._shutdown_event.set()
