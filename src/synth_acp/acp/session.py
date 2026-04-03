"""ACP session wrapping one agent subprocess."""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import logging
import os
import signal
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from acp.client.connection import ClientSideConnection
from acp.contrib.session_state import SessionAccumulator
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    CurrentModeUpdate,
    DeniedOutcome,
    FileSystemCapabilities,
    Implementation,
    McpServerStdio,
    PermissionOption,
    RequestPermissionResponse,
    SessionNotification,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
)
from acp.transports import default_environment

from acp import text_block
from synth_acp.models.agent import (
    TRANSITIONS,
    AgentMode,
    AgentModel,
    AgentState,
    InvalidTransitionError,
)
from synth_acp.models.events import (
    AgentModeChanged,
    AgentModelChanged,
    AgentModelsReceived,
    AgentModesReceived,
    AgentStateChanged,
    AgentThoughtReceived,
    BrokerError,
    BrokerEvent,
    MessageChunkReceived,
    PermissionRequested,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
)

log = logging.getLogger(__name__)

EventSink = Callable[[BrokerEvent], Awaitable[None]]


_SHUTDOWN_TIMEOUT = 2.0


@asynccontextmanager
async def _spawn_isolated_agent(
    client: Any,
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> AsyncIterator[tuple[ClientSideConnection, aio_subprocess.Process]]:
    """Spawn an ACP agent in its own process group.

    Uses ``process_group=0`` so the child calls ``setpgid(0, 0)`` before
    exec — no race with the parent.  This lets ``os.killpg`` safely
    terminate the agent and all its children (e.g. synth-mcp) without
    hitting the synth parent process.
    """
    merged_env = dict(default_environment())
    if env:
        merged_env.update(env)

    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
        stderr=aio_subprocess.PIPE,
        env=merged_env,
        cwd=cwd,
        process_group=0,
    )
    if not process.stdout or not process.stdin:
        raise RuntimeError("Failed to open stdin/stdout pipes for agent subprocess")

    conn = ClientSideConnection(client, process.stdin, process.stdout)
    try:
        yield conn, process
    finally:
        await conn.close()
        # Graceful stdin close, then escalate.
        if process.stdin and not process.stdin.is_closing():
            process.stdin.close()
            with contextlib.suppress(Exception):
                await process.stdin.wait_closed()
        try:
            await asyncio.wait_for(process.wait(), timeout=_SHUTDOWN_TIMEOUT)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=_SHUTDOWN_TIMEOUT)
            except TimeoutError:
                process.kill()
                await process.wait()


class ACPSession:
    """Wraps one ACP agent subprocess.

    Implements the acp SDK Client interface via duck typing — the SDK uses
    Protocol, not inheritance.
    """

    def __init__(
        self,
        agent_id: str,
        binary: str,
        args: list[str],
        cwd: str,
        event_sink: EventSink,
        mcp_servers: list[McpServerStdio] | None = None,
        agent_mode: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.state = AgentState.UNSTARTED
        self._binary = binary
        self._args = args
        self._cwd = cwd
        self._event_sink = event_sink
        self._mcp_servers = mcp_servers or []
        self._conn: Any = None
        self._proc: Any = None
        self._session_id: str | None = None
        self._permission_future: asyncio.Future[str] | None = None
        self._capabilities: Any = None
        self._agent_mode = agent_mode
        self._available_modes: list[AgentMode] = []
        self._current_mode_id: str | None = None
        self._available_models: list[AgentModel] = []
        self._current_model_id: str | None = None
        self._suppress_history_replay: bool = False
        self._accumulator = SessionAccumulator()
        self._accumulator.subscribe(self._on_snapshot)

    async def _set_state(self, new_state: AgentState) -> None:
        """Transition state and notify broker. Awaited to prevent races."""
        old = self.state
        if new_state not in TRANSITIONS[old]:
            raise InvalidTransitionError(f"{self.agent_id}: {old} → {new_state}")
        self.state = new_state
        await self._event_sink(
            AgentStateChanged(agent_id=self.agent_id, old_state=old, new_state=new_state)
        )

    async def run(self) -> None:
        """Main lifecycle — spawns agent, handshakes, waits for exit."""
        try:
            await self._set_state(AgentState.INITIALIZING)
            async with _spawn_isolated_agent(self, self._binary, *self._args, cwd=self._cwd) as (
                conn,
                proc,
            ):
                self._conn = conn
                self._proc = proc

                init_response = await conn.initialize(
                    protocol_version=1,
                    client_capabilities=ClientCapabilities(
                        fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                        terminal=False,
                    ),
                    client_info=Implementation(name="synth", version="0.1.0"),
                )
                self._capabilities = getattr(init_response, "agent_capabilities", None)
                session = await conn.new_session(cwd=self._cwd, mcp_servers=self._mcp_servers)
                self._session_id = session.session_id

                # Capture modes
                if session.modes is not None:
                    self._available_modes = [
                        AgentMode(
                            id=m.id,
                            name=m.name,
                            description=getattr(m, "description", None),
                        )
                        for m in session.modes.available_modes
                    ]
                    self._current_mode_id = session.modes.current_mode_id
                    await self._event_sink(
                        AgentModesReceived(
                            agent_id=self.agent_id,
                            available_modes=self._available_modes,
                            current_mode_id=self._current_mode_id,
                        )
                    )

                # Capture models (UNSTABLE capability — may be absent)
                if session.models is not None:
                    self._available_models = [
                        AgentModel(
                            id=m.model_id,
                            name=m.name,
                            description=getattr(m, "description", None),
                        )
                        for m in session.models.available_models
                    ]
                    self._current_model_id = session.models.current_model_id
                    await self._event_sink(
                        AgentModelsReceived(
                            agent_id=self.agent_id,
                            available_models=self._available_models,
                            current_model_id=self._current_model_id,
                        )
                    )

                # Apply agent_mode from config if advertised
                if self._agent_mode is not None:
                    mode_ids = {m.id for m in self._available_modes}
                    if self._agent_mode in mode_ids:
                        await conn.set_session_mode(
                            mode_id=self._agent_mode, session_id=self._session_id
                        )
                        self._current_mode_id = self._agent_mode
                        await self._event_sink(
                            AgentModeChanged(agent_id=self.agent_id, mode_id=self._agent_mode)
                        )
                        # Re-read model state — mode switch may change the model.
                        # Safe to load_session here because no conversation history
                        # exists yet.
                        try:
                            loaded = await conn.load_session(
                                session_id=self._session_id,
                                cwd=self._cwd,
                                mcp_servers=self._mcp_servers or None,
                            )
                            if loaded.models is not None:
                                new_model = loaded.models.current_model_id
                                if new_model and new_model != self._current_model_id:
                                    self._current_model_id = new_model
                                    await self._event_sink(
                                        AgentModelChanged(
                                            agent_id=self.agent_id, model_id=new_model
                                        )
                                    )
                        except Exception:
                            log.debug(
                                "Model re-read after initial mode switch failed", exc_info=True
                            )
                    else:
                        log.warning(
                            "agent_mode '%s' not in available_modes for %s — skipping",
                            self._agent_mode,
                            self.agent_id,
                        )

                await self._set_state(AgentState.IDLE)
                await proc.wait()
        except Exception as e:
            await self._event_sink(BrokerError(agent_id=self.agent_id, message=f"Agent error: {e}"))

        finally:
            if self._permission_future and not self._permission_future.done():
                self._permission_future.cancel()
            if self.state != AgentState.TERMINATED:
                await self._set_state(AgentState.TERMINATED)

    async def prompt(self, text: str) -> None:
        """Send a prompt to the agent."""
        if not self._conn or not self._session_id:
            return
        await self._set_state(AgentState.BUSY)
        try:
            response = await self._conn.prompt(
                session_id=self._session_id, prompt=[text_block(text)]
            )
            await self._event_sink(
                TurnComplete(
                    agent_id=self.agent_id,
                    stop_reason=response.stop_reason if response else "unknown",
                )
            )
        finally:
            if self.state == AgentState.BUSY:
                await self._set_state(AgentState.IDLE)

    async def set_mode(self, mode_id: str) -> None:
        """Switch the agent's mode, preserving the current model.

        Transitions to CONFIGURING for the duration of the switch. This blocks
        any concurrent prompt from being sent to the agent while set_session_mode,
        set_session_model, and load_session (MCP restore) are in flight — sending
        a prompt concurrently with load_session causes Kiro to hang indefinitely.

        CONFIGURING → IDLE in the finally block ensures the state is always
        restored even if an RPC fails mid-switch.
        """
        if self._conn and self._session_id and self.state == AgentState.IDLE:
            await self._set_state(AgentState.CONFIGURING)
            try:
                await self._conn.set_session_mode(mode_id=mode_id, session_id=self._session_id)
                if self._current_model_id:
                    await self._conn.set_session_model(
                        model_id=self._current_model_id, session_id=self._session_id
                    )
                await self._restore_mcp_servers()
                self._current_mode_id = mode_id
                await self._event_sink(AgentModeChanged(agent_id=self.agent_id, mode_id=mode_id))
            finally:
                if self.state == AgentState.CONFIGURING:
                    await self._set_state(AgentState.IDLE)

    async def _restore_mcp_servers(self) -> None:
        """Re-establish MCP server connections after a mode switch.

        Kiro drops all MCP server connections when session/set_mode is called.
        Calling session/load with mcp_servers causes Kiro to reconnect them.

        session/load requires Kiro to stream the full conversation history back
        to the client as session/update notifications. The _suppress_history_replay
        flag causes session_update to drop all notifications during this call —
        we only need the MCP reconnection side effect, not the replay.

        session/resume (which would avoid the replay entirely) is not supported
        by Kiro — it returns Method not found.

        The flag is always cleared in a finally block so a failed load_session
        cannot leave session_update permanently suppressed.
        """
        if not self._mcp_servers or not self._conn or not self._session_id:
            return
        self._suppress_history_replay = True
        try:
            await self._conn.load_session(
                session_id=self._session_id,
                cwd=self._cwd,
                mcp_servers=self._mcp_servers,
            )
        except Exception:
            log.debug(
                "MCP server restore via load_session failed for %s",
                self.agent_id,
                exc_info=True,
            )
        finally:
            self._suppress_history_replay = False

    async def set_model(self, model_id: str) -> None:
        """Switch the agent's model."""
        if self._conn and self._session_id and self.state == AgentState.IDLE:
            await self._conn.set_session_model(model_id=model_id, session_id=self._session_id)
            self._current_model_id = model_id
            await self._event_sink(AgentModelChanged(agent_id=self.agent_id, model_id=model_id))

    @property
    def available_modes(self) -> list[AgentMode]:
        """Return a copy of available modes, or [] if none received."""
        return list(self._available_modes)

    @property
    def current_mode_id(self) -> str | None:
        """Return the current mode id, or None if not known."""
        return self._current_mode_id

    @property
    def available_models(self) -> list[AgentModel]:
        """Return a copy of available models, or [] if none received."""
        return list(self._available_models)

    @property
    def current_model_id(self) -> str | None:
        """Return the current model id, or None if not known."""
        return self._current_model_id

    async def cancel(self) -> None:
        """Cancel the active prompt turn."""
        if self._conn and self._session_id and self.state == AgentState.BUSY:
            await self._conn.cancel(session_id=self._session_id)

    async def terminate(self) -> None:
        """Terminate the agent and all its children via process group kill.

        Sends SIGTERM to the agent's process group, waits up to 2 seconds,
        then escalates to SIGKILL.  Safe because _spawn_isolated_agent
        creates each agent in its own process group (process_group=0).
        """
        if self._permission_future and not self._permission_future.done():
            self._permission_future.cancel()
        if self._proc is None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=_SHUTDOWN_TIMEOUT)
            except TimeoutError:
                with contextlib.suppress(OSError):
                    os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            with contextlib.suppress(OSError):
                self._proc.terminate()

    # --- ACP Client callbacks (called by SDK) ---
    # ARG002 suppressed: these signatures are required by the acp.interfaces.Client protocol.

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Called by ACP SDK when agent streams a response.

        Ordering: session_id guard → UsageUpdate isinstance check →
        accumulator.apply(). UsageUpdate bypasses the accumulator because
        SessionAccumulator silently ignores it.
        """
        if session_id != self._session_id:
            return

        log.debug("session_update type=%s agent=%s", type(update).__name__, self.agent_id)

        # UsageUpdate bypasses accumulator (it doesn't track usage).
        if isinstance(update, UsageUpdate):
            if self._suppress_history_replay:
                return
            cost = update.cost
            await self._event_sink(
                UsageUpdated(
                    agent_id=self.agent_id,
                    size=update.size or 0,
                    used=update.used or 0,
                    cost_amount=cost.amount if cost else None,
                    cost_currency=cost.currency if cost else None,
                )
            )
            return

        # Everything else goes through the accumulator.
        try:
            notification = SessionNotification(session_id=session_id, update=update)
            self._accumulator.apply(notification)
        except Exception:
            log.warning(
                "accumulator.apply() failed for %s on %s",
                type(update).__name__,
                self.agent_id,
                exc_info=True,
            )

    def _on_snapshot(
        self,
        snapshot: Any,
        notification: SessionNotification,
    ) -> None:
        """Subscriber callback fired by SessionAccumulator after apply().

        Checks suppress flag, then dispatches async event emission via
        create_task with a done-callback that logs exceptions.
        """
        if self._suppress_history_replay:
            return
        task = asyncio.create_task(self._emit_from_notification(notification))
        task.add_done_callback(self._log_task_exception)

    @staticmethod
    def _log_task_exception(task: asyncio.Task[None]) -> None:
        """Done-callback that logs unhandled exceptions from _emit_from_notification."""
        if not task.cancelled() and task.exception():
            log.error("_emit_from_notification failed", exc_info=task.exception())

    async def _emit_from_notification(self, notification: SessionNotification) -> None:
        """Map a notification's update type to the corresponding BrokerEvent.

        Args:
            notification: The SessionNotification whose update to dispatch.
        """
        update = notification.update

        if isinstance(update, AgentMessageChunk):
            content = update.content
            if content:
                text = content.text
                if text:
                    await self._event_sink(MessageChunkReceived(agent_id=self.agent_id, chunk=text))
        elif isinstance(update, AgentThoughtChunk):
            content = update.content
            if content:
                text = content.text
                if text:
                    await self._event_sink(AgentThoughtReceived(agent_id=self.agent_id, chunk=text))
        elif isinstance(update, (ToolCallStart, ToolCallProgress)):
            default_status = "pending" if isinstance(update, ToolCallStart) else "in_progress"
            diffs: list[ToolCallDiff] = []
            text_parts: list[str] = []
            for item in update.content or []:
                if item.type == "diff":
                    diffs.append(
                        ToolCallDiff(
                            path=getattr(item, "path", ""),
                            old_text=getattr(item, "old_text", None),
                            new_text=getattr(item, "new_text", ""),
                        )
                    )
                elif item.type == "content":
                    inner = getattr(item, "content", None)
                    if inner and getattr(inner, "type", None) == "text":
                        text = getattr(inner, "text", None)
                        if text:
                            text_parts.append(text)
            locations: list[ToolCallLocation] = [
                ToolCallLocation(path=loc.path or "", line=loc.line)
                for loc in update.locations or []
            ]
            await self._event_sink(
                ToolCallUpdated(
                    agent_id=self.agent_id,
                    tool_call_id=update.tool_call_id or "",
                    title=update.title or "",
                    kind=update.kind or "other",
                    status=update.status or default_status,
                    diffs=diffs,
                    text_content="\n".join(text_parts) if text_parts else None,
                    locations=locations,
                    raw_input=update.raw_input,
                )
            )
        elif isinstance(update, CurrentModeUpdate):
            mode_id = update.current_mode_id
            if mode_id is not None:
                self._current_mode_id = mode_id
                await self._event_sink(AgentModeChanged(agent_id=self.agent_id, mode_id=mode_id))
        else:
            log.debug("Unhandled update type: %s", type(update).__name__)

    def resolve_permission(self, option_id: str) -> None:
        """Resolve the pending permission Future with the selected option_id.

        No-op if no Future is pending or Future is already done.
        """
        if self._permission_future and not self._permission_future.done():
            self._permission_future.set_result(option_id)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Called by ACP SDK when agent requests permission.

        Creates a Future, transitions to AWAITING_PERMISSION, emits
        PermissionRequested, and awaits resolution.
        """
        if session_id != self._session_id:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        await self._set_state(AgentState.AWAITING_PERMISSION)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._permission_future = future
        await self._event_sink(
            PermissionRequested(
                agent_id=self.agent_id,
                request_id=tool_call.tool_call_id,
                title=tool_call.title or "",
                kind=tool_call.kind or "other",
                options=list(options),
            )
        )
        try:
            option_id = await future
        except asyncio.CancelledError:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        finally:
            self._permission_future = None
        await self._set_state(AgentState.BUSY)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=option_id, outcome="selected")
        )

    def on_connect(self, conn: Any) -> None:
        """Called when the ACP connection is established."""
        self._conn = conn
