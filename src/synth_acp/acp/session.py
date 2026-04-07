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
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeniedOutcome,
    EnvVariable,
    FileSystemCapabilities,
    Implementation,
    KillTerminalResponse,
    McpServerStdio,
    PermissionOption,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    SessionNotification,
    TerminalExitStatus,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    WaitForTerminalExitResponse,
)
from acp.transports import default_environment

from acp import text_block
from synth_acp.acp.state_machine import AgentStateMachine
from synth_acp.models.agent import (
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
    AvailableCommandsReceived,
    BrokerError,
    BrokerEvent,
    MessageChunkReceived,
    PermissionRequested,
    PlanReceived,
    TerminalCreated,
    ToolCallDiff,
    ToolCallLocation,
    ToolCallUpdated,
    TurnComplete,
    UsageUpdated,
)
from synth_acp.terminal.manager import Command, TerminalProcess

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
        self._sm = AgentStateMachine(agent_id, self._on_state_transition)
        self._binary = binary
        self._args = args
        self._cwd = cwd
        self._event_sink = event_sink
        self._mcp_servers = mcp_servers or []
        self._conn: Any = None
        self._proc: Any = None
        self._session_id: str | None = None
        self._permission_futures: dict[str, asyncio.Future[str]] = {}
        self._capabilities: Any = None
        self._agent_mode = agent_mode
        self._available_modes: list[AgentMode] = []
        self._current_mode_id: str | None = None
        self._available_models: list[AgentModel] = []
        self._current_model_id: str | None = None
        self._suppress_history_replay: bool = False
        self._accumulator = SessionAccumulator()
        self._unsubscribe: Callable[[], None] = self._accumulator.subscribe(self._on_snapshot)
        self._terminals: dict[str, TerminalProcess] = {}
        self._terminal_count: int = 0

    @property
    def state(self) -> AgentState:
        return self._sm.state

    @property
    def session_id(self) -> str | None:
        """The ACP session ID, or None if not yet initialized."""
        return self._session_id

    async def force_terminate(self) -> None:
        """Force transition to TERMINATED. Safe for cleanup paths and voluntary exit.

        Other components (broker, lifecycle) call this instead of accessing
        _sm directly to maintain encapsulation.
        """
        await self._sm.force_terminal()

    async def _on_state_transition(self, old: AgentState, new: AgentState) -> None:
        """Callback fired by the state machine after every transition."""
        await self._event_sink(
            AgentStateChanged(agent_id=self.agent_id, old_state=old, new_state=new)
        )

    async def run(self) -> None:
        """Main lifecycle — spawns agent, handshakes, waits for exit."""
        try:
            await self._sm.transition(AgentState.INITIALIZING)
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
                        terminal=True,
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

                await self._sm.transition(AgentState.IDLE)
                await proc.wait()
        except InvalidTransitionError as e:
            log.error("Invalid state transition in session %s: %s", self.agent_id, e, exc_info=True)
            await self._event_sink(
                BrokerError(agent_id=self.agent_id, message=f"Internal state error: {e}", severity="error")
            )
        except asyncio.CancelledError:
            log.debug("Session %s cancelled", self.agent_id)
            raise
        except Exception as e:
            log.error("Session %s raised unexpectedly", self.agent_id, exc_info=True)
            await self._event_sink(BrokerError(agent_id=self.agent_id, message=f"Agent error: {e}"))

        finally:
            for t in self._terminals.values():
                t.kill()
            for fut in self._permission_futures.values():
                if not fut.done():
                    fut.cancel()
            self._permission_futures.clear()
            self._unsubscribe()
            await self._sm.force_terminal()

    async def prompt(self, text: str) -> None:
        """Send a prompt to the agent."""
        if not self._conn or not self._session_id:
            return
        await self._sm.transition(AgentState.BUSY)
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
                await self._sm.transition(AgentState.IDLE)

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
            await self._sm.transition(AgentState.CONFIGURING)
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
                    await self._sm.transition(AgentState.IDLE)

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
        """Switch the agent's model, transitioning through CONFIGURING."""
        if self._conn and self._session_id and self.state == AgentState.IDLE:
            await self._sm.transition(AgentState.CONFIGURING)
            try:
                await self._conn.set_session_model(model_id=model_id, session_id=self._session_id)
                self._current_model_id = model_id
                await self._event_sink(AgentModelChanged(agent_id=self.agent_id, model_id=model_id))
            finally:
                if self.state == AgentState.CONFIGURING:
                    await self._sm.transition(AgentState.IDLE)

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
        for fut in self._permission_futures.values():
            if not fut.done():
                fut.cancel()
        self._permission_futures.clear()
        for t in self._terminals.values():
            t.kill()
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
            log.debug(
                "Tool call %s [%s] agent=%s id=%s title=%r kind=%s status=%s "
                "locations=%s raw_input=%s raw_output=%s content_types=%s field_meta=%s",
                type(update).__name__,
                "start" if isinstance(update, ToolCallStart) else "progress",
                self.agent_id,
                update.tool_call_id,
                update.title,
                update.kind,
                update.status,
                [
                    {"path": loc.path, "line": loc.line}
                    for loc in (update.locations or [])
                ],
                update.raw_input,
                update.raw_output,
                [item.type for item in (update.content or [])],
                update.field_meta,
            )
            default_status = "pending" if isinstance(update, ToolCallStart) else "in_progress"
            diffs: list[ToolCallDiff] = []
            text_parts: list[str] = []
            terminal_id: str | None = None
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
                elif item.type == "terminal":
                    terminal_id = item.terminal_id
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
                    raw_output=update.raw_output,
                    terminal_id=terminal_id,
                )
            )
        elif isinstance(update, CurrentModeUpdate):
            mode_id = update.current_mode_id
            if mode_id is not None:
                self._current_mode_id = mode_id
                await self._event_sink(AgentModeChanged(agent_id=self.agent_id, mode_id=mode_id))
        elif isinstance(update, AgentPlanUpdate):
            await self._event_sink(
                PlanReceived(agent_id=self.agent_id, entries=list(update.entries))
            )
        elif isinstance(update, AvailableCommandsUpdate):
            await self._event_sink(
                AvailableCommandsReceived(
                    agent_id=self.agent_id,
                    commands=list(update.available_commands),
                )
            )
        else:
            log.debug("Unhandled update type: %s", type(update).__name__)

    def resolve_permission(self, request_id: str, option_id: str) -> None:
        """Resolve the pending permission Future for the given request_id.

        No-op if no Future is pending for that request or Future is already done.
        When the last pending permission is resolved, transitions back to BUSY.
        """
        future = self._permission_futures.pop(request_id, None)
        if future and not future.done():
            future.set_result(option_id)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Called by ACP SDK when agent requests permission.

        Creates a Future keyed by tool_call_id, transitions to
        AWAITING_PERMISSION (idempotent for parallel calls), emits
        PermissionRequested, and awaits resolution. Transitions back
        to BUSY only when this is the last pending permission.
        """
        if session_id != self._session_id:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        request_id = tool_call.tool_call_id or ""
        await self._sm.transition(AgentState.AWAITING_PERMISSION)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._permission_futures[request_id] = future
        await self._event_sink(
            PermissionRequested(
                agent_id=self.agent_id,
                request_id=request_id,
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
            self._permission_futures.pop(request_id, None)
        if not self._permission_futures:
            await self._sm.transition(AgentState.BUSY)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=option_id, outcome="selected")
        )

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """Create a terminal process for the agent.

        Args:
            command: Command to execute.
            session_id: ACP session ID.
            args: Command arguments.
            cwd: Working directory.
            env: Environment variables.
            output_byte_limit: Max bytes to retain in output buffer.

        Returns:
            Response containing the terminal ID.
        """
        terminal_env = {e.name: e.value for e in env} if env else {}
        cmd = Command(command=command, args=args or [], env=terminal_env, cwd=cwd or self._cwd)
        terminal = TerminalProcess(cmd, output_byte_limit=output_byte_limit)
        await terminal.start()
        self._terminal_count += 1
        terminal_id = f"terminal-{self._terminal_count}"
        self._terminals[terminal_id] = terminal
        await self._event_sink(
            TerminalCreated(
                agent_id=self.agent_id,
                terminal_id=terminal_id,
                command=str(cmd),
                terminal_process=terminal,
            )
        )
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        """Return buffered output from a terminal.

        Args:
            session_id: ACP session ID.
            terminal_id: Terminal to query.

        Returns:
            Response with output text, truncation flag, and optional exit status.

        Raises:
            KeyError: If terminal_id is unknown.
        """
        terminal = self._terminals[terminal_id]
        state = terminal.tool_state
        exit_status = (
            TerminalExitStatus(exit_code=state.return_code, signal=state.signal)
            if state.return_code is not None
            else None
        )
        return TerminalOutputResponse(
            output=state.output, truncated=state.truncated, exit_status=exit_status
        )

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalResponse | None:
        """Kill a terminal process.

        Args:
            session_id: ACP session ID.
            terminal_id: Terminal to kill.

        Returns:
            Empty response.
        """
        self._terminals[terminal_id].kill()
        return KillTerminalResponse()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        """Kill and release a terminal process.

        Args:
            session_id: ACP session ID.
            terminal_id: Terminal to release.

        Returns:
            Empty response.
        """
        terminal = self._terminals[terminal_id]
        terminal.kill()
        terminal.release()
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal process to exit.

        Args:
            session_id: ACP session ID.
            terminal_id: Terminal to wait on.

        Returns:
            Response with exit code and signal.
        """
        terminal = self._terminals[terminal_id]
        exit_code, terminal_signal = await terminal.wait_for_exit()
        return WaitForTerminalExitResponse(exit_code=exit_code, signal=terminal_signal)

    def on_connect(self, conn: Any) -> None:
        """Called when the ACP connection is established."""
        self._conn = conn
