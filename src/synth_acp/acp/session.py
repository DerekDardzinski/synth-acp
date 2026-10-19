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
from pathlib import Path
from typing import Any, TypeVar

from acp.client.connection import ClientSideConnection
from acp.exceptions import RequestError
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ConfigOptionUpdate,
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
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionResumeCapabilities,
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
from synth_acp.acp.extensions import AcpExtension, get_extension
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
    ConfigOptionChanged,
    ConfigOptionsReceived,
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
from synth_acp.models.extensions import (
    ContextUsage,
    ExtensionCapabilities,
    ProfileUnavailable,
    Signal,
    SteerOutcome,
    UpdateHints,
)
from synth_acp.terminal.manager import Command, TerminalProcess

log = logging.getLogger(__name__)

_T = TypeVar("_T")

EventSink = Callable[[BrokerEvent], Awaitable[None]]


_SHUTDOWN_TIMEOUT = 2.0
DRAIN_PASSES: int = 4
_PIPE_LIMIT = 8 * 1024 * 1024  # 8 MiB — agent tool responses can be large

_STEER_ACCEPTANCE_TIMEOUT = 10.0
"""Seconds to wait for the harness's deterministic steering acceptance response."""

_HANDSHAKE_TIMEOUT = 60.0
"""Seconds to wait for one handshake request before declaring the agent dead.

The ACP SDK has no internal timeout on any request, so without this bound a child that
never answers leaves its session in INITIALIZING for the lifetime of the synth process,
with nothing reported.  That was reachable and was observed: an unpinned package-manager
wrapper in the spawn path, facing an unreachable registry, was measured writing nothing
at all to either stream for 30 seconds, and from here that is indistinguishable from a
harness that is merely slow.

Generous on purpose.  A cold ACP adaptor start plus MCP server connection was measured
between 4.4 and 9.1 seconds, so 60 is well clear of a slow-but-healthy launch while still
bounded.  A handshake timeout is TERMINAL and is never retried: it means the child is dead
rather than transiently busy, and a retry would burn the whole budget again before saying
anything.
"""

_STDERR_TAIL_LIMIT = 8192
"""Bytes of the child's most recent stderr retained for error reports.

Bounded rather than accumulating: the point of reading the pipe is to keep it readable,
and an unbounded buffer would trade a stalled child for unbounded memory on a chatty one.
"""


class _StderrTail:
    """Bounded, continuously drained tail of a child process's stderr.

    Exists for two reasons that happen to share one read loop.

    The diagnostic one: when a handshake times out, the child's last words are the only
    evidence of why, and before this nothing in synth ever read them.

    The correctness one, which is more serious than it looks and was measured rather than
    reasoned about.  ``asyncio`` drains a subprocess pipe into its ``StreamReader``
    without being asked, so a small volume of unread stderr is harmless -- but past
    roughly twice the ``limit`` passed to ``create_subprocess_exec`` the transport pauses
    reading, and from then on the pipe is a real pipe again.  What that breaks here is
    ``run()``, which spends the session's whole life in ``await proc.wait()``.  MEASURED
    with synth's 8 MiB ``_PIPE_LIMIT``, child writes stderr then exits: at 50 KB
    ``wait()`` returns in 0.02s; at 20 MB it NEVER RETURNS, with or without the child
    already dead.  So an agent that logged enough to stderr could exit and synth would
    never learn it had -- the session would sit in IDLE rather than transitioning to
    TERMINATED, with no error and no dead-agent tile.  With the drain running, the same
    20 MB case returns in 0.05s.

    Measured stderr volume for the harnesses synth ships against is tiny -- 105 bytes
    across a full session plus a 95-second tool-heavy turn -- so this is a latent hazard
    rather than today's failure.  It is closed here because the fix is the same loop.
    """

    def __init__(self, limit: int = _STDERR_TAIL_LIMIT) -> None:
        self._limit = limit
        self._buf = bytearray()

    async def drain(self, stream: asyncio.StreamReader) -> None:
        """Read until EOF, retaining only the last ``limit`` bytes."""
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            self._buf.extend(chunk)
            if len(self._buf) > self._limit:
                del self._buf[: len(self._buf) - self._limit]

    def text(self) -> str:
        """Decoded tail, empty when the child wrote nothing."""
        return self._buf.decode("utf-8", "replace")


class HandshakeTimeoutError(Exception):
    """One ACP handshake request did not complete within ``_HANDSHAKE_TIMEOUT``.

    Carries the child's stderr tail because that is the only evidence available at the
    point of failure, and a separate type because the restore path must NOT treat a
    timeout the way it treats other ``load_session`` failures: falling back to
    ``new_session`` there would spend a second full budget before reporting anything.
    """

    def __init__(self, step: str, timeout: float, command: str, stderr: str) -> None:
        self.step = step
        self.timeout = timeout
        self.command = command
        self.stderr = stderr
        # Branches on EMPTINESS, not on strip(): a tail of only whitespace is still
        # output, and reporting it as silence would send an operator looking for a
        # process that never started.  The empty branch claims nothing about stdout,
        # which is consumed by the JSON-RPC reader and never observed here.
        if not stderr:
            detail = "the process wrote nothing to stderr"
        elif stripped := stderr.strip():
            detail = f"stderr tail: {stripped}"
        else:
            # Whitespace only: shown quoted, because unquoted it would read as the
            # empty case it is not.
            detail = f"stderr tail (whitespace only): {stderr!r}"
        super().__init__(
            f"'{step}' did not respond within {timeout:g}s. Command: {command}. {detail}"
        )


@asynccontextmanager
async def _spawn_isolated_agent(
    client: Any,
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> AsyncIterator[tuple[ClientSideConnection, aio_subprocess.Process, _StderrTail]]:
    """Spawn an ACP agent in its own process group.

    Uses ``process_group=0`` so the child calls ``setpgid(0, 0)`` before
    exec — no race with the parent.  This lets ``os.killpg`` safely
    terminate the agent and all its children (e.g. synth-mcp) without
    hitting the synth parent process.

    Yields the drained ``StderrTail`` alongside the connection so a caller reporting a
    failure has the child's own account of it.  The drain task is started immediately
    after spawn and cancelled before the process-group kill, so it cannot outlive the
    stream it reads.
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
        limit=_PIPE_LIMIT,
        env=merged_env,
        cwd=cwd,
        process_group=0,
    )
    if not process.stdout or not process.stdin:
        # Kill before raising: this is the one exit that precedes the cleanup block
        # below, so without it the child outlives the failure that abandoned it.
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=1.0)
        raise RuntimeError("Failed to open stdin/stdout pipes for agent subprocess")

    stderr_tail = _StderrTail()
    drain_task: asyncio.Task[None] | None = None
    if process.stderr:
        drain_task = asyncio.create_task(
            stderr_tail.drain(process.stderr), name=f"stderr-{process.pid}"
        )

    conn = ClientSideConnection(client, process.stdin, process.stdout)
    try:
        yield conn, process, stderr_tail
    finally:
        if drain_task is not None and not drain_task.done():
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
        if process.returncode is None:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=1.0)
        if process.stdin and not process.stdin.is_closing():
            process.stdin.close()


def _allowed_values(
    opt: SessionConfigOptionSelect | SessionConfigOptionBoolean,
) -> frozenset[str] | None:
    """The values a select option advertises, groups flattened; None for a boolean."""
    if not isinstance(opt, SessionConfigOptionSelect):
        return None
    values: set[str] = set()
    for item in opt.options:
        if isinstance(item, SessionConfigSelectGroup):
            values.update(o.value for o in item.options)
        else:
            values.add(item.value)
    return frozenset(values)


def _advertises_resume(init_response: Any) -> bool:
    """Whether the InitializeResponse advertises standard ``session/resume``.

    An isinstance check rather than ``is not None``, so only a parsed
    ``SessionResumeCapabilities`` counts.
    """
    caps = getattr(init_response, "agent_capabilities", None)
    session_caps = getattr(caps, "session_capabilities", None)
    return isinstance(getattr(session_caps, "resume", None), SessionResumeCapabilities)


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
        env: dict[str, str] | None = None,
        extension: AcpExtension | None = None,
    ) -> None:
        """Build an unstarted session.

        Args:
            agent_id: Synth agent id stamped on every event.
            binary: Program to spawn.
            args: Its arguments.
            cwd: Working directory for the agent and its sessions.
            event_sink: Receives every BrokerEvent this session emits.
            mcp_servers: MCP servers passed on session/new and session/load.
            agent_mode: Configured agent profile, or None.
            env: Environment overrides for the subprocess.
            extension: The harness engine's ACP extension; None means
                ``get_extension("standard")``.
        """
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
        self._tool_kinds: dict[str, str] = {}
        """Kind of each live tool call by toolCallId, for permission requests without one."""
        self._capabilities: Any = None
        self._agent_mode = agent_mode
        self._ext = extension if extension is not None else get_extension("standard")
        self._env = env
        self._available_modes: list[AgentMode] = []
        self._current_mode_id: str | None = None
        self._available_models: list[AgentModel] = []
        self._current_model_id: str | None = None
        self._config_options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean] = []
        self._has_native_config_options: bool = False
        self._suppress_history_replay: bool = False
        self._pending_emissions: set[asyncio.Task[None]] = set()
        self._terminals: dict[str, TerminalProcess] = {}
        self._terminal_count: int = 0
        self._on_session_created: Callable[[str, str], Awaitable[None]] | None = None
        self._shutting_down: bool = False
        self._stderr_tail: _StderrTail = _StderrTail()
        """Drained stderr of the current agent process, replaced on each spawn.

        Held on the session rather than passed to each handshake because a parameter is a
        thing a call site can forget: two load_session calls were left unbounded exactly
        that way, and each one hung a state the user could not leave.
        """

    @property
    def state(self) -> AgentState:
        return self._sm.state

    @property
    def session_id(self) -> str | None:
        """The ACP session ID, or None if not yet initialized."""
        return self._session_id

    @property
    def agent_mode(self) -> str | None:
        """The configured agent_mode value, or None."""
        return self._agent_mode

    @property
    def capabilities(self) -> ExtensionCapabilities:
        """What synth may rely on for this session's harness engine."""
        return self._ext.capabilities

    @property
    def extension(self) -> AcpExtension:
        """The harness engine's ACP extension."""
        return self._ext

    async def force_terminate(self) -> None:
        """Force transition to TERMINATED. Safe for cleanup paths and voluntary exit.

        Other components (broker, lifecycle) call this instead of accessing
        _sm directly to maintain encapsulation.
        """
        await self._sm.force_terminal()

    def force_kill(self) -> None:
        """SIGKILL the process group and close stdin. For bulk shutdown only."""
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        if self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.close()

    def rename(self, new_agent_id: str) -> None:
        """Re-stamp this session's identity, for an agent handoff.

        Synchronous and total: BOTH copies of the id are updated.  ``self.agent_id``
        stamps every outgoing event, and the state machine holds its own copy used in
        transition errors and logs, so updating only one leaves this session attributing
        its output to two different agents.

        Exists so callers do not reach into ``self._sm``, following ``force_terminate``.

        Args:
            new_agent_id: The id this session is known by from now on.
        """
        self.agent_id = new_agent_id
        self._sm._agent_id = new_agent_id

    async def _on_state_transition(self, old: AgentState, new: AgentState) -> None:
        """Callback fired by the state machine after every transition."""
        await self._event_sink(
            AgentStateChanged(agent_id=self.agent_id, old_state=old, new_state=new)
        )

    def set_session_created_callback(
        self, cb: Callable[[str, str], Awaitable[None]] | None
    ) -> None:
        """Register a callback invoked after new_session() returns."""
        self._on_session_created = cb

    async def _capture_config_options(self, session: Any) -> None:
        """Capture native config_options or synthesize from modes/models.

        Called after new_session/load_session. Emits ConfigOptionsReceived.
        """
        if session.config_options is not None:
            self._config_options = list(session.config_options)
            self._has_native_config_options = True
        else:
            options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean] = []
            if self._available_modes:
                options.append(
                    SessionConfigOptionSelect(
                        id="mode",
                        name="Mode",
                        category="mode",
                        type="select",
                        current_value=self._current_mode_id or "",
                        options=[
                            SessionConfigSelectOption(name=m.name, value=m.id)
                            for m in self._available_modes
                        ],
                    )
                )
            if self._available_models:
                options.append(
                    SessionConfigOptionSelect(
                        id="model",
                        name="Model",
                        category="model",
                        type="select",
                        current_value=self._current_model_id or "",
                        options=[
                            SessionConfigSelectOption(name=m.name, value=m.id)
                            for m in self._available_models
                        ],
                    )
                )
            self._config_options = options
            self._has_native_config_options = False

        if self._config_options:
            await self._event_sink(
                ConfigOptionsReceived(
                    agent_id=self.agent_id,
                    config_options=self._config_options,
                )
            )

    async def _handshake(self, awaitable: Awaitable[_T], step: str) -> _T:
        """Await one request to the agent under ``_HANDSHAKE_TIMEOUT``.

        THE RULE, rather than a list to conform to: a request belongs here when the session
        holds a state the user cannot leave until the agent answers.  Apply it to the CALL
        SITE, not to the method name, because the same method appears under both states --
        ``session/load`` holds INITIALIZING from ``run_restored`` and CONFIGURING from
        ``_restore_mcp_servers``.  Every call of initialize, session/new, session/load,
        session/set_mode, session/set_model and session/set_config_option
        currently qualifies, which is why all of them are wrapped.

        An earlier version of this docstring listed only the first four methods, which was
        worse than carrying no list: a maintainer conforming to it would have removed the
        bounds on the config requests.

        DO NOT wrap ``prompt`` in this.  A turn legitimately runs for many minutes -- one
        tool-heavy turn was measured at 95 seconds -- so a 60-second bound would abort
        healthy work.  ``cancel`` and ``ext_method`` are likewise unbounded on purpose, and
        a test asserts all three stay out.

        EVERY request synth makes before the agent is usable goes through here.  That is
        wider than the three-method handshake the name suggests, and deliberately so: a
        ``session/load`` issued for its MCP-reconnect side effect hangs CONFIGURING just
        as completely as an unanswered ``initialize`` hangs INITIALIZING.

        Args:
            awaitable: The in-flight SDK call.
            step: Wire method name for the error message, e.g. ``"initialize"``.

        Returns:
            Whatever the request returned.

        Raises:
            HandshakeTimeoutError: The request did not complete in time.
        """
        try:
            return await asyncio.wait_for(awaitable, timeout=_HANDSHAKE_TIMEOUT)
        except TimeoutError:
            command = " ".join([self._binary, *self._args])
            raise HandshakeTimeoutError(
                step, _HANDSHAKE_TIMEOUT, command, self._stderr_tail.text()
            ) from None

    async def run(self) -> None:
        """Main lifecycle — spawns agent, handshakes, waits for exit."""
        try:
            await self._sm.transition(AgentState.INITIALIZING)
            async with _spawn_isolated_agent(self, self._binary, *self._args, cwd=self._cwd, env=self._env) as (
                conn,
                proc,
                stderr_tail,
            ):
                self._conn = conn
                self._proc = proc
                self._stderr_tail = stderr_tail

                init_response = await self._handshake(
                    conn.initialize(
                        protocol_version=1,
                        client_capabilities=ClientCapabilities(
                            fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                            terminal=True,
                        ),
                        client_info=Implementation(name="synth", version="0.1.0"),
                    ),
                    "initialize",
                )
                self._capabilities = getattr(init_response, "agent_capabilities", None)
                if (info := getattr(init_response, "agent_info", None)) is not None:
                    log.info("%s: agent %s %s", self.agent_id, info.name, info.version)
                await self._check_profile(self._agent_mode)
                session = await self._handshake(
                    conn.new_session(
                        cwd=self._cwd,
                        mcp_servers=self._mcp_servers,
                        **self._ext.session_meta("new", self._agent_mode),
                    ),
                    "session/new",
                )
                self._session_id = session.session_id

                if self._on_session_created:
                    await self._on_session_created(self.agent_id, session.session_id)

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
                if self._agent_mode is not None and self.capabilities.profile_switch == "mode":
                    mode_ids = {m.id for m in self._available_modes}
                    if self._agent_mode in mode_ids:
                        await self._handshake(
                            conn.set_session_mode(
                                mode_id=self._agent_mode, session_id=self._session_id
                            ),
                            "session/set_mode",
                        )
                        self._current_mode_id = self._agent_mode
                        await self._event_sink(
                            AgentModeChanged(agent_id=self.agent_id, mode_id=self._agent_mode)
                        )
                        # Re-read model state — mode switch may change the model, and
                        # reconnect the MCP servers the switch dropped.  Only for an
                        # engine that drops them (reload_after_mode_switch).  Safe to
                        # load_session here because no conversation history exists yet.
                        if self.capabilities.reload_after_mode_switch:
                            try:
                                loaded = await self._handshake(
                                    conn.load_session(
                                        session_id=self._session_id,
                                        cwd=self._cwd,
                                        mcp_servers=self._mcp_servers or None,
                                        **self._ext.session_meta("load", self._agent_mode),
                                    ),
                                    "session/load",
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
                            except HandshakeTimeoutError:
                                # Explicit re-raise, because the broad handler below would
                                # otherwise swallow it -- HandshakeTimeoutError is an Exception,
                                # so a comment alone does not exempt it.  An earlier revision
                                # here consisted of exactly that comment and no branch, and the
                                # timeout went on being swallowed while the comment said it did
                                # not.
                                raise
                            # An earlier version reported the timeout and carried on, reasoning
                            # that the agent had already answered three requests so a stall
                            # confined to this best-effort re-read was a harness quirk.  That was
                            # wrong: asyncio.wait_for cancels only synth's local future and sends
                            # no ACP cancellation, so the agent goes on processing the load, and
                            # carrying on to IDLE would permit a prompt concurrent with a live
                            # load_session -- which set_config_option's docstring records as hanging Kiro
                            # indefinitely.  Propagating reaches run()'s handler, which reports
                            # it, and leaving the spawn context kills the process group.
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

                await self._capture_config_options(session)
                await self._sm.transition(AgentState.IDLE)
                await proc.wait()
        except HandshakeTimeoutError as e:
            # Reported rather than logged at error level with a traceback: the cause is in
            # the message and the traceback would only point back at asyncio.wait_for.
            log.warning("Handshake timeout for %s: %s", self.agent_id, e)
            await self._event_sink(
                BrokerError(agent_id=self.agent_id, message=str(e), severity="error")
            )
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
            self._shutting_down = True
            for task in self._pending_emissions:
                if not task.done():
                    task.cancel()
            if self._pending_emissions:
                await asyncio.wait(self._pending_emissions, timeout=1.0)
            self._pending_emissions.clear()
            for t in self._terminals.values():
                t.kill()
                if t._task is not None and not t._task.done():
                    t._task.cancel()
            for fut in self._permission_futures.values():
                if not fut.done():
                    fut.cancel()
            self._permission_futures.clear()
            await self._sm.force_terminal()

    async def run_restored(self, saved_acp_session_id: str) -> None:
        """Restore a previous ACP session instead of creating a new one.

        Key invariant: self._session_id MUST be set before load_session is
        called.  The ACP SDK fires session_update notifications during the
        load_session await, and session_update drops any notification whose
        session_id does not match self._session_id — so setting it afterwards
        would make the guard order below meaningless.

        Restore order, for every harness: standard ``session/resume`` when the
        InitializeResponse advertises ``sessionCapabilities.resume`` (no history
        replay; Claude), otherwise ``session/load`` with the extension's load
        ``_meta`` (Kiro v3 asks for no replay there).

        We set _suppress_history_replay=True for either call so session_update
        returns early without emitting while the agent replays its whole
        conversation.  The broker replays the UI event journal separately.  The
        flag is cleared in a finally block, after _settle_replay_guard() has let
        already-created replay runners reach the guard.

        If the resume or load fails (e.g. the agent has no history for that
        session ID), we fall back to new_session and invoke
        _on_session_created so the fresh acp_session_id is persisted to DB.
        """
        try:
            await self._sm.transition(AgentState.INITIALIZING)
            async with _spawn_isolated_agent(self, self._binary, *self._args, cwd=self._cwd, env=self._env) as (
                conn,
                proc,
                stderr_tail,
            ):
                self._conn = conn
                self._proc = proc
                self._stderr_tail = stderr_tail

                init_response = await self._handshake(
                    conn.initialize(
                        protocol_version=1,
                        client_capabilities=ClientCapabilities(
                            fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                            terminal=True,
                        ),
                        client_info=Implementation(name="synth", version="0.1.0"),
                    ),
                    "initialize",
                )
                self._capabilities = getattr(init_response, "agent_capabilities", None)
                if (info := getattr(init_response, "agent_info", None)) is not None:
                    log.info("%s: agent %s %s", self.agent_id, info.name, info.version)

                # Once per run_restored, before load: the fallback new_session below
                # must not warn a second time.
                await self._check_profile(self._agent_mode)

                # CRITICAL: assign session_id BEFORE calling load_session.
                self._session_id = saved_acp_session_id

                self._suppress_history_replay = True
                try:
                    if _advertises_resume(init_response):
                        session = await self._handshake(
                            conn.resume_session(
                                session_id=saved_acp_session_id,
                                cwd=self._cwd,
                                # A list, never None: the SDK sends None as null
                                # rather than coercing it to [].
                                mcp_servers=list(self._mcp_servers),
                                **self._ext.session_meta("resume", self._agent_mode),
                            ),
                            "session/resume",
                        )
                    else:
                        session = await self._handshake(
                            conn.load_session(
                                session_id=saved_acp_session_id,
                                cwd=self._cwd,
                                mcp_servers=self._mcp_servers,
                                **self._ext.session_meta("load", self._agent_mode),
                            ),
                            "session/load",
                        )
                except HandshakeTimeoutError:
                    # Deliberately NOT the fallback below.  A timeout means the child is
                    # not answering at all, so new_session would spend a second full
                    # budget before anything was reported.  Propagates to run_restored's
                    # own handler, which reports it.
                    raise
                except Exception as exc:
                    log.warning(
                        "Session restore failed for %s (session %s), falling back to new_session: %s",
                        self.agent_id,
                        saved_acp_session_id,
                        exc,
                    )
                    await self._event_sink(
                        BrokerError(
                            agent_id=self.agent_id,
                            message=f"Failed to restore session, starting fresh: {exc}",
                            severity="warning",
                        )
                    )
                    session = await self._handshake(
                        conn.new_session(
                            cwd=self._cwd,
                            mcp_servers=self._mcp_servers,
                            **self._ext.session_meta("new", self._agent_mode),
                        ),
                        "session/new",
                    )
                    self._session_id = session.session_id
                    if self._on_session_created:
                        await self._on_session_created(self.agent_id, session.session_id)
                finally:
                    # Let already-created replay runners reach the suppress
                    # guard before it clears.  Nested finally so the flag is
                    # cleared even if settling raises or is cancelled — a flag
                    # left set would silence the agent for the whole session.
                    try:
                        await self._settle_replay_guard()
                    finally:
                        self._suppress_history_replay = False

                # On the happy path (session_id unchanged), fire the callback
                # to flip DB status → active. The fallback path already calls
                # it inside the except block with the new session_id.
                if self._session_id == saved_acp_session_id and self._on_session_created:
                    await self._on_session_created(self.agent_id, saved_acp_session_id)

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

                # Apply configured agent_mode if it differs from the restored mode
                if (
                    self._agent_mode is not None
                    and self.capabilities.profile_switch == "mode"
                    and self._agent_mode != self._current_mode_id
                    and self._agent_mode in {m.id for m in self._available_modes}
                ):
                    await self._handshake(
                        conn.set_session_mode(
                            mode_id=self._agent_mode, session_id=self._session_id
                        ),
                        "session/set_mode",
                    )
                    self._current_mode_id = self._agent_mode
                    await self._event_sink(
                        AgentModeChanged(agent_id=self.agent_id, mode_id=self._agent_mode)
                    )

                # Capture models
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

                await self._capture_config_options(session)
                await self._sm.transition(AgentState.IDLE)
                await proc.wait()
        except HandshakeTimeoutError as e:
            # Reported rather than logged at error level with a traceback: the cause is in
            # the message and the traceback would only point back at asyncio.wait_for.
            log.warning("Handshake timeout for %s: %s", self.agent_id, e)
            await self._event_sink(
                BrokerError(agent_id=self.agent_id, message=str(e), severity="error")
            )
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
            self._shutting_down = True
            for task in self._pending_emissions:
                if not task.done():
                    task.cancel()
            if self._pending_emissions:
                await asyncio.wait(self._pending_emissions, timeout=1.0)
            self._pending_emissions.clear()
            for t in self._terminals.values():
                t.kill()
                if t._task is not None and not t._task.done():
                    t._task.cancel()
            for fut in self._permission_futures.values():
                if not fut.done():
                    fut.cancel()
            self._permission_futures.clear()
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
            await self._drain_pending_emissions()
            await self._event_sink(
                TurnComplete(
                    agent_id=self.agent_id,
                    stop_reason=response.stop_reason if response else "unknown",
                )
            )
        except ConnectionError:
            log.warning("Connection lost for %s during prompt", self.agent_id)
            await self._event_sink(
                BrokerError(
                    agent_id=self.agent_id,
                    message=f"Connection lost to {self.agent_id}",
                )
            )
        finally:
            if self.state == AgentState.BUSY:
                await self._sm.transition(AgentState.IDLE)
            elif self.state == AgentState.AWAITING_PERMISSION and not self._permission_futures:
                await self._sm.transition(AgentState.BUSY)
                await self._sm.transition(AgentState.IDLE)

    async def _end_session_on_config_timeout(
        self, exc: HandshakeTimeoutError, what: str
    ) -> None:
        """Report a POST-LAUNCH request timeout and end the session.

        Every handshake timeout is terminal.  For the requests made during ``run()`` and
        ``run_restored()`` that is free -- the exception leaves the spawn context manager,
        which kills the process group on the way out.  The requests made after launch, by
        ``set_config_option`` and ``_restore_mcp_servers``, are outside that context, so
        the kill is explicit and lives here rather than in a copy per caller.

        Terminal rather than recoverable, and neither of the two obvious alternatives
        works.  Reporting and returning to IDLE is unsafe, for one MEASURED reason and one
        JUDGMENT that are worth keeping apart.  Measured: ``asyncio.wait_for`` cancels only
        synth's LOCAL future and sends no ACP cancellation (SDK ``connection.py`` sends the
        request then awaits its future with no cancellation path), so the agent goes on
        processing, and for ``session/load`` specifically ``set_config_option``'s docstring records
        that a concurrent prompt hangs Kiro indefinitely.  Judgment, alternatives not
        discriminated: for ``set_session_mode``, ``set_session_model`` and
        ``set_config_option`` no equivalent hang has been observed, and the reason to end
        the session there is that the abandoned request may still succeed, leaving synth and
        the agent disagreeing about the session's mode, model or option set.  Nobody has
        probed a concurrent prompt against each of those.

        Raising to the caller does not work either: every caller wraps its work in a
        ``try/finally`` that transitions CONFIGURING back to IDLE regardless.

        Every caller's ``finally`` is guarded on ``state == CONFIGURING``, so moving to
        TERMINATED here means none of them resurrects a bogus IDLE.

        Args:
            exc: The timeout, whose message already names the step and the command.
            what: Operation name for the report, e.g. ``"Mode switch"``.
        """
        log.warning("%s timed out for %s, terminating: %s", what, self.agent_id, exc)
        await self._event_sink(
            BrokerError(
                agent_id=self.agent_id,
                message=(
                    f"{what}: {exc} The agent stopped answering and was terminated, "
                    "because synth cannot cancel the request it is still processing: if it "
                    "later succeeds, synth and the agent could disagree about the session."
                ),
                severity="error",
            )
        )
        self.force_kill()
        await self.force_terminate()

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
        cannot leave session_update permanently suppressed, and
        _settle_replay_guard() runs first so a replay runner the SDK created but
        has not yet started still meets the guard while it is set.
        """
        if not self._mcp_servers or not self._conn or not self._session_id:
            return
        self._suppress_history_replay = True
        try:
            await self._handshake(
                self._conn.load_session(
                    session_id=self._session_id,
                    cwd=self._cwd,
                    mcp_servers=self._mcp_servers,
                    **self._ext.session_meta("load", self._agent_mode),
                ),
                "session/load",
            )
        except HandshakeTimeoutError as exc:
            await self._end_session_on_config_timeout(exc, "MCP server restore after mode switch")
        except Exception:
            log.debug(
                "MCP server restore via load_session failed for %s",
                self.agent_id,
                exc_info=True,
            )
        finally:
            # See run_restored: settle first, clear unconditionally.
            try:
                await self._settle_replay_guard()
            finally:
                self._suppress_history_replay = False

    async def set_config_option(self, config_id: str, value: str | bool) -> None:
        """Switch a session config option.

        Precondition: state must be IDLE.
        State transitions: IDLE → CONFIGURING → IDLE (in finally block).

        CONFIGURING blocks any concurrent prompt from being sent while the switch's
        requests (set_session_mode, set_session_model, and the load_session MCP restore)
        are in flight: sending a prompt concurrently with load_session causes Kiro to
        hang indefinitely.
        """
        if not self._conn or not self._session_id:
            return
        if self.state != AgentState.IDLE:
            await self._event_sink(
                BrokerError(
                    agent_id=self.agent_id,
                    message=f"Cannot change config while {self.state}",
                    severity="warning",
                )
            )
            return
        # Some agents (Kiro v3) accept any value and any configId without error, so a
        # value the agent never advertised is rejected here rather than sent.  With no
        # options advertised at all there is nothing to check against, and the request
        # passes through as it always has.
        opt = next((o for o in self._config_options if o.id == config_id), None)
        allowed = _allowed_values(opt) if opt is not None else None
        if self._config_options and (opt is None or (allowed is not None and value not in allowed)):
            reason = "Unknown config option" if opt is None else "Value not offered for"
            await self._event_sink(
                BrokerError(
                    agent_id=self.agent_id,
                    message=f"{reason} {config_id}: {value}",
                    severity="warning",
                )
            )
            return

        await self._sm.transition(AgentState.CONFIGURING)
        try:
            if self._has_native_config_options:
                try:
                    resp = await self._handshake(
                        self._conn.set_config_option(config_id, self._session_id, value),
                        "session/set_config_option",
                    )
                except HandshakeTimeoutError:
                    # Explicit re-raise: HandshakeTimeoutError IS an Exception, so without
                    # this the handler below reports it as a recoverable warning and
                    # returns, leaving a live request on an agent presented as usable.
                    raise
                except Exception as exc:
                    log.warning("set_config_option failed for %s: %s", self.agent_id, exc)
                    await self._event_sink(
                        BrokerError(
                            agent_id=self.agent_id,
                            message=f"Failed to set {config_id}={value}: {exc}",
                            severity="warning",
                        )
                    )
                    return
                self._config_options = list(resp.config_options)
                await self._event_sink(
                    ConfigOptionsReceived(
                        agent_id=self.agent_id, config_options=self._config_options
                    )
                )
            elif config_id == "mode":
                await self._handshake(
                    self._conn.set_session_mode(mode_id=value, session_id=self._session_id),
                    "session/set_mode",
                )
                if self._current_model_id:
                    await self._handshake(
                        self._conn.set_session_model(
                            model_id=self._current_model_id,
                            session_id=self._session_id,
                        ),
                        "session/set_model",
                    )
                if self.capabilities.reload_after_mode_switch:
                    await self._restore_mcp_servers()
                # _restore_mcp_servers TERMINATES the session on a timeout, and returns
                # normally rather than raising, because raising would change the contract
                # of a method the broker calls without a handler.  So the success path
                # below has to be guarded, or a killed agent is announced as having
                # switched: subscribers saw AgentModeChanged for a session the same call
                # had just terminated.  CONFIGURING is the right predicate rather than a
                # returned flag -- it is the state this method itself established, and the
                # finally below already trusts it.
                if self.state != AgentState.CONFIGURING:
                    return
                self._current_mode_id = str(value)
                for opt in self._config_options:
                    if opt.id == "mode":
                        self._config_options[self._config_options.index(opt)] = (
                            opt.model_copy(update={"current_value": value})
                        )
                        break
                await self._event_sink(
                    AgentModeChanged(agent_id=self.agent_id, mode_id=str(value))
                )
                await self._event_sink(
                    ConfigOptionChanged(
                        agent_id=self.agent_id, config_id=config_id, value=value
                    )
                )
            elif config_id == "model":
                await self._handshake(
                    self._conn.set_session_model(model_id=value, session_id=self._session_id),
                    "session/set_model",
                )
                self._current_model_id = str(value)
                for opt in self._config_options:
                    if opt.id == "model":
                        self._config_options[self._config_options.index(opt)] = (
                            opt.model_copy(update={"current_value": value})
                        )
                        break
                await self._event_sink(
                    AgentModelChanged(agent_id=self.agent_id, model_id=str(value))
                )
                await self._event_sink(
                    ConfigOptionChanged(
                        agent_id=self.agent_id, config_id=config_id, value=value
                    )
                )
            else:
                await self._event_sink(
                    BrokerError(
                        agent_id=self.agent_id,
                        message=f"Unknown config option: {config_id}",
                        severity="warning",
                    )
                )
        except HandshakeTimeoutError as exc:
            await self._end_session_on_config_timeout(exc, f"Config change {config_id}={value}")
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

    async def steer(self, text: str) -> SteerOutcome:
        """Inject text into the agent's RUNNING turn via the extension's steer hook.

        Args:
            text: The message body to inject, sent to the harness unchanged.

        Returns:
            The extension's outcome.  ``outcome.accepted`` means the harness owns
            delivery; acceptance is NOT confirmation of in-turn delivery.  Status
            ``"failed"`` when there is no connection or session, on an acceptance
            timeout, a JSON-RPC error, or a lost connection, so the caller falls back
            to queueing.  ``"unsupported"`` from an engine that cannot steer.

        Never raises for an acceptance timeout or transport failure. The acp SDK's
        ``RequestError`` is not a ``ConnectionError`` subclass, so it is caught
        explicitly.
        """
        if not self._conn or not self._session_id:
            return SteerOutcome(status="failed")

        loop = asyncio.get_running_loop()
        started = loop.time()
        log.debug(
            "Steer acceptance started agent=%s session=%s timeout_s=%s",
            self.agent_id,
            self._session_id,
            _STEER_ACCEPTANCE_TIMEOUT,
        )
        try:
            outcome = await asyncio.wait_for(
                self._ext.steer(self._conn, self._session_id, text),
                timeout=_STEER_ACCEPTANCE_TIMEOUT,
            )
        except TimeoutError:
            elapsed_ms = (loop.time() - started) * 1000
            log.warning(
                "Steer acceptance outcome=timed_out agent=%s session=%s "
                "elapsed_ms=%.3f timeout_s=%s",
                self.agent_id,
                self._session_id,
                elapsed_ms,
                _STEER_ACCEPTANCE_TIMEOUT,
            )
            try:
                await self._event_sink(
                    BrokerError(
                        agent_id=self.agent_id,
                        message=(
                            "Steering acceptance timed out. The message will use the "
                            "existing queue fallback and may arrive twice if the harness "
                            "accepted it before the response timed out."
                        ),
                        severity="warning",
                    )
                )
            except Exception as exc:
                log.warning(
                    "Steer timeout warning reporting failed agent=%s session=%s "
                    "exception_type=%s",
                    self.agent_id,
                    self._session_id,
                    type(exc).__name__,
                )
            return SteerOutcome(status="failed")
        except (RequestError, ConnectionError, OSError) as exc:
            elapsed_ms = (loop.time() - started) * 1000
            log.debug(
                "Steer acceptance outcome=transport_failed agent=%s session=%s "
                "elapsed_ms=%.3f exception_type=%s",
                self.agent_id,
                self._session_id,
                elapsed_ms,
                type(exc).__name__,
            )
            return SteerOutcome(status="failed")

        elapsed_ms = (loop.time() - started) * 1000
        log.debug(
            "Steer acceptance outcome=%s agent=%s session=%s elapsed_ms=%.3f status=%s",
            "accepted" if outcome.accepted else "not_accepted",
            self.agent_id,
            self._session_id,
            elapsed_ms,
            outcome.status,
        )
        return outcome

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
            if t._task is not None and not t._task.done():
                t._task.cancel()
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

        Dispatches directly to a ``BrokerEvent`` with no ``SessionAccumulator``
        involved.  Guards in order: shutting-down → session_id mismatch →
        ``UsageUpdate`` special case → replay (``_suppress_history_replay`` OR the
        extension's ``UpdateHints.is_replay``).

        The extension's ``interpret_update`` runs synchronously right after the
        session_id guard.  Its signals are emitted from the emission task (or, for a
        ``UsageUpdate``, after the usage event), never awaited ahead of task creation.

        A non-matching ``session_id`` is dropped silently.  ``UsageUpdate``
        emits ``UsageUpdated`` unless ``_suppress_history_replay`` is set, then
        returns.  After that branch this returns early and SYNCHRONOUSLY when
        ``_suppress_history_replay`` is set; that guard is relocated from the
        deleted ``_on_snapshot`` and is load-bearing, because without it
        ``load_session`` history replay during ``run_restored`` and
        ``_restore_mcp_servers`` would emit into the live UI and be journalled,
        duplicating the journal on every restore and mode switch.

        For all other update types an emission task is created SYNCHRONOUSLY:
        no await may execute between entry to the non-``UsageUpdate`` path and
        task creation, because the SDK dispatches one task per notification and
        arrival order is preserved only by that synchronicity.  The suppress
        guard introduces no await.

        Exceptions raised while dispatching are logged and swallowed; nothing
        propagates to the SDK caller.

        Args:
            session_id: ACP session the notification belongs to.
            update: The raw session update to dispatch.
            **kwargs: Ignored; present for the SDK Client protocol.
        """
        if self._shutting_down:
            return
        if session_id != self._session_id:
            return

        log.debug("session_update type=%s agent=%s", type(update).__name__, self.agent_id)

        hints = self._interpret_update(update)

        # UsageUpdate is emitted inline — it carries no streaming content, so
        # the await below cannot reorder chunks relative to each other.
        if isinstance(update, UsageUpdate):
            if self._suppress_history_replay or hints.is_replay:
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
            for sig in hints.signals:
                await self._emit_signal(sig)
            return

        # Relocated from the deleted _on_snapshot.  LOAD-BEARING: load_session
        # replays the whole conversation as session/update notifications, so
        # without this guard run_restored and _restore_mcp_servers would emit
        # every historical notification into the live UI and the broker sink
        # would journal it, duplicating the entire journal on every restore
        # and every mode switch.  Synchronous by necessity — see below.  An
        # extension may also mark an update as replayed (UpdateHints.is_replay).
        if self._suppress_history_replay or hints.is_replay:
            return

        self._record_tool_kind(update)

        # No await may execute between the end of the UsageUpdate branch and
        # task creation below.  The SDK dispatches one task per notification
        # without awaiting it (acp/task/dispatcher.py:90-94), so chunk-to-chunk
        # arrival order is preserved only while emission tasks are created
        # synchronously in arrival order.
        try:
            task = asyncio.create_task(self._emit_from_notification(update, hints))
            self._pending_emissions.add(task)
            task.add_done_callback(self._pending_emissions.discard)
            task.add_done_callback(self._log_task_exception)
        except Exception:
            log.warning(
                "Failed to dispatch %s for %s",
                type(update).__name__,
                self.agent_id,
                exc_info=True,
            )

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Adapt harness-proprietary notifications onto typed broker events.

        The extension's ``on_ext_notification`` translates the notification into
        signals, which map onto existing events (see ``_emit_signal``).  ``method``
        arrives with the SDK-stripped leading underscore (``acp/client/router.py``).
        A method the extension does not recognize yields no signal, which is also the
        SDK's behavior when a client omits this handler entirely.

        Guards, in the same order and for the same reasons as ``session_update``, run
        BEFORE the extension sees anything: shutting-down, then ``sessionId`` mismatch
        -- which includes having no session yet -- then ``_suppress_history_replay``.
        The ``sessionId`` guard is not decorative: without it a delayed notification
        from a session this agent has already left is attributed to the current one,
        storing a usage figure for the wrong conversation and possibly consuming the
        agent's single handoff nudge.
        """
        if self._shutting_down:
            return
        # `is None` first: without it a payload that omits sessionId compares
        # equal while _session_id is still unset, and usage is emitted for an
        # agent that has no ACP session yet. Usage without a session is not a
        # measurement of anything.
        if self._session_id is None or params.get("sessionId") != self._session_id:
            return
        # Same guard as the UsageUpdate branch of session_update: load_session
        # replays history, and a replayed usage figure is not the current one.
        if self._suppress_history_replay:
            return
        for sig in self._ext.on_ext_notification(method, params):
            await self._emit_signal(sig)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """ACP Client callback for agent-to-client extension requests.

        Delegates to the extension's ``on_ext_request``.

        Raises:
            RequestError: ``method_not_found(method)`` when the extension returns None.
        """
        result = await self._ext.on_ext_request(method, params)
        if result is None:
            raise RequestError.method_not_found(method)
        return result

    def _interpret_update(self, update: Any) -> UpdateHints:
        """Run the extension's synchronous ``interpret_update``, never raising.

        ``session_update`` must not raise into the SDK and must reach task creation
        without awaiting, so a failing extension degrades to no hints.
        """
        try:
            return self._ext.interpret_update(update)
        except Exception:
            log.warning(
                "interpret_update failed for %s on %s",
                self.agent_id,
                type(update).__name__,
                exc_info=True,
            )
            return UpdateHints()

    async def _emit_signal(self, sig: Signal) -> None:
        """Map one extension signal onto its existing BrokerEvent."""
        if isinstance(sig, ContextUsage):
            await self._event_sink(
                UsageUpdated(
                    agent_id=self.agent_id,
                    size=sig.size,
                    used=sig.used,
                    cost_amount=sig.cost_amount,
                    cost_currency=sig.cost_currency,
                )
            )
        elif isinstance(sig, ProfileUnavailable):
            fallback = sig.fallback or "the harness default"
            await self._event_sink(
                BrokerError(
                    agent_id=self.agent_id,
                    message=(
                        f"Agent profile '{sig.requested}' is unavailable "
                        f"({sig.reason}); using {fallback}."
                    ),
                    severity="warning",
                )
            )

    async def _check_profile(self, agent_mode: str | None) -> None:
        """Warn once, without stopping the launch, when the extension rejects agent_mode."""
        if not agent_mode:
            return
        unavailable = self._ext.check_profile(agent_mode, Path(self._cwd))
        if unavailable is not None:
            await self._emit_signal(unavailable)

    async def _drain_pending_emissions(self) -> None:
        """Await emission tasks registered during a finite number of passes.

        At most ``DRAIN_PASSES`` iterations of: gather a snapshot of
        ``_pending_emissions``, await it, then ``await asyncio.sleep(0)``.
        Breaks early only when the set is empty AFTER a yield — awaiting an
        empty ``gather()`` does not yield, and a runner the SDK has created
        but not started has registered nothing yet, so deciding quiescence
        without yielding would exit in exactly the state that matters.

        There is deliberately no unbounded loop: under continuous
        registration it would never exit, ``prompt()`` would never reach its
        finally block, and the agent would stay BUSY forever with queued
        prompts wedged.

        Guarantees only that emissions REGISTERED during these passes complete
        before ``TurnComplete``.  It is NOT a guarantee that every chunk
        precedes ``TurnComplete``: a response carries no ``method`` and so
        bypasses the SDK dispatcher queue (acp/client/connection.py:248), and
        ``MessageQueue.join()`` proves runner creation rather than execution,
        so runners still queued inside the SDK cannot be observed from here.
        The same-boundary residual is made correct in the UI layer.

        Returns cleanly on ``CancelledError`` so shutdown cannot wedge.
        """
        try:
            for _ in range(DRAIN_PASSES):
                await asyncio.gather(*tuple(self._pending_emissions), return_exceptions=True)
                await asyncio.sleep(0)
                if not self._pending_emissions:
                    break
        except asyncio.CancelledError:
            return

    async def _settle_replay_guard(self) -> None:
        """Let already-created SDK runners reach the suppress guard.

        Awaited before ``_suppress_history_replay`` is cleared.  The SDK
        creates one task per notification without awaiting it, so a runner
        created during a history replay can first reach ``session_update``
        after the flag has cleared — emitting historical content into the
        live UI and journalling it.  Draining ``_pending_emissions`` does not
        help: a suppressed notification never registers an emission task, so
        the set is empty while the runner has simply not run yet.

        Bounded at ``DRAIN_PASSES`` yields, so it cannot hang.  Like the
        turn-end drain this is a mitigation, not a guarantee — a runner the
        dispatcher has not yet created cannot be waited for.
        """
        for _ in range(DRAIN_PASSES):
            await asyncio.sleep(0)

    @staticmethod
    def _log_task_exception(task: asyncio.Task[None]) -> None:
        """Done-callback that logs unhandled exceptions from _emit_from_notification."""
        if not task.cancelled() and task.exception():
            log.error("_emit_from_notification failed", exc_info=task.exception())

    async def _emit_from_notification(
        self, update: Any, hints: UpdateHints | None = None
    ) -> None:
        """Map a raw session update to the corresponding BrokerEvent and emit it.

        Args:
            update: The raw session update to dispatch.  ``SessionNotification``
                is not constructed on this path.
            hints: What the extension read from the update; its signals are
                emitted first, then the update's own event.  None means read them
                here, for a caller that did not go through ``session_update``.
        """
        if hints is None:
            hints = self._interpret_update(update)
        for sig in hints.signals:
            await self._emit_signal(sig)
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
                    parent_tool_call_id=hints.parent_tool_call_id,
                )
            )
        elif isinstance(update, CurrentModeUpdate):
            mode_id = update.current_mode_id
            if mode_id is not None:
                self._current_mode_id = mode_id
                await self._event_sink(AgentModeChanged(agent_id=self.agent_id, mode_id=mode_id))
        elif isinstance(update, ConfigOptionUpdate):
            old_values = {opt.id: opt.current_value for opt in self._config_options}
            old_shape = {opt.id: _allowed_values(opt) for opt in self._config_options}
            self._config_options = list(update.config_options)
            # Options can appear and disappear with the model (Kiro v3 effortLevel and
            # thinking; Claude effort), which a per-value change cannot express.
            if old_shape != {opt.id: _allowed_values(opt) for opt in self._config_options}:
                await self._event_sink(
                    ConfigOptionsReceived(
                        agent_id=self.agent_id, config_options=self._config_options
                    )
                )
            for opt in self._config_options:
                old_val = old_values.get(opt.id)
                if old_val != opt.current_value:
                    await self._event_sink(
                        ConfigOptionChanged(
                            agent_id=self.agent_id,
                            config_id=opt.id,
                            value=opt.current_value,
                        )
                    )
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

    def _record_tool_kind(self, update: Any) -> None:
        """Remember a tool call's kind by toolCallId until the call completes or fails.

        Runs synchronously in ``session_update``, not in the emission task: the SDK
        dispatches ``request_permission`` as its own task, which can run before the
        tool_call's emission task, so recording there would lose the race.
        """
        if not isinstance(update, (ToolCallStart, ToolCallProgress)) or not update.tool_call_id:
            return
        if update.status in ("completed", "failed"):
            self._tool_kinds.pop(update.tool_call_id, None)
        elif update.kind:
            self._tool_kinds[update.tool_call_id] = update.kind

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
        # Some harnesses (Kiro v3) omit toolCall.kind here; the earlier tool_call has it.
        kind = tool_call.kind or self._tool_kinds.get(request_id) or "other"
        hints = self._ext.permission_hints(tool_call, dict(kwargs))
        await self._sm.transition(AgentState.AWAITING_PERMISSION)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._permission_futures[request_id] = future
        await self._event_sink(
            PermissionRequested(
                agent_id=self.agent_id,
                request_id=request_id,
                title=tool_call.title or "",
                kind=kind,
                options=list(options),
                tool_identity=hints.tool_identity,
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
