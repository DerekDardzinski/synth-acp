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
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    FileSystemCapability,
    Implementation,
    McpServerStdio,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
)
from acp.transports import default_environment

from acp import text_block
from synth_acp.models.agent import TRANSITIONS, AgentState, InvalidTransitionError
from synth_acp.models.events import (
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


def _extract_tool_call_content(update: Any) -> dict[str, Any]:
    """Extract diffs, text_content, locations, and raw_input from an ACP update.

    Args:
        update: ACP SDK ToolCallUpdate or similar object.

    Returns:
        Dict with keys diffs, text_content, locations, raw_input suitable
        for passing to ToolCallUpdated.
    """
    diffs: list[ToolCallDiff] = []
    text_parts: list[str] = []
    for item in getattr(update, "content", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "diff":
            diffs.append(
                ToolCallDiff(
                    path=getattr(item, "path", ""),
                    old_text=getattr(item, "old_text", None),
                    new_text=getattr(item, "new_text", ""),
                )
            )
        elif item_type == "content":
            inner = getattr(item, "content", None)
            if inner and getattr(inner, "type", None) == "text":
                text = getattr(inner, "text", None)
                if text:
                    text_parts.append(text)
    locations: list[ToolCallLocation] = []
    for loc in getattr(update, "locations", None) or []:
        locations.append(
            ToolCallLocation(path=getattr(loc, "path", ""), line=getattr(loc, "line", None))
        )
    text_content = "\n".join(text_parts) if text_parts else None
    return {
        "diffs": diffs,
        "text_content": text_content,
        "locations": locations,
        "raw_input": getattr(update, "raw_input", None),
    }


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
    assert process.stdout and process.stdin

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
            async with _spawn_isolated_agent(
                self, self._binary, *self._args, cwd=self._cwd
            ) as (conn, proc):
                self._conn = conn
                self._proc = proc

                init_response = await conn.initialize(
                    protocol_version=1,
                    client_capabilities=ClientCapabilities(
                        fs=FileSystemCapability(read_text_file=False, write_text_file=False),
                        terminal=False,
                    ),
                    client_info=Implementation(name="synth", version="0.1.0"),
                )
                self._capabilities = getattr(init_response, "agent_capabilities", None)
                session = await conn.new_session(cwd=self._cwd, mcp_servers=self._mcp_servers)
                self._session_id = session.session_id
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
        """Called by ACP SDK when agent streams a response."""
        su = getattr(update, "session_update", None) or getattr(update, "sessionUpdate", None)
        log.debug("session_update type=%s agent=%s", su, self.agent_id)
        if su in ("agent_message_chunk",):
            content = getattr(update, "content", None)
            if content:
                text = getattr(content, "text", None)
                if text:
                    await self._event_sink(MessageChunkReceived(agent_id=self.agent_id, chunk=text))
        elif su in ("tool_call",):
            extra = _extract_tool_call_content(update)
            await self._event_sink(
                ToolCallUpdated(
                    agent_id=self.agent_id,
                    tool_call_id=getattr(update, "tool_call_id", "") or "",
                    title=getattr(update, "title", "") or "",
                    kind=getattr(update, "kind", "other") or "other",
                    status=getattr(update, "status", "pending") or "pending",
                    **extra,
                )
            )
        elif su in ("tool_call_update",):
            extra = _extract_tool_call_content(update)
            await self._event_sink(
                ToolCallUpdated(
                    agent_id=self.agent_id,
                    tool_call_id=getattr(update, "tool_call_id", "") or "",
                    title=getattr(update, "title", "") or "",
                    kind=getattr(update, "kind", "other") or "other",
                    status=getattr(update, "status", "in_progress") or "in_progress",
                    **extra,
                )
            )
        elif su == "agent_thought_chunk":
            content = getattr(update, "content", None)
            if content:
                text = getattr(content, "text", None)
                if text:
                    await self._event_sink(AgentThoughtReceived(agent_id=self.agent_id, chunk=text))
        elif su == "usage_update":
            cost = getattr(update, "cost", None)
            await self._event_sink(
                UsageUpdated(
                    agent_id=self.agent_id,
                    size=getattr(update, "size", 0),
                    used=getattr(update, "used", 0),
                    cost_amount=getattr(cost, "amount", None) if cost else None,
                    cost_currency=getattr(cost, "currency", None) if cost else None,
                )
            )

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
