"""ACP session wrapping one agent subprocess."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from acp.schema import (
    ClientCapabilities,
    DeniedOutcome,
    FileSystemCapability,
    Implementation,
    McpServerStdio,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
)

from acp import spawn_agent_process, text_block
from synth_acp.models.agent import TRANSITIONS, AgentState, InvalidTransitionError
from synth_acp.models.events import (
    AgentStateChanged,
    BrokerError,
    BrokerEvent,
    MessageChunkReceived,
    ToolCallUpdated,
)

log = logging.getLogger(__name__)

EventSink = Callable[[BrokerEvent], Awaitable[None]]


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
        self._session_id: str | None = None
        self._permission_future: asyncio.Future[str] | None = None

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
            async with spawn_agent_process(self, self._binary, *self._args, cwd=self._cwd) as (
                conn,
                proc,
            ):
                self._conn = conn
                await conn.initialize(
                    protocol_version=1,
                    client_capabilities=ClientCapabilities(
                        fs=FileSystemCapability(read_text_file=False, write_text_file=False),
                        terminal=False,
                    ),
                    client_info=Implementation(name="synth", version="0.1.0"),
                )
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
            await self._conn.prompt(session_id=self._session_id, prompt=[text_block(text)])
        finally:
            if self.state == AgentState.BUSY:
                await self._set_state(AgentState.IDLE)

    async def cancel(self) -> None:
        """Cancel the active prompt turn."""
        if self._conn and self._session_id and self.state == AgentState.BUSY:
            await self._conn.cancel(session_id=self._session_id)

    async def terminate(self) -> None:
        """Request termination. The run() finally block handles state."""
        if self._permission_future and not self._permission_future.done():
            self._permission_future.cancel()

    # --- ACP Client callbacks (called by SDK) ---
    # ARG002 suppressed: these signatures are required by the acp.interfaces.Client protocol.

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Called by ACP SDK when agent streams a response."""
        su = getattr(update, "session_update", None) or getattr(update, "sessionUpdate", None)
        if su in ("agent_message_chunk",):
            content = getattr(update, "content", None)
            if content:
                text = getattr(content, "text", None)
                if text:
                    await self._event_sink(MessageChunkReceived(agent_id=self.agent_id, chunk=text))
        elif su in ("tool_call",):
            await self._event_sink(
                ToolCallUpdated(
                    agent_id=self.agent_id,
                    tool_call_id=getattr(update, "tool_call_id", "") or "",
                    title=getattr(update, "title", "") or "",
                    kind=getattr(update, "kind", "other") or "other",
                    status=getattr(update, "status", "pending") or "pending",
                )
            )
        elif su in ("tool_call_update",):
            await self._event_sink(
                ToolCallUpdated(
                    agent_id=self.agent_id,
                    tool_call_id=getattr(update, "tool_call_id", "") or "",
                    title=getattr(update, "title", "") or "",
                    kind=getattr(update, "kind", "other") or "other",
                    status=getattr(update, "status", "in_progress") or "in_progress",
                )
            )

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Called by ACP SDK when agent requests permission. Auto-cancels for now."""
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    def on_connect(self, conn: Any) -> None:
        """Called when the ACP connection is established."""
        self._conn = conn
