"""ACPBroker — owns all agent sessions and routes events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError, BrokerEvent

log = logging.getLogger(__name__)


class ACPBroker:
    """Central orchestration service for agent sessions."""

    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._sessions: dict[str, ACPSession] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._event_queue: asyncio.Queue[BrokerEvent] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

    async def _sink(self, event: BrokerEvent) -> None:
        """Event sink passed to sessions."""
        await self._event_queue.put(event)

    async def launch(self, agent_id: str) -> None:
        """Launch an agent by ID from the config."""
        agent_cfg = next((a for a in self._config.agents if a.id == agent_id), None)
        if not agent_cfg:
            await self._sink(
                BrokerError(agent_id=agent_id, message=f"No config for agent '{agent_id}'")
            )
            return

        session = ACPSession(
            agent_id=agent_cfg.id,
            binary=agent_cfg.binary,
            args=agent_cfg.args,
            cwd=agent_cfg.cwd,
            event_sink=self._sink,
        )
        self._sessions[agent_id] = session
        self._tasks[agent_id] = asyncio.create_task(session.run())

    async def prompt(self, agent_id: str, text: str) -> None:
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
        await session.prompt(text)

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

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel prompts, terminate sessions, signal done."""
        for session in self._sessions.values():
            if session.state == AgentState.BUSY:
                await session.cancel()

        for session in self._sessions.values():
            if session.state != AgentState.TERMINATED:
                await session.terminate()

        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._shutdown_event.set()
