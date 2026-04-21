"""MessageBus — notification-driven inter-agent message delivery."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiosqlite

from synth_acp.db import ensure_schema_async

log = logging.getLogger(__name__)

type DeliverFn = Callable[[str, str, list[str]], Awaitable[bool]]
type CommandFn = Callable[[list[tuple[int, str, str, str]]], Awaitable[None]]


def _format_message(from_agent: str, body: str, kind: str) -> str:
    """Format a message based on its kind for delivery to an agent."""
    if kind == "system":
        return f"[System notification — no action required]: {body}"
    if kind == "response":
        return f"[Response from {from_agent}]: {body}"
    if kind == "request":
        return f"[Request from {from_agent}]: {body}"
    return f"[Message from {from_agent}]: {body}"


class MessageBus:
    """Notification-driven message delivery with fallback polling."""

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        deliver: DeliverFn,
        process_commands: CommandFn | None = None,
        fallback_interval: float = 2.0,
    ) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._deliver = deliver
        self._process_commands = process_commands
        self._fallback_interval = fallback_interval
        self._pending: dict[str, list[tuple[str, str]]] = {}
        self._pending_raw: dict[str, str] = {}
        self._stopped = False
        self._tasks: list[asyncio.Task[None]] = []
        self._server: asyncio.Server | None = None
        self._socket_path = str(Path(tempfile.gettempdir()) / f"synth-{session_id}.sock")
        self._wake_event = asyncio.Event()
        self._delivery_db: aiosqlite.Connection | None = None

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def start(self) -> None:
        """Start the notification listener and delivery loop."""
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=self._socket_path)
        self._tasks.append(asyncio.create_task(self._delivery_loop(), name="msg-bus-delivery"))

    async def stop(self, timeout: float = 2.0) -> None:
        """Stop all listeners and cancel tasks."""
        self._stopped = True
        self._wake_event.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)
        # Safety net: close the delivery loop DB if the task didn't
        # finish in time — prevents a non-daemon aiosqlite thread
        # from keeping the process alive after the event loop exits.
        if self._delivery_db is not None:
            try:
                await self._delivery_db.close()
            except Exception:
                pass
            self._delivery_db = None
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()

    def enqueue_pending(self, agent_id: str, from_agent: str, body: str) -> None:
        """Queue a message for an agent that isn't IDLE yet."""
        self._pending.setdefault(agent_id, []).append((from_agent, body))

    def enqueue_raw(self, agent_id: str, text: str) -> None:
        """Queue a raw prompt for an agent, delivered without formatting."""
        self._pending_raw[agent_id] = text

    def pop_pending(self, agent_id: str) -> str | None:
        """Pop and return combined pending messages, or None if empty."""
        raw = self._pending_raw.pop(agent_id, None)
        messages = self._pending.pop(agent_id, None)
        if raw and messages:
            formatted = "\n\n".join(_format_message(sender, body, "chat") for sender, body in messages)
            return raw + "\n\n" + formatted
        if raw:
            return raw
        if not messages:
            return None
        return "\n\n".join(_format_message(sender, body, "chat") for sender, body in messages)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not self._stopped:
                data = await reader.read(64)
                if not data:
                    break
                self._wake_event.set()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def _delivery_loop(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                self._delivery_db = db
                await ensure_schema_async(db)
                await db.commit()
                await self._deliver_pending(db)
                await self._process_pending_commands(db)
                while not self._stopped:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=self._fallback_interval)
                    except TimeoutError:
                        pass
                    except asyncio.CancelledError:
                        raise
                    if self._stopped:
                        break
                    try:
                        await self._deliver_pending(db)
                        await self._process_pending_commands(db)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception("MessageBus delivery error")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MessageBus connection error")
        finally:
            self._delivery_db = None

    async def _deliver_pending(self, db: aiosqlite.Connection) -> None:
        """Query pending messages, group by recipient, deliver with kind-aware formatting."""
        rows = await db.execute_fetchall(
            "SELECT id, from_agent, to_agent, body, kind FROM messages "
            "WHERE status = 'pending' AND session_id = ? ORDER BY created_at",
            [self._session_id],
        )
        by_agent: dict[str, list[tuple[int, str, str, str, str]]] = {}
        for row in rows:
            by_agent.setdefault(row[2], []).append(row)  # type: ignore[arg-type]
        for agent_id, messages in by_agent.items():
            ids = [m[0] for m in messages]
            placeholders = ",".join("?" * len(ids))
            now = int(time.time() * 1000)
            await db.execute(
                f"UPDATE messages SET status = 'delivered', delivered_at = ? WHERE id IN ({placeholders})",
                [now, *ids],
            )
            await db.commit()

            combined = "\n\n".join(_format_message(m[1], m[3], m[4]) for m in messages)
            senders = list({m[1] for m in messages if m[4] != "system"})
            success = await self._deliver(agent_id, combined, senders)
            if not success:
                await db.execute(
                    f"UPDATE messages SET status = 'pending', delivered_at = NULL WHERE id IN ({placeholders})",
                    ids,
                )
                await db.commit()

    async def _process_pending_commands(self, db: aiosqlite.Connection) -> None:
        """Query pending agent commands and pass them to the command callback."""
        if self._process_commands is None:
            return
        rows = await db.execute_fetchall(
            "SELECT id, from_agent, command, payload FROM agent_commands "
            "WHERE status = 'pending' AND session_id = ? ORDER BY created_at",
            [self._session_id],
        )
        if rows:
            await self._process_commands([(r[0], r[1], r[2], r[3]) for r in rows])
