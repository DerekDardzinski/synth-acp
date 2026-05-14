"""MessageBus — notification-driven inter-agent message delivery."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from synth_acp.db import ensure_schema_sync

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
        self._delivery_backoff: dict[str, float] = {}

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def wake(self, agent_id: str | None = None) -> None:
        """Wake the delivery loop. If agent_id provided, clear backoff for that agent."""
        if agent_id:
            self._delivery_backoff.pop(agent_id, None)
        self._wake_event.set()

    async def start(self) -> None:
        """Start the notification listener and delivery loop."""
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=self._socket_path)
        self._tasks.append(asyncio.create_task(self._delivery_loop(), name="msg-bus-delivery"))

    async def stop(self) -> None:
        """Stop all listeners and cancel tasks."""
        if self._stopped:
            return
        self._stopped = True
        self._wake_event.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
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

    async def _db_op(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run *fn* on a fresh sync sqlite3 connection in a thread-pool thread."""
        db_path = str(self._db_path)

        def _run() -> Any:
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                return fn(conn)

        return await asyncio.to_thread(_run)

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
            await self._db_op(ensure_schema_sync)

            # Startup recovery: revert stale 'processing' commands to 'pending'
            session_id = self._session_id

            def _recover_commands(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "UPDATE agent_commands SET status = 'pending' "
                    "WHERE status = 'processing' AND session_id = ?",
                    (session_id,),
                )
                conn.commit()

            await self._db_op(_recover_commands)

            await self._deliver_pending()
            await self._process_pending_commands()
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
                    await self._deliver_pending()
                    await self._process_pending_commands()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("MessageBus delivery error")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MessageBus connection error")

    async def _deliver_pending(self) -> None:
        """Fetch pending messages, deliver, mark on success. At-least-once semantics.

        Combines in-memory pending (enqueue_pending, enqueue_raw) with DB pending.
        Per-agent lock (via lifecycle.prompt) prevents concurrent delivery to same agent.
        Messages stay pending on failure — retried on next cycle.
        """
        session_id = self._session_id

        def _fetch(conn: sqlite3.Connection) -> list[tuple[int, str, str, str, str]]:
            return conn.execute(
                "SELECT id, from_agent, to_agent, body, kind FROM messages "
                "WHERE status = 'pending' AND session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()

        rows = await self._db_op(_fetch)

        by_agent: dict[str, list[tuple[int, str, str, str, str]]] = {}
        for row in rows:
            by_agent.setdefault(row[2], []).append(row)

        # Include agents with in-memory pending but no DB messages
        for agent_id in set(self._pending.keys()) | set(self._pending_raw.keys()):
            if agent_id not in by_agent:
                by_agent[agent_id] = []

        for agent_id, messages in by_agent.items():
            if time.time() - self._delivery_backoff.get(agent_id, 0) < 2.0:
                continue

            combined_parts: list[str] = []
            senders: set[str] = set()

            # In-memory raw (initial prompt text) — snapshot before await so a
            # concurrent enqueue_raw that overwrites the value during deliver
            # is preserved (we only pop if the value is still the snapshot).
            raw = self._pending_raw.get(agent_id)
            if raw:
                combined_parts.append(raw)

            # In-memory formatted messages — snapshot the count before await so
            # a concurrent enqueue_pending that appends during deliver is
            # preserved (we only consume the prefix we actually sent).
            mem_messages = self._pending.get(agent_id)
            mem_messages_count = len(mem_messages) if mem_messages else 0
            if mem_messages:
                for sender, body in mem_messages:
                    combined_parts.append(_format_message(sender, body, "chat"))
                    senders.add(sender)

            # DB messages
            for m in messages:
                combined_parts.append(_format_message(m[1], m[3], m[4]))
                if m[4] != "system":
                    senders.add(m[1])

            if not combined_parts:
                continue

            combined = "\n\n".join(combined_parts)
            success = await self._deliver(agent_id, combined, list(senders))
            if success:
                # Clear ONLY the in-memory state we actually delivered. Anything
                # enqueued during the deliver await must survive into the next
                # cycle, otherwise we drop messages silently.
                if raw is not None and self._pending_raw.get(agent_id) == raw:
                    self._pending_raw.pop(agent_id, None)
                if mem_messages_count:
                    current = self._pending.get(agent_id)
                    if current is not None:
                        remaining = current[mem_messages_count:]
                        if remaining:
                            self._pending[agent_id] = remaining
                        else:
                            self._pending.pop(agent_id, None)
                # Mark DB messages delivered
                if messages:
                    ids = [m[0] for m in messages]
                    placeholders = ",".join("?" * len(ids))
                    now = int(time.time() * 1000)

                    def _mark(conn: sqlite3.Connection, *, ids: list[int] = ids, now: int = now, ph: str = placeholders) -> None:
                        conn.execute(
                            f"UPDATE messages SET status = 'delivered', delivered_at = ? "
                            f"WHERE id IN ({ph}) AND status = 'pending'",
                            [now, *ids],
                        )
                        conn.commit()

                    await self._db_op(_mark)
                self._delivery_backoff.pop(agent_id, None)
            else:
                self._delivery_backoff[agent_id] = time.time()

    async def _process_pending_commands(self) -> None:
        """Atomically claim and process pending commands. No double-processing.

        Commands transition pending→processing atomically. Callback sets final status.
        """
        if self._process_commands is None:
            return
        session_id = self._session_id

        def _claim(conn: sqlite3.Connection) -> list[tuple[int, str, str, str]]:
            rows = conn.execute(
                "SELECT id, from_agent, command, payload FROM agent_commands "
                "WHERE status = 'pending' AND session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE agent_commands SET status = 'processing' WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()
            return rows

        rows = await self._db_op(_claim)
        if rows:
            await self._process_commands([(r[0], r[1], r[2], r[3]) for r in rows])
