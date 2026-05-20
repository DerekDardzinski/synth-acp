"""MessageBus — notification-driven DB poller for inter-agent messages and commands."""

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

type OnMessageFn = Callable[[str, str, str], Awaitable[None]]
type CommandFn = Callable[[list[tuple[int, str, str, str]]], Awaitable[None]]


class MessageBus:
    """Notification-driven DB poller for inter-agent messages.

    Polls SQLite for pending messages and commands. Messages are handed
    to the broker via on_message callback. The broker decides when/how
    to deliver them to agents. No in-memory buffers, no retry logic.
    """

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        on_message: OnMessageFn,
        process_commands: CommandFn | None = None,
        fallback_interval: float = 2.0,
    ) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._on_message = on_message
        self._process_commands = process_commands
        self._fallback_interval = fallback_interval
        self._stopped = False
        self._tasks: list[asyncio.Task[None]] = []
        self._server: asyncio.Server | None = None
        self._socket_path = str(Path(tempfile.gettempdir()) / f"synth-{session_id}.sock")
        self._wake_event = asyncio.Event()

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def wake(self, agent_id: str | None = None) -> None:  # noqa: ARG002
        """Wake the poll loop."""
        self._wake_event.set()

    async def start(self) -> None:
        """Start the notification listener and poll loop."""
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=self._socket_path)
        self._tasks.append(asyncio.create_task(self._poll_loop(), name="msg-bus-poll"))

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
            await asyncio.wait(self._tasks, timeout=2.0)
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()

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

    async def _poll_loop(self) -> None:
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

            await self._poll_messages()
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
                    await self._poll_messages()
                    await self._process_pending_commands()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("MessageBus poll error")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MessageBus connection error")

    async def _poll_messages(self) -> None:
        """Fetch pending messages from DB, hand to broker, mark delivered."""
        session_id = self._session_id

        def _fetch(conn: sqlite3.Connection) -> list[tuple[int, str, str, str]]:
            return conn.execute(
                "SELECT id, from_agent, to_agent, body FROM messages "
                "WHERE status = 'pending' AND session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()

        rows = await self._db_op(_fetch)
        if not rows:
            return

        for msg_id, from_agent, to_agent, body in rows:
            try:
                await self._on_message(to_agent, body, from_agent)
            except Exception:
                log.debug("Failed to deliver message %d to %s", msg_id, to_agent, exc_info=True)
                continue

            # Mark delivered
            now = int(time.time() * 1000)

            def _mark(conn: sqlite3.Connection, *, mid: int = msg_id, ts: int = now) -> None:
                conn.execute(
                    "UPDATE messages SET status = 'delivered', delivered_at = ? "
                    "WHERE id = ? AND status = 'pending'",
                    (ts, mid),
                )
                conn.commit()

            await self._db_op(_mark)

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
