"""MessagePoller — polls SQLite for new inter-agent messages."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

DeliverFn = Callable[[str, str, list[str]], Awaitable[bool]]


class MessagePoller:
    """Polls SQLite via PRAGMA data_version and delivers pending messages.

    Args:
        db_path: Path to the SQLite database.
        deliver: Callback to deliver combined message text to an agent.
            Returns True on success, False if agent is not idle or delivery fails.
    """

    def __init__(self, db_path: Path, deliver: DeliverFn, session_id: str) -> None:
        self._db_path = db_path
        self._deliver = deliver
        self._session_id = session_id
        self._stopped = False
        self._task: asyncio.Task[None] | None = None
        self._last_version: int = 0

    async def start(self) -> None:
        """Start the polling loop as a background task."""
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop polling and await the current cycle to finish."""
        self._stopped = True
        if self._task:
            await self._task

    async def _poll_loop(self) -> None:
        """Poll for data_version changes at 100ms intervals."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS messages ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "session_id TEXT NOT NULL,"
                    "from_agent TEXT NOT NULL,"
                    "to_agent TEXT NOT NULL,"
                    "body TEXT NOT NULL,"
                    "status TEXT NOT NULL DEFAULT 'pending',"
                    "created_at INTEGER NOT NULL,"
                    "claimed_at INTEGER)"
                )
                await db.commit()

                # Initial sweep + baseline: deliver anything pending, then snapshot version
                await self._deliver_pending(db)
                cursor = await db.execute("PRAGMA data_version")
                row = await cursor.fetchone()
                self._last_version = row[0] if row else 0

                while not self._stopped:
                    try:
                        cursor = await db.execute("PRAGMA data_version")
                        row = await cursor.fetchone()
                        version = row[0] if row else 0
                        if version != self._last_version:
                            self._last_version = version
                            await self._deliver_pending(db)
                    except Exception:
                        log.exception("Poller error")
                    await asyncio.sleep(0.1)
        except Exception:
            log.exception("Poller connection error")

    async def _deliver_pending(self, db: aiosqlite.Connection) -> None:
        """Query pending messages, group by recipient, deliver, mark delivered."""
        rows = await db.execute_fetchall(
            "SELECT id, from_agent, to_agent, body FROM messages "
            "WHERE status = 'pending' AND session_id = ? ORDER BY created_at",
            [self._session_id],
        )
        by_agent: dict[str, list[tuple[int, str, str, str]]] = {}
        for row in rows:
            by_agent.setdefault(row[2], []).append(row)  # type: ignore[arg-type]
        for agent_id, messages in by_agent.items():
            combined = "\n\n".join(f"[Message from {m[1]}]: {m[3]}" for m in messages)
            senders = list({m[1] for m in messages})
            success = await self._deliver(agent_id, combined, senders)
            if success:
                ids = [m[0] for m in messages]
                placeholders = ",".join("?" * len(ids))
                await db.execute(
                    f"UPDATE messages SET status = 'delivered' WHERE id IN ({placeholders})",
                    ids,
                )
                await db.commit()
