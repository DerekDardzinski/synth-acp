"""Tests for MessagePoller."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

from synth_acp.broker.poller import MessagePoller


def _init_db(db_path: Path) -> None:
    """Create the schema in a test database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS agents ("
        "agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'active', registered INTEGER NOT NULL);"
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "from_agent TEXT NOT NULL, to_agent TEXT NOT NULL, body TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', created_at INTEGER NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'chat', "
        "reply_to INTEGER REFERENCES messages(id), "
        "delivered_at INTEGER);"
    )
    conn.commit()
    conn.close()


def _insert_message(db_path: Path, from_agent: str, to_agent: str, body: str) -> int:
    """Insert a pending message and return its ID."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) VALUES ('sess-1', ?, ?, ?, 'pending', 1000)",
        (from_agent, to_agent, body),
    )
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return msg_id  # type: ignore[return-value]


class TestPollerDelivery:
    async def test_poller_when_version_changes_delivers_to_idle(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        _init_db(db_path)

        delivered: list[tuple[str, str]] = []

        async def deliver(agent_id: str, text: str, from_agents: list[str]) -> bool:
            delivered.append((agent_id, text))
            return True

        poller = MessagePoller(db_path, deliver, "sess-1")
        await poller.start()

        # Insert a message after poller has started
        _insert_message(db_path, "agent-a", "agent-b", "hello")

        # Wait for poller to pick it up
        for _ in range(20):
            await asyncio.sleep(0.05)
            if delivered:
                break

        await poller.stop()

        assert len(delivered) == 1
        assert delivered[0][0] == "agent-b"
        assert "hello" in delivered[0][1]

        # Verify status changed to delivered
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT status FROM messages WHERE id = 1").fetchone()
        conn.close()
        assert row[0] == "delivered"

    async def test_poller_when_agent_busy_skips_delivery(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        _init_db(db_path)

        async def deliver(agent_id: str, text: str, from_agents: list[str]) -> bool:
            return False  # Simulate busy agent

        poller = MessagePoller(db_path, deliver, "sess-1")
        await poller.start()

        _insert_message(db_path, "agent-a", "agent-b", "hello")

        # Give poller time to attempt delivery
        await asyncio.sleep(0.3)
        await poller.stop()

        # Message should still be pending
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT status FROM messages WHERE id = 1").fetchone()
        conn.close()
        assert row[0] == "pending"


class TestPollerStop:
    async def test_poller_stop_when_called_awaits_cycle(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        _init_db(db_path)

        deliver_count = 0

        async def deliver(agent_id: str, text: str, from_agents: list[str]) -> bool:
            nonlocal deliver_count
            deliver_count += 1
            return True

        poller = MessagePoller(db_path, deliver, "sess-1")
        await poller.start()
        await poller.stop()

        # Insert a message after stop — should not be delivered
        _insert_message(db_path, "agent-a", "agent-b", "late message")
        await asyncio.sleep(0.2)

        assert deliver_count == 0


def _init_db_with_commands(db_path: Path) -> None:
    """Create the schema including agent_commands table."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "from_agent TEXT NOT NULL, to_agent TEXT NOT NULL, body TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', created_at INTEGER NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'chat', "
        "reply_to INTEGER REFERENCES messages(id), "
        "delivered_at INTEGER);"
        "CREATE TABLE IF NOT EXISTS agent_commands ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "from_agent TEXT NOT NULL, command TEXT NOT NULL, payload TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', error TEXT, "
        "created_at INTEGER NOT NULL);"
    )
    conn.commit()
    conn.close()


class TestPollerCommands:
    async def test_poller_when_command_inserted_calls_command_fn(self, tmp_path: Path) -> None:
        import aiosqlite

        db_path = tmp_path / "test.db"
        _init_db_with_commands(db_path)

        received: list[list[tuple[int, str, str, str]]] = []

        async def process_commands(cmds: list[tuple[int, str, str, str]]) -> None:
            received.append(cmds)

        poller = MessagePoller(
            db_path,
            deliver=AsyncMock(return_value=True),
            session_id="sess-1",
            process_commands=process_commands,
        )

        # Insert a pending command
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) "
            "VALUES ('sess-1', 'agent-a', 'launch', '{\"agent_id\":\"w1\"}', 'pending', 1000)",
        )
        conn.commit()
        conn.close()

        # Call _process_pending_commands directly
        async with aiosqlite.connect(db_path) as db:
            await poller._process_pending_commands(db)

        assert len(received) == 1
        assert len(received[0]) == 1
        cmd_id, from_agent, command, payload = received[0][0]
        assert from_agent == "agent-a"
        assert command == "launch"
        assert "w1" in payload
