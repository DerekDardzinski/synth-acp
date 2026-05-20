"""Tests for MessageBus."""

from __future__ import annotations

import asyncio
from pathlib import Path

from synth_acp.broker.message_bus import MessageBus


async def _noop_on_message(to_agent: str, body: str, from_agent: str) -> None:
    pass


class TestMessageBusLifecycle:
    async def test_stop_does_not_hang_when_delivery_is_slow(self, tmp_path: Path) -> None:
        async def slow_on_message(to_agent: str, body: str, from_agent: str) -> None:
            await asyncio.sleep(10)

        bus = MessageBus(tmp_path / "test.db", "s1", slow_on_message, fallback_interval=0.1)
        await bus.start()
        t0 = asyncio.get_event_loop().time()
        await bus.stop()
        elapsed = asyncio.get_event_loop().time() - t0
        assert elapsed < 2.0

    async def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        bus = MessageBus(tmp_path / "test.db", "s1", _noop_on_message, fallback_interval=0.1)
        await bus.start()
        await bus.stop()
        await bus.stop()  # Should not raise

    async def test_socket_cleaned_up_on_stop(self, tmp_path: Path) -> None:
        bus = MessageBus(tmp_path / "test.db", "s1", _noop_on_message, fallback_interval=0.1)
        await bus.start()
        sock = Path(bus.socket_path)
        assert sock.exists()
        await bus.stop()
        assert not sock.exists()



class TestMessageBusDelivery:
    async def test_notification_triggers_immediate_delivery(self, tmp_path: Path) -> None:
        """A socket byte must wake the delivery loop and deliver within 100ms.
        Without this, inter-agent latency is bounded by fallback_interval."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        delivered: list[str] = []

        async def on_message(to_agent: str, body: str, from_agent: str) -> None:
            delivered.append(to_agent)

        bus = MessageBus(db_path, session_id, on_message, fallback_interval=30.0)
        await bus.start()
        try:
            # Insert a pending message via sync sqlite
            conn = sqlite3.connect(str(db_path))
            ensure_schema_sync(conn)
            now = int(time.time() * 1000)
            conn.execute(
                "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, 'active', ?)",
                ("a1", session_id, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
                "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?)",
                (session_id, now),
            )
            conn.commit()
            conn.close()

            # Send notification byte
            _, writer = await asyncio.open_unix_connection(bus.socket_path)
            writer.write(b"\x01")
            await writer.drain()
            writer.close()

            # Should deliver within 100ms, not 30s
            await asyncio.sleep(0.2)
            assert "a1" in delivered
        finally:
            await bus.stop()

    async def test_fallback_poll_delivers_without_notification(self, tmp_path: Path) -> None:
        """Messages must be delivered even without a socket notification,
        within the fallback interval. Catches the case where MCP server
        fails to send the wake-up byte."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        delivered: list[str] = []

        async def on_message(to_agent: str, body: str, from_agent: str) -> None:
            delivered.append(to_agent)

        bus = MessageBus(db_path, session_id, on_message, fallback_interval=0.3)
        await bus.start()
        try:
            # Insert message after bus started — no notification sent
            conn = sqlite3.connect(str(db_path))
            ensure_schema_sync(conn)
            now = int(time.time() * 1000)
            conn.execute(
                "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, 'active', ?)",
                ("a1", session_id, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
                "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?)",
                (session_id, now),
            )
            conn.commit()
            conn.close()

            # Wait for fallback poll
            await asyncio.sleep(1.0)
            assert "a1" in delivered
        finally:
            await bus.stop()


class TestAtomicCommandClaim:
    """Tests for atomic command claiming and startup recovery."""

    async def test_commands_claimed_atomically(self, tmp_path: Path) -> None:
        """Commands must transition to 'processing' before callback runs."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        processed: list[tuple[int, str, str, str]] = []

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, 'agent1', 'launch_agent', '{}', 'pending', ?)",
            (session_id, now),
        )
        conn.commit()
        cmd_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        async def capture_commands(cmds: list[tuple[int, str, str, str]]) -> None:
            processed.extend(cmds)

        bus = MessageBus(db_path, session_id, _noop_on_message, process_commands=capture_commands, fallback_interval=30.0)
        await bus._process_pending_commands()

        # Command should be 'processing' in DB (callback doesn't change it here)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT status FROM agent_commands WHERE id = ?", (cmd_id,)).fetchone()
        conn.close()
        assert row[0] == "processing"
        assert len(processed) == 1

    async def test_startup_recovery_reverts_processing_to_pending(self, tmp_path: Path) -> None:
        """Stale 'processing' commands must be reverted to 'pending' on startup."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"

        # Pre-create a stale 'processing' command (simulates crash)
        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, 'agent1', 'launch_agent', '{}', 'processing', ?)",
            (session_id, now),
        )
        conn.commit()
        cmd_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        processed: list[tuple[int, str, str, str]] = []

        async def capture_commands(cmds: list[tuple[int, str, str, str]]) -> None:
            processed.extend(cmds)

        # Start bus — startup recovery should revert 'processing' → 'pending'
        # then the first cycle should pick it up
        bus = MessageBus(db_path, session_id, _noop_on_message, process_commands=capture_commands, fallback_interval=30.0)
        await bus.start()
        try:
            # Give the delivery loop time to run startup + first cycle
            await asyncio.sleep(0.2)

            # Command should have been recovered and processed
            assert len(processed) == 1
            assert processed[0][0] == cmd_id
        finally:
            await bus.stop()

