"""Tests for MessageBus."""

from __future__ import annotations

import asyncio
from pathlib import Path

from synth_acp.broker.message_bus import MessageBus


async def _noop_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
    return True


class TestMessageBusLifecycle:
    async def test_stop_does_not_hang_when_delivery_is_slow(self, tmp_path: Path) -> None:
        async def slow_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            await asyncio.sleep(10)
            return True

        bus = MessageBus(tmp_path / "test.db", "s1", slow_deliver, fallback_interval=0.1)
        await bus.start()
        t0 = asyncio.get_event_loop().time()
        await bus.stop(timeout=0.5)
        elapsed = asyncio.get_event_loop().time() - t0
        assert elapsed < 2.0

    async def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        bus = MessageBus(tmp_path / "test.db", "s1", _noop_deliver, fallback_interval=0.1)
        await bus.start()
        await bus.stop()
        await bus.stop()  # Should not raise

    async def test_socket_cleaned_up_on_stop(self, tmp_path: Path) -> None:
        bus = MessageBus(tmp_path / "test.db", "s1", _noop_deliver, fallback_interval=0.1)
        await bus.start()
        sock = Path(bus.socket_path)
        assert sock.exists()
        await bus.stop()
        assert not sock.exists()


class TestPendingMessages:
    def test_enqueue_pending_stores_multiple_messages(self) -> None:
        bus = MessageBus(Path("/tmp/unused.db"), "s1", _noop_deliver)
        bus.enqueue_pending("a1", "sender1", "hello")
        bus.enqueue_pending("a1", "sender2", "world")
        result = bus.pop_pending("a1")
        assert result is not None
        assert "[Message from sender1]: hello" in result
        assert "[Message from sender2]: world" in result


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

        async def deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            delivered.append(agent_id)
            return True

        bus = MessageBus(db_path, session_id, deliver, fallback_interval=30.0)
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

        async def deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            delivered.append(agent_id)
            return True

        bus = MessageBus(db_path, session_id, deliver, fallback_interval=0.3)
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
