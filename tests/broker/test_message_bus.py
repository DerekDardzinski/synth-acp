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
        await bus.stop()
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


class TestDeliverThenMark:
    """Tests for deliver-first, mark-on-success message lifecycle."""

    async def test_delivery_failure_leaves_messages_pending(self, tmp_path: Path) -> None:
        """Delivery failure must not lose messages — they stay pending for retry."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"

        # Set up DB
        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
            "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?, 'chat')",
            (session_id, now),
        )
        conn.commit()
        conn.close()

        async def fail_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            return False

        bus = MessageBus(db_path, session_id, fail_deliver, fallback_interval=30.0)
        await bus._deliver_pending()

        # Verify message is still pending
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT status FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        conn.close()
        assert row[0] == "pending"

    async def test_successful_delivery_marks_delivered(self, tmp_path: Path) -> None:
        """Successful delivery must mark messages as 'delivered' with timestamp."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
            "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?, 'chat')",
            (session_id, now),
        )
        conn.commit()
        conn.close()

        async def ok_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            return True

        bus = MessageBus(db_path, session_id, ok_deliver, fallback_interval=30.0)
        await bus._deliver_pending()

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status, delivered_at FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "delivered"
        assert row[1] is not None

    async def test_concurrent_terminate_does_not_mark_expired_as_delivered(self, tmp_path: Path) -> None:
        """WHERE status='pending' guard prevents marking expired messages as delivered."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        delivery_event = asyncio.Event()
        proceed_event = asyncio.Event()

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
            "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?, 'chat')",
            (session_id, now),
        )
        conn.commit()
        conn.close()

        async def slow_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            delivery_event.set()  # Signal that delivery started
            await proceed_event.wait()  # Wait for terminate to run
            return True

        bus = MessageBus(db_path, session_id, slow_deliver, fallback_interval=30.0)

        # Start delivery in background
        deliver_task = asyncio.create_task(bus._deliver_pending())

        # Wait for delivery to start (message fetched, delivery in progress)
        await delivery_event.wait()

        # Simulate terminate expiring the message between delivery and mark
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE messages SET status = 'expired' WHERE to_agent = 'a1' AND session_id = ? AND status = 'pending'",
            (session_id,),
        )
        conn.commit()
        conn.close()

        # Let delivery complete (returns True)
        proceed_event.set()
        await deliver_task

        # Message should remain expired — WHERE status='pending' guard prevented overwrite
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT status FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        conn.close()
        assert row[0] == "expired"


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

        bus = MessageBus(db_path, session_id, _noop_deliver, process_commands=capture_commands, fallback_interval=30.0)
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
        bus = MessageBus(db_path, session_id, _noop_deliver, process_commands=capture_commands, fallback_interval=30.0)
        await bus.start()
        try:
            # Give the delivery loop time to run startup + first cycle
            await asyncio.sleep(0.2)

            # Command should have been recovered and processed
            assert len(processed) == 1
            assert processed[0][0] == cmd_id
        finally:
            await bus.stop()


class TestBackoffAndWake:
    """Tests for per-agent delivery backoff and wake()."""

    async def test_backoff_prevents_retry_within_2s(self, tmp_path: Path) -> None:
        """Failed delivery must not retry within 2s — prevents tight loop on dead agent."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        deliver_calls: list[str] = []

        async def track_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            deliver_calls.append(agent_id)
            return False  # Always fail

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
            "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?, 'chat')",
            (session_id, now),
        )
        conn.commit()
        conn.close()

        bus = MessageBus(db_path, session_id, track_deliver, fallback_interval=30.0)

        # First call — should attempt delivery (and fail)
        await bus._deliver_pending()
        assert len(deliver_calls) == 1

        # Second call within 2s — should skip due to backoff
        await bus._deliver_pending()
        assert len(deliver_calls) == 1  # No new call

    async def test_wake_clears_backoff_for_agent(self, tmp_path: Path) -> None:
        """wake(agent_id) must clear backoff so next delivery cycle retries."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        deliver_calls: list[str] = []

        async def track_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            deliver_calls.append(agent_id)
            return False

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
            "VALUES (?, 'sender', 'a1', 'hello', 'pending', ?, 'chat')",
            (session_id, now),
        )
        conn.commit()
        conn.close()

        bus = MessageBus(db_path, session_id, track_deliver, fallback_interval=30.0)

        # First call fails — enters backoff
        await bus._deliver_pending()
        assert len(deliver_calls) == 1

        # wake clears backoff
        bus.wake("a1")

        # Now delivery should be attempted again
        await bus._deliver_pending()
        assert len(deliver_calls) == 2


class TestInMemoryAndDbDelivery:
    """Tests for unified in-memory + DB pending delivery."""

    async def test_in_memory_and_db_pending_combined_on_delivery(self, tmp_path: Path) -> None:
        """Both in-memory and DB pending must be delivered together in one prompt."""
        import sqlite3
        import time

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"
        delivered_text: list[str] = []

        async def capture_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            delivered_text.append(text)
            return True

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at, kind) "
            "VALUES (?, 'db-sender', 'a1', 'db-message', 'pending', ?, 'chat')",
            (session_id, now),
        )
        conn.commit()
        conn.close()

        bus = MessageBus(db_path, session_id, capture_deliver, fallback_interval=30.0)
        bus.enqueue_pending("a1", "mem-sender", "mem-message")

        await bus._deliver_pending()

        assert len(delivered_text) == 1
        assert "mem-message" in delivered_text[0]
        assert "db-message" in delivered_text[0]

    async def test_in_memory_pending_cleared_on_success(self, tmp_path: Path) -> None:
        """After successful delivery, in-memory pending must be cleared."""
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "test.db"
        session_id = "s1"

        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        conn.close()

        async def ok_deliver(agent_id: str, text: str, senders: list[str]) -> bool:
            return True

        bus = MessageBus(db_path, session_id, ok_deliver, fallback_interval=30.0)
        bus.enqueue_pending("a1", "sender", "hello")
        bus.enqueue_raw("a1", "raw prompt")

        await bus._deliver_pending()

        assert bus.pop_pending("a1") is None


# ---------------------------------------------------------------------------
# Race condition reproducers: in-flight enqueue preservation
# ---------------------------------------------------------------------------


class TestInFlightEnqueuePreservation:
    """Verify that messages enqueued during an in-flight deliver await are not lost."""

    async def test_in_memory_pending_lost_during_delivery_await(self, tmp_path: Path) -> None:
        """A message enqueued during the await on deliver() must survive."""
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db = tmp_path / "race.db"
        conn = sqlite3.connect(str(db))
        try:
            ensure_schema_sync(conn)
        finally:
            conn.close()

        delivered: list[tuple[str, str, list[str]]] = []
        deliver_started = asyncio.Event()
        allow_deliver = asyncio.Event()

        async def slow_deliver(agent_id: str, text: str, from_agents: list[str]) -> bool:
            deliver_started.set()
            await allow_deliver.wait()
            delivered.append((agent_id, text, from_agents))
            return True

        bus = MessageBus(db, "session-1", slow_deliver, fallback_interval=10.0)

        bus.enqueue_pending("agent-a", "alice", "first")
        deliver_task = asyncio.create_task(bus._deliver_pending())

        await asyncio.wait_for(deliver_started.wait(), timeout=1.0)

        # Race: a second message arrives while the bus is mid-delivery of the first.
        bus.enqueue_pending("agent-a", "bob", "second")

        allow_deliver.set()
        await deliver_task

        assert len(delivered) == 1
        assert delivered[0][0] == "agent-a"
        assert "first" in delivered[0][1]

        # The second message MUST still be pending in memory.
        assert bus._pending.get("agent-a") == [("bob", "second")], (
            f"Bug: second message lost; _pending={bus._pending}"
        )

    async def test_in_memory_raw_overwritten_during_delivery_await(self, tmp_path: Path) -> None:
        """A raw prompt enqueued during the await on deliver() must survive."""
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db = tmp_path / "race.db"
        conn = sqlite3.connect(str(db))
        try:
            ensure_schema_sync(conn)
        finally:
            conn.close()

        delivered: list[tuple[str, str, list[str]]] = []
        deliver_started = asyncio.Event()
        allow_deliver = asyncio.Event()

        async def slow_deliver(agent_id: str, text: str, from_agents: list[str]) -> bool:
            deliver_started.set()
            await allow_deliver.wait()
            delivered.append((agent_id, text, from_agents))
            return True

        bus = MessageBus(db, "session-1", slow_deliver, fallback_interval=10.0)

        bus.enqueue_raw("agent-a", "initial-prompt-v1")
        deliver_task = asyncio.create_task(bus._deliver_pending())

        await asyncio.wait_for(deliver_started.wait(), timeout=1.0)

        # Race: while the bus is mid-delivery of v1, an updated raw prompt arrives.
        bus.enqueue_raw("agent-a", "initial-prompt-v2")

        allow_deliver.set()
        await deliver_task

        assert len(delivered) == 1
        assert delivered[0][0] == "agent-a"
        assert "initial-prompt-v1" in delivered[0][1]

        # v2 MUST still be pending.
        assert bus._pending_raw.get("agent-a") == "initial-prompt-v2", (
            f"Bug: v2 raw prompt lost; _pending_raw={bus._pending_raw}"
        )