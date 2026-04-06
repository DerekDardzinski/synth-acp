"""Tests for BrokerNotifier."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from synth_acp.mcp.notifier import BrokerNotifier


class TestBrokerNotifier:
    async def test_notify_sends_byte_to_socket(self, tmp_path: Path) -> None:
        sock_path = str(Path(tempfile.gettempdir()) / "synth-test-notify.sock")
        if Path(sock_path).exists():
            Path(sock_path).unlink()
        received: list[bytes] = []

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(64)
            received.append(data)
            writer.close()

        server = await asyncio.start_unix_server(handler, path=sock_path)
        try:
            notifier = BrokerNotifier(sock_path)
            await notifier.notify()
            await asyncio.sleep(0.05)
            assert any(b"\x01" in d for d in received)
            await notifier.close()
        finally:
            server.close()
            await server.wait_closed()

    async def test_notify_silently_fails_if_no_server(self) -> None:
        notifier = BrokerNotifier("/tmp/nonexistent-synth-test.sock")
        await notifier.notify()  # Should not raise

    async def test_notify_reconnects_after_disconnect(self) -> None:
        """After the server disconnects and restarts, notify() must reconnect
        and deliver the byte. Without reconnection, all subsequent notifications
        are silently lost."""
        sock_path = str(Path(tempfile.gettempdir()) / "synth-test-reconnect.sock")
        if Path(sock_path).exists():
            Path(sock_path).unlink()
        received: list[bytes] = []

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(64)
            if data:
                received.append(data)
            writer.close()

        # First server — connect and send
        server1 = await asyncio.start_unix_server(handler, path=sock_path)
        notifier = BrokerNotifier(sock_path)
        await notifier.notify()
        await asyncio.sleep(0.05)
        assert len(received) >= 1

        # Kill first server — forces disconnect
        server1.close()
        await server1.wait_closed()
        # Clear the stale writer so reconnect is needed
        notifier._writer = None

        # Start second server
        if Path(sock_path).exists():
            Path(sock_path).unlink()
        server2 = await asyncio.start_unix_server(handler, path=sock_path)
        try:
            received.clear()
            await notifier.notify()
            await asyncio.sleep(0.05)
            assert any(b"\x01" in d for d in received)
            await notifier.close()
        finally:
            server2.close()
            await server2.wait_closed()
