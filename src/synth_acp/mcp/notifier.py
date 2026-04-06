"""BrokerNotifier — socket client for MCP→broker wake-up signals."""

from __future__ import annotations

import asyncio
import contextlib
import logging

log = logging.getLogger(__name__)


class BrokerNotifier:
    """Persistent connection to the broker's notification socket.

    Sends a 1-byte wake-up signal after each SQLite commit.
    """

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._writer: asyncio.StreamWriter | None = None

    async def notify(self) -> None:
        """Send a 1-byte wake-up signal. Silently drops on failure."""
        try:
            if self._writer is None or self._writer.is_closing():
                _, self._writer = await asyncio.open_unix_connection(self._socket_path)
            self._writer.write(b"\x01")
            await self._writer.drain()
        except (OSError, ConnectionError):
            self._writer = None

    async def close(self) -> None:
        """Close the connection if open."""
        if self._writer and not self._writer.is_closing():
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
