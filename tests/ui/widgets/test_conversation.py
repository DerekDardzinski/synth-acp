"""Tests for ConversationFeed terminal mounting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from synth_acp.models.config import SessionConfig
from synth_acp.ui.app import SynthApp
from synth_acp.ui.widgets.tool_call import ToolCallBlock


def _make_config() -> SessionConfig:
    return SessionConfig(
        project="test",
        agents=[{"agent_id": "a1", "harness": "kiro"}],
    )


def _make_broker() -> MagicMock:
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()

    async def _events():
        return
        yield  # pragma: no cover

    broker.events = _events
    return broker


def _mock_process() -> MagicMock:
    """Create a minimal mock TerminalProcess."""
    proc = MagicMock()
    proc.on_output = None
    proc.on_exit = None
    proc.resize_pty = MagicMock()
    proc.return_code = None
    return proc


async def _get_feed(app: SynthApp) -> object:
    """Select the first agent and return its ConversationFeed."""
    await app.select_agent("a1")
    return app._panels["a1"]


class TestConversationTerminal:
    async def test_mount_terminal_when_tool_call_exists_mounts_inside_block(self) -> None:
        """Terminal widget mounts inside matching ToolCallBlock."""
        from synth_acp.ui.widgets.terminal import Terminal

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc1", "Run cmd", "execute", "in_progress",
                terminal_id="t-1",
            )
            proc = _mock_process()
            await feed.mount_terminal("t-1", proc)
            block = app.query_one("#tool-tc1", ToolCallBlock)
            terminals = block.query(Terminal)
            assert len(terminals) == 1

    async def test_mount_terminal_when_no_tool_call_buffers_pending(self) -> None:
        """Terminal stashed in pending; subsequent add_tool_call mounts it."""
        from synth_acp.ui.widgets.terminal import Terminal

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            proc = _mock_process()
            await feed.mount_terminal("t-1", proc)
            assert "t-1" in feed._pending_terminals

            await feed.add_tool_call(
                "tc2", "Run cmd", "execute", "in_progress",
                terminal_id="t-1",
            )
            assert "t-1" not in feed._pending_terminals
            block = app.query_one("#tool-tc2", ToolCallBlock)
            terminals = block.query(Terminal)
            assert len(terminals) == 1
