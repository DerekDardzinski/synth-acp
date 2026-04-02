"""Tests for ToolCallBlock content area rendering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from textual.widgets import Static

from synth_acp.models.config import SessionConfig
from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.app import SynthApp
from synth_acp.ui.widgets.diff_view import DiffView
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
        yield

    broker.events = _events
    return broker


async def _get_feed(app: SynthApp) -> object:
    """Select the first agent and return its ConversationFeed."""
    await app.select_agent("a1")
    return app._panels["a1"]


class TestToolCallBlockContent:
    async def test_tool_call_block_when_locations_present_renders_path(self) -> None:
        """Location renders as file path chip with line number."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc1", "Read file", "read", "in_progress",
                locations=[ToolCallLocation("src/f.py", 10)],
            )
            block = app.query_one("#tool-tc1", ToolCallBlock)
            loc = block.query_one("#tc-location", Static)
            assert "src/f.py:10" in str(loc.content)

    async def test_tool_call_block_when_raw_input_has_command_renders_dollar_prefix(self) -> None:
        """Raw input dict with 'command' key renders as $ command."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc2", "Run cmd", "execute", "in_progress",
                raw_input={"command": "ls"},
            )
            block = app.query_one("#tool-tc2", ToolCallBlock)
            ri = block.query_one("#tc-raw-input", Static)
            assert "$ ls" in str(ri.content)

    async def test_update_content_when_diffs_appended_mounts_diff_views(self) -> None:
        """Calling update_content with diffs twice mounts DiffViews from both calls."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc3", "Edit", "edit", "in_progress",
                diffs=[ToolCallDiff("a.py", "old", "new")],
            )
            block = app.query_one("#tool-tc3", ToolCallBlock)
            await feed.add_tool_call(
                "tc3", "Edit", "edit", "in_progress",
                diffs=[ToolCallDiff("b.py", "x", "y")],
            )
            diff_views = block.query(DiffView)
            assert len(diff_views) == 2

    async def test_update_content_when_locations_already_rendered_does_not_duplicate(self) -> None:
        """Second update_content with locations is a no-op when already rendered."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc4", "Read", "read", "in_progress",
                locations=[ToolCallLocation("a.py", 1)],
            )
            block = app.query_one("#tool-tc4", ToolCallBlock)
            await feed.add_tool_call(
                "tc4", "Read", "read", "in_progress",
                locations=[ToolCallLocation("b.py", 2)],
            )
            loc_widgets = block.query("#tc-location")
            assert len(loc_widgets) == 1
