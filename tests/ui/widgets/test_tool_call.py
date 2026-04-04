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
            ri = block.query_one("#tc-raw-input")
            assert "$ ls" in ri.content.plain

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

    async def test_tool_call_block_renders_raw_output_for_execute_kind(self) -> None:
        """raw_output with execute kind renders the output widget."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc5", "Run", "execute", "completed",
                raw_output={"output": "hello"},
            )
            block = app.query_one("#tool-tc5", ToolCallBlock)
            ro = block.query_one("#tc-raw-output-label")
            assert "hello" in ro.content.plain

    async def test_tool_call_block_does_not_render_raw_output_for_read_kind(self) -> None:
        """raw_output with read kind is suppressed."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc6", "Read", "read", "completed",
                raw_output={"output": "content"},
            )
            block = app.query_one("#tool-tc6", ToolCallBlock)
            assert len(block.query("#tc-raw-output")) == 0

    async def test_tool_call_block_long_raw_output_is_scrollable(self) -> None:
        """Long output is in a scrollable container, not truncated."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc7", "Run", "execute", "completed",
                raw_output={"output": "\n".join(["x"] * 300)},
            )
            block = app.query_one("#tool-tc7", ToolCallBlock)
            block.query_one("#tc-raw-output")  # scrollable container exists
            label = block.query_one("#tc-raw-output-label")
            assert "x" in label.content.plain

    async def test_tool_call_block_renders_kiro_nested_raw_output(self) -> None:
        """Kiro's items[].Json.stdout format is extracted and rendered."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc8", "Run", "execute", "completed",
                raw_output={"items": [{"Json": {"exit_status": "exit status: 0", "stdout": "hello world\n", "stderr": ""}}]},
            )
            block = app.query_one("#tool-tc8", ToolCallBlock)
            ro = block.query_one("#tc-raw-output-label")
            assert "hello world" in ro.content.plain
