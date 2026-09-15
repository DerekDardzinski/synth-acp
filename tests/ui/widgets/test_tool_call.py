"""Tests for ToolCallBlock content area rendering."""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
from unittest.mock import AsyncMock, MagicMock

from textual.containers import VerticalScroll
from textual.widgets import Static

from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import ToolCallDiff, ToolCallLocation
from synth_acp.ui.app import SynthApp
from synth_acp.ui.widgets.conversation import ConversationFeed
from synth_acp.ui.widgets.tool_call import DiffState, ToolCallBlock, _extract_raw_output_text


def _make_config() -> SessionConfig:
    return SessionConfig(
        project="test",
    )


def _make_broker() -> MagicMock:
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    broker._initial_agent = AgentConfig(agent_id="a1", harness="kiro")
    broker.get_usage = MagicMock(return_value=None)

    async def _events():
        return
        yield

    broker.events = _events
    return broker


async def _get_feed(app: SynthApp) -> ConversationFeed:
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
        """execute kind with raw_output produces a VerticalScroll widget."""
        block = ToolCallBlock("tc5", "Run", "execute", "completed", raw_output={"output": "hello"})
        widgets = block._raw_output_widgets({"output": "hello"})
        assert any(isinstance(w, VerticalScroll) for w in widgets)

    async def test_tool_call_block_does_not_render_raw_output_for_read_kind(self) -> None:
        """raw_output with read kind is suppressed."""
        block = ToolCallBlock("tc6", "Read", "read", "completed", raw_output={"output": "content"})
        widgets = block._raw_output_widgets({"output": "content"})
        assert len(widgets) == 0

    async def test_tool_call_block_long_raw_output_not_truncated(self) -> None:
        """Long output is passed through without truncation."""
        long_output = "\n".join(["x"] * 300)
        text = _extract_raw_output_text({"output": long_output})
        assert text == long_output

    async def test_tool_call_block_renders_kiro_nested_raw_output(self) -> None:
        """Kiro's items[].Json.stdout format is extracted correctly."""
        raw = {"items": [{"Json": {"exit_status": "exit status: 0", "stdout": "hello world\n", "stderr": ""}}]}
        text = _extract_raw_output_text(raw)
        assert text is not None
        assert "hello world" in text




class TestDiffWorkState:
    """The single-flight diff work state that the feed's executor drains."""

    @staticmethod
    def _block() -> ToolCallBlock:
        return ToolCallBlock("tc", "Edit", "edit", "in_progress")

    def test_schedule_diffs_is_single_flight(self) -> None:
        """Redelivery must not duplicate work; a distinct diff must always enqueue.

        Silent failure: duplicate queued work runs the ~186ms highlight twice and mounts
        two identical DiffViews, which reads as a rendering quirk.
        """
        block = self._block()
        diff = ToolCallDiff("a.py", "old", "new")

        for state in (DiffState.QUEUED, DiffState.RENDERING, DiffState.RENDERED):
            block._diff_states.clear()
            block.schedule_diffs([diff])
            record = next(iter(block._diff_states.values()))
            record.state = state

            block.schedule_diffs([diff])

            assert len(block._diff_states) == 1
            assert record.state is state

        block.schedule_diffs([ToolCallDiff("a.py", "old", "different")])
        assert len(block._diff_states) == 2

    def test_failed_diff_retries_once_then_stops(self) -> None:
        """A failed diff gets exactly one retry.

        Silent failure: unbounded retry burns the highlight cost on every subsequent
        update for a diff that can never render.
        """
        block = self._block()
        diff = ToolCallDiff("a.py", "old", "new")
        block.schedule_diffs([diff])
        record = next(iter(block._diff_states.values()))

        record.state = DiffState.FAILED
        record.attempts = 1
        block.schedule_diffs([diff])
        assert record.state is DiffState.QUEUED

        record.state = DiffState.FAILED
        record.attempts = 2
        block.schedule_diffs([diff])
        assert record.state is DiffState.FAILED

    def test_release_claim_only_affects_rendering_records(self) -> None:
        """Release returns a claim for redelivery, and leaves other states alone.

        Silent failure: a key left RENDERING after a cancelled worker can never be
        reclaimed, because redelivery of a RENDERING key is a no-op — the diff silently
        never renders and nothing reports an error.
        """
        block = self._block()
        block.schedule_diffs([ToolCallDiff("a.py", "old", "new")])
        key, record = block.claim_next_diff()  # type: ignore[misc]

        block.release_claim(key)
        assert record.state is DiffState.QUEUED

        for state in (DiffState.RENDERED, DiffState.FAILED):
            record.state = state
            block.release_claim(key)
            assert record.state is state

    def test_entry_paths_never_prepare(self) -> None:
        """No synchronous entry path may construct or prepare a DiffView.

        Silent failure: reinstating `await prepare()` on the pump path restores the 186ms
        stall while every behavioural test still passes, because the diff still appears.
        Covers ConversationFeed.add_tool_call too — that is the FIRST-event path the phase
        evidence identifies as unprepared today.
        """
        targets = {
            "synth_acp.ui.widgets.tool_call": {"compose", "update_content"},
            "synth_acp.ui.widgets.conversation": {"add_tool_call"},
        }
        checked: set[str] = set()
        for module_name, names in targets.items():
            module = importlib.import_module(module_name)
            tree = ast.parse(pathlib.Path(inspect.getfile(module)).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name not in names:
                    continue
                checked.add(node.name)
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        name = getattr(inner.func, "attr", None) or getattr(inner.func, "id", None)
                        assert name != "prepare", f"{node.name} awaits prepare()"
                        assert name != "DiffView", f"{node.name} constructs a DiffView"

        assert checked == {"compose", "update_content", "add_tool_call"}


class TestToolCallBlockNested:
    async def test_mount_nested_child_creates_section_on_first_call(self) -> None:
        """First nested child triggers ExpandableSection creation."""
        from synth_acp.ui.widgets.expandable_section import ExpandableSection

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call("parent", "Agent task", "agent", "in_progress")
            parent_block = app.query_one("#tool-parent", ToolCallBlock)
            child = ToolCallBlock("child", "Read file", "read", "completed")
            child.add_class("nested-tool-call")
            await parent_block.mount_nested_child(child)
            assert parent_block._nested_section is not None
            assert isinstance(parent_block._nested_section, ExpandableSection)
            assert child in parent_block._nested_section.content.query(ToolCallBlock)

    async def test_mount_nested_child_updates_preview(self) -> None:
        """Preview updates to latest child's title."""
        from textual.widgets import Static

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call("parent", "Agent task", "agent", "in_progress")
            parent_block = app.query_one("#tool-parent", ToolCallBlock)
            child = ToolCallBlock("child", "Read file", "read", "completed")
            await parent_block.mount_nested_child(child)
            assert parent_block._nested_section is not None
            preview = parent_block._nested_section.query_one("#es-preview", Static)
            assert preview.content == "Read file"

    async def test_finalize_nested_sets_summary(self) -> None:
        """Finalize sets activity=False and preview to summary."""
        from textual.widgets import Static

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call("parent", "Agent task", "agent", "in_progress")
            parent_block = app.query_one("#tool-parent", ToolCallBlock)
            c1 = ToolCallBlock("c1", "Read", "read", "completed")
            c2 = ToolCallBlock("c2", "Write", "edit", "completed")
            await parent_block.mount_nested_child(c1)
            await parent_block.mount_nested_child(c2)
            parent_block.finalize_nested()
            assert parent_block._nested_section is not None
            preview = parent_block._nested_section.query_one("#es-preview", Static)
            assert "2 tool calls" in str(preview.content)
            # `_activity_bar` is authoritative and cleared synchronously; the DOM removal
            # it schedules settles a frame later.
            assert parent_block._nested_section._activity_bar is None
