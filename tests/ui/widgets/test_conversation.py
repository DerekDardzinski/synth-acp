"""Tests for ConversationFeed terminal mounting and viewport visibility."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import SessionConfig
from synth_acp.ui.app import SynthApp
from synth_acp.ui.widgets.tool_call import ToolCallBlock


def _make_config() -> SessionConfig:
    return SessionConfig(
        project="test",
    )


def _make_broker() -> MagicMock:
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    broker._initial_agent = AgentConfig(agent_id="a1", harness="kiro")

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


class TestConversationTurnEvents:
    async def test_record_event_appends_to_current(self) -> None:
        """record_event accumulates events in _current_turn_events."""
        from synth_acp.models.events import MessageChunkReceived

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            e1 = MessageChunkReceived(agent_id="a1", chunk="hello ")
            e2 = MessageChunkReceived(agent_id="a1", chunk="world")
            feed.record_event(e1)
            feed.record_event(e2)
            assert feed._current_turn_events == [e1, e2]

    async def test_finalize_commits_turn_events(self) -> None:
        """finalize_current_message moves _current_turn_events to _turn_events."""
        from synth_acp.models.events import MessageChunkReceived, TurnComplete

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            e1 = MessageChunkReceived(agent_id="a1", chunk="hi")
            tc = TurnComplete(agent_id="a1", stop_reason="end_turn")
            feed.record_event(e1)
            feed.record_event(tc)
            await feed.finalize_current_message()
            assert feed._turn_events == [[e1, tc]]
            assert feed._current_turn_events == []

    async def test_finalize_no_events_no_empty_list(self) -> None:
        """finalize_current_message with no recorded events doesn't append empty list."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.finalize_current_message()
            assert feed._turn_events == []


class TestConversationReplayEvent:
    async def test_replay_event_dispatches_chunk(self) -> None:
        """replay_event with MessageChunkReceived creates an AgentMessage."""
        from synth_acp.models.events import MessageChunkReceived

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            event = MessageChunkReceived(agent_id="a1", chunk="hello")
            await feed.replay_event(event)
            assert feed._current_message is not None

    async def test_replay_event_dispatches_turn_complete(self) -> None:
        """replay_event with TurnComplete finalizes the current turn."""
        from synth_acp.models.events import TurnComplete

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Start a turn first
            await feed.add_chunk("hi")
            assert feed._current_turn is not None
            await feed.replay_event(TurnComplete(agent_id="a1", stop_reason="end_turn"))
            assert feed._current_turn is None

    async def test_replay_event_skips_unknown(self) -> None:
        """replay_event with non-renderable event does not raise."""
        from synth_acp.models.events import AgentStateChanged

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            from synth_acp.models.agent import AgentState

            event = AgentStateChanged(agent_id="a1", old_state=AgentState.BUSY, new_state=AgentState.IDLE)
            await feed.replay_event(event)  # should not raise


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


class TestConversationNestedToolCalls:
    async def test_add_tool_call_with_parent_mounts_inside_parent(self) -> None:
        """Child tool call with parent_tool_call_id mounts inside parent block."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "parent-1", "Agent task", "agent", "in_progress",
            )
            await feed.add_tool_call(
                "child-1", "Read file", "read", "complete",
                parent_tool_call_id="parent-1",
            )
            parent_block = app.query_one("#tool-parent-1", ToolCallBlock)
            child_block = app.query_one("#tool-child-1", ToolCallBlock)
            assert child_block in parent_block.query(ToolCallBlock)
            assert child_block.has_class("nested-tool-call")

    async def test_add_tool_call_buffers_orphan_and_flushes(self) -> None:
        """Child arriving before parent is buffered, then mounted when parent appears."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Child arrives first
            await feed.add_tool_call(
                "child-1", "Read file", "read", "complete",
                parent_tool_call_id="parent-1",
            )
            assert "parent-1" in feed._pending_children
            # Parent arrives
            await feed.add_tool_call(
                "parent-1", "Agent task", "agent", "in_progress",
            )
            assert "parent-1" not in feed._pending_children
            parent_block = app.query_one("#tool-parent-1", ToolCallBlock)
            child_block = app.query_one("#tool-child-1", ToolCallBlock)
            assert child_block in parent_block.query(ToolCallBlock)

    async def test_add_tool_call_grandchild_out_of_order(self) -> None:
        """Grandchild → child → parent arrival order mounts correctly at depth > 1."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Grandchild arrives first
            await feed.add_tool_call(
                "grandchild-1", "Grep", "search", "complete",
                parent_tool_call_id="child-1",
            )
            # Child arrives second
            await feed.add_tool_call(
                "child-1", "Read file", "read", "complete",
                parent_tool_call_id="parent-1",
            )
            # Parent arrives last
            await feed.add_tool_call(
                "parent-1", "Agent task", "agent", "in_progress",
            )
            parent_block = app.query_one("#tool-parent-1", ToolCallBlock)
            child_block = app.query_one("#tool-child-1", ToolCallBlock)
            grandchild_block = app.query_one("#tool-grandchild-1", ToolCallBlock)
            assert child_block in parent_block.query(ToolCallBlock)
            assert grandchild_block in child_block.query(ToolCallBlock)

    async def test_add_tool_call_no_parent_mounts_in_turn(self) -> None:
        """Top-level tool call (no parent) mounts in turn container, not inside another block."""
        from synth_acp.ui.widgets.conversation import TurnContainer

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "tc-top", "Write file", "write", "complete",
            )
            block = app.query_one("#tool-tc-top", ToolCallBlock)
            assert isinstance(block.parent, TurnContainer)

    async def test_nested_child_mounted_inside_expandable_section(self) -> None:
        """Nested children end up inside ExpandableSection, not directly on parent."""

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call("parent-1", "Agent task", "agent", "in_progress")
            await feed.add_tool_call(
                "child-1", "Read file", "read", "complete",
                parent_tool_call_id="parent-1",
            )
            parent_block = app.query_one("#tool-parent-1", ToolCallBlock)
            assert parent_block._nested_section is not None
            child_block = app.query_one("#tool-child-1", ToolCallBlock)
            assert child_block in parent_block._nested_section.content.query(ToolCallBlock)

    async def test_flush_pending_uses_expandable_section(self) -> None:
        """Out-of-order children are flushed into ExpandableSection."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call(
                "child-1", "Read file", "read", "complete",
                parent_tool_call_id="parent-1",
            )
            await feed.add_tool_call("parent-1", "Agent task", "agent", "in_progress")
            parent_block = app.query_one("#tool-parent-1", ToolCallBlock)
            assert parent_block._nested_section is not None
            child_block = app.query_one("#tool-child-1", ToolCallBlock)
            assert child_block in parent_block._nested_section.content.query(ToolCallBlock)

    async def test_parent_completion_calls_finalize_nested(self) -> None:
        """Parent status change to completed triggers finalize_nested."""
        from textual.widgets import Static

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            await feed.add_tool_call("parent-1", "Agent task", "agent", "in_progress")
            await feed.add_tool_call(
                "child-1", "Read file", "read", "complete",
                parent_tool_call_id="parent-1",
            )
            # Update parent to completed
            await feed.add_tool_call("parent-1", "Agent task", "agent", "completed")
            parent_block = app.query_one("#tool-parent-1", ToolCallBlock)
            preview = parent_block._nested_section.query_one("#es-preview", Static)
            assert preview.content == "✓ 1 tool calls"


class TestPruningScrollContainer:
    async def test_near_top_posted_when_scroll_near_top(self) -> None:
        """NearTop fires when scroll_y transitions to <= threshold while scrolling up."""
        from unittest.mock import patch

        from textual.app import App, ComposeResult

        from synth_acp.ui.widgets.conversation import PruningScrollContainer

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield PruningScrollContainer()

        app = TestApp()
        async with app.run_test(headless=True, size=(120, 40)):
            scroll = app.query_one(PruningScrollContainer)
            messages: list = []
            with patch.object(scroll, "post_message", side_effect=messages.append):
                scroll.watch_scroll_y(30.0, 15.0)
            assert any(isinstance(m, PruningScrollContainer.NearTop) for m in messages)

    async def test_near_top_not_posted_when_scrolling_down(self) -> None:
        """NearTop NOT posted when scrolling down even if below threshold."""
        from unittest.mock import patch

        from textual.app import App, ComposeResult

        from synth_acp.ui.widgets.conversation import PruningScrollContainer

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield PruningScrollContainer()

        app = TestApp()
        async with app.run_test(headless=True, size=(120, 40)):
            scroll = app.query_one(PruningScrollContainer)
            messages: list = []
            with patch.object(scroll, "post_message", side_effect=messages.append):
                scroll.watch_scroll_y(10.0, 15.0)
            assert not any(isinstance(m, PruningScrollContainer.NearTop) for m in messages)


class TestConversationPruning:
    async def test_check_prune_removes_oldest_turns(self) -> None:
        """After exceeding HIGH_MARK turns, oldest are pruned to LOW_MARK."""
        from synth_acp.models.events import MessageChunkReceived
        from synth_acp.ui.widgets.conversation import TurnContainer

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Create 41 turns (exceeds HIGH_MARK=40)
            for i in range(41):
                await feed.add_chunk(f"msg {i}")
                feed.record_event(MessageChunkReceived(agent_id="a1", chunk=f"msg {i}"))
                await feed.finalize_current_message()
            turns = [c for c in feed._scroll.children if isinstance(c, TurnContainer)]
            assert len(turns) == 30
            assert feed._mounted_start_idx == 11

    async def test_check_prune_cleans_tool_call_blocks(self) -> None:
        """Pruned turns have their tool_call_blocks entries removed."""
        from synth_acp.models.events import MessageChunkReceived, TurnComplete

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # First turn has a tool call
            await feed.add_tool_call("tc-early", "Read", "read", "completed")
            feed.record_event(TurnComplete(agent_id="a1", stop_reason="end_turn"))
            await feed.finalize_current_message()
            assert "tc-early" in feed._tool_call_blocks
            # Create 40 more turns to trigger prune
            for i in range(40):
                await feed.add_chunk(f"msg {i}")
                feed.record_event(MessageChunkReceived(agent_id="a1", chunk=f"msg {i}"))
                await feed.finalize_current_message()
            assert "tc-early" not in feed._tool_call_blocks

    async def test_check_prune_skips_when_scrolled_up(self) -> None:
        """Pruning does not fire when user is scrolled up."""
        from unittest.mock import PropertyMock, patch

        from synth_acp.models.events import MessageChunkReceived
        from synth_acp.ui.widgets.conversation import TurnContainer

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Create 41 turns
            for i in range(41):
                await feed.add_chunk(f"msg {i}")
                feed.record_event(MessageChunkReceived(agent_id="a1", chunk=f"msg {i}"))
                # Patch scroll position to simulate user scrolled up before last finalize
                if i == 40:
                    with patch.object(
                        type(feed._scroll), "scroll_y", new_callable=PropertyMock, return_value=0
                    ), patch.object(
                        type(feed._scroll), "max_scroll_y", new_callable=PropertyMock, return_value=100
                    ):
                        await feed.finalize_current_message()
                else:
                    await feed.finalize_current_message()
            turns = [c for c in feed._scroll.children if isinstance(c, TurnContainer)]
            assert len(turns) == 41
            assert feed._mounted_start_idx == 0


class TestConversationRestore:
    async def test_restore_turns_replays_batch(self) -> None:
        """After pruning, _restore_turns mounts turns back and decreases _mounted_start_idx."""
        from synth_acp.models.events import MessageChunkReceived
        from synth_acp.ui.widgets.conversation import TurnContainer

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Create 41 turns to trigger prune (leaves 30 mounted, 11 pruned)
            for i in range(41):
                await feed.add_chunk(f"msg {i}")
                feed.record_event(MessageChunkReceived(agent_id="a1", chunk=f"msg {i}"))
                await feed.finalize_current_message()
            assert feed._mounted_start_idx == 11
            turns_before = len([c for c in feed._scroll.children if isinstance(c, TurnContainer)])
            # Restore
            await feed._restore_turns()
            assert feed._mounted_start_idx == 1
            turns_after = len([c for c in feed._scroll.children if isinstance(c, TurnContainer)])
            assert turns_after == turns_before + 10

    async def test_restore_turns_debounced(self) -> None:
        """Second call while _loading_more=True returns immediately."""
        from synth_acp.models.events import MessageChunkReceived

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Create 41 turns to trigger prune
            for i in range(41):
                await feed.add_chunk(f"msg {i}")
                feed.record_event(MessageChunkReceived(agent_id="a1", chunk=f"msg {i}"))
                await feed.finalize_current_message()
            feed._loading_more = True
            idx_before = feed._mounted_start_idx
            await feed._restore_turns()
            assert feed._mounted_start_idx == idx_before

    async def test_restore_turns_resets_loading_on_error(self) -> None:
        """_loading_more is reset to False even if _restore_turns encounters an error."""
        from unittest.mock import AsyncMock, patch

        from synth_acp.models.events import MessageChunkReceived

        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            # Create 41 turns to trigger prune
            for i in range(41):
                await feed.add_chunk(f"msg {i}")
                feed.record_event(MessageChunkReceived(agent_id="a1", chunk=f"msg {i}"))
                await feed.finalize_current_message()
            # Make replay_event raise — exception should be caught, not propagated
            with patch.object(feed, "replay_event", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
                await feed._restore_turns()
            assert feed._loading_more is False

    async def test_restore_turns_no_op_when_all_mounted(self) -> None:
        """_restore_turns with _mounted_start_idx=0 does nothing."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            assert feed._mounted_start_idx == 0
            await feed._restore_turns()
            assert feed._loading_more is False
