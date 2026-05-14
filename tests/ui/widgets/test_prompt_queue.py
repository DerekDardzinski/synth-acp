"""Tests for PromptQueue widget."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Button

from synth_acp.ui.widgets.prompt_queue import PromptQueue


class QueueApp(App):
    """Minimal app for testing PromptQueue."""

    def compose(self) -> ComposeResult:
        yield PromptQueue()


class TestEnqueueAndDrain:
    async def test_enqueue_adds_item_and_shows_widget(self) -> None:
        """enqueue makes widget visible and adds to internal queue."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            assert not queue.has_items
            assert queue.display is False

            queue.enqueue("hello", "user", None)
            await pilot.pause()

            assert queue.has_items
            assert len(queue._queue) == 1
            assert queue.display is True

    async def test_drain_next_pops_fifo(self) -> None:
        """First enqueued item is returned first, queue shrinks."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            queue.enqueue("first", "user", None)
            queue.enqueue("second", "mcp", "agent-1")
            await pilot.pause()

            result = queue.drain_next()
            assert result is not None
            assert result.text == "first"
            assert result.source == "user"
            assert len(queue._queue) == 1

    async def test_drain_next_returns_none_when_editing(self) -> None:
        """drain_next returns None when queue[0].editing is True."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            queue.enqueue("editing this", "user", None)
            await pilot.pause()

            queue._queue[0].editing = True
            result = queue.drain_next()
            assert result is None
            assert len(queue._queue) == 1

    async def test_drain_next_returns_none_when_empty(self) -> None:
        """drain_next returns None on empty queue."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            await pilot.pause()

            result = queue.drain_next()
            assert result is None

    async def test_widget_hidden_when_empty_after_drain(self) -> None:
        """Widget hides after draining last item."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            queue.enqueue("only item", "user", None)
            await pilot.pause()
            assert queue.display is True

            queue.drain_next()
            await pilot.pause()
            assert queue.display is False


class TestEditSaveDelete:
    async def test_edit_save_toggle_updates_editing_flag(self) -> None:
        """Edit sets editing=True, save sets editing=False."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            queue.enqueue("test prompt", "user", None)
            await pilot.pause()

            btn = queue.query_one("#queue-edit-btn", Button)
            # Press Edit
            btn.press()
            await pilot.pause()
            assert queue._queue[0].editing is True

            # Press Save
            btn.press()
            await pilot.pause()
            assert queue._queue[0].editing is False

    async def test_save_posts_drain_ready(self) -> None:
        """DrainReady message posted when save is clicked."""
        messages: list[PromptQueue.DrainReady] = []

        class TrackApp(App):
            def compose(self) -> ComposeResult:
                yield PromptQueue()

            def on_prompt_queue_drain_ready(self, event: PromptQueue.DrainReady) -> None:
                messages.append(event)

        app = TrackApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            queue.enqueue("test", "user", None)
            await pilot.pause()

            btn = queue.query_one("#queue-edit-btn", Button)
            # Enter edit mode
            btn.press()
            await pilot.pause()

            # Save (exit edit mode)
            btn.press()
            await pilot.pause()

            assert len(messages) == 1

    async def test_delete_removes_item_from_queue(self) -> None:
        """Delete removes item from queue."""
        app = QueueApp()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            queue = app.query_one(PromptQueue)
            queue.enqueue("to delete", "user", None)
            queue.enqueue("to keep", "mcp", "agent-1")
            await pilot.pause()

            # Delete active (first) item
            del_btn = queue.query_one("#queue-delete-btn", Button)
            del_btn.press()
            await pilot.pause()

            assert len(queue._queue) == 1
            assert queue._queue[0].text == "to keep"
