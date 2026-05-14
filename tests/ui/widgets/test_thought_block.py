"""Tests for ThoughtBlock widget."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Static

from synth_acp.ui.widgets.expandable_section import ExpandableSection
from synth_acp.ui.widgets.gradient_bar import ActivityBar
from synth_acp.ui.widgets.thought_block import ThoughtBlock


class _TestApp(App):
    def __init__(self, block: ThoughtBlock) -> None:
        super().__init__()
        self._block = block

    def compose(self) -> ComposeResult:
        yield self._block


class TestThoughtBlock:
    async def test_append_chunk_sets_activity_true(self) -> None:
        """Activity indicator activates on first chunk."""
        block = ThoughtBlock()
        app = _TestApp(block)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await block.append_chunk("thinking")
            await pilot.pause()
            bar = block.query_one(".es-activity", ActivityBar)
            assert bar.active is True

    async def test_append_chunk_streams_and_debounces_preview(self) -> None:
        """Streaming content reaches Markdown and preview updates after debounce."""
        block = ThoughtBlock()
        app = _TestApp(block)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await block.append_chunk("Hello ")
            await block.append_chunk("world, this is a thought block test")
            # Wait for debounce (200ms)
            await asyncio.sleep(0.3)
            await pilot.pause()
            preview = block.query_one("#es-preview", Static)
            assert "thought block test" in str(preview.content)

    async def test_finalize_collapses_and_sets_activity(self) -> None:
        """Finalize collapses section, sets activity to False, freezes preview."""
        block = ThoughtBlock()
        app = _TestApp(block)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await block.append_chunk("some thought content here")
            await pilot.pause()
            await block.finalize()
            await pilot.pause()

            section = block.query_one(ExpandableSection)
            assert section.collapsed is True
            bar = block.query_one(".es-activity", ActivityBar)
            assert bar.active is False
            preview = block.query_one("#es-preview", Static)
            assert "some thought content here" in str(preview.content)
