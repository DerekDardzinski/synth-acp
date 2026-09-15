"""Tests for ThoughtBlock widget."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.widgets.markdown import Markdown

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
            assert len(block.query(ActivityBar)) == 1

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
            assert len(block.query(ActivityBar)) == 0
            preview = block.query_one("#es-preview", Static)
            assert "some thought content here" in str(preview.content)


class TestThoughtBlockReopen:
    """Reopening a finalized thought so a late reasoning chunk continues into it."""

    async def test_reopen_restores_stream_activity_and_preview(self) -> None:
        """Reopen must restore everything finalize stopped.

        Silent failure: a reopened thought that streams text with no activity indicator and
        a frozen preview looks finished while it is still running.
        """
        block = ThoughtBlock()
        app = _TestApp(block)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await block.append_chunk("thinking hard")
            await pilot.pause()
            original = block._stream
            await block.finalize()
            await pilot.pause()

            calls = {"n": 0}
            real_update = Markdown.update

            def _counting(self: Markdown, markdown: str):
                calls["n"] += 1
                return real_update(self, markdown)

            with patch.object(Markdown, "update", _counting):
                await block.reopen()
            await pilot.pause()

            assert calls["n"] == 0
            assert block._stream is not None
            assert block._stream is not original
            assert block._stream.markdown_widget is block._markdown
            assert len(block.query(ActivityBar)) == 1
            assert block._preview_timer is not None

            stream = block._stream
            await block.reopen()
            assert block._stream is stream

    async def test_reopen_restores_the_collapsed_state_held_while_streaming(self) -> None:
        """The state the user held while streaming must survive finalize + reopen.

        Silent failure: hard-coding collapsed=False silently reopens a thought the user
        deliberately collapsed, every time a late chunk arrives.
        """
        for collapsed_while_streaming in (True, False):
            block = ThoughtBlock()
            app = _TestApp(block)

            async with app.run_test(headless=True, size=(80, 24)) as pilot:
                await block.append_chunk("thinking")
                await pilot.pause()
                block._section.collapsed = collapsed_while_streaming
                await block.finalize()
                await pilot.pause()
                assert block._section.collapsed is True

                await block.reopen()
                await pilot.pause()

                assert block._section.collapsed is collapsed_while_streaming

    async def test_two_chunks_after_reopen_append_in_order(self) -> None:
        """The thought path has its own stream, so ordering is asserted separately.

        Silent failure: the AgentMessage test cannot see a ThoughtBlock stream that drops
        or reorders fragments after reopen.
        """
        block = ThoughtBlock()
        app = _TestApp(block)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await block.append_chunk("seed")
            await block.finalize()
            await block.reopen()
            await block.append_chunk("one")
            await block.append_chunk("two")
            await pilot.pause()

            assert "".join(block._chunks) == "seedonetwo"
