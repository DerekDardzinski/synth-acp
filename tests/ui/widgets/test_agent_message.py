"""Tests for AgentMessage streaming lifecycle."""

from __future__ import annotations

from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.widgets.markdown import Markdown

from synth_acp.ui.widgets.agent_message import AgentMessage


class _TestApp(App):
    def __init__(self, message: AgentMessage) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield self._message


class TestReopen:
    """Reopening a finalized message so a late chunk can continue into it."""

    async def test_reopen_uses_a_fresh_stream_over_the_same_widget(self) -> None:
        """A fresh MarkdownStream is required; the stopped one cannot be reused.

        Silent failure: MarkdownStream.stop() sets a one-way `_stopped` latch, so reusing
        it raises RuntimeError on the next chunk — the user loses the tail of the response
        and the error surfaces far from its cause.
        """
        message = AgentMessage("a1")
        app = _TestApp(message)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await message.append_chunk("first ")
            original = message._stream
            await message.finalize()
            assert message._stream is None

            await message.reopen()
            await pilot.pause()

            assert message._stream is not None
            assert message._stream is not original
            assert message._stream.markdown_widget is message._md
            await message.append_chunk("second")

    async def test_reopen_does_not_reparse_the_document(self) -> None:
        """Reopen must not call Markdown.update.

        Silent failure: a full re-parse per late chunk reintroduces exactly the cost class
        this phase exists to remove, and the rendered output looks identical.
        """
        message = AgentMessage("a1")
        app = _TestApp(message)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await message.append_chunk("first ")
            await message.finalize()

            calls = {"n": 0}
            original = Markdown.update

            def _counting(self: Markdown, markdown: str):
                calls["n"] += 1
                return original(self, markdown)

            with patch.object(Markdown, "update", _counting):
                await message.reopen()
            await pilot.pause()

            assert calls["n"] == 0

    async def test_reopen_is_idempotent_and_later_chunks_keep_order(self) -> None:
        """Repeated reopens keep one stream, and following chunks arrive in order.

        Silent failure: a second reopen replacing the stream mid-flight drops or reorders
        buffered fragments, which reads as a garbled sentence rather than an error.
        """
        message = AgentMessage("a1")
        app = _TestApp(message)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await message.append_chunk("seed")
            await message.finalize()

            await message.reopen()
            stream = message._stream
            await message.reopen()
            assert message._stream is stream

            await message.append_chunk("one")
            await message.append_chunk("two")
            await pilot.pause()

            assert "".join(message._chunks) == "seedonetwo"
