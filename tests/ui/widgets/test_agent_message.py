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


_MERMAID = "```mermaid\nflowchart LR\n    Alpha --> Beta\n```\n"


def _fence_text(message: AgentMessage) -> str:
    from textual.widgets import Label
    from textual.widgets.markdown import MarkdownFence

    fence = message.query_one(MarkdownFence)
    return str(fence.query_one("#code-content", Label).render())


class TestMermaidDiagrams:
    """Closed mermaid fences are drawn once the message finishes streaming."""

    async def test_streaming_shows_source_and_finalize_draws_the_diagram(self) -> None:
        """Silent failure: a diagram drawn from a half-streamed fence is a wrong picture,
        and a finished one left as source means the feature quietly does nothing."""
        message = AgentMessage("a1")
        app = _TestApp(message)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await message.append_chunk("Here:\n\n```mermaid\nflowchart LR\n    Alpha --> ")
            await pilot.pause()
            assert "-->" in _fence_text(message)

            await message.append_chunk("Beta\n```\n")
            await message.finalize()
            await pilot.pause()

            drawn = _fence_text(message)
            assert "-->" not in drawn
            assert "Alpha" in drawn and "Beta" in drawn
            assert "►" in drawn

    async def test_unparseable_mermaid_falls_back_to_source(self) -> None:
        """Silent failure: termaid returns an empty drawing for text it cannot parse, so
        without the fallback the block renders blank and the content is lost."""
        message = AgentMessage("a1")
        app = _TestApp(message)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await message.append_chunk("```mermaid\nnot a diagram at all\n```\n")
            await message.finalize()
            await pilot.pause()

            assert "not a diagram at all" in _fence_text(message)

    async def test_missing_termaid_falls_back_to_source(self) -> None:
        """Silent failure: without the optional extra installed, a mermaid block must
        still show its source rather than raising or rendering empty."""
        message = AgentMessage("a1")
        app = _TestApp(message)

        with patch.dict("sys.modules", {"termaid": None}):
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await message.append_chunk(_MERMAID)
                await message.finalize()
                await pilot.pause()

                assert "Alpha --> Beta" in _fence_text(message)


_MATH_REPLY = (
    "Let $\\alpha_1$ be small; it costs $5 and $10 in $HOME.\n"
    "The root is:\n"
    "$$\n"
    "\\frac{-b}{2a}\n"
    "$$\n"
    "\n"
    "```latex\n"
    "\\documentclass{article}\n"
    "```\n"
)


class TestMath:
    """LaTeX math is drawn once the message finishes streaming."""

    async def test_streaming_shows_source_and_finalize_draws_math(self) -> None:
        """Silent failures: math drawn mid-stream from a partial span; a `$$` block right
        after a line of text (no blank line, the usual model output) left as source or
        duplicated; prices and shell variables eaten as math; LaTeX source in a
        ```latex fence redrawn as an equation."""
        from textual.widgets._markdown import MarkdownParagraph
        from textual.widgets.markdown import MarkdownFence

        message = AgentMessage("a1")
        app = _TestApp(message)

        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await message.append_chunk(_MATH_REPLY)
            await pilot.pause()
            first = message.query(MarkdownParagraph).first()
            assert "$\\alpha_1$" in str(first._content)

            await message.finalize()
            await pilot.pause()

            paragraph = str(message.query(MarkdownParagraph).first()._content)
            assert paragraph == ("Let α₁ be small; it costs $5 and $10 in $HOME. The root is:")
            fences = list(message.query(MarkdownFence))
            assert len(fences) == 2
            from textual.widgets import Label

            drawn, latex = (str(f.query_one("#code-content", Label).render()) for f in fences)
            assert (
                "\\frac" not in drawn
                and "2\N{MATHEMATICAL ITALIC SMALL A}" in drawn
                and "─" in drawn
            )
            assert latex == "\\documentclass{article}"


class TestBlockRewrites:
    """HTML, GitHub alerts and task lists render the way agents intend."""

    async def test_html_block_text_is_shown_not_dropped(self) -> None:
        """Silent failure: Textual has no case for html_block, so the text inside a
        `<details>` or `<p>` vanished with no sign anything was there."""
        from textual.widgets._markdown import MarkdownParagraph

        message = AgentMessage("a1")
        app = _TestApp(message)
        async with app.run_test(headless=True, size=(100, 40)) as pilot:
            await message.append_chunk(
                "<details><summary>More</summary>\n\nhidden\n\n</details>\n\n"
                "<p>one<br>two</p>\n\n<!-- note to self -->\n\nPress <kbd>Ctrl</kbd>+C<br>now\n"
            )
            await message.finalize()
            await pilot.pause()

            texts = [str(p._content) for p in message.query(MarkdownParagraph)]
            # "hidden" is inside the <details> fold (tests/ui/widgets/test_drawn_markdown.py).
            assert texts == ["hidden", "one\ntwo", "Press Ctrl+C\nnow"]

    async def test_alert_marker_becomes_a_label_and_a_colored_quote(self) -> None:
        """Silent failure: `[!WARNING]` showed as literal text in a plain quote, so a
        caveat the agent flagged looked like any other quote. A marker that is not
        alone on its line is not an alert on GitHub and must stay text."""
        from textual.widgets._markdown import MarkdownParagraph

        from synth_acp.ui.widgets.drawn_markdown import AlertQuote

        message = AgentMessage("a1")
        app = _TestApp(message)
        async with app.run_test(headless=True, size=(100, 40)) as pilot:
            await message.append_chunk("> [!WARNING]\n> careful **now**\n\n> [!tip] not alone\n")
            await message.finalize()
            await pilot.pause()

            alert, plain = message.query(AlertQuote)
            assert alert.has_class("-alert-warning")
            assert not plain.has_class("-alert")
            texts = [str(p._content) for p in message.query(MarkdownParagraph)]
            assert texts == ["Warning\ncareful now", "[!tip] not alone"]

    async def test_task_items_get_checkbox_bullets(self) -> None:
        """Silent failure: `[x]`/`[ ]` showed as text after a normal bullet, so done
        and open items in an agent's plan looked alike at a glance. Ordered items keep
        their number."""
        from textual.widgets._markdown import MarkdownBullet, MarkdownParagraph

        message = AgentMessage("a1")
        app = _TestApp(message)
        async with app.run_test(headless=True, size=(100, 40)) as pilot:
            await message.append_chunk("- [x] done\n- [ ] todo\n- plain\n\n1. [x] kept\n")
            await message.finalize()
            await pilot.pause()

            symbols = [b.symbol for b in message.query(MarkdownBullet)]
            assert symbols[:3] == ["☑ ", "☐ ", "• "]
            texts = [str(p._content) for p in message.query(MarkdownParagraph)]
            assert texts == ["done", "todo", "plain", "[x] kept"]
