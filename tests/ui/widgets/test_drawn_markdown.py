"""Tests for math detection and the math fallbacks in drawn_markdown."""

from __future__ import annotations

from unittest.mock import patch

from synth_acp.ui.widgets.drawn_markdown import DrawnMarkdown, render_math


def _math_spans(text: str) -> list[str]:
    """Return the inline math the parser finds, before it is converted to text."""
    parser = DrawnMarkdown._make_parser(DrawnMarkdown.__new__(DrawnMarkdown))
    parser.core.ruler.disable("synth_math")
    return [
        child.content
        for token in parser.parse(text)
        for child in token.children or []
        if child.type == "math_inline"
    ]


class TestRenderMathFallbacks:
    """Each case must return None so the caller shows the LaTeX source."""

    def test_unknown_command_is_not_drawn_as_a_question_mark(self) -> None:
        """Silent failure: TeXicode draws an unknown command as `?`, so `\\foo{x}` would
        appear as `?` followed by an italic x, and the reader loses what the agent wrote."""
        assert render_math("\\notacommand{x}", inline=False) is None

    def test_inline_math_needing_several_rows_is_not_drawn(self) -> None:
        """Silent failure: a stacked fraction squeezed into a line of prose collapses to
        an unreadable run of box characters."""
        assert render_math("\\frac{a}{b}", inline=True) is None
        assert render_math("\\frac{a}{b}", inline=False) is not None

    def test_missing_texicode_returns_none(self) -> None:
        """Silent failure: without the optional extra, math must stay as source rather
        than raise inside the markdown parser and blank the whole message."""
        with patch.dict("sys.modules", {"texicode": None, "texicode.pipeline": None}):
            assert render_math("x^2", inline=True) is None

    def test_drawn_inline_math(self) -> None:
        """Silent failure: a TeXicode upgrade that changes inline output (spacing, glyph
        choice) would otherwise only show up as odd-looking prose."""
        assert render_math("\\alpha_1 + x^2", inline=True) == "α₁+\N{MATHEMATICAL ITALIC SMALL X}²"


class TestDollarDetection:
    """Which `$` spans the parser treats as math."""

    def test_prices_and_shell_variables_in_quotes_stay_text(self) -> None:
        """Silent failure: `$10" and "$` passed the plugin's strict check and was drawn
        as italic math, turning a sentence about prices into `10"and"HOME`."""
        assert _math_spans('"$5 and $10" and "$HOME", or $PATH:$HOME.') == []

    def test_letter_prefixed_currency_does_not_swallow_later_math(self) -> None:
        """Silent failure: `US$`/`R$` amounts pair with a later `$n$` and typeset the
        prose between them (reported against Hermes and DeepSeek harness renderers)."""
        assert _math_spans("US$ 500 and R$800, where $n$ is the count.") == ["n"]
        assert _math_spans("Costs $5; outcome is $4\\in A$.") == ["4\\in A"]

    def test_math_next_to_punctuation_is_still_math(self) -> None:
        """Silent failure: an over-strict boundary rule would quietly stop drawing the
        common cases of math followed by a comma or wrapped in parentheses."""
        assert _math_spans("Let $x$, and ($y_1$) hold.") == ["x", "y_1"]


class TestRenderTable:
    def test_csv_with_quoted_comma_and_numeric_column(self) -> None:
        """Silent failures: a quoted comma splitting a cell shifts every later column,
        and numbers left-aligned make magnitudes hard to compare."""
        from synth_acp.ui.widgets.drawn_markdown import render_table

        table = render_table('name,score,note\nalpha,1.5,"a, b"\nbeta,10,ok\n', ",")
        assert table is not None
        assert table.plain == (
            "┌───────┬───────┬──────┐\n"
            "│ name  │ score │ note │\n"
            "├───────┼───────┼──────┤\n"
            "│ alpha │   1.5 │ a, b │\n"
            "│ beta  │    10 │ ok   │\n"
            "└───────┴───────┴──────┘"
        )

    def test_one_column_or_one_row_is_not_a_table(self) -> None:
        """Silent failure: a ```csv block holding a plain list, or only a header, drawn
        as a box adds borders and hides nothing useful; the source reads better."""
        from synth_acp.ui.widgets.drawn_markdown import render_table

        assert render_table("one\ntwo\nthree", ",") is None
        assert render_table("a,b,c", ",") is None

    def test_rows_past_the_cap_are_counted_not_drawn(self) -> None:
        """Silent failure: an agent pasting a 10,000-row CSV would mount a drawing tall
        enough to stall the conversation view."""
        from synth_acp.ui.widgets.drawn_markdown import MAX_TABLE_ROWS, render_table

        rows = "\n".join(f"{n},x" for n in range(MAX_TABLE_ROWS + 5))
        table = render_table("n,v\n" + rows, ",")
        assert table is not None
        assert table.plain.endswith("… 5 more rows")
        assert table.plain.count("│ x │") == MAX_TABLE_ROWS


class TestSourceToggle:
    async def test_toggle_switches_between_drawing_and_source(self) -> None:
        """Silent failure: without a working toggle a drawing that looks wrong cannot
        be checked against what the agent wrote. Plain code gets no toggle."""
        from textual.app import App, ComposeResult
        from textual.widgets import Label, Static

        from synth_acp.ui.widgets.drawn_markdown import DrawnFence

        class _App(App):
            def compose(self) -> ComposeResult:
                yield DrawnMarkdown("```csv\na,b\n1,2\n```\n\n```py\nx = 1\n```\n", drawing=True)

        app = _App()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await pilot.pause()
            drawn, code = app.query(DrawnFence)

            def shown(fence: DrawnFence) -> str:
                return str(fence.query_one("#code-content", Label).render())

            assert "│ a │ b │" in shown(drawn)
            assert len(code.query(".drawn-toggle")) == 0

            await pilot.click(".drawn-toggle")
            assert shown(drawn) == "a,b\n1,2"
            assert str(drawn.query_one(".drawn-toggle", Static).render()) == "⇄ drawing"

            await pilot.click(".drawn-toggle")
            assert "│ a │ b │" in shown(drawn)
