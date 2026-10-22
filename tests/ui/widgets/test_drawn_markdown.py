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
