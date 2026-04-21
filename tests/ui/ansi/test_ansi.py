"""Tests for ANSI parser and terminal state."""

from __future__ import annotations

from textual.color import Color

from synth_acp.ui.ansi._ansi import ANSIScrollMargin, ANSIStream, TerminalState


async def _noop_stdin(text: str) -> None:
    pass


class TestParseSgr:
    def test_parse_sgr_when_reset_followed_by_color_returns_color_style(self) -> None:
        """ESC[0;32m must yield green foreground, not discard it on reset."""
        result = ANSIStream._parse_sgr("0;32")
        assert result is not None
        style, did_reset = result
        assert did_reset is True
        assert style.foreground == Color(0, 128, 0, ansi=2)

    def test_parse_sgr_when_pure_reset_returns_none(self) -> None:
        """ESC[0m must signal a full reset to the caller."""
        assert ANSIStream._parse_sgr("0") is None

    def test_parse_csi_when_scroll_region_bottom_only_sets_bottom(self) -> None:
        """ESC[;40r must set bottom=39, not produce a full reset."""
        result = ANSIStream._parse_csi("[;40r")
        assert result == ANSIScrollMargin(None, 39)


class TestTerminalStateWrite:
    async def test_write_when_sgr_reset_then_color_applies_only_new_color(self) -> None:
        """Bold then reset+green must yield green without bold."""
        state = TerminalState(_noop_stdin, width=80, height=24)
        await state.write("\x1b[1m")       # bold
        assert state.style.bold is True
        await state.write("\x1b[0;32m")    # reset + green
        assert state.style.foreground == Color(0, 128, 0, ansi=2)
        assert state.style.bold is not True

    async def test_write_when_insert_mode_enabled_inserts_rather_than_overwrites(self) -> None:
        """ESC[4h enables insert mode — new text shifts existing content right."""
        state = TerminalState(_noop_stdin, width=80, height=24)
        await state.write("ABCDE")
        await state.write("\r")            # cursor to column 0
        await state.write("\x1b[4h")       # set insert mode (replace_mode=False)
        assert state.replace_mode is False
        await state.write("XY")
        line = state.buffer.lines[0].content.plain
        assert line.startswith("XYABCDE")
