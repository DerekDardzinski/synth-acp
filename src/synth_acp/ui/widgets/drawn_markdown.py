"""Markdown that draws mermaid diagrams and LaTeX math as text.

Two optional libraries do the drawing:

- ``termaid`` (``synth-acp[diagrams]``) draws closed ```` ```mermaid ```` fences.
- ``TeXicode`` (``synth-acp[math]``) draws ``$…$`` inline math, ``$$…$$`` display math
  and ```` ```math ```` fences. ```` ```latex ```` fences are left alone, because agents
  use them for LaTeX source code as often as for equations.

Drawing is off while a message streams, because neither library rejects unfinished
input: each draws what it has, which would show a wrong picture. The owner turns
``drawing`` on once the message is complete and re-parses it.

Any failure shows the source instead: the library missing, an exception, an empty
drawing, an error row from TeXicode, or a ``?`` TeXicode put in for an unknown command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from textual.content import Content
from textual.widgets.markdown import Markdown, MarkdownBlock, MarkdownFence

if TYPE_CHECKING:
    from markdown_it.rules_block import StateBlock
    from markdown_it.rules_core import StateCore
    from markdown_it.rules_inline import StateInline
    from markdown_it.token import Token

MATH_LANGUAGE = "math"


def render_diagram(code: str) -> Content | None:
    """Draw Mermaid source as text, or return None when it cannot be drawn.

    Args:
        code: Mermaid source, without the fence lines.

    Returns:
        The drawing as Content, or None if termaid is not installed, raised, or drew
        nothing.
    """
    try:
        from termaid import render_rich
    except ImportError:
        return None
    try:
        text = render_rich(code)
    except Exception:  # third-party parser on model output: any failure means "show code"
        return None
    if not text.plain.strip():
        return None
    return Content.from_rich_text(text)


def render_math(tex: str, *, inline: bool) -> str | None:
    """Draw LaTeX math as Unicode text, or return None when it cannot be drawn.

    Args:
        tex: The math, without its ``$`` delimiters.
        inline: Require a single-row drawing, so it fits inside a line of prose.

    Returns:
        The drawing, or None if TeXicode is not installed, raised, reported an error,
        drew an unknown command as ``?``, drew nothing, or (inline) needed several rows.
    """
    try:
        from texicode.pipeline import render_tex_rows
    except ImportError:
        return None
    try:
        rows = [row.rstrip() for row in render_tex_rows(tex.strip(), False)]
    except Exception:  # third-party parser on model output: any failure means "show source"
        return None
    while rows and not rows[-1]:
        rows.pop()
    drawn = "\n".join(rows)
    if not drawn.strip() or any(row.lstrip().startswith("TeXicode:") for row in rows):
        return None
    if "?" in drawn and "?" not in tex:
        return None
    if inline:
        if len(rows) != 1:
            return None
        return drawn.strip()
    return drawn


class DrawnFence(MarkdownFence):
    """A fence that shows a drawing instead of code when it can."""

    # Layout lives in app.tcss: its MarkdownFence rules wrap code lines, and app CSS
    # outranks DEFAULT_CSS, so the no-wrap rule for drawings must sit beside them.

    def __init__(self, markdown: Markdown, token: Token, code: str) -> None:
        super().__init__(markdown, token, code)
        self._drawing: Content | None = None
        if getattr(markdown, "drawing", False):
            language = (self.lexer or "").strip().split(maxsplit=1)
            name = language[0].lower() if language else ""
            if name == "mermaid":
                self._drawing = render_diagram(code)
            elif name == MATH_LANGUAGE:
                drawn = render_math(code, inline=False)
                self._drawing = Content(drawn) if drawn is not None else None
        if self._drawing is not None:
            self._highlighted_code = self._drawing
            self.add_class("-drawn")

    def notify_style_update(self) -> None:
        """Keep the drawing when the theme changes; the base class re-highlights code."""
        super().notify_style_update()
        if self._drawing is not None:
            # The base can run before compose, so restore the stored content too:
            # compose builds the Label from _highlighted_code, not from set_content.
            self._highlighted_code = self._drawing
            self.set_content(self._drawing)

    def _copy_context(self, block: MarkdownBlock) -> None:
        super()._copy_context(block)
        if isinstance(block, DrawnFence):
            self._drawing = block._drawing
            self.set_class(block._drawing is not None, "-drawn")


def _interrupting(rule: Any) -> Any:
    """Make dollarmath's ``$$`` block rule safe to run as a paragraph terminator.

    markdown-it calls terminators with ``silent=True`` and expects no side effects, but
    the plugin's rule ignores ``silent`` and always pushes a token. Undo its effects in
    that mode so ``$$`` can end a paragraph without a blank line before it.
    """

    def wrapped(state: StateBlock, start: int, end: int, silent: bool) -> bool:
        if not silent:
            return rule(state, start, end, silent)
        n_tokens, line = len(state.tokens), state.line
        found = rule(state, start, end, silent)
        del state.tokens[n_tokens:]
        state.line = line
        return found

    return wrapped


def _word_bounded(rule: Any) -> Any:
    """Reject inline ``$…$`` that touches a letter or digit on the outside.

    The plugin's strict mode still accepts ``"$10" and "$HOME"`` as math, because the
    span ``$10" and "$`` has no space inside its dollars and no digit after the
    closing one. Real inline math almost never sits flush against a word, so a
    match whose opening ``$`` follows, or whose closing ``$`` precedes, a letter or
    digit is undone and left as text.
    """

    def wrapped(state: StateInline, silent: bool) -> bool:
        start, n_tokens = state.pos, len(state.tokens)
        if not rule(state, silent):
            return False
        src, end = state.src, state.pos
        before = src[start - 1] if start > 0 else ""
        after = src[end] if end < len(src) else ""
        if before.isalnum() or after.isalnum():
            del state.tokens[n_tokens:]
            state.pos = start
            return False
        return True

    return wrapped


class DrawnMarkdown(Markdown):
    """Markdown that draws diagrams and math when ``drawing`` is True."""

    BLOCKS: ClassVar[dict[str, type[MarkdownBlock]]] = {
        **Markdown.BLOCKS,
        "fence": DrawnFence,
    }

    def __init__(self, *args: Any, drawing: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("parser_factory", self._make_parser)
        super().__init__(*args, **kwargs)
        self.drawing = drawing

    def _make_parser(self) -> MarkdownIt:
        """Build the gfm-like parser plus strict ``$`` math.

        Strict means no space just inside the dollars, no digit right after the
        closing one, and no letter or digit touching either dollar from outside, so
        prices ("$5 and $10") and shell variables ("$HOME") stay text.
        """
        parser = MarkdownIt("gfm-like").use(
            dollarmath_plugin,
            allow_space=False,
            allow_digits=False,
            allow_labels=False,
            allow_blank_lines=False,
        )
        rule = next(r.fn for r in parser.block.ruler.__rules__ if r.name == "math_block")
        parser.block.ruler.at(
            "math_block",
            _interrupting(rule),
            {"alt": ["paragraph", "reference", "blockquote", "list"]},
        )
        inline = next(r.fn for r in parser.inline.ruler.__rules__ if r.name == "math_inline")
        parser.inline.ruler.at("math_inline", _word_bounded(inline))
        parser.core.ruler.push("synth_math", self._convert_math_tokens)
        return parser

    def _convert_math_tokens(self, state: StateCore) -> None:
        """Turn math tokens into types Textual renders; it drops types it does not know.

        ``$$`` blocks become ```` ```math ```` fences, which DrawnFence draws. Inline
        math becomes text: the drawing when drawing is on and it fits on one row,
        otherwise the original ``$…$`` source.
        """
        for token in state.tokens:
            if token.type == "math_block":
                token.type = "fence"
                token.tag = "code"
                token.info = MATH_LANGUAGE
                token.markup = "```"
                token.content = token.content.strip() + "\n"
            elif token.type == "inline" and token.children:
                for child in token.children:
                    if child.type != "math_inline":
                        continue
                    drawn = render_math(child.content, inline=True) if self.drawing else None
                    child.type = "text"
                    child.content = drawn if drawn is not None else f"${child.content}$"
