"""Markdown that draws mermaid diagrams and LaTeX math as text.

It also rewrites three constructs Textual's Markdown shows badly, always, with no
optional library: HTML blocks (Textual drops them and their text), GitHub alerts
(``> [!WARNING]``) and task lists (``- [x] done``).

Two optional libraries do the drawing:

- ``termaid`` (``synth-acp[diagrams]``) draws closed ```` ```mermaid ```` fences.
- ``TeXicode`` (``synth-acp[math]``) draws ``$…$`` inline math, ``$$…$$`` display math
  and ```` ```math ```` fences. ```` ```latex ```` fences are left alone, because agents
  use them for LaTeX source code as often as for equations.

```` ```csv ```` and ```` ```tsv ```` fences are drawn as tables with no extra library.
Every drawn fence has a "⇄ source" label that switches between drawing and source.

Drawing is off while a message streams, because neither library rejects unfinished
input: each draws what it has, which would show a wrong picture. The owner turns
``drawing`` on once the message is complete and re-parses it.

Any failure shows the source instead: the library missing, an exception, an empty
drawing, an error row from TeXicode, or a ``?`` TeXicode put in for an unknown command.
"""

from __future__ import annotations

import csv
import io
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, ClassVar

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from rich.cells import cell_len
from rich.text import Text
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Static
from textual.widgets._markdown import MarkdownBlockQuote, MarkdownUnorderedListItem
from textual.widgets.markdown import Markdown, MarkdownBlock, MarkdownFence

if TYPE_CHECKING:
    from markdown_it.rules_block import StateBlock
    from markdown_it.rules_core import StateCore
    from markdown_it.rules_inline import StateInline
    from textual.app import ComposeResult
    from textual.events import Click

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)

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


TABLE_DELIMITERS = {"csv": ",", "tsv": "\t"}
MAX_TABLE_ROWS = 200
_NUMBER = re.compile(r"[-+]?(\d[\d,_]*)?(\.\d+)?([eE][-+]?\d+)?%?")


def render_table(code: str, delimiter: str) -> Content | None:
    """Draw CSV or TSV text as a box-drawn table, or None when it is not a table.

    The first row is the header. Columns whose body cells are all numbers are
    right-aligned. Rows past ``MAX_TABLE_ROWS`` are counted, not drawn.

    Returns:
        The table, or None when the text does not parse, has fewer than two rows, or
        has fewer than two columns (then it is a list, and the source reads better).
    """
    try:
        rows = list(csv.reader(io.StringIO(code.strip("\n")), delimiter=delimiter))
    except csv.Error:
        return None
    rows = [[cell.strip() for cell in row] for row in rows if row]
    if len(rows) < 2:
        return None
    n_cols = max(len(row) for row in rows)
    if n_cols < 2:
        return None
    rows = [row + [""] * (n_cols - len(row)) for row in rows]
    header, body = rows[0], rows[1:]
    hidden = max(0, len(body) - MAX_TABLE_ROWS)
    body = body[:MAX_TABLE_ROWS]
    widths = [max(cell_len(row[c]) for row in [header, *body]) for c in range(n_cols)]
    numeric = [
        any(row[c] for row in body) and all(not row[c] or _NUMBER.fullmatch(row[c]) for row in body)
        for c in range(n_cols)
    ]

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right + "\n"

    def cells(row: list[str], right_align: list[bool]) -> str:
        parts = []
        for c, cell in enumerate(row):
            pad = " " * (widths[c] - cell_len(cell))
            parts.append(f" {pad}{cell} " if right_align[c] else f" {cell}{pad} ")
        return "│" + "│".join(parts) + "│\n"

    text = Text(line("┌", "┬", "┐"))
    text.append(cells(header, [False] * n_cols), style="bold")
    text.append(line("├", "┼", "┤"))
    for row in body:
        text.append(cells(row, numeric))
    text.append(line("└", "┴", "┘").rstrip("\n"))
    if hidden:
        text.append(f"\n… {hidden} more rows", style="dim")
    return Content.from_rich_text(text)


class _DrawnToggle(Static):
    """Clickable label that switches a drawn fence between drawing and source."""

    def on_click(self, event: Click) -> None:
        event.stop()
        fence = self.parent
        if isinstance(fence, DrawnFence):
            fence.toggle_source()


class DrawnFence(MarkdownFence):
    """A fence that shows a drawing instead of code when it can.

    A small label above a drawing switches between the drawing and its source.
    """

    # Layout lives in app.tcss: its MarkdownFence rules wrap code lines, and app CSS
    # outranks DEFAULT_CSS, so the no-wrap rule for drawings must sit beside them.

    def __init__(self, markdown: Markdown, token: Token, code: str) -> None:
        super().__init__(markdown, token, code)
        self._source = self._highlighted_code
        self._showing_source = False
        self._drawing: Content | None = None
        if getattr(markdown, "drawing", False):
            language = (self.lexer or "").strip().split(maxsplit=1)
            name = language[0].lower() if language else ""
            if name == "mermaid":
                self._drawing = render_diagram(code)
            elif name == MATH_LANGUAGE:
                drawn = render_math(code, inline=False)
                self._drawing = Content(drawn) if drawn is not None else None
            elif name in TABLE_DELIMITERS:
                self._drawing = render_table(code, TABLE_DELIMITERS[name])
        if self._drawing is not None:
            self._highlighted_code = self._drawing
            self.add_class("-drawn")

    def compose(self) -> ComposeResult:
        # Only drawn fences get a toggle, so plain code blocks cost no extra widget.
        if self._drawing is not None:
            yield _DrawnToggle(self._toggle_text(), classes="drawn-toggle")
        yield from super().compose()

    def _toggle_text(self) -> str:
        return "⇄ drawing" if self._showing_source else "⇄ source"

    def _shown(self) -> Content:
        if self._drawing is None or self._showing_source:
            return self._source
        return self._drawing

    def toggle_source(self) -> None:
        """Switch between the drawing and its source. Does nothing without a drawing."""
        if self._drawing is None:
            return
        self._showing_source = not self._showing_source
        self._refresh_view()

    def _refresh_view(self) -> None:
        # compose builds the Label from _highlighted_code, not from set_content, so
        # both must hold what is shown.
        self._highlighted_code = self._shown()
        self.set_content(self._highlighted_code)
        try:
            toggle = self.query_one(_DrawnToggle)
        except NoMatches:
            # A streaming re-parse can hand a drawing to a fence composed without one.
            if self._drawing is not None and self.is_mounted:
                self.mount(_DrawnToggle(self._toggle_text(), classes="drawn-toggle"), before=0)
            return
        toggle.display = self._drawing is not None
        toggle.update(self._toggle_text())

    def notify_style_update(self) -> None:
        """Keep the shown view when the theme changes; the base class re-highlights."""
        super().notify_style_update()
        # The base just re-highlighted the source for the new theme into
        # _highlighted_code, and it can run before compose.
        if self._drawing is not None and self._highlighted_code is not self._drawing:
            self._source = self._highlighted_code
        self._refresh_view()

    def _copy_context(self, block: MarkdownBlock) -> None:
        super()._copy_context(block)
        if isinstance(block, DrawnFence):
            self._source = block._source
            self._drawing = block._drawing
            self._showing_source = False
            self.set_class(block._drawing is not None, "-drawn")
            self._refresh_view()


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


ALERT_KINDS = ("note", "tip", "important", "warning", "caution")
_ALERT_MARKER = re.compile(r"\[!(" + "|".join(ALERT_KINDS) + r")\]", re.IGNORECASE)
_TASK_MARKER = re.compile(r"\[([ xX])\](?: |$)")
_HTML_BLOCK_TAGS = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    ]
)


class _HTMLText(HTMLParser):
    """Collect the readable text of an HTML fragment, one entry per visual line."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = [""]
        self._skip = 0

    def _newline(self) -> None:
        if self.lines[-1].strip():
            self.lines.append("")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002 - HTMLParser signature
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _HTML_BLOCK_TAGS:
            self._newline()
            if tag == "summary":
                self.lines[-1] += "▸ "
            elif tag == "li":
                self.lines[-1] += "• "

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _HTML_BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.lines[-1] += data


def html_to_lines(html: str) -> list[str]:
    """Return the readable text of an HTML block as lines, tags and comments removed."""
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in parser.lines)
    return [line for line in lines if line and line not in {"▸", "•"}]


def _convert_html_block(tokens: list[Token], index: int) -> int:
    """Replace the html_block at ``index`` with a paragraph of its text.

    Textual's Markdown widget has no case for html_block, so without this the block,
    and every word inside it, silently disappears.

    Returns:
        The number of tokens now occupying the block's position (0 if it had no text).
    """
    block = tokens[index]
    lines = html_to_lines(block.content)
    if not lines:
        del tokens[index]
        return 0
    children: list[Token] = []
    for n, line in enumerate(lines):
        if n:
            children.append(Token("hardbreak", "br", 0))
        children.append(Token("text", "", 0, content=line))
    opened = Token("paragraph_open", "p", 1, map=block.map, block=True)
    inline = Token("inline", "", 0, map=block.map, content="\n".join(lines), children=children)
    closed = Token("paragraph_close", "p", -1, block=True)
    tokens[index : index + 1] = [opened, inline, closed]
    return 3


def _first_paragraph_inline(tokens: list[Token], index: int) -> Token | None:
    """Return the inline token of the paragraph that opens right after ``index``."""
    if index + 2 < len(tokens) and tokens[index + 1].type == "paragraph_open":
        inline = tokens[index + 2]
        if inline.type == "inline" and inline.children:
            return inline
    return None


def _mark_alert(tokens: list[Token], index: int) -> None:
    """Turn ``> [!WARNING]`` into a labeled alert quote, as GitHub renders it.

    The marker must be the whole first line of the quote. It is replaced by a bold
    label, and the quote token is tagged so AlertQuote can color its border.
    """
    inline = _first_paragraph_inline(tokens, index)
    if inline is None or inline.children is None:
        return
    children = inline.children
    first = children[0]
    match = _ALERT_MARKER.fullmatch(first.content.strip()) if first.type == "text" else None
    if match is None or (len(children) > 1 and children[1].type != "softbreak"):
        return
    kind = match.group(1).lower()
    tokens[index].meta["alert"] = kind
    label = [
        Token("strong_open", "strong", 1),
        Token("text", "", 0, content=kind.capitalize()),
        Token("strong_close", "strong", -1),
    ]
    rest = children[2:]
    inline.children = label + ([Token("hardbreak", "br", 0), *rest] if rest else [])


def _mark_task(tokens: list[Token], index: int) -> None:
    """Turn ``- [x] done`` into a list item whose bullet is a checkbox."""
    if tokens[index].info:  # ordered list item: keep its number
        return
    inline = _first_paragraph_inline(tokens, index)
    if inline is None or inline.children is None or inline.children[0].type != "text":
        return
    first = inline.children[0]
    match = _TASK_MARKER.match(first.content)
    if match is None:
        return
    tokens[index].meta["task"] = match.group(1) != " "
    first.content = first.content[match.end() :]


class AlertQuote(MarkdownBlockQuote):
    """A block quote that shows a GitHub alert (note, tip, warning, ...)."""

    def __init__(self, markdown: Markdown, token: Token) -> None:
        super().__init__(markdown, token)
        kind = token.meta.get("alert") if token.meta else None
        if kind:
            self.add_class("-alert", f"-alert-{kind}")


class TaskListItem(MarkdownUnorderedListItem):
    """A bullet list item whose bullet is a checkbox when it is a task."""

    def __init__(self, markdown: Markdown, token: Token, bullet: str) -> None:
        task = token.meta.get("task") if token.meta else None
        if task is not None:
            bullet = "☑ " if task else "☐ "
        super().__init__(markdown, token, bullet)


class DrawnMarkdown(Markdown):
    """Markdown that draws diagrams and math when ``drawing`` is True.

    It also always shows HTML blocks as text, GitHub alerts as labeled quotes, and
    task lists with checkboxes.
    """

    BLOCKS: ClassVar[dict[str, type[MarkdownBlock]]] = {
        **Markdown.BLOCKS,
        "fence": DrawnFence,
        "blockquote_open": AlertQuote,
        "list_item_unordered_open": TaskListItem,
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
        parser.core.ruler.push("synth_blocks", self._convert_block_tokens)
        return parser

    @staticmethod
    def _convert_block_tokens(state: StateCore) -> None:
        """Rewrite HTML, alert and task-list tokens into ones Textual renders.

        These run while streaming too: unlike diagrams and math, a half-written one
        just shows as text until its line is complete.
        """
        tokens = state.tokens
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.type == "html_block":
                index += _convert_html_block(tokens, index)
                continue
            if token.type == "blockquote_open":
                _mark_alert(tokens, index)
            elif token.type == "list_item_open":
                _mark_task(tokens, index)
            elif token.type == "inline" and token.children:
                for child in token.children:
                    if child.type == "html_inline" and _BR.fullmatch(child.content):
                        child.type = "hardbreak"
            index += 1

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
