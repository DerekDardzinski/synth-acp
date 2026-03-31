"""Syntax-highlighted diff view with line numbers and split/unified modes."""

from __future__ import annotations

import asyncio
import difflib
from collections.abc import Iterable
from typing import ClassVar, Literal

from rich.segment import Segment
from textual import containers, events, highlight
from textual.app import ComposeResult
from textual.content import Content, Span
from textual.css.styles import RulesMap
from textual.geometry import Size
from textual.reactive import reactive, var
from textual.selection import Selection
from textual.strip import Strip
from textual.style import Style
from textual.visual import RenderOptions, Visual
from textual.widget import Widget
from textual.widgets import Static

type Annotation = Literal["+", "-", "/", " "]


def loop_last[T](values: Iterable[T]) -> Iterable[tuple[bool, T]]:
    """Iterate yielding (is_last, value) tuples."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    for value in iter_values:
        yield False, previous_value
        previous_value = value
    yield True, previous_value


def fill_lists[T](list_a: list[T], list_b: list[T], fill_value: T) -> None:
    """Pad the shorter list with fill_value so both have equal length."""
    diff = len(list_a) - len(list_b)
    if diff > 0:
        list_b.extend([fill_value] * diff)
    elif diff < 0:
        list_a.extend([fill_value] * -diff)


class DiffScrollContainer(containers.HorizontalGroup):
    """Horizontally scrollable container that links scroll position to a peer."""

    scroll_link: var[Widget | None] = var(None)

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        if self.scroll_link:
            self.scroll_link.scroll_x = new_value


class LineContent(Visual):
    """Custom Visual that renders syntax-highlighted code lines with soft wrapping."""

    def __init__(
        self,
        code_lines: list[Content | None],
        line_styles: list[str],
        width: int | None = None,
    ) -> None:
        self.code_lines = code_lines
        self.line_styles = line_styles
        self._width = width

    def _wrap_lines(self, width: int) -> list[tuple[list[Content], str]]:
        """Wrap each logical line, returning (wrapped_parts, style) per logical line."""
        result: list[tuple[list[Content], str]] = []
        for line, color in zip(self.code_lines, self.line_styles, strict=False):
            if line is None:
                result.append(([Content.styled("╲" * width, "$foreground 15%")], color))
            elif line.cell_length > width > 0:
                result.append((line.wrap(width), color))
            else:
                result.append(([line], color))
        return result

    def render_strips(
        self, width: int, height: int | None, style: Style, options: RenderOptions  # noqa: ARG002
    ) -> list[Strip]:
        strips: list[Strip] = []
        wrapped = self._wrap_lines(width)
        y = 0
        for wrapped_parts, color in wrapped:
            for part in wrapped_parts:
                if part.cell_length < width:
                    part = part.pad_right(width - part.cell_length)
                part = part.stylize_before(color).stylize_before(style)
                segments = [
                    Segment(text, rich_style)
                    for text, rich_style, _ in part.render_segments()
                ]
                strips.append(Strip(segments, part.cell_length))
                y += 1
        return strips

    def rows_per_line(self, width: int) -> list[int]:
        """Return the number of visual rows each logical line occupies at the given width."""
        return [len(parts) for parts, _ in self._wrap_lines(width)]

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:  # noqa: ARG002
        if self._width is not None:
            return self._width
        return max(
            (line.cell_length for line in self.code_lines if line is not None), default=1
        )

    def get_minimal_width(self, rules: RulesMap) -> int:  # noqa: ARG002
        return 1

    def get_height(self, rules: RulesMap, width: int) -> int:  # noqa: ARG002
        if width <= 0:
            return len(self.line_styles)
        return sum(self.rows_per_line(width))


class LineAnnotations(Widget):
    """Vertical gutter showing line numbers or annotation symbols."""

    numbers: reactive[list[Content]] = reactive(list)

    def __init__(
        self,
        numbers: Iterable[Content],
        *,
        rows_per_line: list[int] | None = None,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.numbers = list(numbers)
        self._rows_per_line = rows_per_line

    def _expanded_numbers(self) -> list[Content]:
        """Expand numbers list to account for wrapped continuation rows."""
        if self._rows_per_line is None:
            return self.numbers
        result: list[Content] = []
        for num, rows in zip(self.numbers, self._rows_per_line, strict=False):
            result.append(num)
            for _ in range(rows - 1):
                result.append(Content(" " * num.cell_length))
        return result

    @property
    def total_width(self) -> int:
        return max((n.cell_length for n in self.numbers), default=0)

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        return self.total_width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:  # noqa: ARG002
        return len(self._expanded_numbers())

    def render_line(self, y: int) -> Strip:
        width = self.total_width
        rich_style = self.visual_style.rich_style
        expanded = self._expanded_numbers()
        try:
            number = expanded[y]
        except IndexError:
            number = Content.empty()
        strip = Strip(number.render_segments(self.visual_style), cell_length=number.cell_length)
        return strip.adjust_cell_length(width, rich_style)


class DiffCode(Static):
    """Code container with text selection support."""

    ALLOW_SELECT = True

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        visual = self._render()
        if isinstance(visual, LineContent):
            text = "\n".join("" if line is None else line.plain for line in visual.code_lines)
            return selection.extract(text), "\n"
        return None


class DiffView(containers.VerticalGroup):
    """Syntax-highlighted diff with unified and split view modes."""

    code_before: reactive[str] = reactive("")
    code_after: reactive[str] = reactive("")
    path1: reactive[str] = reactive("")
    path2: reactive[str] = reactive("")
    split: reactive[bool] = reactive(True, recompose=True)
    annotations: var[bool] = var(False, toggle_class="-with-annotations")
    auto_split: var[bool] = var(True)

    NUMBER_STYLES: ClassVar[dict[str, str]] = {
        "+": "$text-success 80% on $success 20%",
        "-": "$text-error 80% on $error 20%",
        " ": "$foreground 30% on $foreground 3%",
    }
    LINE_STYLES: ClassVar[dict[str, str]] = {
        "+": "on $success 10%",
        "-": "on $error 10%",
        " ": "",
        "/": "",
    }
    EDGE_STYLES: ClassVar[dict[str, str]] = {
        "+": "$text-success 30% on $success 20%",
        "-": "$text-error 30% on $error 20%",
        " ": "$foreground 10% on $foreground 3%",
    }

    def __init__(
        self,
        path1: str,
        path2: str,
        code_before: str,
        code_after: str,
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.set_reactive(DiffView.path1, path1)
        self.set_reactive(DiffView.path2, path2)
        self.set_reactive(DiffView.code_before, code_before.expandtabs())
        self.set_reactive(DiffView.code_after, code_after.expandtabs())
        self._grouped_opcodes: list[list[tuple[str, int, int, int, int]]] | None = None
        self._highlighted_code_lines: tuple[list[Content], list[Content]] | None = None

    async def prepare(self) -> None:
        """Offload CPU-heavy highlighting to a thread."""
        def _work() -> None:
            self.grouped_opcodes  # noqa: B018
            self.highlighted_code_lines  # noqa: B018
        await asyncio.to_thread(_work)

    @property
    def grouped_opcodes(self) -> list[list[tuple[str, int, int, int, int]]]:
        if self._grouped_opcodes is None:
            matcher = difflib.SequenceMatcher(
                lambda c: c in {" ", "\t"},
                self.code_before.splitlines(),
                self.code_after.splitlines(),
                autojunk=True,
            )
            self._grouped_opcodes = list(matcher.get_grouped_opcodes())
        return self._grouped_opcodes

    @property
    def counts(self) -> tuple[int, int]:
        """Return (additions, removals)."""
        additions = removals = 0
        for group in self.grouped_opcodes:
            for tag, i1, i2, j1, j2 in group:
                if tag == "delete":
                    removals += i2 - i1
                elif tag == "replace":
                    additions += j2 - j1
                    removals += i2 - i1
                elif tag == "insert":
                    additions += j2 - j1
        return additions, removals

    @classmethod
    def _highlight_diff_lines(
        cls, lines_a: list[Content], lines_b: list[Content]
    ) -> tuple[list[Content], list[Content]]:
        """Character-level diff highlighting within changed line groups."""
        code_a = Content("\n").join(lines_a)
        code_b = Content("\n").join(lines_b)
        matcher = difflib.SequenceMatcher(
            lambda c: c in {" ", "\t"}, code_a.plain, code_b.plain, autojunk=True
        )
        spans_a: list[Span] = []
        spans_b: list[Span] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in {"delete", "replace"}:
                spans_a.append(Span(i1, i2, "on $error 30%"))
            if tag in {"insert", "replace"}:
                spans_b.append(Span(j1, j2, "on $success 30%"))
        return code_a.add_spans(spans_a).split("\n"), code_b.add_spans(spans_b).split("\n")

    @property
    def highlighted_code_lines(self) -> tuple[list[Content], list[Content]]:
        """Syntax-highlighted lines with character-level diff spans."""
        if self._highlighted_code_lines is None:
            lang1 = highlight.guess_language(self.code_before, self.path1)
            lang2 = highlight.guess_language(self.code_after, self.path2)
            text_a = self.code_before.splitlines()
            text_b = self.code_after.splitlines()
            lines_a = highlight.highlight(
                "\n".join(text_a), language=lang1, path=self.path1
            ).split("\n")
            lines_b = highlight.highlight(
                "\n".join(text_b), language=lang2, path=self.path2
            ).split("\n")
            if self.code_before:
                for group in self.grouped_opcodes:
                    for tag, i1, i2, j1, j2 in group:
                        if tag == "replace" and (j2 - j1) == (i2 - i1):
                            da, db = self._highlight_diff_lines(
                                lines_a[i1:i2], lines_b[j1:j2]
                            )
                            lines_a[i1:i2] = da
                            lines_b[j1:j2] = db
            self._highlighted_code_lines = (lines_a, lines_b)
        return self._highlighted_code_lines

    def get_title(self) -> Content:
        additions, removals = self.counts
        return Content.from_markup(
            "📄 [dim]$path[/dim]"
            " ([$text-success][b]+$additions[/b][/],"
            " [$text-error][b]-$removals[/b][/])",
            path=self.path2,
            additions=additions,
            removals=removals,
        ).stylize_before("$text")

    def compose(self) -> ComposeResult:
        yield Static(self.get_title(), classes="diff-title")
        if self.split:
            yield from self.compose_split()
        else:
            yield from self.compose_unified()

    def _check_auto_split(self, width: int) -> None:
        if self.auto_split:
            lines_a, lines_b = self.highlighted_code_lines
            split_width = max(
                (line.cell_length for line in lines_a + lines_b), default=0
            ) * 2
            split_width += 4 + 2 * max(len(str(len(lines_a))), len(str(len(lines_b))))
            split_width += 3 * 2 if self.annotations else 2
            self.split = width >= split_width

    async def on_resize(self, event: events.Resize) -> None:
        self._check_auto_split(event.size.width)

    async def on_mount(self) -> None:
        self._check_auto_split(self.size.width)

    def compose_unified(self) -> ComposeResult:
        lines_a, lines_b = self.highlighted_code_lines
        num_styles = self.NUMBER_STYLES
        line_styles = self.LINE_STYLES
        edge_styles = self.EDGE_STYLES
        for last, group in loop_last(self.grouped_opcodes):
            line_numbers_a: list[int | None] = []
            line_numbers_b: list[int | None] = []
            annots: list[str] = []
            code_lines: list[Content | None] = []
            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for off, line in enumerate(lines_a[i1:i2], 1):
                        annots.append(" ")
                        line_numbers_a.append(i1 + off)
                        line_numbers_b.append(j1 + off)
                        code_lines.append(line)
                    continue
                if tag in {"delete", "replace"}:
                    for off, line in enumerate(lines_a[i1:i2], 1):
                        annots.append("-")
                        line_numbers_a.append(i1 + off)
                        line_numbers_b.append(None)
                        code_lines.append(line)
                if tag in {"insert", "replace"}:
                    for off, line in enumerate(lines_b[j1:j2], 1):
                        annots.append("+")
                        line_numbers_a.append(None)
                        line_numbers_b.append(j1 + off)
                        code_lines.append(line)

            lnw = max(
                len("" if n is None else str(n))
                for n in line_numbers_a + line_numbers_b
            )
            with containers.HorizontalGroup(classes="diff-group"):
                yield LineAnnotations([
                    (
                        Content(f"▎{' ' * lnw} ") if n is None
                        else Content(f"▎{n:>{lnw}} ")
                    )
                    .stylize(num_styles[a], 1)
                    .stylize(edge_styles[a], 0, 1)
                    for n, a in zip(line_numbers_a, annots, strict=False)
                ])
                yield LineAnnotations([
                    (
                        Content(f" {' ' * lnw} ") if n is None
                        else Content(f" {n:>{lnw}} ")
                    ).stylize(num_styles[a])
                    for n, a in zip(line_numbers_b, annots, strict=False)
                ])
                yield LineAnnotations(
                    [
                        Content(f" {a} ").stylize(line_styles[a]).stylize("bold")
                        for a in annots
                    ],
                    classes="annotations",
                )
                with DiffScrollContainer():
                    yield DiffCode(
                        LineContent(code_lines, [line_styles[a] for a in annots])
                    )
            if not last:
                yield Static("⋮", classes="ellipsis")

    def compose_split(self) -> ComposeResult:
        lines_a, lines_b = self.highlighted_code_lines
        ann_hatch = Content.styled("╲" * 3, "$foreground 15%")
        ann_blank = Content(" " * 3)

        def make_ann(ann: Annotation, which: Literal["+", "-"]) -> Content:
            if ann == which:
                return (
                    Content(f" {ann} ")
                    .stylize(self.LINE_STYLES[ann])
                    .stylize("bold")
                )
            if ann == "/":
                return ann_hatch
            return ann_blank

        for last, group in loop_last(self.grouped_opcodes):
            ln_a: list[int | None] = []
            ln_b: list[int | None] = []
            ann_a: list[Annotation] = []
            ann_b: list[Annotation] = []
            cl_a: list[Content | None] = []
            cl_b: list[Content | None] = []
            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for off, line in enumerate(lines_a[i1:i2], 1):
                        ann_a.append(" ")
                        ann_b.append(" ")
                        ln_a.append(i1 + off)
                        ln_b.append(j1 + off)
                        cl_a.append(line)
                        cl_b.append(line)
                else:
                    if tag in {"delete", "replace"}:
                        for num, line in enumerate(lines_a[i1:i2], i1 + 1):
                            ann_a.append("-")
                            ln_a.append(num)
                            cl_a.append(line)
                    if tag in {"insert", "replace"}:
                        for num, line in enumerate(lines_b[j1:j2], j1 + 1):
                            ann_b.append("+")
                            ln_b.append(num)
                            cl_b.append(line)
                    fill_lists(cl_a, cl_b, None)
                    fill_lists(ann_a, ann_b, "/")
                    fill_lists(ln_a, ln_b, None)

            lnw = max(
                (0 if n is None else len(str(n)) for n in ln_a + ln_b), default=1
            )
            _hatch = Content.styled("╲" * (2 + lnw), "$foreground 15%")
            _lnw = lnw

            def fmt_num(n: int | None, a: str, *, _h: Content = _hatch, _w: int = _lnw) -> Content:
                if n is None:
                    return _h
                return (
                    Content(f"▎{n:>{_w}} ")
                    .stylize(self.NUMBER_STYLES[a], 1)
                    .stylize(self.EDGE_STYLES[a], 0, 1)
                )

            line_width = max(
                (line.cell_length for line in cl_a + cl_b if line is not None),
                default=1,
            )
            with containers.HorizontalGroup(classes="diff-group"):
                yield LineAnnotations(map(fmt_num, ln_a, ann_a))
                yield LineAnnotations(
                    [make_ann(a, "-") for a in ann_a],
                    classes="annotations",
                )
                with DiffScrollContainer() as sc_a:
                    yield DiffCode(
                        LineContent(
                            cl_a,
                            [self.LINE_STYLES[a] for a in ann_a],
                            width=line_width,
                        )
                    )
                yield LineAnnotations(map(fmt_num, ln_b, ann_b))
                yield LineAnnotations(
                    [make_ann(a, "+") for a in ann_b],
                    classes="annotations",
                )
                with DiffScrollContainer() as sc_b:
                    yield DiffCode(
                        LineContent(
                            cl_b,
                            [self.LINE_STYLES[a] for a in ann_b],
                            width=line_width,
                        )
                    )
                sc_a.scroll_link = sc_b
                sc_b.scroll_link = sc_a
            if not last:
                with containers.HorizontalGroup():
                    yield Static("⋮", classes="ellipsis")
                    yield Static("⋮", classes="ellipsis")
