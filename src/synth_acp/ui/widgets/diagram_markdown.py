"""Markdown that draws closed ```mermaid fences as text diagrams.

Diagrams are drawn with the optional ``termaid`` library (``synth-acp[diagrams]``).
Drawing is off while a message streams, because an unfinished fence would draw as a
half-finished picture: termaid does not reject incomplete input, it draws what it has.
The owner turns drawing on once the message is complete and re-parses it.

Any failure falls back to the normal highlighted code block: termaid missing, an
exception while drawing, or an empty drawing (termaid's answer to text it cannot parse).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.content import Content
from textual.widgets.markdown import Markdown, MarkdownBlock, MarkdownFence

if TYPE_CHECKING:
    from markdown_it.token import Token

DIAGRAM_LANGUAGES = frozenset({"mermaid"})


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


class DiagramFence(MarkdownFence):
    """A fence that shows a drawn diagram instead of code when it can."""

    # Layout lives in app.tcss: its MarkdownFence rules wrap code lines, and app CSS
    # outranks DEFAULT_CSS, so the no-wrap rule for drawings must sit beside them.

    def __init__(self, markdown: Markdown, token: Token, code: str) -> None:
        super().__init__(markdown, token, code)
        self._diagram: Content | None = None
        language = (self.lexer or "").strip().split(maxsplit=1)[0:1]
        if (
            language
            and language[0].lower() in DIAGRAM_LANGUAGES
            and getattr(markdown, "render_diagrams", False)
        ):
            self._diagram = render_diagram(code)
        if self._diagram is not None:
            self._highlighted_code = self._diagram
            self.add_class("-diagram")

    def notify_style_update(self) -> None:
        """Keep the drawing when the theme changes; the base class re-highlights code."""
        super().notify_style_update()
        if self._diagram is not None:
            # The base can run before compose, so restore the stored content too:
            # compose builds the Label from _highlighted_code, not from set_content.
            self._highlighted_code = self._diagram
            self.set_content(self._diagram)

    def _copy_context(self, block: MarkdownBlock) -> None:
        super()._copy_context(block)
        if isinstance(block, DiagramFence):
            self._diagram = block._diagram
            self.set_class(block._diagram is not None, "-diagram")


class DiagramMarkdown(Markdown):
    """Markdown whose mermaid fences are drawn when ``render_diagrams`` is True."""

    BLOCKS: ClassVar[dict[str, type[MarkdownBlock]]] = {
        **Markdown.BLOCKS,
        "fence": DiagramFence,
    }

    def __init__(self, *args: Any, render_diagrams: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.render_diagrams = render_diagrams
