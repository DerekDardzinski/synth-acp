"""Right-aligned user message bubble with markdown rendering."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from synth_acp.ui.widgets.copy_button import CopyButton
from synth_acp.ui.widgets.drawn_markdown import DrawnMarkdown


class PromptBubble(Vertical, can_focus=False):
    """Right-aligned user prompt with $primary border and markdown rendering.

    Args:
        text: The user's message text (rendered as markdown).
        timestamp: Display timestamp string.
    """

    def __init__(self, text: str, timestamp: str) -> None:
        super().__init__()
        self._text = text
        self._timestamp = timestamp

    def compose(self) -> ComposeResult:
        yield CopyButton(lambda: self._text)
        yield DrawnMarkdown(self._text, open_links=False, drawing=True)
        yield Static(f"[dim]{self._timestamp}[/dim]", classes="bubble-ts")
