"""Right-aligned user message bubble with markdown rendering."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.widgets.markdown import Markdown

from synth_acp.ui.widgets.copy_button import CopyButton


class PromptBubble(Vertical):
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
        yield Markdown(self._text, open_links=False)
        yield Static(f"[dim]{self._timestamp}[/dim]", classes="bubble-ts")
