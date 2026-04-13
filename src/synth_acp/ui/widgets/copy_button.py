"""Clipboard copy button that briefly shows a checkmark on success."""

from __future__ import annotations

from collections.abc import Callable

from textual.widgets import Static


class CopyButton(Static):
    """Small button that copies text to the system clipboard on click.

    Args:
        text_source: Callable returning the text to copy.
    """

    DEFAULT_CSS = """
    CopyButton {
        dock: right;
        width: 3;
        height: 1;
        content-align: right top;
        color: $text-muted;
        background: transparent;
    }
    CopyButton:hover { color: $foreground; }
    """

    def __init__(self, text_source: Callable[[], str]) -> None:
        super().__init__("⎘", classes="copy-btn")
        self._text_source = text_source

    async def on_click(self) -> None:
        """Copy text to clipboard and flash a checkmark."""
        self.app.copy_to_clipboard(self._text_source())
        self.update("[green]✓[/green]")
        self.set_timer(1.0, lambda: self.update("⎘"))
