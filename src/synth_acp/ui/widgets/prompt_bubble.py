"""Right-aligned user message bubble."""

from __future__ import annotations

from textual.widgets import Static


class PromptBubble(Static):
    """Right-aligned user prompt with $primary border.

    Args:
        text: The user's message text.
        timestamp: Display timestamp string.
    """

    def __init__(self, text: str, timestamp: str) -> None:
        super().__init__(f"{text}\n[dim]{timestamp}[/dim]")
