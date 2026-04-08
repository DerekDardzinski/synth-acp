"""SessionPickerScreen — modal for selecting a session to restore."""

from __future__ import annotations

import time
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


def _relative_time(ms_timestamp: int) -> str:
    """Convert a millisecond epoch timestamp to a human-readable relative string."""
    delta = int(time.time()) - (ms_timestamp // 1000)
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = delta // 60
        return f"{m}m ago"
    if delta < 86400:
        h = delta // 3600
        return f"{h}h ago"
    d = delta // 86400
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


class SessionPickerScreen(ModalScreen[str | None]):
    """Modal that lists restorable sessions and returns the selected session_id."""

    DEFAULT_CSS = """
    SessionPickerScreen {
        align: center middle;
    }
    #picker-container {
        width: 70;
        max-height: 20;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    #picker-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Close")]

    def __init__(self, sessions: list[dict]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-container"):
            yield Label("Restore Session", id="picker-title")
            if not self._sessions:
                yield Label("No restorable sessions found. Press ESC.")
            else:
                options = OptionList(id="session-list")
                for s in self._sessions:
                    agents = ", ".join(s["agents"])
                    when = _relative_time(s["last_active"])
                    label = f"{s['session_id']}  ({s['agent_count']} agents, {when})\n  {agents}"
                    options.add_option(Option(label, id=s["session_id"]))
                yield options

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
