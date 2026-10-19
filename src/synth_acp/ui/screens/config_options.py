"""ConfigOptionsScreen — modal for the session config options not shown inline."""

from __future__ import annotations

from typing import ClassVar

from acp.schema import (
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
)
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.markup import escape
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, Switch

from synth_acp.ui.ids import widget_id

ConfigOption = SessionConfigOptionSelect | SessionConfigOptionBoolean


def select_items(opt: SessionConfigOptionSelect) -> list[tuple[str, str]]:
    """Return ``(value, display name)`` pairs for a select option, groups flattened.

    Values that share a display name get the value appended, so each stays distinguishable.
    """
    items: list[tuple[str, str]] = []
    for entry in opt.options:
        if isinstance(entry, SessionConfigSelectGroup):
            items.extend((sub.value, sub.name) for sub in entry.options)
        else:
            items.append((entry.value, entry.name))
    counts: dict[str, int] = {}
    for _, name in items:
        counts[name] = counts.get(name, 0) + 1
    return [(value, f"{name} ({value})" if counts[name] > 1 else name) for value, name in items]


class ConfigOptionsScreen(ModalScreen[dict[str, str | bool] | None]):
    """Edit an agent's remaining config options, one row per option, in the agent's order.

    Dismisses with ``{config_id: new value}`` for the options the user changed (empty when
    nothing changed), or None on cancel.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Close")]

    def __init__(self, agent_id: str, options: list[ConfigOption]) -> None:
        super().__init__()
        self._agent_id = agent_id
        self._options = options
        # widget id -> config option id; option ids are agent-chosen, so escaped.
        self._ids = {widget_id("config-option", opt.id): opt.id for opt in options}

    def compose(self) -> ComposeResult:
        """Yield one labelled control per option, then Apply."""
        with Vertical(id="config-options-container"):
            yield Label(f"Session options · {escape(self._agent_id)}", id="config-options-title")
            for opt in self._options:
                wid = widget_id("config-option", opt.id)
                label = escape(opt.name)
                if opt.description:
                    label += f" [dim]{escape(opt.description)}[/dim]"
                if isinstance(opt, SessionConfigOptionBoolean):
                    with Horizontal(classes="config-option-row"):
                        yield Label(label, classes="field-label")
                        yield Switch(value=opt.current_value, id=wid)
                    continue
                yield Label(label, classes="field-label")
                items = select_items(opt)
                known = any(value == opt.current_value for value, _ in items)
                yield Select(
                    [(name, value) for value, name in items],
                    value=opt.current_value if known else Select.BLANK,
                    allow_blank=not known,
                    id=wid,
                )
            yield Button("Apply", id="config-options-apply", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the options whose value differs from the one shown on open."""
        if event.button.id != "config-options-apply":
            return
        event.stop()
        current = {opt.id: opt.current_value for opt in self._options}
        changes: dict[str, str | bool] = {}
        for wid, config_id in self._ids.items():
            widget = self.query_one(f"#{wid}")
            if isinstance(widget, Switch):
                value: object = widget.value
            elif isinstance(widget, Select):
                value = widget.value
            else:
                continue
            if isinstance(value, str | bool) and value != current[config_id]:
                changes[config_id] = value
        self.dismiss(changes)

    def action_dismiss_none(self) -> None:
        """Close without changing anything."""
        self.dismiss(None)
