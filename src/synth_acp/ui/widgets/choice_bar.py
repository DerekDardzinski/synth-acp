"""ChoiceBar — inline option picker mounted above the InputBar."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, NamedTuple

from textual import events, on
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup
from textual.content import Content
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import Label

log = logging.getLogger(__name__)


class Choice(NamedTuple):
    """One option in a ChoiceBar."""

    choice_id: str
    label: str
    hotkey: str | None = None


class _NonSelectableLabel(Label):
    ALLOW_SELECT = False


class _OptionRow(HorizontalGroup):
    """Single option row with caret, hotkey, and label."""

    ALLOW_SELECT = False

    class Selected(Message):
        """The option was clicked."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    selected: reactive[bool] = reactive(False, toggle_class="-selected")

    def __init__(self, index: int, name: str, key: str | None) -> None:
        super().__init__()
        self.index = index
        self._name = name
        self._key = key

    def compose(self):
        """Yield caret, hotkey index, and label."""
        yield _NonSelectableLabel("❯", id="caret")  # noqa: RUF001
        if self._key:
            yield _NonSelectableLabel(Content.styled(self._key, "b"), id="index")
        else:
            yield _NonSelectableLabel(Content(" "), id="index")
        yield _NonSelectableLabel(self._name, id="label", markup=False)

    def on_click(self) -> None:
        """Post selected message on click."""
        self.post_message(_OptionRow.Selected(self.index))


class ChoiceBar(VerticalGroup, can_focus=True):
    """Inline picker: arrows and Enter, a hotkey per choice, Escape to cancel.

    Posts ``ChoiceBar.Chosen`` with the chosen ``choice_id`` (``cancel_id`` on Escape)
    and then removes itself.
    """

    ALLOW_SELECT = False

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    # DOM ids of the composed parts; PermissionBar keeps its own so its CSS is unchanged.
    _CONTENTS_ID: ClassVar[str] = "choice-contents"
    _TITLE_ID: ClassVar[str] = "choice-title"
    _OPTIONS_ID: ClassVar[str] = "choice-option-container"

    selection: reactive[int] = reactive(0, init=False)
    selected: var[bool] = var(False, toggle_class="-selected")
    blink: var[bool] = var(False)

    class Chosen(Message):
        """Posted when the user picks a choice, or cancels."""

        def __init__(self, bar_id: str, choice_id: str) -> None:
            super().__init__()
            self.bar_id = bar_id
            self.choice_id = choice_id

    def __init__(
        self,
        *,
        bar_id: str,
        title: str,
        choices: list[Choice],
        cancel_id: str,
        position: str = "",
        id: str | None = None,  # noqa: A002
    ) -> None:
        super().__init__(id=id)
        self._bar_id = bar_id
        self._title = title
        self._choices = choices
        self._cancel_id = cancel_id
        self._position = position
        self._blink_timer = None

    def compose(self):
        """Yield the title and the option rows."""
        display_title = f"({self._position}) {self._title}" if self._position else self._title
        with VerticalGroup(id=self._CONTENTS_ID):
            yield Label(display_title, id=self._TITLE_ID, markup=False)
        with VerticalGroup(id=self._OPTIONS_ID):
            for i, choice in enumerate(self._choices):
                row = _OptionRow(i, choice.label, choice.hotkey)
                if i == 0:
                    row.add_class("-active")
                yield row

    def on_mount(self) -> None:
        """Ring bell, grab focus, start blink timer."""
        self.app.bell()
        self.focus()

        def _toggle_blink() -> None:
            if self.has_focus:
                self.blink = not self.blink
            else:
                self.blink = False

        self._blink_timer = self.set_interval(0.5, _toggle_blink)

    def _reset_blink(self) -> None:
        self.blink = False
        if self._blink_timer is not None:
            self._blink_timer.reset()

    def watch_blink(self, blink: bool) -> None:
        """Toggle blink class on option container."""
        try:
            self.query_one(f"#{self._OPTIONS_ID}", VerticalGroup).set_class(blink, "-blink")
        except Exception:
            log.debug("Choice bar blink toggle failed", exc_info=True)

    def watch_selection(self, old: int, new: int) -> None:
        """Toggle -active class on option rows."""
        rows = list(self.query(_OptionRow))
        if 0 <= old < len(rows):
            rows[old].remove_class("-active")
        if 0 <= new < len(rows):
            rows[new].add_class("-active")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:  # noqa: ARG002
        """Guard against input after selection."""
        return not (self.selected and action in ("move_up", "move_down"))

    def on_key(self, event: events.Key) -> None:
        """Select the choice whose hotkey matches the typed character."""
        if self.selected or event.character is None:
            return
        for i, choice in enumerate(self._choices):
            if choice.hotkey == event.character:
                event.stop()
                event.prevent_default()
                self.selection = i
                self.action_select()
                return

    def action_move_up(self) -> None:
        """Move selection up."""
        self._reset_blink()
        self.selection = max(0, self.selection - 1)

    def action_move_down(self) -> None:
        """Move selection down."""
        self._reset_blink()
        self.selection = min(len(self._choices) - 1, self.selection + 1)

    def action_select(self) -> None:
        """Confirm the current selection."""
        self._reset_blink()
        if not self.selected:
            self._do_select(self.selection)

    def action_cancel(self) -> None:
        """Select the cancel choice, or resolve with cancel_id when no row carries it."""
        for i, choice in enumerate(self._choices):
            if choice.choice_id == self._cancel_id:
                self._do_select(i)
                return
        self._resolve(self._cancel_id)

    @on(_OptionRow.Selected)
    def _on_option_clicked(self, event: _OptionRow.Selected) -> None:
        """Handle click on an option row."""
        event.stop()
        self._reset_blink()
        if not self.selected:
            self.selection = event.index

    def _do_select(self, index: int) -> None:
        """Mark as selected and resolve after delay."""
        self.selected = True
        choice_id = self._choices[index].choice_id

        async def _delayed_resolve() -> None:
            await asyncio.sleep(0.4)
            self._resolve(choice_id)

        self.run_worker(_delayed_resolve(), exclusive=True)

    def _result_message(self, choice_id: str) -> Message:
        """The message posted when the bar resolves."""
        return ChoiceBar.Chosen(self._bar_id, choice_id)

    def _resolve(self, choice_id: str) -> None:
        """Post the result message and remove self."""
        self.post_message(self._result_message(choice_id))
        self.remove()
