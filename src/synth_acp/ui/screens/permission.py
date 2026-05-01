"""PermissionBar — inline widget for resolving agent permission requests."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from acp.schema import PermissionOption
from textual import on
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup
from textual.content import Content
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import Label

log = logging.getLogger(__name__)

_KIND_KEYS: dict[str, str] = {
    "allow_once": "a",
    "allow_always": "A",
    "reject_once": "r",
    "reject_always": "R",
}


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


class PermissionBar(VerticalGroup, can_focus=True):
    """Inline permission picker mounted above the InputBar.

    Posts a `PermissionBar.Resolved` message with the selected option_id,
    or "" if cancelled.
    """

    ALLOW_SELECT = False

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("a", "select_kind('allow_once')", "Allow once", show=False, priority=True),
        Binding("A", "select_kind('allow_always')", "Allow always", show=False, priority=True),
        Binding("r", "select_kind('reject_once')", "Reject once", show=False, priority=True),
        Binding("R", "select_kind('reject_always')", "Reject always", show=False, priority=True),
    ]

    selection: reactive[int] = reactive(0, init=False)
    selected: var[bool] = var(False, toggle_class="-selected")
    blink: var[bool] = var(False)

    class Resolved(Message):
        """Posted when the user selects a permission option."""

        def __init__(self, agent_id: str, request_id: str, option_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.request_id = request_id
            self.option_id = option_id

    def __init__(self, agent_id: str, request_id: str, title: str, options: list[PermissionOption], *, position: str = "") -> None:
        super().__init__(id=f"perm-{request_id}")
        self._agent_id = agent_id
        self._request_id = request_id
        self._title = title
        self._position = position
        self._options = options
        self._blink_timer = None

    def compose(self):
        """Yield the permission bar layout."""
        display_title = f"({self._position}) {self._title}" if self._position else self._title
        with VerticalGroup(id="perm-contents"):
            yield Label(display_title, id="perm-title", markup=False)
        seen_kinds: set[str] = set()
        with VerticalGroup(id="perm-option-container"):
            for i, opt in enumerate(self._options):
                key = None
                if opt.kind and opt.kind not in seen_kinds:
                    key = _KIND_KEYS.get(opt.kind)
                    seen_kinds.add(opt.kind)
                row = _OptionRow(i, opt.name, key)
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
            self.query_one("#perm-option-container", VerticalGroup).set_class(blink, "-blink")
        except Exception:
            log.debug("Permission blink toggle failed", exc_info=True)

    def watch_selection(self, old: int, new: int) -> None:
        """Toggle -active class on option rows."""
        rows = list(self.query(_OptionRow))
        if 0 <= old < len(rows):
            rows[old].remove_class("-active")
        if 0 <= new < len(rows):
            rows[new].add_class("-active")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Guard against input after selection."""
        if self.selected and action in ("move_up", "move_down"):
            return False
        if action == "select_kind":
            kinds = {opt.kind for opt in self._options if opt.kind is not None}
            return any(isinstance(p, str) and p in kinds for p in parameters)
        return True

    def action_move_up(self) -> None:
        """Move selection up."""
        self._reset_blink()
        self.selection = max(0, self.selection - 1)

    def action_move_down(self) -> None:
        """Move selection down."""
        self._reset_blink()
        self.selection = min(len(self._options) - 1, self.selection + 1)

    def action_select(self) -> None:
        """Confirm the current selection."""
        self._reset_blink()
        if not self.selected:
            self._do_select(self.selection)

    def action_cancel(self) -> None:
        """Resolve with the first reject_once option, or empty string."""
        for i, opt in enumerate(self._options):
            if opt.kind == "reject_once":
                self._do_select(i)
                return
        self._resolve("")

    def action_select_kind(self, kind: str) -> None:
        """Select the first option matching the given kind."""
        for i, opt in enumerate(self._options):
            if opt.kind == kind:
                self.selection = i
                self.action_select()
                return

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
        option_id = self._options[index].option_id

        async def _delayed_resolve() -> None:
            await asyncio.sleep(0.4)
            self._resolve(option_id)

        self.run_worker(_delayed_resolve(), exclusive=True)

    def _resolve(self, option_id: str) -> None:
        """Post the resolved message and remove self."""
        self.post_message(PermissionBar.Resolved(self._agent_id, self._request_id, option_id))
        self.remove()
