"""PermissionBar — inline widget for resolving agent permission requests."""

from __future__ import annotations

from typing import ClassVar

from acp.schema import PermissionOption
from textual.binding import Binding
from textual.message import Message

from synth_acp.ui.widgets.choice_bar import Choice, ChoiceBar

_KIND_KEYS: dict[str, str] = {
    "allow_once": "a",
    "allow_always": "A",
    "reject_once": "r",
    "reject_always": "R",
}


def _choices(options: list[PermissionOption]) -> list[Choice]:
    """Map options to choices; the first option of each kind gets the kind's hotkey."""
    seen_kinds: set[str] = set()
    choices: list[Choice] = []
    for opt in options:
        key = None
        if opt.kind and opt.kind not in seen_kinds:
            key = _KIND_KEYS.get(opt.kind)
            seen_kinds.add(opt.kind)
        choices.append(Choice(opt.option_id, opt.name, key))
    return choices


class PermissionBar(ChoiceBar):
    """Inline permission picker mounted above the InputBar.

    Posts a `PermissionBar.Resolved` message with the selected option_id,
    or "" if cancelled.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("a", "select_kind('allow_once')", "Allow once", show=False, priority=True),
        Binding("A", "select_kind('allow_always')", "Allow always", show=False, priority=True),
        Binding("r", "select_kind('reject_once')", "Reject once", show=False, priority=True),
        Binding("R", "select_kind('reject_always')", "Reject always", show=False, priority=True),
    ]

    _CONTENTS_ID: ClassVar[str] = "perm-contents"
    _TITLE_ID: ClassVar[str] = "perm-title"
    _OPTIONS_ID: ClassVar[str] = "perm-option-container"

    class Resolved(Message):
        """Posted when the user selects a permission option."""

        def __init__(self, agent_id: str, request_id: str, option_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.request_id = request_id
            self.option_id = option_id

    def __init__(self, agent_id: str, request_id: str, title: str, options: list[PermissionOption], *, position: str = "") -> None:
        cancel_id = next((opt.option_id for opt in options if opt.kind == "reject_once"), "")
        super().__init__(
            bar_id=f"perm-{request_id}",
            title=title,
            choices=_choices(options),
            cancel_id=cancel_id,
            position=position,
            id=f"perm-{request_id}",
        )
        self._agent_id = agent_id
        self._request_id = request_id
        self._options = options

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Allow a kind hotkey only when an option of that kind exists."""
        if action == "select_kind":
            kinds = {opt.kind for opt in self._options if opt.kind is not None}
            return any(isinstance(p, str) and p in kinds for p in parameters)
        return super().check_action(action, parameters)

    def action_select_kind(self, kind: str) -> None:
        """Select the first option matching the given kind."""
        for i, opt in enumerate(self._options):
            if opt.kind == kind:
                self.selection = i
                self.action_select()
                return

    def _result_message(self, choice_id: str) -> Message:
        """Post the permission-specific Resolved message instead of Chosen."""
        return PermissionBar.Resolved(self._agent_id, self._request_id, choice_id)
