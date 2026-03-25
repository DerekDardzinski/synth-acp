"""Inline permission request widget with action buttons."""

from __future__ import annotations

from acp.schema import PermissionOption
from textual.containers import Vertical
from textual.widgets import Button, Static

from synth_acp.models.commands import RespondPermission

_ALLOW_KINDS = {"allow_once", "allow_always"}


class PermissionRequest(Vertical):
    """Yellow-bordered permission box with title, kind, and option buttons.

    Args:
        agent_id: Agent requesting permission.
        request_id: Unique permission request identifier.
        title: Human-readable permission title.
        kind: Permission kind description.
        options: List of ACP PermissionOption objects.
    """

    def __init__(
        self,
        agent_id: str,
        request_id: str,
        title: str,
        kind: str,
        options: list[PermissionOption],
    ) -> None:
        super().__init__(id=f"perm-{request_id}", classes="permission-box")
        self._agent_id = agent_id
        self._request_id = request_id
        self._title = title
        self._kind = kind
        self._options = options

    def compose(self):
        """Yield title label and option buttons."""
        yield Static(f"[bold yellow]⚠  Permission required[/bold yellow]\n[dim]{self._title}[/dim]")
        for opt in self._options:
            variant = "success" if opt.kind in _ALLOW_KINDS else "error"
            yield Button(opt.name, variant=variant, id=f"perm-btn-{opt.option_id}")

    def on_mount(self) -> None:
        """Ring terminal bell on mount."""
        self.app.bell()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Resolve the permission and remove this widget.

        Args:
            event: The button press event.
        """
        option_id = event.button.id
        if option_id and option_id.startswith("perm-btn-"):
            option_id = option_id[len("perm-btn-") :]

        from synth_acp.ui.app import SynthApp

        app = self.app
        assert isinstance(app, SynthApp)
        app.run_worker(
            app.broker.handle(RespondPermission(agent_id=self._agent_id, option_id=option_id or ""))
        )
        self.remove()
