"""Expandable section with single-line header and bounded scrollable content."""

from __future__ import annotations

from typing import Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from synth_acp.ui.widgets.gradient_bar import ActivityBar


class _ToggleLabel(Static, can_focus=False):
    """Clickable toggle label that posts a Toggle message on click."""

    class Toggle(Message):
        """Posted when the toggle is clicked."""

    DEFAULT_CSS = """
    _ToggleLabel {
        width: auto;
        height: 1;
        padding: 0 1 0 0;
        color: $text-muted;
    }
    _ToggleLabel:hover {
        color: $foreground;
    }
    """

    def on_click(self) -> None:
        self.post_message(self.Toggle())


class _Header(Horizontal):
    """Single-line header row: [toggle label] [preview...]."""


class ExpandableSection(Vertical, can_focus=False):
    """Collapsible section with single-line header (toggle label + preview)
    and a bounded scrollable content region. An activity bar below the header
    animates while active.
    """

    DEFAULT_CSS = """
    ExpandableSection {
        height: auto;
    }
    ExpandableSection .es-header {
        height: auto;
    }
    ExpandableSection #es-preview {
        width: 1fr;
        color: $text-muted;
    }
    ExpandableSection .es-activity {
        height: 1;
    }
    ExpandableSection .es-activity GradientBar {
        height: 1;
    }
    ExpandableSection .es-activity .activity-bar-bg {
        height: 1;
    }
    ExpandableSection .es-body {
        height: auto;
        max-height: 20;
        scrollbar-size-vertical: 1;
        overflow-x: hidden;
    }
    ExpandableSection .es-body.-collapsed {
        display: none;
    }
    """

    class Toggled(Message):
        """Posted when collapsed state changes."""

        def __init__(self, expandable_section: ExpandableSection, collapsed: bool) -> None:
            self.expandable_section = expandable_section
            self.collapsed = collapsed
            super().__init__()

    collapsed: reactive[bool] = reactive(True)

    def __init__(
        self,
        *children: Widget,
        start_expanded: bool = False,
        toggle_position: Literal["top", "bottom"] = "top",
        max_content_height: int = 20,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Args:
            *children: Initial content widgets mounted into the scrollable body.
            start_expanded: If True, body is visible on mount. Default collapsed.
            toggle_position: Header at "top" (expand downward) or "bottom" (expand upward).
            max_content_height: Max height in lines for the scrollable body.
        """
        super().__init__(id=id, classes=classes)
        self._content_children = children
        self._toggle_position = toggle_position
        self._max_content_height = max_content_height
        self.collapsed = not start_expanded

    def compose(self) -> ComposeResult:
        """Compose header, activity bar, and scrollable body."""
        header = _Header(
            _ToggleLabel("▶ Expand", id="es-toggle"),
            Static("", id="es-preview"),
            classes="es-header",
        )
        activity = ActivityBar(classes="es-activity")
        activity.active = False
        body = VerticalScroll(*self._content_children, classes="es-body")
        if self._max_content_height != 20:
            body.styles.max_height = self._max_content_height
        if self._toggle_position == "top":
            yield header
            yield body
            yield activity
        else:
            yield activity
            yield body
            yield header

    def on_mount(self) -> None:
        """Apply initial collapsed state."""
        body = self.query_one(".es-body", VerticalScroll)
        if self.collapsed:
            body.add_class("-collapsed")
            self._set_toggle_label(collapsed=True)
        else:
            body.remove_class("-collapsed")
            self._set_toggle_label(collapsed=False)

    def _set_toggle_label(self, *, collapsed: bool) -> None:
        """Update the toggle label text."""
        self.query_one("#es-toggle", _ToggleLabel).update(
            "▶ Expand" if collapsed else "▼ Collapse"
        )

    def watch_collapsed(self, collapsed: bool) -> None:
        """React to collapsed state changes."""
        try:
            body = self.query_one(".es-body", VerticalScroll)
        except Exception:
            return
        if collapsed:
            body.add_class("-collapsed")
        else:
            body.remove_class("-collapsed")
        try:
            self._set_toggle_label(collapsed=collapsed)
        except Exception:
            pass
        self.post_message(self.Toggled(self, collapsed))

    @property
    def content(self) -> VerticalScroll:
        """The scrollable content container. Use for dynamic mounting:
        await section.content.mount(widget)
        """
        return self.query_one(".es-body", VerticalScroll)

    def set_preview(self, text: str) -> None:
        """Update the preview text in the header. Truncates to fit one line."""
        self.query_one("#es-preview", Static).update(text)

    def set_activity(self, active: bool) -> None:
        """Set activity state: active=True shows animated gradient bar, False hides it."""
        self.query_one(".es-activity", ActivityBar).active = active

    def toggle(self) -> None:
        """Flip collapsed state programmatically."""
        self.collapsed = not self.collapsed

    def on__toggle_label_toggle(self) -> None:
        """Handle toggle label click."""
        self.toggle()
