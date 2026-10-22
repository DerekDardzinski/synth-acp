"""Expandable section with a toggle line, an optional preview line, and bounded content."""

from __future__ import annotations

import logging
from typing import Literal

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from synth_acp.ui.widgets.gradient_bar import ActivityBar

log = logging.getLogger(__name__)


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


class _Header(Vertical):
    """Header: the toggle on its own line, then the preview line when it has text."""


class ExpandableSection(Vertical, can_focus=False):
    """Collapsible section with a header and a bounded scrollable content region.

    The header is the toggle on its own line (``▶ Expand``, or ``▶ <label>`` when a
    label is given), then a muted preview line, shown only while it has text. An
    activity bar below the header animates while active.
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
        height: auto;
        color: $text-muted;
    }
    ExpandableSection #es-preview.-empty {
        display: none;
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
        label: str | None = None,
        toggle_position: Literal["top", "bottom"] = "top",
        max_content_height: int = 20,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Args:
        *children: Initial content widgets mounted into the scrollable body.
        start_expanded: If True, body is visible on mount. Default collapsed.
        label: Toggle text after the arrow. None shows "Expand" / "Collapse".
        toggle_position: Header at "top" (expand downward) or "bottom" (expand upward).
        max_content_height: Max height in lines for the scrollable body.
        """
        super().__init__(id=id, classes=classes)
        self._content_children = children
        self._toggle_position = toggle_position
        self._max_content_height = max_content_height
        self._label = label
        self._activity_bar: ActivityBar | None = None
        self.collapsed = not start_expanded

    def compose(self) -> ComposeResult:
        """Compose header and scrollable body.

        The ActivityBar is NOT composed here — ``set_activity`` mounts one on demand and
        removes it again, so an idle section holds no gradient timer.
        """
        header = _Header(
            _ToggleLabel(self._toggle_text(collapsed=self.collapsed), id="es-toggle"),
            Static("", id="es-preview", classes="-empty"),
            classes="es-header",
        )
        body = VerticalScroll(*self._content_children, classes="es-body")
        if self._max_content_height != 20:
            body.styles.max_height = self._max_content_height
        if self._toggle_position == "top":
            yield header
            yield body
        else:
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

    def _toggle_text(self, *, collapsed: bool) -> str:
        arrow = "▶" if collapsed else "▼"
        if self._label is not None:
            return f"{arrow} {self._label}"
        return f"{arrow} {'Expand' if collapsed else 'Collapse'}"

    def _set_toggle_label(self, *, collapsed: bool) -> None:
        """Update the toggle label text."""
        self.query_one("#es-toggle", _ToggleLabel).update(self._toggle_text(collapsed=collapsed))

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
        """Update the preview line under the toggle; empty text hides the line."""
        preview = self.query_one("#es-preview", Static)
        preview.update(text)
        preview.set_class(not text, "-empty")

    def set_activity(self, active: bool) -> None:
        """Show or hide the streaming activity indicator. SYNCHRONOUS.

        On True an ActivityBar is mounted; on False the mounted one is removed. Removal,
        not hiding, reclaims the widgets and guarantees no orphaned animation timer.

        ``_activity_bar`` is the authoritative state and is assigned BEFORE the unawaited
        mount and cleared BEFORE the unawaited remove, so two activations in one frame
        cannot schedule duplicate bars. The DOM settles a frame later.

        Idempotent in both directions. A mount failure resets the slot and is logged
        rather than propagated: the callers are App message-pump paths with no local guard.

        Args:
            active: True to show the animated gradient bar, False to remove it.
        """
        if active:
            if self._activity_bar is not None:
                return
            bar = ActivityBar(classes="es-activity")
            self._activity_bar = bar
            try:
                # Anchored on the body widget, never on an index and never on an
                # ActivityBar: Textual marks a removed widget for pruning synchronously
                # but takes it out of the NodeList later, so a rapid True->False->True
                # could otherwise anchor against a dying bar.
                body = self.query_one(".es-body", VerticalScroll)
                if self._toggle_position == "top":
                    self.mount(bar, after=body)
                else:
                    self.mount(bar, before=body)
            except Exception:
                self._activity_bar = None
                log.warning("Failed to mount activity bar", exc_info=True)
        else:
            bar = self._activity_bar
            if bar is None:
                return
            self._activity_bar = None
            bar.remove()

    def toggle(self) -> None:
        """Flip collapsed state programmatically."""
        self.collapsed = not self.collapsed

    def on__toggle_label_toggle(self) -> None:
        """Handle toggle label click."""
        self.toggle()
