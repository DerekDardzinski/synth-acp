"""Tests for ExpandableSection widget."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Static

from synth_acp.ui.widgets.expandable_section import ExpandableSection, _ToggleLabel


class _TestApp(App):
    """Minimal app for testing ExpandableSection in a live widget tree."""

    def __init__(self, section: ExpandableSection) -> None:
        super().__init__()
        self._section = section

    def compose(self) -> ComposeResult:
        yield self._section


class TestExpandableSection:
    async def test_toggle_flips_state_label_and_visibility(self) -> None:
        """Toggle flips collapsed, changes button label, toggles body CSS class, posts Toggled."""
        section = ExpandableSection(Static("content"), id="sec")
        app = _TestApp(section)
        messages: list[ExpandableSection.Toggled] = []

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            # Starts collapsed
            assert section.collapsed is True
            body = section.query_one(".es-body")
            toggle = section.query_one("#es-toggle", _ToggleLabel)
            assert "-collapsed" in body.classes
            assert "Expand" in str(toggle.content)

            # Capture Toggled messages
            original_post = section.post_message

            def _capture(msg: Message) -> bool:
                if isinstance(msg, ExpandableSection.Toggled):
                    messages.append(msg)
                return original_post(msg)

            section.post_message = _capture  # type: ignore[assignment]

            # Toggle to expanded
            section.toggle()
            await pilot.pause()
            assert section.collapsed is False
            assert "-collapsed" not in body.classes
            assert "Collapse" in str(toggle.content)
            assert len(messages) == 1
            assert messages[0].collapsed is False
            assert messages[0].expandable_section is section

            # Toggle back to collapsed
            section.toggle()
            await pilot.pause()
            assert section.collapsed is True
            assert "-collapsed" in body.classes
            assert "Expand" in str(toggle.content)
            assert len(messages) == 2
            assert messages[1].collapsed is True

    async def test_set_preview_updates_text(self) -> None:
        """set_preview updates the #es-preview Static content."""
        section = ExpandableSection(id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_preview("Loading files...")
            await pilot.pause()
            preview = section.query_one("#es-preview", Static)
            assert "Loading files..." in str(preview.content)

    async def test_set_activity_updates_indicator(self) -> None:
        """set_activity toggles the ActivityBar between active and inactive."""
        from synth_acp.ui.widgets.gradient_bar import ActivityBar

        section = ExpandableSection(id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_activity(True)
            await pilot.pause()
            bar = section.query_one(".es-activity", ActivityBar)
            assert bar.active is True

            section.set_activity(False)
            await pilot.pause()
            assert bar.active is False

    async def test_start_expanded_shows_body(self) -> None:
        """start_expanded=True means body is visible (no -collapsed class)."""
        section = ExpandableSection(Static("visible"), start_expanded=True, id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)):
            assert section.collapsed is False
            body = section.query_one(".es-body")
            assert "-collapsed" not in body.classes

    async def test_toggle_position_bottom_puts_header_below(self) -> None:
        """toggle_position='bottom' places header after body."""
        section = ExpandableSection(
            Static("content"), toggle_position="bottom", id="sec"
        )
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)):
            children = list(section.children)
            # Activity first, body second, header last
            assert children[1] == section.query_one(".es-body")
            assert children[-1] == section.query_one(".es-header")

    async def test_dynamic_mount_into_content(self) -> None:
        """await section.content.mount(widget) places child inside the VerticalScroll body."""
        section = ExpandableSection(start_expanded=True, id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            new_widget = Static("dynamically added", id="dynamic-child")
            await section.content.mount(new_widget)
            await pilot.pause()
            # Verify the widget is in the DOM inside the body
            found = section.query_one("#dynamic-child", Static)
            assert found is new_widget
            assert found.parent == section.content
