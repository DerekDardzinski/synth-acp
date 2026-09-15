"""Tests for ExpandableSection widget."""

from __future__ import annotations

import ast
import inspect
import pathlib
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Static

from synth_acp.ui.widgets.expandable_section import ExpandableSection, _ToggleLabel
from synth_acp.ui.widgets.gradient_bar import ActivityBar


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
            # Body first, header last: no ActivityBar is composed eagerly.
            assert children[0] == section.query_one(".es-body")
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


class TestSetActivity:
    """The lazy ActivityBar: mounted on activation, REMOVED on deactivation."""

    async def test_two_activations_settle_to_one_bar(self) -> None:
        """The slot is claimed before the unawaited mount.

        Silent failure: two mounts scheduled in one frame produce two bars and two 15Hz
        timers — the leak this phase removes, at reduced scale, with `_activity_bar`
        still looking correct.
        """
        section = ExpandableSection(id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_activity(True)
            section.set_activity(True)
            await pilot.pause()

            bars = section.query(ActivityBar)
            assert len(bars) == 1
            assert section._activity_bar is bars.first()

    async def test_mount_failure_rolls_back_slot(self) -> None:
        """A synchronous mount failure must not poison the slot.

        Silent failure: without rollback the non-None slot short-circuits every later
        activation, so the section never shows activity again for the rest of its life.
        """
        section = ExpandableSection(id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            with patch.object(
                ExpandableSection, "mount", side_effect=RuntimeError("boom")
            ):
                section.set_activity(True)  # must not raise
            assert section._activity_bar is None

            section.set_activity(True)
            await pilot.pause()
            assert len(section.query(ActivityBar)) == 1

    async def test_deactivate_removes_the_bar(self) -> None:
        """Deactivation reclaims the widgets rather than hiding them.

        Silent failure: leaving the bar mounted-but-hidden reproduces the exact timer
        leak this phase fixes while every `active`-flag assertion stays green.
        """
        section = ExpandableSection(id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_activity(True)
            await pilot.pause()
            assert len(section.query(ActivityBar)) == 1

            section.set_activity(False)
            assert section._activity_bar is None  # cleared in the same tick
            await pilot.pause()
            assert len(section.query(ActivityBar)) == 0

            section.set_activity(False)  # must not raise

    async def test_rapid_toggle_settles_to_one_new_bar(self) -> None:
        """True -> False -> True inside one frame settles to a single fresh bar.

        Silent failure: anchoring against the dying bar, or leaving the removed instance
        mounted with its timer running, produces a plausible single-bar DOM only some of
        the time.
        """
        section = ExpandableSection(id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_activity(True)
            first = section._activity_bar
            section.set_activity(False)
            section.set_activity(True)
            await pilot.pause()

            bars = section.query(ActivityBar)
            assert len(bars) == 1
            assert bars.first() is not first
            assert section._activity_bar is bars.first()

    async def test_bar_is_last_child_for_top_position(self) -> None:
        """With the header on top the bar mounts after the body.

        Silent failure: an index-based anchor puts the bar in the wrong slot, which reads
        as a cosmetic glitch rather than a bug.
        """
        section = ExpandableSection(Static("c"), toggle_position="top", id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_activity(True)
            await pilot.pause()

            children = list(section.children)
            assert children[0] is section.query_one(".es-header")
            assert children[-1] is section._activity_bar

    async def test_bar_is_first_child_for_bottom_position(self) -> None:
        """With the header at the bottom the bar mounts before the body."""
        section = ExpandableSection(Static("c"), toggle_position="bottom", id="sec")
        app = _TestApp(section)

        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            section.set_activity(True)
            await pilot.pause()

            children = list(section.children)
            assert children[0] is section._activity_bar
            assert children[-1] is section.query_one(".es-header")

    def test_mount_anchors_on_a_widget_not_an_index(self) -> None:
        """Placement must anchor on the body widget, never on an integer index.

        Silent failure: Textual marks a removed widget for pruning synchronously but
        takes it out of the NodeList later, so an index anchor drifts only during rapid
        toggling — the steady-state DOM tests above would still pass.
        """
        source = pathlib.Path(inspect.getfile(ExpandableSection)).read_text()
        tree = ast.parse(source)
        set_activity = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "set_activity"
        )
        mount_calls = [
            node
            for node in ast.walk(set_activity)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mount"
        ]

        assert len(mount_calls) == 2
        for call in mount_calls:
            anchors = [kw for kw in call.keywords if kw.arg in {"before", "after"}]
            assert len(anchors) == 1, ast.dump(call)
            assert isinstance(anchors[0].value, (ast.Name, ast.Attribute))
            assert not call.args[1:]
