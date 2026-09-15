"""Tests for GradientBar timer pause/resume on visibility and UsageBar."""

from __future__ import annotations

from unittest.mock import Mock

from textual.app import App, ComposeResult
from textual.color import Color, Gradient
from textual.style import Style
from textual.visual import RenderOptions

from synth_acp.ui.widgets.gradient_bar import (
    ActivityBar,
    GradientBar,
    GradientBarVisual,
    UsageBar,
    UsageBarVisual,
)


def _render_options() -> RenderOptions:
    """Create a minimal RenderOptions for testing (unused by UsageBarVisual)."""
    return RenderOptions(get_style=Mock(), rules={})


class _TestApp(App):
    def compose(self) -> ComposeResult:
        yield ActivityBar()


class _InactiveApp(App):
    """An ActivityBar that is inactive from its very first layout.

    Its GradientBar is therefore ``display: none`` before the compositor has ever
    mapped it, which is exactly the case Textual never posts ``events.Hide`` to.
    """

    def compose(self) -> ComposeResult:
        bar = ActivityBar()
        bar.active = False
        yield bar


class TestGradientBarArming:
    async def test_displayed_bar_still_animates(self) -> None:
        """Gating arming on display must not leave a visible bar static.

        Silent failure: every gradient in the app freezes while all functional
        assertions pass — nothing else in the suite renders two frames.
        """
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await pilot.pause()
            gradient = app.query_one(ActivityBar).query_one(GradientBar)
            assert gradient.auto_refresh == 1 / 15

        # The visual always emits the same "━" character and animates by moving the
        # gradient offset, so comparing segment TEXT across ticks would fail a correct
        # implementation. The styles are what move.
        clock = {"t": 0.0}
        visual = GradientBarVisual(
            Gradient.from_colors("#ff0000", "#00ff00", "#0000ff"),
            get_time=lambda: clock["t"],
        )
        first = visual.render_strips(20, 1, Style(), _render_options())[0]
        clock["t"] = 0.5
        second = visual.render_strips(20, 1, Style(), _render_options())[0]

        assert "".join(s.text for s in first._segments) == "".join(
            s.text for s in second._segments
        )
        assert [s.style for s in first._segments] != [s.style for s in second._segments]

    async def test_bar_hidden_from_first_layout_never_arms(self) -> None:
        """A bar that is display:none from its first layout must hold no timer.

        Silent failure: this is the original defect. Arming in on_mount and relying on
        a Hide event Textual never posts for such a widget leaked 500 of 521 timers at
        21 agents, and every tick forced a full compositor map rebuild.
        """
        app = _InactiveApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await pilot.pause()
            gradient = app.query_one(ActivityBar).query_one(GradientBar)
            assert gradient.display is False
            assert gradient.auto_refresh is None
            assert gradient._auto_refresh_timer is None

    async def test_hide_clears_and_reshow_restores_the_timer(self) -> None:
        """A bar that WAS mapped and is then hidden must stop, and resume on re-show.

        Silent failure: routing both events through one assignment point could drop the
        transition handling that stops a bar the user has toggled off, leaving a hidden
        bar ticking — the leak this phase removes, arriving by a different route.
        """
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            bar = app.query_one(ActivityBar)
            gradient = bar.query_one(GradientBar)

            bar.active = False
            await pilot.pause()
            assert gradient.auto_refresh is None

            bar.active = True
            await pilot.pause()
            assert gradient.auto_refresh == 1 / 15


class TestUsageBarVisual:
    def test_renders_bar_with_label(self) -> None:
        """Filled/empty chars are proportional and label is right-aligned."""
        fill_color = Color.parse("#00ff00")
        empty_color = Color.parse("#333333")
        label_color = Color.parse("#ffffff")
        visual = UsageBarVisual(
            used=50,
            size=100,
            cost_text="$1.23",
            fill_color=fill_color,
            empty_color=empty_color,
            label_color=label_color,
        )
        strips = visual.render_strips(40, 1, Style(), _render_options())
        assert len(strips) == 1
        strip = strips[0]
        assert strip.cell_length == 40

        # Reconstruct text from segments
        text = "".join(seg.text for seg in strip._segments)
        # Label "50% $1.23" is 9 chars + 1 space = 10, so bar_width = 30
        assert "50% $1.23" in text
        # Bar should have fill chars (50% of 30 = 15) and empty chars (15)
        bar_text = text[:30]
        assert bar_text.count("━") == 15
        assert bar_text.count("─") == 15

    def test_size_zero_shows_cost_only(self) -> None:
        """When size=0, only cost_text is shown as the label with no bar."""
        fill_color = Color.parse("#00ff00")
        empty_color = Color.parse("#333333")
        label_color = Color.parse("#ffffff")
        visual = UsageBarVisual(
            used=0,
            size=0,
            cost_text="$2.50",
            fill_color=fill_color,
            empty_color=empty_color,
            label_color=label_color,
        )
        strips = visual.render_strips(40, 1, Style(), _render_options())
        strip = strips[0]
        text = "".join(seg.text for seg in strip._segments)
        # Label is "$2.50", no percentage
        assert "$2.50" in text
        assert "%" not in text
        # No fill chars since pct=0 and size=0
        assert "━" not in text

    def test_no_data_empty(self) -> None:
        """No data (used=0, size=0, cost_text='') renders empty strip."""
        fill_color = Color.parse("#00ff00")
        empty_color = Color.parse("#333333")
        label_color = Color.parse("#ffffff")
        visual = UsageBarVisual(
            used=0,
            size=0,
            cost_text="",
            fill_color=fill_color,
            empty_color=empty_color,
            label_color=label_color,
        )
        strips = visual.render_strips(40, 1, Style(), _render_options())
        strip = strips[0]
        assert list(strip._segments) == []


class TestUsageBar:
    async def test_pick_color_thresholds(self) -> None:
        """Color thresholds return correct theme colors at boundaries."""
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            usage_bar = app.query_one(UsageBar)
            await pilot.pause()

            theme = app.current_theme
            success = Color.parse(theme.success)
            warning = Color.parse(theme.warning)
            error = Color.parse(theme.error)

            assert usage_bar._pick_color(0.5) == success
            assert usage_bar._pick_color(0.69) == success
            assert usage_bar._pick_color(0.7) == warning
            assert usage_bar._pick_color(0.89) == warning
            assert usage_bar._pick_color(0.9) == error
            assert usage_bar._pick_color(1.0) == error

    async def test_update_triggers_rebuild(self) -> None:
        """update() stores values and creates a visual with matching data."""
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            usage_bar = app.query_one(UsageBar)
            await pilot.pause()

            usage_bar.update(500, 1000, "$1.23")
            assert usage_bar._used == 500
            assert usage_bar._context_size == 1000
            assert usage_bar._cost_text == "$1.23"
            assert isinstance(usage_bar._visual, UsageBarVisual)
            assert usage_bar._visual.used == 500
            assert usage_bar._visual.size == 1000


class TestActivityBarIntegration:
    async def test_activity_bar_compose_is_unchanged(self) -> None:
        """The shared ActivityBar must keep BOTH children.

        Silent failure: dropping the UsageBar blanks the static context/cost readout for
        every agent tile and input bar, which no ExpandableSection test would notice.
        """
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            bar = app.query_one(ActivityBar)
            await pilot.pause()
            assert len(bar.query(GradientBar)) == 1
            usage_bars = bar.query(UsageBar)
            assert len(usage_bars) == 1
            assert usage_bars.first().has_class("activity-bar-bg")

    async def test_composes_usage_bar(self) -> None:
        """ActivityBar contains UsageBar child, not Static."""
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            bar = app.query_one(ActivityBar)
            await pilot.pause()
            assert bar.query_one(UsageBar) is not None
            # No Static widget should be present
            from textual.widgets import Static

            assert len(bar.query(Static)) == 0

    async def test_update_usage_delegates_to_usage_bar(self) -> None:
        """ActivityBar.update_usage passes through to UsageBar.update."""
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            bar = app.query_one(ActivityBar)
            await pilot.pause()

            bar.update_usage(50, 100, "$1.23")
            usage_bar = bar.query_one(UsageBar)
            assert usage_bar._used == 50
            assert usage_bar._context_size == 100
            assert usage_bar._cost_text == "$1.23"
