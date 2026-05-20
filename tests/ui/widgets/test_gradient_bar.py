"""Tests for GradientBar timer pause/resume on visibility and UsageBar."""

from __future__ import annotations

from unittest.mock import Mock

from textual.app import App, ComposeResult
from textual.color import Color
from textual.style import Style
from textual.visual import RenderOptions

from synth_acp.ui.widgets.gradient_bar import (
    ActivityBar,
    UsageBar,
    UsageBarVisual,
)


def _render_options() -> RenderOptions:
    """Create a minimal RenderOptions for testing (unused by UsageBarVisual)."""
    return RenderOptions(get_style=Mock(), rules={})


class _TestApp(App):
    def compose(self) -> ComposeResult:
        yield ActivityBar()


class TestGradientBarVisibility:
    async def test_on_hide_pauses_auto_refresh(self) -> None:
        """Hidden GradientBar stops firing timer events."""
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            bar = app.query_one(ActivityBar)
            bar.active = False
            await pilot.pause()
            gradient = bar.query_one("GradientBar")
            assert gradient.auto_refresh is None

    async def test_on_show_resumes_auto_refresh(self) -> None:
        """Re-shown GradientBar resumes animation timer."""
        app = _TestApp()
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            bar = app.query_one(ActivityBar)
            bar.active = False
            await pilot.pause()
            bar.active = True
            await pilot.pause()
            gradient = bar.query_one("GradientBar")
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
