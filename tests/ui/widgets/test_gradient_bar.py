"""Tests for GradientBar timer pause/resume on visibility."""

from __future__ import annotations

from textual.app import App, ComposeResult

from synth_acp.ui.widgets.gradient_bar import ActivityBar


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
