"""Tests for ChoiceBar."""

from __future__ import annotations

from textual.app import App, ComposeResult

from synth_acp.ui.widgets.choice_bar import Choice, ChoiceBar

_CHOICES = [
    Choice("first", "First", "f"),
    Choice("second", "Second", "s"),
    Choice("cancel", "Cancel", "c"),
]


class _App(App):
    def __init__(self) -> None:
        super().__init__()
        self.chosen: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield ChoiceBar(bar_id="b", title="Pick one", choices=_CHOICES, cancel_id="cancel")

    def on_choice_bar_chosen(self, message: ChoiceBar.Chosen) -> None:
        self.chosen.append((message.bar_id, message.choice_id))


async def _press_and_settle(pilot, *keys: str) -> None:
    await pilot.pause()
    await pilot.press(*keys)
    # The bar resolves 0.4 s after the selection.
    await pilot.pause(0.6)


class TestChoiceBar:
    async def test_arrows_and_enter_post_highlighted_choice(self) -> None:
        app = _App()
        async with app.run_test(headless=True, size=(80, 20)) as pilot:
            await _press_and_settle(pilot, "down", "enter")
            assert app.chosen == [("b", "second")]

    async def test_hotkey_posts_its_choice(self) -> None:
        app = _App()
        async with app.run_test(headless=True, size=(80, 20)) as pilot:
            await _press_and_settle(pilot, "s")
            assert app.chosen == [("b", "second")]

    async def test_escape_posts_cancel_id_and_bar_removes_itself(self) -> None:
        app = _App()
        async with app.run_test(headless=True, size=(80, 20)) as pilot:
            await _press_and_settle(pilot, "down", "escape")
            assert app.chosen == [("b", "cancel")]
            assert list(app.query(ChoiceBar)) == []
