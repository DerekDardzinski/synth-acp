"""Tests for ConfigOptionsScreen."""

from __future__ import annotations

from acp.schema import (
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
)
from textual.app import App
from textual.widgets import Button, Select, Switch

from synth_acp.ui.screens.config_options import ConfigOptionsScreen


def _options() -> list:
    return [
        SessionConfigOptionSelect(
            id="autopilot",
            name="Autopilot",
            type="select",
            current_value="on",
            options=[
                SessionConfigSelectOption(value="on", name="Autopilot"),
                SessionConfigSelectOption(value="off", name="Supervised"),
            ],
        ),
        SessionConfigOptionSelect(
            id="memoryReflection",
            name="Memory reflection",
            type="select",
            current_value="on",
            options=[
                SessionConfigSelectOption(value="on", name="On"),
                SessionConfigSelectOption(value="off", name="Off"),
            ],
        ),
        SessionConfigOptionBoolean(id="ns:verbose", name="Verbose", type="boolean", current_value=False),
    ]


class TestConfigOptionsScreen:
    async def test_apply_returns_only_changed_options_including_booleans(self) -> None:
        """Silent failures guarded: sending unchanged options (a set per option, each
        resetting agent state) and dropping boolean options. The id ``ns:verbose`` is not
        a valid widget id, so it also checks agent-chosen ids are escaped and mapped back."""
        results: list = []
        app = App()
        async with app.run_test(headless=True, size=(100, 40)) as pilot:
            await app.push_screen(ConfigOptionsScreen("lab-lead", _options()), results.append)
            await pilot.pause()
            screen = app.screen
            screen.query(Select).first().value = "off"
            screen.query_one(Switch).value = True
            await pilot.pause()
            screen.query_one("#config-options-apply", Button).press()
            await pilot.pause()

        assert results == [{"autopilot": "off", "ns:verbose": True}]
