"""Tests for the read-only Terminal widget."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from synth_acp.terminal.manager import Command, TerminalProcess
from synth_acp.ui.widgets.terminal import Terminal


def _lines(terminal: Terminal) -> list[str]:
    buffer = terminal.state.scrollback_buffer
    return [str(line.content.plain).rstrip() for line in buffer.lines if line.content.plain.strip()]


class _TerminalApp(App[None]):
    def __init__(self, process: TerminalProcess) -> None:
        super().__init__()
        self._process = process

    def compose(self) -> ComposeResult:
        yield Terminal(self._process)


async def _settle(pilot: object, terminal: Terminal) -> None:
    for _ in range(50):
        await pilot.pause()  # type: ignore[attr-defined]
        if terminal.is_finalized:
            return


class TestTerminalMountAfterOutput:
    """The ToolCallBlock hosting a Terminal often mounts after the process produced output.

    Silent failure guarded: output written before mount went only to the process buffer,
    so the widget showed an empty region while the copy button had the full output.
    """

    async def test_process_that_exited_before_mount_shows_its_output(self) -> None:
        process = TerminalProcess(
            Command(command="printf 'hello\\nworld\\n'", args=[], env={}, cwd="/tmp")
        )
        await process.start()
        await process.wait_for_exit()

        app = _TerminalApp(process)
        async with app.run_test(headless=True, size=(80, 20)) as pilot:
            terminal = app.query_one(Terminal)
            await _settle(pilot, terminal)
            assert _lines(terminal) == ["hello", "world"]
            assert terminal.is_finalized

    async def test_output_before_and_after_mount_keeps_its_order(self) -> None:
        process = TerminalProcess(
            Command(
                command="printf 'before\\n'; sleep 0.5; printf 'after\\n'",
                args=[],
                env={},
                cwd="/tmp",
            )
        )
        await process.start()
        for _ in range(100):
            if process.get_output()[0]:
                break
            await asyncio.sleep(0.01)
        assert process.get_output()[0].strip() == "before"

        app = _TerminalApp(process)
        async with app.run_test(headless=True, size=(80, 20)) as pilot:
            terminal = app.query_one(Terminal)
            await process.wait_for_exit()
            await _settle(pilot, terminal)
            assert _lines(terminal) == ["before", "after"]
