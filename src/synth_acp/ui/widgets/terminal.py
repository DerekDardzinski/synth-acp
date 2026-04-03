"""Read-only terminal widget for displaying PTY output."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import events
from textual.cache import LRUCache
from textual.geometry import Region, Size
from textual.message import Message
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.style import Style

from synth_acp.ui import ansi

if TYPE_CHECKING:
    from synth_acp.terminal.manager import TerminalProcess


class Terminal(ScrollView, can_focus=False):
    """Read-only terminal widget rendering ANSI output from a TerminalProcess.

    Adapted from Toad's Terminal widget with all input handlers stripped.

    Args:
        process: The TerminalProcess whose output to display.
        name: Widget name.
        id: Widget ID.
        classes: CSS classes.
        disabled: Whether the widget is disabled.
    """

    CURSOR_STYLE = Style.parse("reverse")

    hide_cursor = reactive(False)

    @dataclass
    class Finalized(Message):
        """Terminal was finalized."""

        terminal: Terminal

        @property
        def control(self) -> Terminal:
            return self.terminal

    @dataclass
    class AlternateScreenChanged(Message):
        """Terminal enabled or disabled alternate screen."""

        terminal: Terminal
        enabled: bool

        @property
        def control(self) -> Terminal:
            return self.terminal

    def __init__(
        self,
        process: TerminalProcess,
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.set_reactive(Terminal.auto_links, False)
        self._process = process
        self._state = ansi.TerminalState(self._noop_stdin)
        self._width = 80
        self._height = 24
        self._finalized = False
        self._alternate_screen = False
        self._terminal_render_cache: LRUCache[tuple, Strip] = LRUCache(1024)
        self._write_to_stdin: Callable[[str], Awaitable] | None = None

    @staticmethod
    async def _noop_stdin(text: str) -> None:
        """No-op stdin callback for read-only terminal."""

    @property
    def state(self) -> ansi.TerminalState:
        """The terminal state machine."""
        return self._state

    @property
    def is_finalized(self) -> bool:
        """Finalized terminals will not accept writes or receive input."""
        return self._finalized

    @property
    def width(self) -> int:
        """Width of the terminal."""
        return self._width

    @property
    def height(self) -> int:
        """Height of the terminal."""
        return self._height

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    @property
    def alternate_screen(self) -> bool:
        return self._alternate_screen

    def notify_style_update(self) -> None:
        """Clear cache when theme changes."""
        self._terminal_render_cache.clear()
        super().notify_style_update()

    def set_state(self, state: ansi.TerminalState) -> None:
        """Set the terminal state, if this terminal is to inherit an existing state.

        Args:
            state: Terminal state object.
        """
        self._state = state

    def set_write_to_stdin(self, write_to_stdin: Callable[[str], Awaitable]) -> None:
        """Set a callable which is invoked with input, to be sent to stdin.

        Args:
            write_to_stdin: Callable which takes a string.
        """
        self._write_to_stdin = write_to_stdin

    def finalize(self) -> None:
        """Finalize the terminal.

        The finalized terminal will reject new writes.
        Adds the TCSS class `-finalized`.
        """
        if not self._finalized:
            self._finalized = True
            self._state.show_cursor = False
            self.add_class("-finalized")
            self._terminal_render_cache.clear()
            self.refresh()
            self.post_message(self.Finalized(self))
            if not self._state.buffer.height:
                self.display = False

    def _on_resize(self, event: events.Resize) -> None:  # noqa: ARG002
        width, height = self.scrollable_content_region.size
        self.update_size(width, height)
        self._process.resize_pty(width, height)

    def update_size(self, width: int, height: int) -> None:
        """Update terminal dimensions.

        Args:
            width: New width in columns.
            height: New height in rows.
        """
        self._terminal_render_cache.grow(height * 2)
        self._width = width or 80
        self._height = height or 24
        self._state.update_size(self._width, height)
        self._terminal_render_cache.clear()
        self.refresh()

    def on_mount(self) -> None:
        """Wire up process callbacks and set initial size."""
        self.anchor()
        width, height = self.scrollable_content_region.size
        self.update_size(width, height)
        self._process.on_output = self._handle_output
        self._process.on_exit = self._handle_exit
        self._process.resize_pty(width, height)

    async def _handle_output(self, text: str) -> None:
        """Async callback for TerminalProcess output.

        Args:
            text: Decoded text from the PTY.
        """
        await self.write(text)

    def _handle_exit(self, return_code: int | None) -> None:
        """Sync callback for TerminalProcess exit.

        Args:
            return_code: Process exit code.
        """
        self.call_later(self._do_finalize, return_code)

    def _do_finalize(self, return_code: int | None) -> None:
        """Finalize and apply exit status CSS class.

        Args:
            return_code: Process exit code.
        """
        self.finalize()
        if return_code == 0:
            self.add_class("-success")
        else:
            self.add_class("-error")

    async def write(self, text: str, hide_output: bool = False) -> bool:
        """Write sequences to the terminal.

        Args:
            text: Text with ANSI escape sequences.
            hide_output: Do not update the buffers with visible text.

        Returns:
            True if the state visuals changed, False if no visual change.
        """
        scrollback_delta, alternate_delta = await self._state.write(
            text, hide_output=hide_output
        )
        self._update_from_state(scrollback_delta, alternate_delta)
        scrollback_changed = bool(scrollback_delta is None or scrollback_delta)
        alternate_changed = bool(alternate_delta is None or alternate_delta)

        if self._alternate_screen != self._state.alternate_screen:
            self.post_message(
                self.AlternateScreenChanged(self, enabled=self._state.alternate_screen)
            )
        self._alternate_screen = self._state.alternate_screen
        return scrollback_changed or alternate_changed

    def _update_from_state(
        self, scrollback_delta: set[int] | None, alternate_delta: set[int] | None
    ) -> None:
        if self._state.current_directory:
            self.finalize()
        width = self._state.width
        height = self._state.scrollback_buffer.height

        if self._state.alternate_screen:
            height += self._state.alternate_buffer.height
        self.virtual_size = Size(min(self._state.buffer.max_line_width, width), height)
        if self._anchored and not self._anchor_released:
            self.scroll_y = self.max_scroll_y

        scroll_y = int(self.scroll_y)
        visible_lines = frozenset(range(scroll_y, scroll_y + height))

        if scrollback_delta is None and alternate_delta is None:
            self.refresh()
        else:
            window_width = self.region.width
            scrollback_height = self._state.scrollback_buffer.height
            if scrollback_delta is None:
                self.refresh(Region(0, 0, window_width, scrollback_height))
            else:
                refresh_lines = [
                    Region(0, y - scroll_y, window_width, 1)
                    for y in sorted(scrollback_delta & visible_lines)
                ]
                if refresh_lines:
                    self.refresh(*refresh_lines)
            alternate_height = self._state.alternate_buffer.height
            if alternate_delta is None:
                self.refresh(
                    Region(
                        0,
                        scrollback_height - scroll_y,
                        window_width,
                        scrollback_height + alternate_height,
                    )
                )
            else:
                alternate_delta = {
                    line_no + scrollback_height for line_no in alternate_delta
                }
                refresh_lines = [
                    Region(0, y - scroll_y, window_width, 1)
                    for y in sorted(alternate_delta & visible_lines)
                ]
                if refresh_lines:
                    self.refresh(*refresh_lines)

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        return self._render_line(scroll_x, scroll_y + y, self._width)

    def _render_line(self, x: int, y: int, width: int) -> Strip:
        visual_style = self.visual_style
        rich_style = visual_style.rich_style

        state = self._state
        buffer = state.scrollback_buffer
        buffer_offset = 0
        if y >= buffer.height and state.alternate_screen:
            buffer_offset = buffer.height
            buffer = state.alternate_buffer
        try:
            folded_line_ = buffer.folded_lines[y - buffer_offset]
            line_no, _line_offset, offset, line, updates = folded_line_
        except IndexError:
            return Strip.blank(width, rich_style)

        line_record = buffer.lines[line_no]
        cache_key: tuple | None = (
            self._state.alternate_screen,
            y,
            line_record.updates,
            updates,
        )

        if (
            not self.hide_cursor
            and state.show_cursor
            and buffer.cursor_line == y - buffer_offset
        ):
            if buffer.cursor_offset >= len(line):
                line = line.pad_right(buffer.cursor_offset - len(line) + 1)
            line_cursor_offset = buffer.cursor_offset
            line = line.stylize(
                self.CURSOR_STYLE, line_cursor_offset, line_cursor_offset + 1
            )
            cache_key = None

        if (
            cache_key is not None
            and (strip := self._terminal_render_cache.get(cache_key))
        ):
            strip = strip.crop(x, x + width)
            strip = strip.adjust_cell_length(
                width, (visual_style + line_record.style).rich_style
            )
            strip = strip.apply_offsets(x + offset, line_no)
            return strip  # noqa: RET504

        try:
            strip = Strip(
                line.render_segments(visual_style), cell_length=line.cell_length
            )
        except Exception:
            strip = Strip.blank(line.cell_length)

        if cache_key is not None:
            self._terminal_render_cache[cache_key] = strip

        strip = strip.crop(x, x + width)
        strip = strip.adjust_cell_length(
            width, (visual_style + line_record.style).rich_style
        )
        strip = strip.apply_offsets(x + offset, line_no)

        return strip  # noqa: RET504

    async def write_process_stdin(self, input: str) -> None:  # noqa: A002
        """Write to process stdin (no-op for read-only terminal).

        Args:
            input: Text to write.
        """
        if self._write_to_stdin is not None:
            await self._write_to_stdin(input)
