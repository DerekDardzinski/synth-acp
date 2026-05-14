"""Input bar with multiline support."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from acp.schema import (
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
)
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.highlight import highlight
from textual.markup import escape
from textual.message import Message
from textual.widgets import Button, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from synth_acp.models.commands import CancelTurn, SendPrompt, SetConfigOption
from synth_acp.ui.file_discovery import FileEntry, discover_files, estimate_tokens, filter_files
from synth_acp.ui.widgets.gradient_bar import ActivityBar

if TYPE_CHECKING:
    from synth_acp.ui.widgets.prompt_queue import PromptQueue, QueuedPrompt

log = logging.getLogger(__name__)

def _short_path(cwd: str) -> str:
    """Collapse a cwd to use ~ for the home directory."""
    try:
        resolved = Path(cwd).resolve()
        rel = resolved.relative_to(Path.home())
        return "~" if str(rel) == "." else "~/" + str(rel)
    except ValueError:
        return str(Path(cwd).resolve())


def _git_branch(cwd: str) -> str | None:
    """Return the current git branch for cwd, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# Matches /slash-command at start of single-line input
RE_SLASH_COMMAND = re.compile(r"^(/\S+)")

class PromptTextArea(TextArea):
    """TextArea where Enter submits and Ctrl+J inserts newlines."""

    class AtTrigger(Message):
        """Posted when @ is typed at a word boundary."""

        def __init__(self, query: str, cursor_row: int, at_pos: int) -> None:
            self.query = query
            self.cursor_row = cursor_row
            self.at_pos = at_pos
            super().__init__()

    class AtDismiss(Message):
        """Posted when @ context is lost (space after query, cursor moved away)."""

    class PickerKey(Message):
        """Posted when a picker navigation key is pressed while @ is active."""

        def __init__(self, key: str) -> None:
            self.key = key
            super().__init__()

    def __init__(self, **kwargs: object) -> None:
        self._highlight_cache: list[Content] | None = None
        self._at_active: bool = False
        super().__init__(
            soft_wrap=True,
            show_line_numbers=True,
            highlight_cursor_line=False,
            **kwargs,
        )
        self.compact = True

    # --- Highlighting ---

    def _get_highlighted_lines(self) -> list[Content]:
        if self._highlight_cache is not None:
            return self._highlight_cache

        text = self.text

        # Slash command: entire single line goes green
        if text.startswith("/") and "\n" not in text:
            self._highlight_cache = [Content.styled(text, "$text-success")]
            return self._highlight_cache

        # Shell command: entire single line goes warning color
        if text.startswith("!") and "\n" not in text:
            self._highlight_cache = [Content.styled(text, "$text-warning")]
            return self._highlight_cache

        # Markdown highlighting
        content = highlight(text + "\n```", language="markdown")
        self._highlight_cache = content.split("\n", allow_blank=True)[:-1]
        return self._highlight_cache

    def get_line(self, line_index: int) -> Text:
        """Override to inject custom highlighting."""
        lines = self._get_highlighted_lines()
        try:
            line = lines[line_index]
        except IndexError:
            return Text("", end="", no_wrap=True)

        rendered = list(line.render_segments(self.visual_style))
        return Text.assemble(
            *[(text, str(style) if style else "") for text, style, _ in rendered],
            end="",
            no_wrap=True,
        )

    @on(TextArea.Changed)
    def _on_text_changed(self, event: TextArea.Changed) -> None:  # noqa: ARG002
        self._highlight_cache = None
        self._update_suggestion()
        self._check_at_trigger()

    def _check_at_trigger(self) -> None:
        """Detect @ at word boundary and post AtTrigger or AtDismiss."""
        text = self.text
        cursor_row, cursor_col = self.cursor_location
        lines = text.split("\n")
        if cursor_row >= len(lines):
            if self._at_active:
                self._at_active = False
                self.post_message(self.AtDismiss())
            return
        line = lines[cursor_row]
        text_before_cursor = line[:cursor_col]

        # Find the last @ at word boundary before cursor
        at_pos = None
        for i, ch in enumerate(text_before_cursor):
            if ch == "@" and (i == 0 or text_before_cursor[i - 1] in " \t"):
                at_pos = i

        if at_pos is not None:
            query = text_before_cursor[at_pos + 1:]
            if " " not in query:
                self._at_active = True
                self.post_message(self.AtTrigger(query, cursor_row, at_pos))
                return

        if self._at_active:
            self._at_active = False
            self.post_message(self.AtDismiss())

    def notify_style_update(self) -> None:
        self._highlight_cache = None
        super().notify_style_update()

    # --- Placeholder & ghost text ---

    def on_mount(self) -> None:
        self.placeholder = Content.assemble(
            "Ask anything\t".expandtabs(8),
            ("▌@▐", "r"), " file  ",
            ("▌!▐", "r"), " shell  ",
            ("▌ctrl+j▐", "r"), " newline",
        )

    def _update_suggestion(self) -> None:
        """Set inline ghost text based on the current prefix character."""
        text = self.text
        if text == "!":
            self.suggestion = "command..."
        elif text == "/":
            self.suggestion = "command"
        else:
            self.suggestion = ""

    # --- Key handling ---

    def _on_key(self, event: events.Key) -> None:
        """Enter submits, Ctrl+J inserts newline."""
        # Delegate to file picker if active
        if self._at_active and event.key in ("up", "down", "enter", "escape"):
            event.prevent_default()
            event.stop()
            self.post_message(self.PickerKey(event.key))
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(PromptTextArea.Submitted(self))
            return
        if event.key == "ctrl+j":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key == "escape" and self.suggestion:
            self.suggestion = ""
            event.prevent_default()
            event.stop()
            return
        super()._on_key(event)

    class Submitted(Message):
        """Posted when the user submits the prompt."""

        def __init__(self, text_area: PromptTextArea) -> None:
            self.text_area = text_area
            super().__init__()


class _PickerPopup(OptionList):
    """OptionList popup that notifies its owning InputBar on selection."""

    def __init__(self, picker_id: str, input_bar: InputBar, anchor_region: object, *options: Option, **kwargs: object) -> None:
        super().__init__(*options, **kwargs)
        self._picker_id = picker_id
        self._input_bar = input_bar
        self._anchor_region = anchor_region
        # Mount offscreen; repositioned after first layout
        self.styles.offset = (-9999, -9999)

    def _reposition(self) -> None:
        """Compute and apply position relative to anchor label."""
        region = self._anchor_region
        y = region.y - self.outer_size.height
        popup_width = 40
        screen_width = self.screen.size.width
        if region.x + popup_width > screen_width:
            x = region.x + region.width - popup_width
        else:
            x = region.x
        self.styles.offset = (x, y)

    def on_mount(self) -> None:
        """Position after first layout completes."""
        self.call_after_refresh(self._reposition)

    def on_resize(self) -> None:
        """Reposition after size changes."""
        self._reposition()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.remove()
        if event.option_id:
            self._input_bar._on_picker_selected(self._picker_id, event.option_id)

    def on_blur(self) -> None:
        """Dismiss when focus leaves the popup."""
        self.remove()


class _FilePickerPopup(OptionList):
    """File picker popup that shows fuzzy-filtered file results."""

    def __init__(self, input_bar: InputBar, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._input_bar = input_bar
        # Mount offscreen; repositioned after first layout
        self.styles.offset = (-9999, -9999)

    def _reposition(self) -> None:
        """Compute and apply position relative to the @ character."""
        anchor = self._input_bar._get_at_screen_position()
        if anchor is None:
            return
        screen_x, screen_y = anchor
        y = screen_y - self.outer_size.height
        screen_width = self.screen.size.width
        x = min(screen_x, screen_width - self.outer_size.width)
        x = max(0, x)
        self.styles.offset = (x, y)

    def on_mount(self) -> None:
        """Position after first layout completes."""
        self.call_after_refresh(self._reposition)

    def on_resize(self) -> None:
        """Reposition after size changes."""
        self._reposition()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option_id:
            self._input_bar._on_file_selected(event.option_id)

    def on_blur(self) -> None:
        """Dismiss when focus leaves the popup."""
        self._input_bar._close_file_picker()


class _PickerLabel(Static):
    """Clickable label that opens an OptionList popup for selection."""

    def __init__(self, prefix: str, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._prefix = prefix
        self._items: list[tuple[str, str]] = []  # (id, display_name)
        self._current_id: str | None = None

    def set_items(self, items: list[tuple[str, str]], current_id: str | None) -> None:
        """Replace the option list and set the active item."""
        self._items = items
        self._current_id = current_id
        self._refresh_label()

    def set_current(self, current_id: str) -> None:
        """Update the active item without replacing the list."""
        self._current_id = current_id
        self._refresh_label()

    def _refresh_label(self) -> None:
        if not self._items or self._current_id is None:
            self.display = False
            return
        self.display = True
        name = next((n for vid, n in self._items if vid == self._current_id), self._current_id)
        self.update(f"[dim]{self._prefix}:[/] [$accent]{escape(name)}[/] [dim]▾[/]")

    def on_click(self) -> None:
        """Toggle the popup OptionList on the screen."""
        popup_id = f"{self.id}-popup"
        try:
            existing = self.screen.query_one(f"#{popup_id}")
            existing.remove()
            return
        except Exception:
            pass

        if len(self._items) < 2:
            return

        # Walk up to find the InputBar
        input_bar: InputBar | None = None
        for ancestor in self.ancestors_with_self:
            if isinstance(ancestor, InputBar):
                input_bar = ancestor
                break
        if not input_bar:
            return

        options = [Option(name, id=vid) for vid, name in self._items]
        popup = _PickerPopup(self.id, input_bar, self.region, *options, id=popup_id, classes="config-picker-popup")
        self.screen.mount(popup)
        popup.focus()


class InputBar(Vertical):
    """Bottom input bar for sending prompts to agents.

    Args:
        agent_id: Default target agent.
        agent_name: Display name for the agent.
        harness: Harness name for display.
    """

    class DrainReady(Message):
        """Posted when queue edit completes and drain should be re-attempted."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    class CancelClicked(Message):
        """Posted when the user cancels the current turn."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    def __init__(self, agent_id: str, agent_name: str, harness: str = "", cwd: str = "", **kwargs: object) -> None:
        super().__init__(classes="input-bar", **kwargs)
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._harness = harness
        self._cwd = cwd
        self._cwd_display = _short_path(cwd) if cwd else ""
        self._git_branch: str | None = None
        self._busy = False
        self._slash_commands: list[object] = []
        self._file_cache: list[FileEntry] = []
        self._file_picker: _FilePickerPopup | None = None
        self._at_pos: int | None = None
        self._at_row: int = 0
        self._at_textarea: PromptTextArea | None = None
        self._filter_timer: asyncio.TimerHandle | None = None

    def on_mount(self) -> None:
        """Start polling git branch if cwd is set."""
        cwd = self.query_one("#cwd-label", Static)
        if self._cwd_display:
            cwd.update(self._build_cwd_label())
        else:
            cwd.display = False
        if self._cwd:
            self.run_worker(self._async_poll_git_branch(), name="git-poll-init")
            self.set_interval(5, self._poll_git_branch)
            self.run_worker(self._refresh_file_cache(), exclusive=True, group="file-cache", name="file-cache-init")
            self.set_interval(30, self._schedule_file_cache_refresh)

    def _poll_git_branch(self) -> None:
        """Timer callback — spawn async worker to avoid blocking the event loop."""
        self.run_worker(self._async_poll_git_branch(), exclusive=True, name="git-poll")

    async def _async_poll_git_branch(self) -> None:
        """Fetch git branch in a thread and update the label if it changed."""
        branch = await asyncio.to_thread(_git_branch, self._cwd)
        if branch != self._git_branch:
            self._git_branch = branch
            try:
                label = self.query_one("#cwd-label", Static)
                text = self._build_cwd_label()
                label.update(text)
                label.display = bool(text)
            except Exception:
                pass

    def _schedule_file_cache_refresh(self) -> None:
        """Timer callback — spawn async worker to refresh file cache."""
        self.run_worker(self._refresh_file_cache(), exclusive=True, group="file-cache", name="file-cache-refresh")

    async def _refresh_file_cache(self) -> None:
        """Discover project files and cache the result."""
        self._file_cache = await discover_files(Path(self._cwd))

    # --- File picker ---

    def on_prompt_text_area_at_trigger(self, event: PromptTextArea.AtTrigger) -> None:
        """Handle @ trigger: open or update file picker."""
        self._at_pos = event.at_pos
        self._at_row = event.cursor_row
        self._at_textarea = event._sender  # type: ignore[assignment]
        if self._file_picker is None:
            self._open_file_picker(event.query)
        else:
            self._schedule_filter(event.query)
            self._file_picker.on_resize()

    def on_prompt_text_area_at_dismiss(self, event: PromptTextArea.AtDismiss) -> None:  # noqa: ARG002
        """Handle @ dismiss: close file picker."""
        self._close_file_picker()

    def on_prompt_text_area_picker_key(self, event: PromptTextArea.PickerKey) -> None:
        """Handle picker navigation keys from PromptTextArea."""
        if self._file_picker is None:
            return
        if event.key == "up":
            self._file_picker.action_cursor_up()
        elif event.key == "down":
            self._file_picker.action_cursor_down()
        elif event.key == "enter":
            highlighted = self._file_picker.highlighted
            if highlighted is not None:
                option = self._file_picker.get_option_at_index(highlighted)
                if option.id:
                    self._on_file_selected(option.id)
        elif event.key == "escape":
            self._close_file_picker()
            try:
                ta = self.query_one("#prompt-input", PromptTextArea)
                ta._at_active = False
            except Exception:
                pass

    def _open_file_picker(self, query: str) -> None:
        """Mount the file picker popup above the input bar."""
        if self._file_picker is not None:
            return
        results = filter_files(query, self._file_cache) if query else self._file_cache[:15]
        options = [
            Option(f"{e.rel_path}  [dim]{estimate_tokens(e.size_bytes)}[/]", id=e.rel_path)
            for e in results
        ]
        if not options:
            return
        popup = _FilePickerPopup(self, id="file-picker-popup")
        for opt in options:
            popup.add_option(opt)
        self.screen.mount(popup)
        self._file_picker = popup

    def _get_at_screen_position(self) -> tuple[int, int] | None:
        """Compute the current screen position of the @ trigger character."""
        if self._at_pos is None:
            return None
        ta = self._at_textarea
        if ta is None:
            return None
        at_location = (self._at_row, self._at_pos)
        virtual_offset = ta.wrapped_document.location_to_offset(at_location)
        content_region = ta.content_region
        scroll_x, scroll_y = ta.scroll_offset
        screen_x = content_region.x + virtual_offset.x - scroll_x + ta.gutter_width
        screen_y = content_region.y + virtual_offset.y - scroll_y
        return (screen_x, screen_y)

    def _close_file_picker(self) -> None:
        """Remove the file picker popup and reset state."""
        if self._filter_timer is not None:
            self._filter_timer.cancel()
            self._filter_timer = None
        if self._file_picker is not None:
            try:
                self._file_picker.remove()
            except Exception:
                pass
            self._file_picker = None
        self._at_pos = None
        self._at_textarea = None

    def _schedule_filter(self, query: str) -> None:
        """Debounce filter updates at ~80ms."""
        if self._filter_timer is not None:
            self._filter_timer.cancel()
        loop = asyncio.get_event_loop()
        self._filter_timer = loop.call_later(0.08, self._run_filter, query)

    def _run_filter(self, query: str) -> None:
        """Spawn the async filter worker."""
        self.run_worker(self._async_filter(query), exclusive=True, group="file-filter", name="file-filter")

    async def _async_filter(self, query: str) -> None:
        """Run filter_files off-thread and update the picker."""
        results = await asyncio.to_thread(filter_files, query, self._file_cache)
        if self._file_picker is None:
            return
        self._file_picker.clear_options()
        for e in results:
            self._file_picker.add_option(
                Option(f"{e.rel_path}  [dim]{estimate_tokens(e.size_bytes)}[/]", id=e.rel_path)
            )

    def _on_file_selected(self, rel_path: str) -> None:
        """Insert @rel_path + trailing space at the @ position."""
        ta = self._at_textarea
        if ta is None:
            self._close_file_picker()
            return
        at_pos = self._at_pos
        self._close_file_picker()
        if at_pos is None:
            return
        # Replace from @ to cursor with @rel_path + space
        text = ta.text
        cursor_row, cursor_col = ta.cursor_location
        lines = text.split("\n")
        line = lines[cursor_row]
        new_line = line[:at_pos] + f"@{rel_path} " + line[cursor_col:]
        lines[cursor_row] = new_line
        ta.load_text("\n".join(lines))
        # Move cursor after the inserted text
        new_col = at_pos + len(rel_path) + 2  # @ + path + space
        ta.move_cursor((cursor_row, new_col))
        ta.focus()

    def compose(self) -> ComposeResult:
        """Yield the text area and info bar."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        yield PromptQueue()
        yield Horizontal(id="config-pickers", classes="info-bar info-bar-pickers")
        yield PromptTextArea(id="prompt-input")
        with Horizontal(classes="info-bar"):
            yield Static(
                self._build_info_label(),
                id="info-label",
                classes="info-bar-left",
            )
            yield Button("Submit ⏎", id="submit-btn", classes="info-bar-right")
            yield Button("Drain Queue ▶", id="drain-btn", classes="info-bar-right")
            yield Button("Cancel ■", id="cancel-btn", classes="info-bar-right cancel-btn")
        yield Static("", id="cwd-label", classes="info-cwd")
        yield ActivityBar(classes="input-activity")

    def _build_info_label(self) -> str:
        """Build the static info label text."""
        return (
            f"[dim]agent:[/] [$primary]{escape(self._agent_id)}[/]"
            f" · [dim]harness:[/] {escape(self._harness)}"
        )

    def _build_cwd_label(self) -> str:
        """Build the cwd/branch label text."""
        if self._cwd_display:
            cwd_part = escape(self._cwd_display)
            if self._git_branch:
                cwd_part += f" ([$accent]{escape(self._git_branch)}[/])"
            return cwd_part
        if self._git_branch:
            return f"[$accent]{escape(self._git_branch)}[/]"
        return ""

    # --- Config option pickers ---

    _CATEGORY_ORDER: ClassVar[dict[str, int]] = {"mode": 0, "model": 1, "thought_level": 2}

    def update_config_options(self, options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean]) -> None:
        """Rebuild pickers from config options. Only select-type options get pickers.

        Uses a diff approach: updates existing pickers in place, removes stale
        ones, and adds new ones. This avoids the Textual DuplicateIds bug that
        occurs when remove_children() (async) hasn't completed before mount()
        adds widgets with the same IDs.
        """
        select_opts = [o for o in options if isinstance(o, SessionConfigOptionSelect)]
        select_opts.sort(key=lambda o: self._CATEGORY_ORDER.get(o.category or "", 99))
        try:
            container = self.query_one("#config-pickers", Horizontal)
        except Exception:
            return

        # Build desired state: ordered list of (id, opt)
        desired = {f"picker-{opt.id}": opt for opt in select_opts}

        # Remove pickers that are no longer in the desired set
        for child in list(container.children):
            if child.id and child.id not in desired:
                child.remove()

        # Update existing or add new pickers in order
        for picker_id, opt in desired.items():
            items: list[tuple[str, str]] = []
            for entry in opt.options:
                if isinstance(entry, SessionConfigSelectGroup):
                    for sub in entry.options:
                        items.append((sub.value, sub.name))
                else:
                    items.append((entry.value, entry.name))

            try:
                existing = container.query_one(f"#{picker_id}", _PickerLabel)
                existing.set_items(items, opt.current_value)
            except Exception:
                picker = _PickerLabel(opt.name, id=picker_id, classes="info-bar-picker")
                container.mount(picker)
                picker.set_items(items, opt.current_value)

    def update_config_option_value(self, config_id: str, value: str) -> None:
        """Update a single picker's current value."""
        try:
            self.query_one(f"#picker-{config_id}", _PickerLabel).set_current(value)
        except Exception:
            log.debug("Config option picker update failed for %s", config_id, exc_info=True)

    # --- Picker selection handler ---

    def _on_picker_selected(self, picker_id: str, option_id: str) -> None:
        """Called by _PickerPopup when the user selects an option."""
        try:
            picker = self.query_one(f"#{picker_id}", _PickerLabel)
            picker.set_current(option_id)
        except Exception:
            pass
        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
            return
        # picker_id is "picker-{config_id}"
        config_id = picker_id.removeprefix("picker-")
        app.run_worker(
            app.broker.handle(SetConfigOption(agent_id=self._agent_id, config_id=config_id, value=option_id))
        )

    # --- Slash commands ---

    def update_slash_commands(self, commands: list[object]) -> None:
        """Store available slash commands received from the agent.

        Args:
            commands: List of AvailableCommand from the ACP SDK.
        """
        self._slash_commands = list(commands)

    # --- Existing functionality ---

    @on(Button.Pressed, "#submit-btn")
    def _on_submit_click(self, event: Button.Pressed) -> None:
        event.stop()
        ta = self.query_one("#prompt-input", PromptTextArea)
        ta.post_message(PromptTextArea.Submitted(ta))

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel_click(self, event: Button.Pressed) -> None:
        event.stop()
        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
                    return
        app.run_worker(app.broker.handle(CancelTurn(agent_id=self._agent_id)))
        self.post_message(self.CancelClicked(agent_id=self._agent_id))

    @on(Button.Pressed, "#drain-btn")
    def _on_drain_click(self, event: Button.Pressed) -> None:
        event.stop()
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        self.set_drain_pending(False)
        queued = self.query_one(PromptQueue).drain_next()
        if queued:
            from synth_acp.ui.app import SynthApp

            app = self.app
            if isinstance(app, SynthApp):
                app.run_worker(app.broker.handle(SendPrompt(agent_id=self._agent_id, text=queued.text)))

    def set_busy(self, busy: bool) -> None:
        """Toggle between submit/cancel button and activity bar."""
        self._busy = busy
        self.query_one("#submit-btn").display = not busy
        self.query_one("#cancel-btn").display = busy
        self.query_one("#drain-btn").display = False
        self.query_one(ActivityBar).active = busy

    def set_drain_pending(self, pending: bool) -> None:
        """Show/hide the Drain Queue button."""
        self.query_one("#drain-btn").display = pending

    def on_prompt_text_area_submitted(self, message: PromptTextArea.Submitted) -> None:
        """Send prompt to the focused agent."""
        text = message.text_area.text.strip()
        if not text:
            return
        message.text_area.clear()

        # If busy, enqueue instead of dispatching
        if self._busy:
            from synth_acp.ui.widgets.prompt_queue import PromptQueue

            self.query_one(PromptQueue).enqueue(text, "user", None)
            return

        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
                    return

        # Shell command: !<command>
        if text.startswith("!"):
            command = text[1:].strip()
            if command and self._agent_id in app._panels:
                feed = app._panels[self._agent_id]
                app.run_worker(feed.run_shell_command(command))
            return

        app.run_worker(app.broker.handle(SendPrompt(agent_id=self._agent_id, text=text)))

    def set_disabled(self, disabled: bool, hint: str) -> None:  # noqa: ARG002
        """Enable or disable the input.

        Args:
            disabled: Whether to disable the input.
            hint: Placeholder text to show.
        """
        try:
            inp = self.query_one("#prompt-input", PromptTextArea)
            inp.disabled = disabled
        except Exception:
            log.debug("Input disable failed", exc_info=True)

    # --- Queue API ---

    def enqueue(self, text: str, source: Literal["user", "mcp"], from_agent: str | None) -> None:
        """Public API for app.py to enqueue MCP messages into this agent's queue."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        self.query_one(PromptQueue).enqueue(text, source, from_agent)

    def drain_next(self) -> QueuedPrompt | None:
        """Public API for app.py drain trigger. Delegates to PromptQueue."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        return self.query_one(PromptQueue).drain_next()

    @property
    def is_composing(self) -> bool:
        """True if the prompt textarea has non-empty stripped text."""
        return bool(self.query_one("#prompt-input", PromptTextArea).text.strip())

    @property
    def has_queue_items(self) -> bool:
        """True if the prompt queue has pending items."""
        from synth_acp.ui.widgets.prompt_queue import PromptQueue

        return self.query_one(PromptQueue).has_items

    def on_prompt_queue_drain_ready(self, message: PromptQueue.DrainReady) -> None:
        """Handle DrainReady from PromptQueue — bubble as InputBar.DrainReady."""
        message.stop()
        self.post_message(self.DrainReady(agent_id=self._agent_id))
