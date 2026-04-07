"""Input bar with multiline support and @agent-id routing."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.highlight import highlight
from textual.message import Message
from textual.widgets import Button, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from synth_acp.models.agent import AgentMode, AgentModel
from synth_acp.models.commands import CancelTurn, SendPrompt, SetAgentMode, SetAgentModel
from synth_acp.ui.widgets.gradient_bar import ActivityBar

log = logging.getLogger(__name__)

def _short_path(cwd: str) -> str:
    """Collapse a cwd to use ~ for the home directory."""
    try:
        return "~/" + str(Path(cwd).relative_to(Path.home()))
    except ValueError:
        return cwd


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


# Matches @agent-id and @"agent id with spaces"
RE_AGENT_MENTION = re.compile(r'(@\S+)|@"(.*?)"')
# Matches /slash-command at start of single-line input
RE_SLASH_COMMAND = re.compile(r"^(/\S+)")


class PromptTextArea(TextArea):
    """TextArea where Enter submits and Ctrl+J inserts newlines."""

    def __init__(self, **kwargs: object) -> None:
        self._highlight_cache: list[Content] | None = None
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

        # Markdown highlighting + @mention overlay
        content = highlight(text + "\n```", language="markdown")
        content = content.highlight_regex(RE_AGENT_MENTION, style="$primary")
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

    def notify_style_update(self) -> None:
        self._highlight_cache = None
        super().notify_style_update()

    # --- Placeholder & ghost text ---

    def on_mount(self) -> None:
        self.placeholder = Content.assemble(
            "Ask anything\t".expandtabs(8),
            ("▌@▐", "r"), " route  ",
            ("▌!▐", "r"), " shell  ",
            ("▌ctrl+j▐", "r"), " newline",
        )

    def _update_suggestion(self) -> None:
        """Set inline ghost text based on the current prefix character."""
        text = self.text
        if text == "@":
            self.suggestion = "agent-id message..."
        elif text == "!":
            self.suggestion = "command..."
        elif text == "/":
            self.suggestion = "command"
        else:
            self.suggestion = ""

    # --- Key handling ---

    def _on_key(self, event: events.Key) -> None:
        """Enter submits, Ctrl+J inserts newline."""
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

    def __init__(self, picker_id: str, input_bar: InputBar, *options: Option, **kwargs: object) -> None:
        super().__init__(*options, **kwargs)
        self._picker_id = picker_id
        self._input_bar = input_bar

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.remove()
        if event.option_id:
            self._input_bar._on_picker_selected(self._picker_id, event.option_id)

    def on_blur(self) -> None:
        """Dismiss when focus leaves the popup."""
        self.remove()


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
        self.update(f"[dim]{self._prefix}:[/] [$accent]{name}[/] [dim]▾[/]")

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
        popup = _PickerPopup(self.id, input_bar, *options, id=popup_id)
        self.screen.mount(popup)
        popup.focus()


class InputBar(Vertical):
    """Bottom input bar for sending prompts to agents.

    Args:
        agent_id: Default target agent.
        agent_name: Display name for the agent.
        harness: Harness name for display.
    """

    def __init__(self, agent_id: str, agent_name: str, harness: str = "", cwd: str = "", **kwargs: object) -> None:
        super().__init__(classes="input-bar", **kwargs)
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._harness = harness
        self._cwd = cwd
        self._cwd_display = _short_path(cwd) if cwd else ""
        self._git_branch = _git_branch(cwd) if cwd else None
        self._slash_commands: list[object] = []

    def on_mount(self) -> None:
        """Start polling git branch if cwd is set."""
        if self._cwd:
            self.set_interval(5, self._poll_git_branch)

    def _poll_git_branch(self) -> None:
        """Check for branch changes and update the info label."""
        branch = _git_branch(self._cwd)
        if branch != self._git_branch:
            self._git_branch = branch
            try:
                self.query_one("#info-label", Static).update(self._build_info_label())
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        """Yield the text area and info bar."""
        yield PromptTextArea(id="prompt-input")
        with Horizontal(classes="info-bar"):
            yield Static(
                self._build_info_label(),
                id="info-label",
                classes="info-bar-left",
            )
            yield _PickerLabel("mode", id="mode-picker", classes="info-bar-picker")
            yield _PickerLabel("model", id="model-picker", classes="info-bar-picker")
            yield Button("Submit ⏎", id="submit-btn", classes="info-bar-right")
            yield Button("Cancel ■", id="cancel-btn", classes="info-bar-right cancel-btn")
        yield ActivityBar(classes="input-activity")

    def _build_info_label(self) -> str:
        """Build the static info label text."""
        line1 = [
            f"[dim]agent:[/] [$primary]{self._agent_id}[/]",
            f"[dim]harness:[/] {self._harness}",
        ]
        line2: list[str] = []
        if self._cwd_display:
            cwd_part = self._cwd_display
            if self._git_branch:
                cwd_part += f" ([$accent]{self._git_branch}[/])"
            line2.append(cwd_part)
        elif self._git_branch:
            line2.append(f"[$accent]{self._git_branch}[/]")
        result = " · ".join(line1)
        if line2:
            result += "\n" + " · ".join(line2)
        return result

    # --- Mode / model updates ---

    def update_mode_info(self, modes: list[AgentMode], current_id: str | None) -> None:
        """Push available modes and current selection to the picker."""
        try:
            picker = self.query_one("#mode-picker", _PickerLabel)
            picker.set_items([(m.id, m.name) for m in modes], current_id)
        except Exception:
            log.debug("Mode picker update failed", exc_info=True)

    def update_model_info(self, models: list[AgentModel], current_id: str | None) -> None:
        """Push available models and current selection to the picker."""
        try:
            picker = self.query_one("#model-picker", _PickerLabel)
            picker.set_items([(m.id, m.name) for m in models], current_id)
        except Exception:
            log.debug("Model picker update failed", exc_info=True)

    def update_current_mode(self, mode_id: str) -> None:
        """Update just the active mode without replacing the list."""
        try:
            self.query_one("#mode-picker", _PickerLabel).set_current(mode_id)
        except Exception:
            log.debug("Mode picker current update failed", exc_info=True)

    def update_current_model(self, model_id: str) -> None:
        """Update just the active model without replacing the list."""
        try:
            self.query_one("#model-picker", _PickerLabel).set_current(model_id)
        except Exception:
            log.debug("Model picker current update failed", exc_info=True)

    # --- Picker selection handler ---

    def _on_picker_selected(self, picker_id: str, option_id: str) -> None:
        """Called by _PickerPopup when the user selects an option."""
        # Update the label
        try:
            picker = self.query_one(f"#{picker_id}", _PickerLabel)
            picker.set_current(option_id)
        except Exception:
            pass
        # Forward to broker
        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
            return
        if picker_id == "mode-picker":
            app.run_worker(
                app.broker.handle(SetAgentMode(agent_id=self._agent_id, mode_id=option_id))
            )
        elif picker_id == "model-picker":
            app.run_worker(
                app.broker.handle(SetAgentModel(agent_id=self._agent_id, model_id=option_id))
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

    def set_busy(self, busy: bool) -> None:
        """Toggle between submit/cancel button and activity bar."""
        self.query_one("#submit-btn").display = not busy
        self.query_one("#cancel-btn").display = busy
        self.query_one(ActivityBar).active = busy

    def on_prompt_text_area_submitted(self, message: PromptTextArea.Submitted) -> None:
        """Parse @agent-id routing and send prompt to broker."""
        text = message.text_area.text.strip()
        if not text:
            return
        message.text_area.clear()

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

        target = self._agent_id
        if text.startswith("@"):
            parts = text.split(" ", 1)
            candidate = parts[0][1:]
            known_ids = [a.agent_id for a in app.config.agents]
            if candidate in known_ids:
                target = candidate
                text = parts[1] if len(parts) > 1 else ""
                if not text:
                    return
            else:
                app.notify(f"Unknown agent: {candidate}", severity="warning")
                return

        app.run_worker(app.broker.handle(SendPrompt(agent_id=target, text=text)))

        if target in app._panels:
            feed = app._panels[target]
            feed.add_prompt(text)

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
