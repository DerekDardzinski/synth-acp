"""Input bar with multiline support and @agent-id routing."""

from __future__ import annotations

import logging
import re

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.highlight import highlight
from textual.message import Message
from textual.widgets import Button, Static, TextArea

from synth_acp.models.commands import CancelTurn, SendPrompt
from synth_acp.ui.widgets.gradient_bar import ActivityBar

log = logging.getLogger(__name__)

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

    def notify_style_update(self) -> None:
        self._highlight_cache = None
        super().notify_style_update()

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
        super()._on_key(event)

    class Submitted(Message):
        """Posted when the user submits the prompt."""

        def __init__(self, text_area: PromptTextArea) -> None:
            self.text_area = text_area
            super().__init__()


class InputBar(Vertical):
    """Bottom input bar for sending prompts to agents.

    Args:
        agent_id: Default target agent.
        agent_name: Display name for the agent.
        color: Hex color for the agent prompt indicator.
    """

    def __init__(self, agent_id: str, agent_name: str, project: str = "", **kwargs: object) -> None:
        super().__init__(classes="input-bar", **kwargs)
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._project = project

    def compose(self) -> ComposeResult:
        """Yield the text area and info bar."""
        yield PromptTextArea(id="prompt-input")
        with Horizontal(classes="info-bar"):
            yield Static(
                f"[dim]harness:[/] [$primary]{self._project}[/]"
                f" · [dim]agent_name:[/] [$primary]{self._agent_name}[/]"
                f" · [dim]agent_id:[/] {self._agent_id}",
                classes="info-bar-left",
            )
            yield Button("Submit ⏎", id="submit-btn", classes="info-bar-right")
            yield Button("Cancel ■", id="cancel-btn", classes="info-bar-right cancel-btn")
        yield ActivityBar(classes="input-activity")

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

        target = self._agent_id
        if text.startswith("@"):
            parts = text.split(" ", 1)
            candidate = parts[0][1:]
            known_ids = [a.id for a in app.config.agents]
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
