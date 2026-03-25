"""Input bar with multiline support and @agent-id routing."""

from __future__ import annotations

from textual import events
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import TextArea

from synth_acp.models.commands import SendPrompt


class PromptTextArea(TextArea):
    """TextArea where Enter submits and Ctrl+J inserts newlines."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(language=None, soft_wrap=True, show_line_numbers=False, **kwargs)

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
        color: Hex color for the agent prompt indicator.
    """

    def __init__(self, agent_id: str, color: str, **kwargs: object) -> None:
        super().__init__(classes="input-bar", **kwargs)
        self._agent_id = agent_id
        self._color = color

    def compose(self):
        """Yield the text area widget."""
        yield PromptTextArea(id="prompt-input")

    def on_prompt_text_area_submitted(self, message: PromptTextArea.Submitted) -> None:
        """Parse @agent-id routing and send prompt to broker."""
        text = message.text_area.text.strip()
        if not text:
            return
        message.text_area.clear()

        from synth_acp.ui.app import SynthApp

        app = self.app
        assert isinstance(app, SynthApp)

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
            pass
