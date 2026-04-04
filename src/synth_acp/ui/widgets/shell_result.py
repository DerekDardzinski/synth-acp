"""Widget displaying the output of a shell command run from the input bar."""

from __future__ import annotations

from textual.containers import Vertical, VerticalScroll
from textual.highlight import highlight
from textual.widgets import Label, Rule, Static


class ShellResultBlock(Vertical):
    """Shows a shell command and its syntax-highlighted output."""

    DEFAULT_CSS = """
    ShellResultBlock {
        height: auto;
        background: $background;
        border-left: wide $warning;
        padding: 0 1;
        color: $text-muted;
    }
    #shell-output { height: auto; max-height: 20; padding: 0 1; }
    #shell-output-label { width: 1fr; }
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self):
        """Yield header label."""
        content = highlight(f"$ {self._command}", language="bash")
        yield Label(content, id="shell-input")
        yield Static("[dim]running…[/dim]", id="shell-status")

    def set_output(self, output: str, return_code: int) -> None:
        """Replace the placeholder with syntax-highlighted output."""
        self.query_one("#shell-status").remove()
        self.mount(Rule(line_style="dashed", id="shell-sep"))
        text = output.rstrip()
        if text:
            content = highlight(text, language="bash")
            label = Label(content, id="shell-output-label")
            self.mount(VerticalScroll(label, id="shell-output"))
        exit_style = "success" if return_code == 0 else "error"
        self.mount(Rule(line_style="dashed", classes=f"shell-exit-{exit_style}"))
        self.query_one("#shell-sep").add_class(f"shell-exit-{exit_style}")
