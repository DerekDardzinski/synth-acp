"""Prompt queue widget — tabbed, editable queue of pending prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Tab, Tabs, TextArea
from textual.widgets.markdown import Markdown

from synth_acp.ui.widgets.input_bar import PromptTextArea


@dataclass
class QueuedPrompt:
    """A prompt waiting in the queue."""

    id: str = field(default_factory=lambda: f"q-{uuid4().hex[:8]}")
    source: Literal["user", "mcp"] = "user"
    text: str = ""
    from_agent: str | None = None
    editing: bool = False


class PromptQueue(Vertical):
    """Tabbed queue of pending prompts displayed above the input bar."""

    class DrainReady(Message):
        """Posted when editing completes and drain should be re-attempted."""

    DEFAULT_CSS = """
    PromptQueue {
        height: auto;
        max-height: 14;
        margin: 0 3;
        border: heavy $surface-lighten-1;
        padding: 0 1;
    }
    PromptQueue Tabs {
        height: auto;
        max-height: 2;
    }
    PromptQueue #tabs-scroll {
        height: auto;
    }
    PromptQueue .queue-content {
        height: auto;
        max-height: 10;
    }
    PromptQueue .queue-display {
        width: 1fr;
        height: auto;
        max-height: 9;
        overflow-y: auto;
    }
    PromptQueue PromptTextArea {
        max-height: 4;
    }
    PromptQueue .queue-actions {
        height: auto;
        layout: horizontal;
        align: right middle;
    }
    PromptQueue #queue-delete-btn {
        color: $warning;
    }
    PromptQueue #queue-delete-btn:hover {
        color: $error;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: list[QueuedPrompt] = []
        self._active_id: str | None = None
        self.display = False

    def compose(self) -> ComposeResult:
        yield Tabs(id="queue-tabs")
        with Horizontal(classes="queue-content"):
            yield Markdown("", classes="queue-display")
        with Horizontal(classes="queue-actions"):
            yield Button("Edit ✎", id="queue-edit-btn", classes="info-bar-right")
            yield Button("Delete ✕", id="queue-delete-btn", classes="info-bar-right")

    def enqueue(
        self, text: str, source: Literal["user", "mcp"], from_agent: str | None
    ) -> None:
        """Add a prompt to the queue. Shows widget if first item."""
        item = QueuedPrompt(source=source, text=text, from_agent=from_agent)
        self._queue.append(item)
        label = self._tab_label(item)
        tabs = self.query_one("#queue-tabs", Tabs)
        tabs.add_tab(Tab(label, id=item.id))
        if len(self._queue) == 1:
            self.display = True
            self._active_id = item.id
            self._show_item(item)

    def drain_next(self) -> QueuedPrompt | None:
        """Pop and return queue[0] if not editing. Returns None if empty or head is being edited."""
        if not self._queue:
            return None
        if self._queue[0].editing:
            return None
        item = self._queue.pop(0)
        tabs = self.query_one("#queue-tabs", Tabs)
        tabs.remove_tab(item.id)
        if not self._queue:
            self.display = False
            self._active_id = None
        else:
            self._active_id = self._queue[0].id
            self._show_item(self._queue[0])
        return item

    @property
    def has_items(self) -> bool:
        """Whether queue has any items."""
        return bool(self._queue)

    def _get_item(self, item_id: str) -> QueuedPrompt | None:
        """Find a queued item by id."""
        for item in self._queue:
            if item.id == item_id:
                return item
        return None

    def _tab_label(self, item: QueuedPrompt) -> str:
        """Generate tab label from item source and text preview."""
        preview = item.text[:10].replace("\n", " ")
        ellipsis = "…" if len(item.text) > 10 else ""
        prefix = "MCP" if item.source == "mcp" else "User:"
        return f"{prefix} {preview}{ellipsis}"

    def _update_active_tab_label(self) -> None:
        """Update the active tab's label to reflect current text."""
        if not self._active_id:
            return
        item = self._get_item(self._active_id)
        if not item:
            return
        tabs = self.query_one("#queue-tabs", Tabs)
        try:
            tab = tabs.query_one(f"#{item.id}", Tab)
            tab.label = self._tab_label(item)
        except Exception:
            pass

    def _show_item(self, item: QueuedPrompt) -> None:
        """Display the given item's content in the content area."""
        content = self.query_one(".queue-content", Horizontal)
        edit_btn = self.query_one("#queue-edit-btn", Button)
        content.remove_children()
        if item.editing:
            ta = PromptTextArea(classes="queue-display")
            content.mount(ta)
            ta.text = item.text
            edit_btn.label = "Save ✓"
        else:
            content.mount(Markdown(item.text, classes="queue-display"))
            edit_btn.label = "Edit ✎"

    def _enter_edit_mode(self, item: QueuedPrompt) -> None:
        """Switch active item to edit mode."""
        item.editing = True
        self._show_item(item)
        try:
            self.query_one(".queue-content PromptTextArea", PromptTextArea).focus()
        except Exception:
            pass

    def _exit_edit_mode(self, item: QueuedPrompt) -> None:
        """Save edits and switch back to read-only mode."""
        try:
            ta = self.query_one(".queue-content PromptTextArea", PromptTextArea)
            item.text = ta.text
        except Exception:
            pass
        item.editing = False
        self._update_active_tab_label()
        self._show_item(item)
        self.post_message(self.DrainReady())

    def _delete_active(self) -> None:
        """Remove the active item from the queue."""
        if not self._active_id:
            return
        item = self._get_item(self._active_id)
        if not item:
            return
        self._queue.remove(item)
        tabs = self.query_one("#queue-tabs", Tabs)
        tabs.remove_tab(item.id)
        if not self._queue:
            self.display = False
            self._active_id = None
        else:
            self._active_id = self._queue[0].id
            self._show_item(self._queue[0])

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle tab switch — save edits if needed, show new item."""
        event.stop()
        tab_id = event.tab.id
        if tab_id is None:
            return
        # Save current edit if switching away
        if self._active_id and self._active_id != tab_id:
            prev = self._get_item(self._active_id)
            if prev and prev.editing:
                try:
                    ta = self.query_one(".queue-content PromptTextArea", PromptTextArea)
                    prev.text = ta.text
                except Exception:
                    pass
                prev.editing = False
        self._active_id = tab_id
        item = self._get_item(tab_id)
        if item:
            self._show_item(item)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Edit/Save and Delete button presses."""
        event.stop()
        if event.button.id == "queue-edit-btn":
            if not self._active_id:
                return
            item = self._get_item(self._active_id)
            if not item:
                return
            if item.editing:
                self._exit_edit_mode(item)
            else:
                self._enter_edit_mode(item)
        elif event.button.id == "queue-delete-btn":
            self._delete_active()

    def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        """Intercept Enter in edit mode — save instead of submitting."""
        event.stop()
        if not self._active_id:
            return
        item = self._get_item(self._active_id)
        if item and item.editing:
            self._exit_edit_mode(item)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update tab label dynamically as user edits."""
        event.stop()
        if not self._active_id:
            return
        item = self._get_item(self._active_id)
        if item and item.editing:
            item.text = event.text_area.text
            self._update_active_tab_label()
