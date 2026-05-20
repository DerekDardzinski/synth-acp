"""Prompt queue widget — tabbed, editable queue of pending prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Tab, Tabs, TextArea
from textual.widgets.markdown import Markdown

from synth_acp.ui.widgets.input_bar import PromptTextArea

if TYPE_CHECKING:
    from synth_acp.models.events import QueueItemSnapshot


class PromptQueue(Vertical):
    """Tabbed queue of pending prompts displayed above the input bar.

    This widget is reactive — it receives state snapshots from the broker
    via reconcile() and posts messages for user actions (edit, save, delete).
    """

    class EditRequested(Message):
        """User wants to edit a queue item."""

        def __init__(self, item_id: str) -> None:
            self.item_id = item_id
            super().__init__()

    class EditCommitted(Message):
        """User saved an edit to a queue item."""

        def __init__(self, item_id: str, text: str) -> None:
            self.item_id = item_id
            self.text = text
            super().__init__()

    class DeleteRequested(Message):
        """User wants to delete a queue item."""

        def __init__(self, item_id: str) -> None:
            self.item_id = item_id
            super().__init__()

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
        self._items: list[QueueItemSnapshot] = []
        self._active_id: str | None = None
        self._editing_id: str | None = None  # locally tracked edit state
        self.display = False

    def compose(self) -> ComposeResult:
        yield Tabs(id="queue-tabs")
        with Horizontal(classes="queue-content"):
            yield Markdown("", classes="queue-display")
        with Horizontal(classes="queue-actions"):
            yield Button("Edit ✎", id="queue-edit-btn", classes="info-bar-right")
            yield Button("Delete ✕", id="queue-delete-btn", classes="info-bar-right")

    def reconcile(self, items: list[QueueItemSnapshot]) -> None:
        """Reconcile widget state from a broker queue snapshot.

        This is the only way the widget receives state. It diffs against
        current items and updates tabs/content accordingly.
        """
        if not items:
            self._items = []
            self._active_id = None
            self._editing_id = None
            self.display = False
            # Clear tabs
            tabs = self.query_one("#queue-tabs", Tabs)
            tabs.clear()
            self._show_empty()
            return

        old_ids = {i.id for i in self._items}
        new_ids = {i.id for i in items}
        self._items = list(items)

        tabs = self.query_one("#queue-tabs", Tabs)

        # Remove tabs for items no longer in queue
        for removed_id in old_ids - new_ids:
            tabs.remove_tab(removed_id)

        # Add tabs for new items
        for item in items:
            if item.id not in old_ids:
                label = self._tab_label(item)
                tabs.add_tab(Tab(label, id=item.id))
            else:
                # Update existing tab label
                try:
                    tab = tabs.query_one(f"#{item.id}", Tab)
                    tab.label = self._tab_label(item)
                except Exception:
                    pass

        # Set active to first item if current active is gone
        if self._active_id not in new_ids:
            self._active_id = items[0].id
            self._editing_id = None

        # Show the active item
        active_item = next((i for i in items if i.id == self._active_id), items[0])
        self._show_item(active_item)
        self.display = True

    @property
    def has_items(self) -> bool:
        """Whether queue has any items."""
        return bool(self._items)

    def _tab_label(self, item: QueueItemSnapshot) -> str:
        """Generate tab label from item source and text preview."""
        preview = item.text[:10].replace("\n", " ")
        ellipsis = "…" if len(item.text) > 10 else ""
        prefix = "MCP" if item.source == "mcp" else "User:"
        return f"{prefix} {preview}{ellipsis}"

    def _show_empty(self) -> None:
        """Clear the content area."""
        content = self.query_one(".queue-content", Horizontal)
        content.remove_children()

    def _show_item(self, item: QueueItemSnapshot) -> None:
        """Display the given item's content in the content area."""
        content = self.query_one(".queue-content", Horizontal)
        edit_btn = self.query_one("#queue-edit-btn", Button)
        content.remove_children()
        if self._editing_id == item.id:
            ta = PromptTextArea(classes="queue-display")
            content.mount(ta)
            ta.text = item.text
            edit_btn.label = "Save ✓"
        else:
            content.mount(Markdown(item.text, classes="queue-display"))
            edit_btn.label = "Edit ✎"

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle tab switch — show new item."""
        event.stop()
        tab_id = event.tab.id
        if tab_id is None:
            return
        # If switching away from edit, commit it first
        if self._editing_id and self._editing_id != tab_id:
            self._commit_current_edit()
        self._active_id = tab_id
        item = next((i for i in self._items if i.id == tab_id), None)
        if item:
            self._show_item(item)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Edit/Save and Delete button presses."""
        event.stop()
        if event.button.id == "queue-edit-btn":
            if not self._active_id:
                return
            if self._editing_id == self._active_id:
                # Save
                self._commit_current_edit()
            else:
                # Enter edit mode
                self._editing_id = self._active_id
                self.post_message(self.EditRequested(item_id=self._active_id))
                item = next((i for i in self._items if i.id == self._active_id), None)
                if item:
                    self._show_item(item)
        elif event.button.id == "queue-delete-btn":
            if self._active_id:
                self.post_message(self.DeleteRequested(item_id=self._active_id))

    def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        """Intercept Enter in edit mode — save instead of submitting."""
        event.stop()
        if self._editing_id:
            self._commit_current_edit()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Stop propagation of TextArea changes within the queue edit."""
        event.stop()

    def _commit_current_edit(self) -> None:
        """Save the current edit and post EditCommitted."""
        if not self._editing_id:
            return
        try:
            ta = self.query_one(".queue-content PromptTextArea", PromptTextArea)
            new_text = ta.text
        except Exception:
            new_text = ""
        item_id = self._editing_id
        self._editing_id = None
        self.post_message(self.EditCommitted(item_id=item_id, text=new_text))
        # Re-show in read mode
        item = next((i for i in self._items if i.id == item_id), None)
        if item:
            self._show_item(item)
