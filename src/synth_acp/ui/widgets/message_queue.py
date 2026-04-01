"""MCP messages panel with thread list and metadata detail."""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Collapsible, Static

from synth_acp.models.events import McpMessageDelivered

log = logging.getLogger(__name__)


class ThreadDetailHeader(Static):
    """Header bar showing participant names and message count."""

    def __init__(self) -> None:
        super().__init__("", id="thread-detail-header")


class ThreadDetail(Vertical):
    """Thread detail area: header + scrollable message metadata."""

    def __init__(self) -> None:
        super().__init__(id="thread-detail")

    def compose(self) -> ComposeResult:
        """Yield header and scrollable message list."""
        yield ThreadDetailHeader()
        yield ScrollableContainer(id="thread-detail-scroll")

    def show_thread(
        self,
        thread_key: tuple[str, str],
        messages: list[McpMessageDelivered],
    ) -> None:
        """Populate the detail pane with messages for a thread.

        Args:
            thread_key: Sorted (agent_a, agent_b) pair.
            messages: List of delivered messages in this thread.
        """
        a, b = thread_key
        n = len(messages)
        header = self.query_one("#thread-detail-header", ThreadDetailHeader)
        header.update(
            f" [$primary bold]{a}[/] [dim]→[/dim] [$primary bold]{b}[/]"
            f"  [dim]{n} message{'s' if n != 1 else ''}[/dim]"
        )

        scroll = self.query_one("#thread-detail-scroll", ScrollableContainer)
        scroll.remove_children()
        prev_from: str | None = None
        for msg in messages:
            ts = msg.timestamp.strftime("%H:%M")
            if msg.from_agent != prev_from:
                scroll.mount(
                    Static(
                        f"[$primary bold]● {msg.from_agent}[/] [dim]→ {msg.to_agent}   {ts}[/dim]",
                        classes="tmsg-from",
                    )
                )
            prev_from = msg.from_agent
            preview = msg.preview[:80] + "…" if len(msg.preview) > 80 else msg.preview
            if preview:
                scroll.mount(
                    Collapsible(
                        Static(msg.preview, classes="tmsg-body"),
                        title=preview,
                        collapsed=True,
                    )
                )
            else:
                scroll.mount(Static("[dim]delivered[/dim]", classes="tmsg-body"))


class ThreadItem(Static):
    """Clickable thread item showing agent pair, pending badge, and last timestamp.

    Args:
        thread_key: Sorted (agent_a, agent_b) pair.
        messages: Messages in this thread.
    """

    def __init__(
        self,
        thread_key: tuple[str, str],
        messages: list[McpMessageDelivered],
    ) -> None:
        self._thread_key = thread_key
        a, b = thread_key
        last = messages[-1]
        ts = last.timestamp.strftime("%H:%M")
        snippet = last.preview[:30] + "…" if len(last.preview) > 30 else last.preview
        preview_line = f"  [dim]{snippet}[/dim]" if snippet else f"  [dim]{ts}[/dim]"
        content = (
            f"[$primary bold]{a}[/] [dim]→[/dim] [$primary bold]{b}[/]\n{preview_line}"
        )
        key_id = f"titem-{a}-{b}"
        super().__init__(content, id=key_id, classes="thread-item")

    def on_click(self) -> None:
        """Select this thread in the parent MessageQueue."""
        from synth_acp.ui.app import SynthApp

        app = self.app
        if not isinstance(app, SynthApp):
                    return
        try:
            panel = app.query_one("#messages", MessageQueue)
            panel.show_thread(self._thread_key)
        except Exception:
            log.debug("Thread panel query failed", exc_info=True)


class MessageQueue(Vertical):
    """MCP messages panel with thread list (left) and detail pane (right).

    Args:
        threads: Thread data keyed by sorted agent pair.
    """

    def __init__(
        self,
        threads: dict[tuple[str, str], list[McpMessageDelivered]],
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="right-panel", **kwargs)
        self._threads = threads
        self._active_key: tuple[str, str] | None = None

    def compose(self) -> ComposeResult:
        """Yield header, thread list, and detail pane."""
        total = sum(len(msgs) for msgs in self._threads.values())
        yield Static(
            f" [bold]MCP Messages[/bold]  [dim]{total} messages[/dim]",
            classes="panel-header-static",
        )
        with Horizontal(id="msg-body"):
            with ScrollableContainer(id="thread-list"):
                for key, msgs in self._threads.items():
                    yield ThreadItem(key, msgs)
            yield ThreadDetail()

    async def update_threads(
        self, threads: dict[tuple[str, str], list[McpMessageDelivered]]
    ) -> None:
        """Rebuild the thread list from current data.

        Args:
            threads: Updated thread data.
        """
        self._threads = threads
        try:
            thread_list = self.query_one("#thread-list", ScrollableContainer)
        except Exception:
            log.debug("Thread list query failed", exc_info=True)
            return
        await thread_list.remove_children()
        for key, msgs in threads.items():
            thread_list.mount(ThreadItem(key, msgs))
        # Re-select active thread if still present
        if self._active_key and self._active_key in threads:
            self.show_thread(self._active_key)

    def show_thread(self, thread_key: tuple[str, str]) -> None:
        """Show a thread's detail and highlight it in the list.

        Args:
            thread_key: Sorted (agent_a, agent_b) pair.
        """
        self._active_key = thread_key
        # Clear active class from all thread items
        for item in self.query(".thread-item"):
            item.remove_class("thread-active")
        # Highlight selected
        a, b = thread_key
        try:
            self.query_one(f"#titem-{a}-{b}", ThreadItem).add_class("thread-active")
        except Exception:
            log.debug("Thread item highlight failed", exc_info=True)
        # Populate detail
        msgs = self._threads.get(thread_key, [])
        try:
            detail = self.query_one("#thread-detail", ThreadDetail)
            detail.show_thread(thread_key, msgs)
        except Exception:
            log.debug("Thread detail update failed", exc_info=True)
