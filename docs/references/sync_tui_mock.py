#!/usr/bin/env python3
"""
TEAM-ACP TUI — Visual Mock v4
Usage: uv run --with textual python team_acp_tui.py
   or: python team_acp_tui.py  (if textual is installed)
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Static

# ── Mock data ─────────────────────────────────────────────────────────────────

AGENTS = [
    {"id": "coordinator", "status": "IDLE"},
    {"id": "kiro-auth", "status": "BUSY"},
    {"id": "kiro-db", "status": "AWAITING_PERMISSION"},
    {"id": "researcher", "status": "IDLE"},
    {"id": "kiro-api", "status": "TERMINATED"},
]

STATUS_MARKUP = {
    "IDLE": "[green]●[/green]",
    "BUSY": "[yellow]●[/yellow]",
    "AWAITING_PERMISSION": "[bold yellow]●[/bold yellow]",
    "TERMINATED": "[dim]○[/dim]",
}

STATUS_LABEL = {
    "IDLE": "[dim green]IDLE[/dim green]",
    "BUSY": "[yellow]BUSY[/yellow]",
    "AWAITING_PERMISSION": "[bold yellow]AWAITING PERMISSION[/bold yellow]",
    "TERMINATED": "[dim]TERMINATED[/dim]",
}

AGENT_COLOR = {
    "coordinator": "#3b82f6",
    "kiro-auth": "#a78bfa",
    "kiro-db": "#f97316",
    "researcher": "#2dd4bf",
    "kiro-api": "#f472b6",
}

# Tool kind → (icon, color)  — matches JSX kindIcon / kindColor
TOOL_KIND_STYLE: dict[str, tuple[str, str]] = {
    "◎": ("◎", "#3b82f6"),  # read  → blue
    "✎": ("✎", "#a78bfa"),  # edit  → purple
    "⚡": ("⚡", "#f97316"),  # exec  → orange
    "✕": ("✕", "#f87171"),  # delete→ red
    "◈": ("◈", "#64748b"),  # other → gray
    "▶": ("▶", "#64748b"),  # fallback
}

# (kind, text, ts)  or  (kind, text, ts, extra, icon)
CONVERSATIONS: dict[str, list[tuple]] = {
    "coordinator": [
        ("you", "Spin up auth, db and api agents — refactor stack to JWT.", "10:40"),
        (
            "agent",
            "Understood. Dispatching to kiro-auth, kiro-db, and kiro-api now.",
            "10:40",
        ),
        ("tool", "send_message → kiro-auth", "10:40", "done", "◈"),
        ("tool", "send_message → kiro-db", "10:40", "done", "◈"),
        ("tool", "send_message → kiro-api", "10:41", "done", "◈"),
    ],
    "kiro-auth": [
        (
            "you",
            "Refactor the auth module to use JWT tokens instead of sessions.",
            "10:42",
        ),
        ("tool", "Reading src/auth/session.py", "10:42", "done", "◎"),
        ("tool", "Reading src/auth/middleware.py", "10:42", "done", "◎"),
        (
            "agent",
            "I've reviewed the session-based auth. Replacing with JWT using\npython-jose. Starting with the token generation module.",
            "10:43",
        ),
        ("tool", "Modifying src/auth/tokens.py", "10:43", "in_progress", "✎"),
    ],
    "kiro-db": [
        ("you", "Run pending migrations and add a refresh_tokens column.", "10:41"),
        ("tool", "Reading migrations/env.py", "10:41", "done", "◎"),
        (
            "agent",
            "I see 2 pending migrations. Adding refresh_tokens column now.",
            "10:42",
        ),
        ("tool", "$ alembic upgrade head", "10:43", "permission", "⚡"),
    ],
    "researcher": [
        ("you", "Research best practices for JWT refresh token rotation.", "10:39"),
        (
            "agent",
            "Best practices for JWT refresh token rotation:\n  1. Short-lived access tokens (15 min)\n  2. httpOnly refresh tokens (7 days)\n  3. Rotate token on every use\n  4. Token families to detect reuse attacks",
            "10:40",
        ),
    ],
    "kiro-api": [
        ("you", "Update API endpoints to validate JWT on every request.", "10:41"),
        ("agent", "Starting middleware update…", "10:41"),
    ],
}

MCP_THREADS = [
    {
        "id": "t1",
        "a": "coordinator",
        "b": "kiro-auth",
        "msgs": [
            (
                "coordinator",
                "kiro-auth",
                "Refactor auth to JWT. Pull researcher findings if needed.",
                "10:40",
                "delivered",
            ),
            (
                "kiro-auth",
                "coordinator",
                "Acknowledged. Starting token generation module now.",
                "10:43",
                "delivered",
            ),
        ],
    },
    {
        "id": "t2",
        "a": "coordinator",
        "b": "kiro-db",
        "msgs": [
            (
                "coordinator",
                "kiro-db",
                "Run pending migrations, add refresh_tokens to users table.",
                "10:40",
                "delivered",
            ),
            (
                "kiro-db",
                "coordinator",
                "On it. Will need permission to run alembic.",
                "10:42",
                "delivered",
            ),
        ],
    },
    {
        "id": "t3",
        "a": "coordinator",
        "b": "kiro-api",
        "msgs": [
            (
                "coordinator",
                "kiro-api",
                "Update API endpoints to validate JWT. Coordinate with kiro-auth on token schema.",
                "10:41",
                "delivered",
            ),
        ],
    },
    {
        "id": "t4",
        "a": "kiro-auth",
        "b": "researcher",
        "msgs": [
            (
                "kiro-auth",
                "researcher",
                "Do you have findings on JWT refresh token rotation?",
                "10:42",
                "delivered",
            ),
            (
                "researcher",
                "kiro-auth",
                "Yes — 15m access, 7d httpOnly refresh, rotate on use, track families.",
                "10:42",
                "delivered",
            ),
        ],
    },
    {
        "id": "t5",
        "a": "kiro-auth",
        "b": "kiro-api",
        "msgs": [
            (
                "kiro-auth",
                "kiro-api",
                "Using HS256. Payload: { sub, iat, exp, jti }. Schema in src/auth/tokens.py.",
                "10:43",
                "pending",
            ),
        ],
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _dot(status: str) -> str:
    return STATUS_MARKUP.get(status, "[dim]○[/dim]")


def _color(agent_id: str) -> str:
    return AGENT_COLOR.get(agent_id, "#94a3b8")


def _pending_mcp_count() -> int:
    return sum(1 for t in MCP_THREADS for m in t["msgs"] if m[4] == "pending")


def _active_agent_count() -> int:
    """Count agents that are not TERMINATED (matches JSX behaviour)."""
    return sum(1 for a in AGENTS if a["status"] != "TERMINATED")


# ── Message widgets ───────────────────────────────────────────────────────────


class MsgYou(Static):
    """Right-aligned user bubble — pushed right with margin-left."""

    DEFAULT_CSS = """
    MsgYou {
        height: auto;
        margin-left: 18;
        background: $background;
        border: round $primary;
        padding: 0 1;
        color: $foreground;
    }
    """

    def __init__(self, text: str, ts: str, **kwargs) -> None:
        super().__init__(f"{text}\n[dim]{ts}[/dim]", **kwargs)


class MsgAgent(Static):
    """Agent message: colored name header + indented text on $surface card."""

    DEFAULT_CSS = """
    MsgAgent {
        height: auto;
        margin-right: 8;
        background: $background;
        border: round $surface-lighten-2;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, agent_id: str, text: str, ts: str, **kwargs) -> None:
        color = _color(agent_id)
        lines = text.split("\n")
        body = "\n  ".join(lines)
        super().__init__(
            f"[bold {color}]● {agent_id}[/bold {color}]\n  {body}\n  [dim]{ts}[/dim]",
            **kwargs,
        )
        self.styles.border = ("round", color)


class MsgTool(Static):
    """Tool call row: icon + label, with kind-specific color and status badge."""

    DEFAULT_CSS = """
    MsgTool {
        height: auto;
        background: $background;
        border: round $surface-lighten-1;
        padding: 0 1;
        color: $text-muted;
    }
    MsgTool.tool-permission {
        border: round $warning;
    }
    """

    def __init__(
        self,
        icon: str,
        label: str,
        ts: str,
        badge: str,
        is_permission: bool = False,
        **kwargs,
    ) -> None:
        _icon_char, icon_color = TOOL_KIND_STYLE.get(icon, ("▶", "#64748b"))
        super().__init__(
            f"[{icon_color}]{icon}[/{icon_color}] {label}  [dim]{ts}[/dim]  {badge}",
            **kwargs,
        )
        if is_permission:
            self.add_class("tool-permission")


# ── Sidebar widgets ───────────────────────────────────────────────────────────


class AgentTile(Static):
    """Clickable agent tile in the sidebar."""

    def __init__(self, agent: dict, **kwargs) -> None:
        self.agent_data = agent
        color = _color(agent["id"])
        warn = (
            "  [bold yellow]⚠[/bold yellow]"
            if agent["status"] == "AWAITING_PERMISSION"
            else ""
        )
        dot = _dot(agent["status"])
        conv = CONVERSATIONS.get(agent["id"], [])
        last = next((m for m in reversed(conv) if m[0] in ("agent", "tool")), None)

        # Preview truncation — sidebar is 32 wide, ~26 usable chars for
        # the preview line after the 2-char indent.
        max_preview = 26

        if agent["status"] == "TERMINATED":
            preview = "[dim italic]terminated[/dim italic]"
        elif agent["status"] == "AWAITING_PERMISSION":
            preview = "[bold yellow italic]awaiting permission…[/bold yellow italic]"
        elif last and last[0] == "tool":
            t = last[1]
            preview = (
                f"[dim]{t[:max_preview]}{'…' if len(t) > max_preview else ''}[/dim]"
            )
        elif last:
            t = last[1].replace("\n", " ")
            preview = (
                f"[dim]{t[:max_preview]}{'…' if len(t) > max_preview else ''}[/dim]"
            )
        else:
            preview = "[dim italic]idle[/dim italic]"

        content = f"{dot} [bold {color}]{agent['id']}[/bold {color}]{warn}\n  {preview}"
        super().__init__(content, id=f"tile-{agent['id']}", **kwargs)
        if agent["status"] == "AWAITING_PERMISSION":
            self.add_class("tile-permission")

    def on_click(self) -> None:
        app = self.app
        assert isinstance(app, TeamACPApp)
        app.select_agent(self.agent_data["id"])


class LaunchButton(Static):
    """'+ launch agent' button at the bottom of the agent list (matches JSX)."""

    DEFAULT_CSS = """
    LaunchButton {
        height: 3;
        border: dashed $surface-lighten-1;
        color: $text-muted;
        text-align: center;
        content-align: center middle;
    }
    LaunchButton:hover {
        border: dashed $surface-lighten-3;
        color: $foreground;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("[dim]+ launch agent[/dim]", id="launch-btn", **kwargs)

    def on_click(self) -> None:
        self.app.notify(
            "Launch agent dialog — not implemented in mock", title="Mock UI"
        )


class MCPButton(Static):
    def __init__(self, **kwargs) -> None:
        count = _pending_mcp_count()
        badge = f"  [bold #f97316]{count}[/bold #f97316]" if count else ""
        super().__init__(f"◈  MCP Messages{badge}", id="mcp-btn", **kwargs)

    def on_click(self) -> None:
        app = self.app
        assert isinstance(app, TeamACPApp)
        app.show_messages()


# ── Agent panel ───────────────────────────────────────────────────────────────


class AgentPanel(Vertical):
    def __init__(self, agent_id: str, **kwargs) -> None:
        super().__init__(id=f"panel-{agent_id}", classes="right-panel", **kwargs)
        self.agent_id = agent_id

    def compose(self) -> ComposeResult:
        agent = next((a for a in AGENTS if a["id"] == self.agent_id), {})
        status = agent.get("status", "IDLE")
        color = _color(self.agent_id)

        # Header: dot + name + status + optional cancel
        cancel = "  [dim]✕ cancel[/dim]" if status == "BUSY" else ""
        yield Static(
            f" {_dot(status)} [bold {color}]{self.agent_id}[/bold {color}]"
            f"  {STATUS_LABEL.get(status, status)}{cancel}",
            classes="panel-header-static",
        )

        with ScrollableContainer(classes="conv-scroll"):
            yield from self._build_messages()

        if status == "AWAITING_PERMISSION":
            yield Static(
                "[bold yellow]⚠  Permission required[/bold yellow]\n"
                "[dim]$ alembic upgrade head[/dim]\n \n"
                " [bold green][ allow once ][/bold green]  "
                "[bold green][ always allow ][/bold green]  "
                "[bold red][ reject ][/bold red]  "
                "[bold red][ always reject ][/bold red]",
                classes="permission-box",
            )

        busy = status in ("BUSY", "AWAITING_PERMISSION")
        hint = (
            f"[italic dim]{self.agent_id} is {status.lower().replace('_', ' ')}…[/italic dim]"
            if busy
            else f"[italic dim]Message {self.agent_id}…[/italic dim]"
        )
        yield Static(f" [{color}]›[/{color}]  {hint}", classes="input-bar")

    def _build_messages(self):
        for entry in CONVERSATIONS.get(self.agent_id, []):
            kind = entry[0]
            text = entry[1]
            ts = entry[2]
            extra = entry[3] if len(entry) > 3 else None
            icon = entry[4] if len(entry) > 4 else "▶"

            if kind == "you":
                yield MsgYou(text, ts)

            elif kind == "agent":
                yield MsgAgent(self.agent_id, text, ts)

            elif kind == "tool":
                is_perm = extra == "permission"
                if extra == "done":
                    badge = "[green]✓[/green]"
                elif extra == "in_progress":
                    badge = "[yellow]⟳[/yellow]"
                elif is_perm:
                    badge = "[bold yellow]⏸[/bold yellow]"
                else:
                    badge = "[dim]·[/dim]"
                yield MsgTool(icon, text, ts, badge, is_permission=is_perm)


# ── MCP messages panel ────────────────────────────────────────────────────────


class ThreadItem(Static):
    def __init__(self, thread: dict, **kwargs) -> None:
        self.thread = thread
        a, b = thread["a"], thread["b"]
        ca, cb = _color(a), _color(b)
        pending = sum(1 for m in thread["msgs"] if m[4] == "pending")
        badge = f"  [bold #f97316]{pending}[/bold #f97316]" if pending else ""
        last = thread["msgs"][-1]
        lc = _color(last[0])
        preview = last[2][:30] + ("…" if len(last[2]) > 30 else "")
        content = (
            f"[bold {ca}]{a}[/bold {ca}] [dim]→[/dim] [bold {cb}]{b}[/bold {cb}]{badge}\n"
            f"  [dim {lc}]{last[0].split('-')[0]}:[/dim {lc}] [dim]{preview}[/dim]"
        )
        super().__init__(
            content, id=f"titem-{thread['id']}", classes="thread-item", **kwargs
        )
        if pending:
            self.add_class("thread-pending")

    def on_click(self) -> None:
        app = self.app
        assert isinstance(app, TeamACPApp)
        app.select_thread(self.thread["id"])


class ThreadDetailHeader(Static):
    """Header bar for the active thread: participants + message count (matches JSX)."""

    DEFAULT_CSS = """
    ThreadDetailHeader {
        height: 3;
        background: $background;
        border-bottom: tall $surface;
        padding: 0 1;
        content-align: left middle;
        color: $foreground;
    }
    """


class ThreadDetail(Vertical):
    """Thread detail area: header + scrollable messages."""

    DEFAULT_CSS = """
    ThreadDetail {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(id="thread-detail", **kwargs)

    def compose(self) -> ComposeResult:
        yield ThreadDetailHeader(id="thread-detail-header")
        yield ScrollableContainer(id="thread-detail-scroll")

    def show_thread(self, thread_id: str) -> None:
        thread = next((t for t in MCP_THREADS if t["id"] == thread_id), None)
        if not thread:
            return

        # Update header — matches JSX: AgentPill → AgentPill  N messages
        a, b = thread["a"], thread["b"]
        ca, cb = _color(a), _color(b)
        n = len(thread["msgs"])
        header = self.query_one("#thread-detail-header", ThreadDetailHeader)
        header.update(
            f" [bold {ca}]{a}[/bold {ca}] [dim]→[/dim] [bold {cb}]{b}[/bold {cb}]"
            f"  [dim]{n} message{'s' if n != 1 else ''}[/dim]"
        )

        # Update messages
        scroll = self.query_one("#thread-detail-scroll", ScrollableContainer)
        scroll.remove_children()
        prev_from = None
        for frm, to, text, ts, status in thread["msgs"]:
            fc = _color(frm)
            if frm != prev_from:
                scroll.mount(
                    Static(
                        f"[bold {fc}]● {frm}[/bold {fc}] [dim]→ {to}   {ts}[/dim]",
                        classes="tmsg-from",
                    )
                )
            prev_from = frm
            status_m = (
                "[dim]✓[/dim]"
                if status == "delivered"
                else "[bold #f97316]pending[/bold #f97316]"
            )
            scroll.mount(Static(f"{text}  {status_m}", classes="tmsg-body"))


class MessagesPanel(Vertical):
    def __init__(self, **kwargs) -> None:
        super().__init__(id="messages-panel", classes="right-panel", **kwargs)

    def compose(self) -> ComposeResult:
        count = _pending_mcp_count()
        yield Static(
            f" [bold]MCP Messages[/bold]  [dim]{count} pending[/dim]",
            classes="panel-header-static",
        )
        with Horizontal(id="msg-body"):
            with ScrollableContainer(id="thread-list"):
                for t in MCP_THREADS:
                    yield ThreadItem(t)
            yield ThreadDetail()


# ── App ───────────────────────────────────────────────────────────────────────


class TeamACPApp(App):
    TITLE = "TEAM-ACP"
    THEME = "catppuccin-mocha"

    CSS = """
    Screen { background: $background; layout: vertical; }

    /* ── Topbar ── */
    #topbar {
        height: 3;
        background: $background;
        border-bottom: tall $surface;
        layout: horizontal;
        align: left middle;
        padding: 0 2;
    }
    #tb-title   { width: auto; color: $primary; text-style: bold; padding-right: 1; }
    #tb-sep     { width: auto; color: $surface-lighten-2; padding: 0 1; }
    #tb-session { width: auto; color: $text-muted; }
    #tb-right   { width: 1fr; content-align: right middle; color: $text-muted; }

    /* ── Main layout ── */
    #main { height: 1fr; layout: horizontal; }

    /* ── Sidebar ── */
    #sidebar {
        width: 32;
        border-right: tall $surface;
        background: $background;
        layout: vertical;
    }
    #sidebar-label {
        height: 3;
        background: $background;
        border-bottom: tall $surface;
        padding: 0 2;
        color: $text-muted;
        content-align: center middle;
    }
    #agent-list { height: 1fr; overflow-y: auto; padding: 0 1; }

    AgentTile {
        height: auto;
        width: 1fr;
        padding: 0 1;
        background: $background;
        border: round $surface-lighten-1;
    }
    AgentTile:hover { border: round $surface-lighten-3; }
    AgentTile.tile-active {
        background: $background;
        border: round $primary;
    }
    AgentTile.tile-permission { border: round $warning; }
    AgentTile.tile-permission.tile-active { border: round $warning; }

    #mcp-btn {
        height: 3;
        background: $background;
        border-top: tall $surface;
        padding: 0 2;
        color: $text-muted;
        content-align: left middle;
    }
    #mcp-btn:hover { color: $foreground; }
    #mcp-btn.btn-active { border-top: tall $primary; color: $foreground; }

    /* ── Right panels ── */
    #right { height: 1fr; width: 1fr; layout: vertical; }

    .right-panel {
        height: 1fr;
        width: 1fr;
        layout: vertical;
        background: $background;
    }

    /* ── Panel header ── */
    .panel-header-static {
        height: 3;
        background: $background;
        border-bottom: tall $surface;
        padding: 0 1;
        content-align: left middle;
        color: $foreground;
    }

    /* ── Conversation scroll area ── */
    .conv-scroll { height: 1fr; padding: 0 1; }

    /* ── Permission box ── */
    .permission-box {
        height: auto;
        margin: 0 1;
        background: $background;
        border: round $warning;
        padding: 0 1;
        color: $text-muted;
    }

    /* ── Input bar ── */
    .input-bar {
        height: 3;
        background: $background;
        border-top: tall $surface;
        content-align: left middle;
        color: $text-muted;
        padding: 0 1;
    }

    /* ── Messages panel ── */
    #msg-body { height: 1fr; layout: horizontal; }
    #thread-list {
        width: 34;
        border-right: tall $surface;
        overflow-y: auto;
    }
    .thread-item {
        height: auto;
        padding: 0 1;
        margin: 0 1;
        border: round $surface-lighten-1;
    }
    .thread-item:hover { border: round $surface-lighten-3; }
    .thread-item.thread-active { border: round $primary; }
    .thread-item.thread-pending { border: round $warning; }
    .thread-item.thread-pending.thread-active { border: round $warning; }

    #thread-detail { height: 1fr; }
    #thread-detail-scroll { height: 1fr; padding: 0 1; overflow-y: auto; }
    .tmsg-from { height: auto; color: $text-muted; }
    .tmsg-body {
        height: auto;
        padding: 0 1;
        background: $background;
        border: round $surface-lighten-1;
        margin-bottom: 1;
        color: $text-muted;
    }

    Footer { background: $background; color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "next_agent", "Next agent", show=False),
        Binding("m", "messages", "MCP messages"),
        Binding("l", "launch", "Launch agent"),
        Binding("f1", "help", "Help"),
    ]

    selected_agent: reactive[str] = reactive("kiro-auth")
    selected_thread: reactive[str] = reactive("t1")

    def compose(self) -> ComposeResult:
        n_perm = sum(1 for a in AGENTS if a["status"] == "AWAITING_PERMISSION")
        n_active = _active_agent_count()

        with Horizontal(id="topbar"):
            yield Static("TEAM-ACP", id="tb-title")
            yield Static("│", id="tb-sep")
            perm_part = (
                f"  [dim]│[/dim]  [bold #f97316]⚠ {n_perm} awaiting permission[/bold #f97316]"
                if n_perm
                else ""
            )
            yield Static(f"session: dev-project{perm_part}", id="tb-session")
            yield Static(f"[dim]{n_active} active   F1 help[/dim]", id="tb-right")

        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Static("AGENTS", id="sidebar-label")
                with ScrollableContainer(id="agent-list"):
                    for a in AGENTS:
                        yield AgentTile(a)
                    yield LaunchButton()
                yield MCPButton()
            yield Vertical(id="right")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tile-kiro-auth", AgentTile).add_class("tile-active")
        self.query_one("#right").mount(AgentPanel("kiro-auth"))

    # ── Panel switching ───────────────────────────────────────────────────────

    def _swap_panel(self, new_widget: Widget) -> None:
        right = self.query_one("#right")
        for child in list(right.children):
            child.remove()
        right.mount(new_widget)

    def _clear_sidebar(self) -> None:
        for a in AGENTS:
            self.query_one(f"#tile-{a['id']}", AgentTile).remove_class("tile-active")
        self.query_one("#mcp-btn").remove_class("btn-active")

    def select_agent(self, agent_id: str) -> None:
        self._clear_sidebar()
        self.query_one(f"#tile-{agent_id}", AgentTile).add_class("tile-active")
        self._swap_panel(AgentPanel(agent_id))
        self.selected_agent = agent_id

    def show_messages(self) -> None:
        self._clear_sidebar()
        self.query_one("#mcp-btn").add_class("btn-active")
        panel = MessagesPanel()
        self._swap_panel(panel)
        self.call_after_refresh(self._load_thread, self.selected_thread)

    def _load_thread(self, thread_id: str) -> None:
        try:
            for t in MCP_THREADS:
                self.query_one(f"#titem-{t['id']}", ThreadItem).remove_class(
                    "thread-active"
                )
            self.query_one(f"#titem-{thread_id}", ThreadItem).add_class("thread-active")
            self.query_one(ThreadDetail).show_thread(thread_id)
        except Exception:
            pass

    def select_thread(self, thread_id: str) -> None:
        self._load_thread(thread_id)
        self.selected_thread = thread_id

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_next_agent(self) -> None:
        ids = [a["id"] for a in AGENTS]
        idx = ids.index(self.selected_agent) if self.selected_agent in ids else 0
        self.select_agent(ids[(idx + 1) % len(ids)])

    def action_messages(self) -> None:
        self.show_messages()

    def action_launch(self) -> None:
        self.notify("Launch agent dialog — not implemented in mock", title="Mock UI")

    def action_help(self) -> None:
        self.notify(
            "Tab: cycle agents   m: MCP messages   l: launch   q: quit",
            title="Keybindings",
            timeout=4,
        )


if __name__ == "__main__":
    TeamACPApp().run()
