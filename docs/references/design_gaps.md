# SYNTH — Gap Analysis & Feature Design

**Version:** 0.1-draft  
**Date:** 2026-03-24  
**Relates to:** `DESIGN.md` v0.2-draft

This document catalogs every ACP SDK capability and Textual feature that the current codebase does not leverage, with a recommended design for each and an explicit priority rating.

---

## Priority Legend

| Label | Meaning |
|---|---|
| 🔴 **Critical** | Correctness bug. Must fix before the tool is reliable. |
| 🟠 **High** | Significant missing capability that defines the product's core value. |
| 🟡 **Medium** | Meaningful improvement that's worth doing in Phase 2 or 3. |
| 🟢 **Low** | Nice-to-have polish. Defer until higher-priority work is stable. |

---

## Part I — ACP SDK Gaps

### ACP-1 · Unhandled `session_update` Types

**Priority: 🟠 High (thoughts), 🟠 High (usage), 🟡 Medium (plan), 🟢 Low (mode/commands)**

`ACPSession.session_update` handles exactly three of the eleven `session_update` variants the ACP Client protocol defines. The remaining eight fire silently and are dropped. Five of them matter in practice.

#### ACP-1a · `agent_thought_chunk` — 🟠 High

Agents like Kiro CLI and Claude Code emit internal reasoning as `AgentThoughtChunk` notifications before and during response generation. These are currently dropped entirely. For an orchestration dashboard, losing agent reasoning is a major observability gap — it's the primary signal for understanding whether an agent is on track or stuck.

**Design:**

Add a new broker event:

```python
class AgentThoughtReceived(BrokerEvent):
    """A streaming chunk of agent internal reasoning."""
    chunk: str
```

Handle in `session.py`:

```python
elif su == "agent_thought_chunk":
    content = getattr(update, "content", None)
    text = getattr(content, "text", None) if content else None
    if text:
        await self._event_sink(AgentThoughtReceived(agent_id=self.agent_id, chunk=text))
```

Add a `ThoughtBlock` widget (a `Collapsible` wrapping a `MarkdownStream`) to `ConversationFeed`. Mount it when the first thought chunk arrives for a turn; populate via streaming; collapse it when `TurnComplete` fires. Title it `"Thinking…"` during streaming, then `"Thought"` after finalization. Style with `color: $text-muted` and `border: round $surface` to visually separate it from the agent response.

The thought block should be mounted *above* the `AgentMessage` it precedes, mirroring how Claude Code and Kiro present reasoning in their own UIs.

---

#### ACP-1b · `usage_update` — 🟠 High

`UsageUpdate` carries `input_tokens`, `output_tokens`, `cached_read_tokens`, `cached_write_tokens`, and an optional `cost: Cost` object (ISO 4217 currency + amount). These fire at the end of every turn. For a multi-agent orchestrator, token burn rate and cumulative cost are mission-critical — a runaway agent can exhaust a budget before the operator notices.

**Design:**

Add a new broker event:

```python
class UsageUpdated(BrokerEvent):
    """Cumulative usage stats after a turn."""
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    cost_amount: float | None = None
    cost_currency: str | None = None
```

The broker should maintain per-agent cumulative totals by summing `UsageUpdated` events. Expose a `get_usage(agent_id)` method on `ACPBroker`.

In the TUI, populate the currently-empty `#tb-right` `Static` in the topbar with the selected agent's token count and running cost. Format as `[dim]32k tok  $0.14[/dim]`. Update it on every `UsageUpdated` for the currently-selected agent.

Additionally, update `AgentTile`'s preview line to show the last turn's token count when the agent is `IDLE`. This gives a quick scan of which agents are expensive.

---

#### ACP-1c · `plan` (`AgentPlanUpdate`) — 🟡 Medium

When agents emit a plan, SYNTH receives an `AgentPlanUpdate` with a `Plan` containing a list of `PlanEntry` objects. Each entry has:
- `content: str` — human-readable description of the task
- `priority: "high" | "medium" | "low"`
- `status: "pending" | "in_progress" | "completed"`

The entire plan is replaced on each update (not diffed). Kiro emits plans actively at the start of complex tasks.

**Design:**

Add a new broker event:

```python
class PlanUpdated(BrokerEvent):
    """Agent emitted or updated its execution plan."""
    entries: list[PlanEntry]
```

Add a `PlanPanel` widget that renders a `Tree` or formatted list of entries with status icons (`○` pending, `⟳` in_progress, `✓` completed) and priority color coding (red/yellow/dim). Mount it as a collapsible section at the top of `ConversationFeed`, hidden by default, shown when a `PlanUpdated` event arrives for that agent.

The plan panel replaces its content in full on each update (matching ACP semantics). It does not need to animate diffs.

---

#### ACP-1d · `current_mode_update` — 🟢 Low

Agents that support mode switching (e.g. Kiro's `auto` vs `ask` modes) broadcast their current mode via `CurrentModeUpdate`. Currently dropped.

**Design:**

Store `current_mode_id: str | None` in `ACPSession`. Emit a `SessionModeChanged` broker event. Display the mode as a small badge on `AgentTile` (e.g. `[dim]ask[/dim]`). This is cosmetic; the more important capability is *changing* the mode, which is covered in ACP-6.

---

#### ACP-1e · `available_commands_update` — 🟢 Low

Agents broadcast slash commands they accept via `AvailableCommandsUpdate`. Currently dropped.

**Design:**

Store the command list in `ACPSession`. This data feeds `InputBar` autocompletion (see Textual-9). Worth implementing only once the autocompletion feature is built — tracking it in isolation has no value.

---

### ACP-2 · `SessionAccumulator` Not Used

**Priority: 🟡 Medium**

`acp.contrib.session_state.SessionAccumulator` is a first-party SDK utility that merges all `SessionNotification` objects into a `SessionSnapshot` containing merged `tool_calls`, `plan_entries`, `current_mode_id`, `available_commands`, and the full chunk streams for user messages, agent messages, and agent thoughts.

The current `session.py` implements partial, manual equivalents of this logic: it handles `agent_message_chunk` and `tool_call` / `tool_call_update` by emitting events downstream. As ACP-1 issues are fixed and more update types are handled, this manual handling will grow in complexity and fragility.

**Design:**

Each `ACPSession` should own a `SessionAccumulator` instance:

```python
from acp.contrib.session_state import SessionAccumulator

class ACPSession:
    def __init__(self, ...):
        self._accumulator = SessionAccumulator()
```

Subscribe to the accumulator rather than switch-casing `session_update` strings:

```python
self._accumulator.subscribe(self._on_snapshot)

async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
    self._accumulator.apply(update)
    # _on_snapshot is called synchronously by the accumulator after apply

def _on_snapshot(self, snapshot: SessionSnapshot, notification: SessionNotification) -> None:
    asyncio.create_task(self._emit_from_snapshot(snapshot, notification))
```

The subscriber callback receives both the new snapshot and the raw notification that triggered it. Emit broker events based on `type(notification.update)` rather than string matching on `session_update`. This is more robust — it uses the SDK's typed discriminated union rather than duck-typing string attributes.

This refactor unblocks clean implementations of ACP-1c, ACP-1d, and ACP-1e because the accumulator handles all of them internally.

**Trade-off:** The accumulator buffers all chunk text in memory for the session lifetime. For very long sessions with large agent outputs this may grow. The raw event approach is more memory-efficient but requires more code. Accept the trade-off for now; it can be revisited if memory becomes a concern.

---

### ACP-3 · `InitializeResponse` Capabilities Discarded

**Priority: 🟠 High**

`conn.initialize()` returns an `InitializeResponse` containing `agent_capabilities: AgentCapabilities`. The return value is currently discarded:

```python
# session.py — current
await conn.initialize(protocol_version=1, client_capabilities=..., client_info=...)
# return value not captured
```

`AgentCapabilities` contains:

| Field | Type | Meaning |
|---|---|---|
| `load_session` | `bool` | Agent supports `session/load` for resumption |
| `mcp_capabilities.http` | `bool` | Agent supports HTTP MCP servers |
| `mcp_capabilities.sse` | `bool` | Agent supports SSE MCP servers |

Without capturing this, SYNTH cannot know whether session resumption is available (ACP-4), and it cannot adapt the MCP server injection strategy for agents that only support SSE (currently all agents are assumed to support stdio MCP servers).

**Design:**

Capture and store the capabilities on `ACPSession`:

```python
init_response = await conn.initialize(...)
self._capabilities = init_response.agent_capabilities
```

Add an `AgentCapabilities`-typed field to `ACPSession`. Expose it via the broker as part of a new `AgentInfo` model (alongside `AgentState`) returned by a `get_agent_info(agent_id)` method. Gate session resumption calls (ACP-4) on `capabilities.load_session`. Gate MCP server type selection on `capabilities.mcp_capabilities`.

---

### ACP-4 · No Session Persistence or Resumption

**Priority: 🟠 High**

Every time SYNTH starts (or an agent is re-launched), a new ACP session is created unconditionally via `conn.new_session(...)`. If an agent was mid-task and SYNTH was restarted — deliberately or due to a crash — all context is lost and the agent restarts from scratch.

The ACP SDK provides three session management methods on the `Agent` protocol that are never called:

- `conn.list_sessions(cwd)` — enumerate existing sessions in a working directory
- `conn.load_session(cwd, session_id, mcp_servers)` — reattach to an existing session
- `conn.resume_session(cwd, session_id, mcp_servers)` — resume a session that ended mid-turn

**Design:**

On `_launch` in `broker.py`, after the connection handshake and capability check (ACP-3), query existing sessions before creating a new one:

```python
if self._capabilities.load_session:
    sessions_response = await conn.list_sessions(cwd=agent_cfg.cwd)
    existing = sessions_response.sessions if sessions_response else []
else:
    existing = []

if existing:
    # Emit a new broker event asking the UI to decide
    await self._sink(SessionResumeAvailable(
        agent_id=agent_id,
        sessions=[s.session_id for s in existing],
    ))
    # Block until UI responds (similar to permission flow)
    choice = await self._session_resume_future[agent_id]
    if choice:
        session = await conn.load_session(cwd=agent_cfg.cwd, session_id=choice, mcp_servers=mcp_servers)
    else:
        session = await conn.new_session(cwd=agent_cfg.cwd, mcp_servers=mcp_servers)
else:
    session = await conn.new_session(cwd=agent_cfg.cwd, mcp_servers=mcp_servers)
```

The `SessionResumeAvailable` event triggers a `ModalScreen` in the TUI (see Textual-1) showing the available sessions with their IDs and timestamps. The user picks one or "Start fresh". This mirrors exactly how the permission system works: the broker blocks an async Future while the UI resolves it.

On graceful shutdown, persist the active `session_id` per agent to `~/.synth/sessions.json` (this path is already noted in the design doc under section 10.3 but never written). On next launch, if `load_session` is supported and a matching entry exists in `sessions.json`, default to resuming rather than prompting.

---

### ACP-5 · No `set_session_mode` or `set_session_model`

**Priority: 🟡 Medium**

The ACP `Agent` protocol exposes two configuration methods that are never called:

```python
await conn.set_session_mode(mode_id="auto", session_id=session_id)
await conn.set_session_model(model_id="claude-opus-4-5", session_id=session_id)
```

An orchestrator that can't change agent modes or models mid-session loses a significant control surface. Toggling Kiro between `auto` (autonomous) and `ask` (approval-seeking) is the canonical use case.

**Design:**

Add two new broker commands:

```python
class SetAgentMode(BrokerCommand):
    agent_id: str
    mode_id: str

class SetAgentModel(BrokerCommand):
    agent_id: str
    model_id: str
```

Handle them in `ACPBroker.handle()` by delegating to the corresponding `ACPSession` method. Gate on agent capabilities where applicable.

In the TUI, expose these via the command palette (see Textual-4) rather than adding UI chrome. "Set mode: auto", "Set mode: ask", "Switch model: claude-opus-4-5" are natural command palette entries. This avoids adding buttons or dropdowns to a UI that is already information-dense.

---

### ACP-6 · `set_config_option` and `config_option_update` Not Handled

**Priority: 🟢 Low**

`conn.set_config_option(config_id, value, session_id)` lets SYNTH set agent-specific configuration at runtime (verbosity, auto-approve patterns, etc.). The `config_option_update` session notification fires when the agent changes its own config internally. Neither is implemented.

**Design:**

Track config state in `ACPSession` using the `SessionAccumulator` (which handles `ConfigOptionUpdate` internally). Expose `get_config(agent_id, config_id)` on the broker for UI inspection.

For setting config from the TUI, use the command palette (`set config verbose true`). This requires knowing what config options an agent supports, which means also implementing `available_commands_update` (ACP-1e).

Defer this until ACP-2 (`SessionAccumulator`) and Textual-4 (command palette) are in place, since both are prerequisites for doing it cleanly.

---

### ACP-7 · `allow_always` / `reject_always` Never Persisted — Permission Engine Bug

**Priority: 🔴 Critical**

This is the only correctness bug in the current codebase. Two issues compound:

**Issue 1:** `PermissionDecision` only has `allow` and `reject`. The ACP schema defines four `PermissionOptionKind` values: `allow_once`, `allow_always`, `reject_once`, `reject_always`. The `_find_option_id` method in `broker.py` maps `allow` → `allow_once` and `reject` → `reject_once`. There is no way to auto-resolve using `allow_always` because that decision can never be stored.

**Issue 2:** `PermissionEngine.persist()` is never called anywhere. When a user clicks "Approve for session" (which sends an `allow_always` option_id), the broker correctly forwards it to the session via `resolve_permission()`, but the rule is never written to disk. On the next permission request of the same kind, `PermissionEngine.check()` returns `None` and the user is prompted again. The "persist rules" feature is built but disconnected.

**Design:**

Extend `PermissionDecision`:

```python
class PermissionDecision(StrEnum):
    allow_once = "allow_once"
    allow_always = "allow_always"
    reject_once = "reject_once"
    reject_always = "reject_always"
```

Update `_find_option_id` to map directly by kind string rather than translating:

```python
@staticmethod
def _find_option_id(options: list[PermissionOption], decision: PermissionDecision) -> str | None:
    for opt in options:
        if opt.kind == decision.value:
            return opt.option_id
    return None
```

Update `PermissionEngine.check` to return `allow_once` for `allow_always` hits (the auto-resolve fires with `allow_once` semantics, since `allow_always` means "auto-approve in the future", not "send a different option").

Call `PermissionEngine.persist()` in `_sink` when intercepting a `PermissionRequested` event and the option resolved is `allow_always` or `reject_always`:

```python
if decision in (PermissionDecision.allow_always, PermissionDecision.reject_always):
    self._permission_engine.persist(PermissionRule(
        agent_id=event.agent_id,
        tool_kind=event.kind,
        decision=decision,
    ))
```

Also add a `RespondPermission`-side hook: when the *user* chooses `allow_always` or `reject_always` in the TUI, the broker receives `RespondPermission` and should persist before forwarding:

```python
async def _resolve_permission(self, agent_id: str, option_id: str) -> None:
    session = self._sessions.get(agent_id)
    if not session:
        return
    # Find the option kind from the pending permission event
    option_kind = self._pending_permissions.get((agent_id, option_id))
    if option_kind in ("allow_always", "reject_always"):
        self._permission_engine.persist(PermissionRule(
            agent_id=agent_id,
            tool_kind=...,  # stored from PermissionRequested
            decision=PermissionDecision(option_kind),
        ))
    session.resolve_permission(option_id)
```

The broker needs a small in-memory map of pending permission requests (keyed by `agent_id`) to know the `tool_kind` at resolution time. It already has the data via the `PermissionRequested` event; it just needs to hold it until resolved.

---

### ACP-8 · Client Filesystem Callbacks Hardcoded to `False`

**Priority: 🟢 Low**

The ACP handshake advertises:

```python
ClientCapabilities(
    fs=FileSystemCapability(read_text_file=False, write_text_file=False),
    terminal=False,
)
```

If these are set to `True`, agents can route file operations and terminal sessions through SYNTH rather than accessing the filesystem directly. This matters for:
- Sandboxing: SYNTH can intercept and approve/reject file writes before they happen
- Remote operation: SYNTH running on a remote host with the UI on a local machine
- Audit logging: all file ops are visible in the broker event stream

**Design:**

Make these configurable in `.synth.json` at the session level:

```json
{
  "session": "my-project",
  "capabilities": {
    "fs": { "readTextFile": true, "writeTextFile": false }
  }
}
```

Implement `read_text_file` and `write_text_file` on `ACPSession`. `read_text_file` reads from the agent's `cwd`; `write_text_file` emits a `FileWriteRequested` broker event and optionally waits for operator approval (integrating with the permission system). 

Defer until session resumption and the permission system are solid. This is a power feature for advanced deployments, not a baseline requirement.

---

### ACP-9 · Terminal Management Callbacks Unimplemented

**Priority: 🟢 Low**

The ACP `Client` protocol includes five terminal management callbacks (`create_terminal`, `terminal_output`, `wait_for_terminal_exit`, `kill_terminal`, `release_terminal`). If an agent requests a terminal session, `ACPSession` has no implementation for these methods — the SDK will likely raise or return an error.

**Design:**

In the short term, add stub implementations on `ACPSession` that return `None` or a graceful error. This prevents unexpected exceptions if an agent requests a terminal:

```python
async def create_terminal(self, command: str, session_id: str, **kwargs: Any) -> None:
    log.warning("Terminal requested by %s but not supported", self.agent_id)
    return None
```

Full terminal support (actually running a terminal subprocess and streaming output back) is Phase 4 work. The TUI would need a new `TerminalPanel` widget backed by a PTY. This is non-trivial and not a requirement for the core orchestration use case.

---

### ACP-10 · Rich Prompt Content Types Unused

**Priority: 🟢 Low**

`conn.prompt()` accepts a heterogeneous content list:

```python
prompt: list[TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock]
```

SYNTH always sends `[text_block(text)]`. File attachments, image context, and resource references are not supported.

**Design:**

Add `@file:path` syntax to `InputBar` parsing. When the input contains `@file:/path/to/file.py`, read the file and include it as an `embedded_text_resource` block in the prompt. This is a natural extension of the existing `@agent-id` routing syntax.

Defer until the core feature set is stable. This is a quality-of-life feature for users who want to give agents file context without pasting content.

---

## Part II — Textual Gaps

### Textual-1 · No `ModalScreen` Implementations

**Priority: 🟠 High**

Three features are stub-implemented with `self.notify(...)` instead of a proper modal:

1. `action_launch` → `"Launch agent dialog — not yet implemented"`
2. `action_help` → a single-line notification listing keybindings
3. Session resume choice (ACP-4 above)

`dashboard.py` is an empty file — clearly a placeholder for a screen that was never built.

**Design:**

Textual's `ModalScreen` captures focus, blocks background interaction, and returns a typed result via `push_screen_wait()`. Use it for:

**`LaunchAgentScreen`:** A form showing unstarted agents from the config with a "Launch" button per agent. Agents already running are shown as disabled. Returns the selected `agent_id` or `None`:

```python
async def action_launch(self) -> None:
    result = await self.push_screen_wait(LaunchAgentScreen(self.config, launched=set(self._sessions)))
    if result:
        await self.broker.handle(LaunchAgent(agent_id=result))
```

**`HelpScreen`:** A proper keybinding reference rendered as a table, replacing the single-line notification. Include all bindings, the `@agent-id` routing syntax, and a note on slash commands. Use `ModalScreen` with an `Escape`-to-dismiss binding.

**`SessionResumeScreen`:** Used by ACP-4 when existing sessions are found. Shows a list of resumable sessions with IDs and timestamps. Returns the chosen session ID or `None` for "Start fresh".

Move these into `ui/screens/` to populate the currently empty directory.

---

### Textual-2 · `ContentSwitcher` Not Used — Manual Display Toggling

**Priority: 🟡 Medium**

`select_agent()` and `show_messages()` manually iterate all children of `#right` and set `display = False`, then show the target:

```python
for child in right.children:
    child.display = False
feed.display = True
```

This is exactly what `ContentSwitcher` does, natively and correctly. The manual approach has subtle issues: it doesn't handle newly mounted widgets, it doesn't participate in Textual's layout recalculation cleanly, and it makes the `selected_agent` reactive useless (see Textual-3).

**Design:**

Replace `#right Vertical` with a `ContentSwitcher`:

```python
# In compose()
yield ContentSwitcher(id="right")
```

Each `ConversationFeed` and `MessageQueue` panel is mounted into the switcher with a unique ID. Switching is a single reactive assignment:

```python
def watch_selected_agent(self, agent_id: str) -> None:
    self.query_one(ContentSwitcher).current = f"feed-{agent_id}"
```

`select_agent()` becomes a 3-line method. `show_messages()` similarly becomes trivial. The current ~80 lines of display-toggling logic across both methods collapses.

**Note:** The panel creation-on-first-selection pattern (lazy mounting + event buffer drain) is compatible with `ContentSwitcher` — mount the panel into the switcher on first access, then switch `current` to it.

---

### Textual-3 · `reactive` Declared but Never Watched

**Priority: 🟡 Medium**

```python
selected_agent: reactive[str] = reactive("")
selected_thread: reactive[str] = reactive("")
```

Both reactives are written to (`self.selected_agent = agent_id`) but no `watch_selected_agent` or `watch_selected_thread` methods exist. The reactive values are set but nothing observes them — all consequent logic is imperative, driven by calling `select_agent()` directly.

This is inconsistent with Textual's intended architecture. Reactive values exist to separate state mutation from state reaction. Any part of the app that needs to respond to agent selection currently has to be called manually from `select_agent()`.

**Design:**

Once Textual-2 (`ContentSwitcher`) is in place:

```python
def watch_selected_agent(self, agent_id: str) -> None:
    """React to agent selection: switch panel, update tile styles, update input state."""
    # Switch panel
    self.query_one(ContentSwitcher).current = f"feed-{agent_id}"
    # Update tile active styles
    for tile in self.query(AgentTile):
        tile.set_class(tile._agent_id == agent_id, "tile-active")
    # Update topbar session info
    self.query_one("#tb-session", Static).update(f"session: {self.config.session}  ·  {agent_id}")
```

`selected_thread` should similarly drive the `MessageQueue` thread detail panel reactively.

---

### Textual-4 · No Command Palette

**Priority: 🟡 Medium**

Textual has a built-in command palette activated by Ctrl+P. It is defined via a `COMMANDS` class variable on the `App` and populated by `Provider` subclasses that yield `Hit` objects for a search query.

SYNTH currently exposes its feature set only via the keybinding strip at the bottom (Footer). As more features are added (mode switching, model selection, config options, session management), the keybinding approach doesn't scale — there aren't enough keys.

**Design:**

```python
class SynthApp(App):
    COMMANDS = {SynthCommandProvider}
```

```python
class SynthCommandProvider(Provider):
    async def search(self, query: str) -> Hits:
        app = self.app
        assert isinstance(app, SynthApp)

        commands = [
            ("Launch agent", app.action_launch),
            ("Show MCP messages", app.action_messages),
            ("Show help", app.action_help),
        ]
        # Dynamic commands based on current agent state
        for agent in app.config.agents:
            state = app._agent_states.get(agent.id)
            if state == AgentState.IDLE:
                commands.append((f"Terminate {agent.id}", lambda a=agent.id: ...))
            if state is None or state == AgentState.TERMINATED:
                commands.append((f"Launch {agent.id}", lambda a=agent.id: ...))

        # Mode switching (ACP-5) once implemented
        # Model switching (ACP-5) once implemented

        matcher = self.matcher(query)
        for label, action in commands:
            score = matcher.match(label)
            if score > 0:
                yield Hit(score, matcher.highlight(label), action)
```

This provides a discoverable, keyboard-first interface for all features. It also decouples feature invocation from keybinding — features can be added to the palette without claiming a keyboard shortcut.

---

### Textual-5 · `LoadingIndicator` Missing During `INITIALIZING`

**Priority: 🟡 Medium**

When an agent is in `AgentState.INITIALIZING`, the selected conversation panel is empty — there is no feedback that anything is happening. The spinner only starts when the first message chunk arrives. This can look like the agent failed to launch.

**Design:**

In `ConversationFeed.compose()`, yield a `LoadingIndicator` inside the scroll container:

```python
def compose(self):
    with ScrollableContainer(classes="conv-scroll"):
        yield LoadingIndicator(id="loading-spinner")
    yield InputBar(self._agent_id, self._color)
```

The spinner is visible by default. In `SynthApp._route_event_to_feed()`, when `AgentStateChanged` arrives with `new_state == IDLE`, hide the spinner:

```python
elif isinstance(event, AgentStateChanged) and event.new_state == AgentState.IDLE:
    try:
        feed.query_one("#loading-spinner", LoadingIndicator).display = False
    except NoMatches:
        pass
```

This costs two lines. The visual improvement for first-launch UX is significant — users know the agent is starting rather than wondering if their config is broken.

---

### Textual-6 · `ThoughtBlock` Should Use `Collapsible`

**Priority: 🟡 Medium** *(Depends on ACP-1a)*

This is the TUI design for displaying `agent_thought_chunk` events (ACP-1a). Calling it out separately as the Textual-specific design.

**Design:**

```python
class ThoughtBlock(Collapsible):
    """Collapsible agent reasoning block."""

    def __init__(self, agent_id: str, color: str) -> None:
        super().__init__(title="Thinking…", collapsed=True, classes="thought-block")
        self._stream: MarkdownStream | None = None

    def compose(self) -> ComposeResult:
        yield Markdown("", id="thought-content")

    async def append_chunk(self, chunk: str) -> None:
        md = self.query_one("#thought-content", Markdown)
        if self._stream is None:
            self._stream = Markdown.get_stream(md)
            self.collapsed = False  # Expand while streaming
        await self._stream.write(chunk)

    async def finalize(self) -> None:
        if self._stream:
            await self._stream.stop()
            self._stream = None
        self.title = "Thought"
        self.collapsed = True  # Collapse when done
```

Mount `ThoughtBlock` in `ConversationFeed` before the `AgentMessage` it precedes, using the same lazy-creation pattern as `AgentMessage`. The `Collapsible` widget handles expand/collapse toggle natively; no additional event handling required.

Style with `border: round $surface` and `color: $text-muted` at 70% opacity to make clear it is internal reasoning, not output.

---

### Textual-7 · `@on` Decorator Not Used — Potential Event Bubbling Issues

**Priority: 🟢 Low**

`PermissionRequest.on_button_pressed` matches any `Button.Pressed` event that reaches the widget. Currently this is safe because there's only one button type. But as more `Button` instances are added to the feed (e.g. inline "Retry" buttons, tool call expand/collapse controls), button presses could unintentionally match.

**Design:**

Replace the string-prefix check with a scoped `@on` handler:

```python
# Before
def on_button_pressed(self, event: Button.Pressed) -> None:
    option_id = event.button.id
    if option_id and option_id.startswith("perm-btn-"):
        option_id = option_id[len("perm-btn-"):]
        ...

# After
@on(Button.Pressed, ".permission-box Button")
def handle_permission_button(self, event: Button.Pressed) -> None:
    option_id = event.button.id.removeprefix("perm-btn-")
    ...
```

Low urgency because the current code works. Apply during a general refactor pass rather than as standalone work.

---

### Textual-8 · `AgentTile` Inherits `Static` Instead of `Widget`

**Priority: 🟢 Low**

`AgentTile(Static)` handles click events and manages interactive state, but `Static` is semantically a non-interactive display widget. The practical consequence is that `AgentTile` cannot receive keyboard focus — users cannot navigate agents with arrow keys or select an agent with Enter/Space.

**Design:**

Change the base class to `Widget` with `can_focus=True`:

```python
class AgentTile(Widget, can_focus=True):
    ...

    def on_key(self, event: events.Key) -> None:
        if event.key in ("enter", "space"):
            self.app.run_worker(self.app.select_agent(self._agent_id))

    DEFAULT_CSS = """
    AgentTile:focus { border: round $border; }
    """
```

With this change, `Tab` cycling can navigate individual tiles rather than cycling through all agents wholesale (the current `action_next_agent` behavior). The existing `on_click` is unchanged.

---

### Textual-9 · `InputBar` Has No `@agent-id` Autocompletion

**Priority: 🟢 Low**

The `@agent-id` routing prefix is parsed manually from raw text, but there's no completion or feedback while typing. Users must remember agent IDs exactly.

`TextArea` does not natively support suggestions, but the routing prefix is a single token at the start of the message. One option is to detect `@` at position 0 and show a floating suggestion list.

**Design:**

Use Textual's `Tooltip` or a custom overlay `ListView` that appears below the input when `@` is the first character. Populate with matching agent IDs from `app.config.agents`. Dismiss on selection or on any non-`@` prefix.

Alternatively, consider a dedicated agent selector `Select` widget in the `InputBar` alongside the `TextArea` rather than inline syntax. The `Select` widget is built-in and handles the routing concern explicitly — the text area then always sends to the selected agent, with `@agent-id` as a power-user override. This is simpler to implement and more discoverable.

Defer until core features are stable. The current `@agent-id` syntax works; this is ergonomic polish.

---

### Textual-10 · Worker Error Handling Missing

**Priority: 🟠 High**

`run_worker(self._consume_broker_events(), exit_on_error=False)` silently swallows worker exceptions. If `_consume_broker_events()` throws (e.g., a broker shutdown race, a pydantic validation error on an unexpected event shape), the TUI goes silent — no more events are processed, no agents respond, but the UI stays up showing nothing is wrong.

**Design:**

Add `on_worker_state_changed` to `SynthApp`:

```python
def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
    if event.state == WorkerState.ERROR and event.worker.error is not None:
        self.notify(
            f"Internal error: {event.worker.error}",
            severity="error",
            title="SYNTH worker error",
            timeout=0,  # Stay until dismissed
        )
        # Attempt restart if it's the broker consumer
        if event.worker.name == "broker-consumer":
            self.run_worker(
                self._consume_broker_events(),
                exit_on_error=False,
                name="broker-consumer",
            )
```

Name the broker consumer worker (`name="broker-consumer"`) so it can be identified and restarted. This is a one-time error — if broker events fail repeatedly, something is fundamentally broken and the user should see that.

---

### Textual-11 · `RichLog` Not Used for High-Volume Output

**Priority: 🟢 Low**

Tool call blocks, MCP notifications, permission auto-resolved messages, and error notifications are each mounted as individual `Static` widgets. In a busy multi-agent session with many concurrent tool calls, the conversation feed may contain hundreds of `Static` widgets, each requiring layout calculation on mount.

`RichLog` is designed for this: it renders all entries in a single widget, uses virtual scrolling, and supports `max_lines` to prevent unbounded growth.

**Design:**

This is an optimization, not a correctness fix. The current approach works correctly for expected session lengths. Revisit if performance becomes observable — measured by frame rate degradation in sessions with 100+ tool calls.

If implemented, `RichLog` replaces the individual `ToolCallBlock` and MCP notification widgets in `ConversationFeed` with a single log pane. However, `ToolCallBlock` currently needs in-place mutation (status updates after mount). `RichLog` does not support mutation of existing entries — the entry would need to be re-written. This makes the swap non-trivial and is further reason to defer.

---

## Summary Table

| ID | Area | Priority | One-Line Summary |
|---|---|---|---|
| ACP-7 | ACP | 🔴 Critical | `allow_always` / `reject_always` never persisted — permission rules don't work |
| ACP-1a | ACP | 🟠 High | `agent_thought_chunk` dropped — no agent reasoning visible |
| ACP-1b | ACP | 🟠 High | `usage_update` dropped — no token count or cost visibility |
| ACP-3 | ACP | 🟠 High | `InitializeResponse` discarded — agent capabilities unknown |
| ACP-4 | ACP | 🟠 High | No `load_session` / `list_sessions` — session context lost on every restart |
| Textual-1 | TUI | 🟠 High | No `ModalScreen` — launch dialog and help are notification stubs |
| Textual-10 | TUI | 🟠 High | Worker error swallowed silently — broker consumer failures are invisible |
| ACP-2 | ACP | 🟡 Medium | `SessionAccumulator` unused — manual event dispatch growing in fragility |
| ACP-1c | ACP | 🟡 Medium | `AgentPlanUpdate` dropped — no plan visibility |
| ACP-5 | ACP | 🟡 Medium | No `set_session_mode` / `set_session_model` — can't control agents at runtime |
| Textual-2 | TUI | 🟡 Medium | Manual display toggling instead of `ContentSwitcher` |
| Textual-3 | TUI | 🟡 Medium | `reactive` declared but never watched — state/reaction coupling is manual |
| Textual-4 | TUI | 🟡 Medium | No command palette — feature discoverability doesn't scale with features |
| Textual-5 | TUI | 🟡 Medium | No `LoadingIndicator` during `INITIALIZING` — launch looks like a hang |
| Textual-6 | TUI | 🟡 Medium | `ThoughtBlock` design for ACP-1a (depends on ACP-1a) |
| ACP-6 | ACP | 🟢 Low | `set_config_option` / `config_option_update` unimplemented |
| ACP-8 | ACP | 🟢 Low | Filesystem callbacks hardcoded `False` — no sandboxing or remote file ops |
| ACP-9 | ACP | 🟢 Low | Terminal callbacks unimplemented — agents requesting terminals will error |
| ACP-10 | ACP | 🟢 Low | Prompt content types limited to text — no file attachment support |
| ACP-1d | ACP | 🟢 Low | `current_mode_update` dropped |
| ACP-1e | ACP | 🟢 Low | `available_commands_update` dropped |
| Textual-7 | TUI | 🟢 Low | `@on` scoping not used — latent event bubbling risk |
| Textual-8 | TUI | 🟢 Low | `AgentTile` extends `Static` — no keyboard focus on tiles |
| Textual-9 | TUI | 🟢 Low | No `@agent-id` autocompletion in `InputBar` |
| Textual-11 | TUI | 🟢 Low | Individual `Static` mounts vs `RichLog` for high-volume output |

---

## Recommended Implementation Order

### Immediate (Fix Before Wider Use)

1. **ACP-7** — Fix the permission persistence bug. This is a correctness issue that affects any session longer than a few turns.
2. **Textual-10** — Add worker error handler. Silent failures in the broker consumer make the tool appear broken with no diagnosis path.

### Phase 2 — Core Capability

3. **ACP-3** — Capture `InitializeResponse` capabilities. Prerequisite for ACP-4.
4. **ACP-4** — Implement session resume via `list_sessions` / `load_session`. Transforms SYNTH from a session starter into a session manager.
5. **ACP-1a + Textual-6** — Handle `agent_thought_chunk` and render `ThoughtBlock`. The highest-visibility missing feature for observability.
6. **ACP-1b** — Handle `usage_update` and display in topbar. Token cost visibility is essential for multi-agent workloads.
7. **Textual-1** — Implement `LaunchAgentScreen` and `HelpScreen` as proper modals. The stubs undermine the polish of the rest of the UI.
8. **Textual-5** — Add `LoadingIndicator`. Very low effort, high impact on first-launch UX.

### Phase 2 — Refactor

9. **ACP-2** — Adopt `SessionAccumulator`. Refactor cleans up session.py and unblocks ACP-1c, ACP-1d, ACP-1e for free.
10. **Textual-2 + Textual-3** — `ContentSwitcher` + reactive watchers. Reduces app.py complexity significantly and aligns with Textual's design model.

### Phase 3 — Control Surface

11. **ACP-5** — `set_session_mode` / `set_session_model` + Textual-4 command palette. These are co-dependent: mode/model switching is only usable if exposed through the command palette.
12. **ACP-1c** — Plan panel. Valuable but requires `SessionAccumulator` (ACP-2) first.
13. **Textual-8** — `AgentTile` keyboard focus. Enables full keyboard navigation.

### Phase 4 — Polish and Advanced Features

14. ACP-6 · ACP-8 · ACP-10 · Textual-7 · Textual-9 · Textual-11
