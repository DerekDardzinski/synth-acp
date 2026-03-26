# SYNTH Design Document

**SYNTH** — **SY**nchronized **N**etwork of **T**eamed **H**arnesses over ACP

Version: 0.4
Date: 2026-03-25

---

## 1. Overview

SYNTH is a multi-agent orchestration dashboard that manages teams of AI coding agents through the Agent Client Protocol (ACP). It provides a Textual-based TUI for real-time session management, streaming markdown output, permission handling, and inter-agent messaging.

### Project Identity

| Attribute | Value |
|---|---|
| PyPI package | `synth_acp` |
| CLI command | `synth` |
| MCP server entrypoint | `synth-mcp` |
| Config file | `.synth.toml` (`.synth.json` for backward compat) |
| Data directory | `~/.synth/` |
| Repo name | `synth-acp` |

### Design Principles

- **Feel like the underlying tool, not a wrapper.** Running `synth` should feel as natural as running `kiro-cli` or `claude` directly. Multi-agent capability is additive.
- **Config file is an output, not a prerequisite.** First-run interactive setup guides the user; the resulting `.synth.toml` is a project artifact worth committing.
- **The orchestrator is primary.** The default workflow is one orchestrator that the user talks to directly, which dynamically spawns workers as needed. Static team configs are supported but secondary.

### Goals

- Manage teams of ACP-compatible agents (Kiro CLI, Claude Code, Gemini CLI, etc.) from a single process
- Surface permission requests to the operator in real time with persistent auto-resolve rules
- Stream agent responses with rendered markdown, thought blocks, and token usage visibility
- Support flexible agent topologies: human-dispatch, orchestrator, peer-to-peer
- Enable agent-to-agent communication via a bundled MCP server
- Support session resumption across restarts

### Non-Goals

- Building a new AI agent (SYNTH manages existing agents)
- Supporting non-ACP agents (no tmux fallback)
- Multi-user authentication (single operator, single session)

---

## 2. Architecture

### 2.1 System Diagram

```
synth (single process: broker + TUI)
┌──────────────────────────────────────────────────┐
│                                                  │
│  CLI (argparse)                                  │
│    ├── first-run picker (no config)              │
│    ├── --harness/--agent (transient config)      │
│    └── .synth.toml / .synth.json (project config)│
│                                                  │
│  ACPBroker ─────────────────────────────────┐    │
│    ├── ACPSession (agent-1)                 │    │
│    ├── ACPSession (agent-2)                 │    │
│    ├── ACPSession (agent-N)                 │    │
│    ├── PermissionEngine (persisted rules)   │    │
│    └── MessagePoller (SQLite watcher)       │    │
│              │                              │    │
│              │ AsyncIterator[BrokerEvent]    │    │
│              ▼                              │    │
│  SynthApp (Textual) ◄─────────────────────┘    │
│    ├── AgentList (sidebar)                       │
│    ├── ConversationFeed (per agent, lazy)        │
│    │     ├── PromptBubble                        │
│    │     ├── AgentMessage (MarkdownStream)       │
│    │     ├── ThoughtBlock (Collapsible)          │
│    │     ├── ToolCallBlock                       │
│    │     └── PermissionRequest                   │
│    ├── MessageQueue                              │
│    └── InputBar                                  │
│                                                  │
└──────────────────────────────────────────────────┘
     │ spawns N agent subprocesses via ACP stdio
     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ kiro-cli │  │ kiro-cli │  │ claude   │
│   acp    │  │   acp    │  │   mcp    │
│ MCP:     │  │ MCP:     │  │ MCP:     │
│ synth-mcp│  │ synth-mcp│  │ synth-mcp│
└──────────┘  └──────────┘  └──────────┘
      │              │              │
      └──────────────┴──────────────┘
                     │
              SQLite (messages table)
```

### 2.2 Three-Layer Design

| Layer | Package | Responsibility | Imports from |
|---|---|---|---|
| 3 — Frontend | `synth_acp.ui` | Textual TUI rendering | `models`, `broker` |
| 2 — Broker | `synth_acp.broker` | Session lifecycle, routing, permissions | `models`, `acp` |
| 1 — ACP | `synth_acp.acp` | ACP SDK wrapper, subprocess management | `models` |
| Shared | `synth_acp.models` | Pydantic models for events, commands, config | (none) |

Layers 1 and 2 have zero Textual imports. The frontend communicates with the broker exclusively through typed events and commands defined in `models/`.

### 2.3 Why a Single Process

The broker, UI, and event loop all run in one Python process:

- The broker owns all agent stdio channels. If it dies, all agents die. No failure mode where UI survives but broker doesn't.
- Textual's event loop is asyncio. Running broker and UI in the same loop avoids cross-process IPC for the hot path.
- A future web UI replaces `SynthApp` with a FastAPI/websocket server in the same process, consuming the same `broker.events()` / `broker.handle()` interface.

---

## 3. Component Design

### 3.1 ACPSession (Layer 1)

Wraps the `agent-client-protocol` SDK's `spawn_agent_process` for a single agent subprocess. Implements the SDK's `Client` protocol via duck typing.

Responsibilities:
- Spawn agent subprocess and perform ACP handshake (`initialize` → `session/new`)
- Capture `InitializeResponse` agent capabilities (load_session, mcp_capabilities)
- Track session state via a finite state machine
- Forward `session_update` notifications as typed broker events
- Handle `request_permission` via Future-based blocking
- Emit `TurnComplete` on prompt completion

```python
class ACPSession:
    agent_id: str
    state: AgentState
    _conn: Any                 # from spawn_agent_process
    _event_sink: EventSink
    _permission_future: asyncio.Future[str] | None = None
    _capabilities: AgentCapabilities | None = None
```

#### State Machine

```
UNSTARTED → INITIALIZING → IDLE → BUSY → IDLE → ... → TERMINATED
                                    ↑
                              AWAITING_PERMISSION
```

Transitions are enforced at runtime. State change notifications are `await`ed (not fire-and-forget) to prevent race conditions where the broker's view of agent state is stale when the first streaming chunk arrives.

#### Session Update Handling

Currently handles `agent_message_chunk`, `tool_call`, `tool_call_update`, `agent_thought_chunk`, and `usage_update`:

| Update Type | Status | Priority |
|---|---|---|
| `agent_message_chunk` | ✅ Implemented | — |
| `tool_call` / `tool_call_update` | ✅ Implemented | — |
| `agent_thought_chunk` | ✅ Implemented | — |
| `usage_update` | ✅ Implemented | — |
| `plan` | ❌ Not handled | 🟡 Medium |
| `current_mode_update` | ❌ Not handled | 🟢 Low |
| `available_commands_update` | ❌ Not handled | 🟢 Low |

#### ACP Client Capabilities

SYNTH declares filesystem and terminal capabilities as `False`:

```python
ClientCapabilities(
    fs=FileSystemCapability(read_text_file=False, write_text_file=False),
    terminal=False,
)
```

Target harnesses (Kiro CLI, Claude Code, Gemini CLI) use their own file editing and shell execution tools. They report results via `session/update` notifications with `ToolCallContentDiff` and `ToolCallContentTerminal`. SYNTH receives full diff data and terminal output without touching the filesystem.

#### Subprocess Crash Handling

The `run()` method wraps `spawn_agent_process` in try/finally. On any exit:
1. Cancel pending permission Future
2. Transition to TERMINATED
3. Emit `BrokerError` with details

#### Permission Handling

When the SDK calls `request_permission()`:
1. Create `asyncio.Future[str]` and store as `_permission_future`
2. Transition to `AWAITING_PERMISSION`
3. Emit `PermissionRequested` event (Future stays on session, not on event)
4. Await the Future
5. On resolution: return `AllowedOutcome` with the selected `option_id`
6. On cancellation: return `DeniedOutcome(outcome="cancelled")`
7. Transition back to `BUSY`

The broker resolves the Future via `session.resolve_permission(option_id)`.

### 3.2 ACPBroker (Layer 2)

Central orchestration service. Owns all `ACPSession` instances.

```python
class ACPBroker:
    _sessions: dict[str, ACPSession]
    _event_queue: asyncio.Queue[BrokerEvent]
    _config: SessionConfig
    _permission_engine: PermissionEngine
    _poller: MessagePoller | None
    _session_id: str  # "{config.project}-{uuid4.hex[:8]}"
    _agent_parents: dict[str, str | None]  # agent_id → parent agent_id (None for config-defined)
    _pending_initial_prompts: dict[str, str]  # agent_id → message (consumed on first IDLE)
```

Key behaviors:
- `handle(command)` — match/case dispatch for all `BrokerCommand` subclasses
- `events()` — async iterator over the event queue, terminates on shutdown
- `_sink(event)` — intercepts `PermissionRequested` for auto-resolve before forwarding to queue; watches for `AgentStateChanged(new_state=IDLE)` to send initial prompts for dynamically launched agents
- `get_agent_states()` — returns `{agent_id: state}` for all launched sessions
- `get_agent_configs()` — returns all `AgentConfig` from the session config
- Prompt dispatch is fire-and-forget (`create_task`) so `handle(SendPrompt)` doesn't block
- Pre-registers all config agents in SQLite at startup so `list_agents` works immediately
- MCP server config injected into each agent's `session/new` using `McpServerStdio` + `EnvVariable`
- `_process_commands(commands)` — `CommandFn` implementation; dispatches launch/terminate commands from the `agent_commands` table
- `_handle_launch_command` — resolves harness via `load_harness_registry()`, validates agent_id, checks `SYNTH_MAX_AGENTS` limit, spawns `ACPSession` with parentage tracking
- `_handle_terminate_command` — enforces parentage (only parent can terminate children), orphans children by setting their `parent` to NULL
- `_build_mcp_env` — constructs `EnvVariable` list for `McpServerStdio` including `SYNTH_COMMUNICATION_MODE`, `SYNTH_MAX_AGENTS`, and `AgentConfig.env`
- `_get_visible_agents_for(agent_id)` — computes visibility from in-memory `_agent_parents` (used for join broadcasts without DB round-trip)
- `_send_join_broadcast(agent_id, task)` — inserts system messages into `messages` table for all visible agents

### 3.3 PermissionEngine

Session-scoped permission engine backed by SQLite (`~/.synth/synth.db`). Keyed on `(agent_id, tool_kind, session_id)`.

```python
class PermissionEngine:
    def __init__(self, db_path: Path, session_id: str) -> None: ...
    def check(self, agent_id: str, tool_kind: str, session_id: str) -> PermissionDecision | None: ...
    def persist(self, rule: PermissionRule) -> None: ...
```

`PermissionDecision` has four values: `allow_once`, `allow_always`, `reject_once`, `reject_always`. Cache starts empty on each new session — no pre-loading from SQLite. `persist()` writes to both in-memory cache and SQLite `rules` table. `check()` returns the stored decision as-is (no translation). The SQLite write stores the per-run `session_id` for future session resume support.

### 3.4 MessagePoller

Polls SQLite via `PRAGMA data_version` at 100ms intervals using a persistent `aiosqlite` connection.

Key behaviors:
- Accepts a `DeliverFn` callback (not a broker reference) to avoid circular dependency
- Accepts an optional `CommandFn` callback for processing `agent_commands` rows
- Combined delivery: multiple pending messages for one agent become a single `session/prompt`
- Two-phase status: messages marked `delivered` only after `session/prompt` succeeds
- Only delivers to IDLE agents; leaves messages pending for BUSY agents
- Initial sweep on startup to catch messages from before the poller started
- Session-scoped queries (ignores messages from other sessions)
- On each poll cycle where `data_version` changed: calls `_deliver_pending` then `_process_pending_commands`
- `stop()` awaits current poll cycle completion (guarantees no in-flight deliveries)

```python
DeliverFn = Callable[[str, str, list[str]], Awaitable[bool]]
CommandFn = Callable[[list[tuple[int, str, str, str]]], Awaitable[None]]
#                     list of (id, from_agent, command, payload)
```

### 3.5 MCP Server (`synth-mcp`)

FastMCP server using `mcp.server.fastmcp.FastMCP` (from the `mcp` package, not a separate `fastmcp` package). Each agent spawns its own instance. All instances share the same SQLite database.

Tools provided:

| Tool | Description |
|---|---|
| `send_message` | Send a message to a visible teammate (`to_agent="*"` broadcasts to visible set) |
| `check_delivery` | Check delivery status of a sent message |
| `list_agents` | List visible agents with status, parent, and task |
| `deregister_agent` | Mark this agent as inactive (does not delete the row) |
| `launch_agent` | Request the broker to spawn a new agent (caller becomes parent) |
| `terminate_agent` | Request the broker to terminate a child agent |

`pull_messages` was removed — the broker's poller handles all delivery. Agents receive messages between turns automatically via `session/prompt`.

Auto-registers the agent on first tool call via `INSERT OR IGNORE`. WAL mode enabled.

#### Communication Modes

`_get_visible_agents()` controls what each agent can see and message:

- **MESH** (default): all active agents except self
- **LOCAL**: parent, children, and siblings (agents sharing the same parent) only

Mode is set via `SYNTH_COMMUNICATION_MODE` env var (passed to each MCP server instance). `list_agents` returns only visible agents. `send_message` validates target is in visible set. Broadcast `"*"` expands to visible agents only.

---

## 4. Data Strategy

### 4.1 Design Rationale

The broker is a mandatory parent process that owns every agent's stdio channel. If the broker dies, all agents die. In-flight state cannot outlive the broker. SYNTH uses in-memory state for everything the broker owns, and persists only what must survive restarts.

### 4.2 What Lives Where

| Concern | Storage | Rationale |
|---|---|---|
| Agent state | Broker memory | Dies with broker (and so do agents) |
| Conversation history | Broker memory + TUI event buffers | Streaming through broker; TUI buffers per-agent for panel replay |
| Inter-agent messages | SQLite (write by MCP server, read by broker) | MCP servers are separate processes with no direct IPC to broker |
| Permission rules | SQLite (`~/.synth/synth.db`, `rules` table) | Session-scoped, keyed on `(agent_id, tool_kind, session_id)` |
| Session resume IDs | JSON file (`~/.synth/sessions.json`) | Written on graceful shutdown |
| Token usage | Broker memory (per-agent cumulative) | Ephemeral; displayed in TUI topbar |

### 4.3 SQLite Schema

```sql
CREATE TABLE agents (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL,
    parent      TEXT,       -- agent_id of launcher, NULL for config-defined
    task        TEXT        -- one-line task summary
);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  INTEGER NOT NULL,
    claimed_at  INTEGER
);

CREATE TABLE agent_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    command     TEXT NOT NULL,       -- 'launch' | 'terminate'
    payload     TEXT NOT NULL,       -- JSON
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | processed | rejected
    error       TEXT,
    created_at  INTEGER NOT NULL
);
```

WAL mode enabled. Broadcast `"*"` is expanded to individual rows at send time. The `agent_commands` table serves as a durable command queue — the poller's `PRAGMA data_version` detects writes and the broker processes pending commands on each poll cycle.

---

## 5. Inter-Agent Messaging

### 5.1 Why MCP (Not ACP) for Agent-to-Agent

ACP is client↔agent. Tool discovery flows from agent to MCP server, not from agent to client. The broker cannot expose tools as ACP-native capabilities. Agents need an actual MCP server for `send_message`, `check_delivery`, etc.

### 5.2 Message Flow

```
agent-1 calls send_message(to="agent-2", body="...")
  → synth-mcp writes to SQLite messages table
  → Broker's MessagePoller detects PRAGMA data_version change (~100ms)
  → Broker reads pending messages, groups by recipient
  → If agent-2 is IDLE: combined delivery via session/prompt
  → If agent-2 is BUSY: stays queued, delivered when IDLE
```

### 5.3 MCP Server Config Injection

```python
from acp.schema import McpServerStdio, EnvVariable

mcp_servers = [McpServerStdio(
    name="synth-mcp",
    command="synth-mcp",
    args=[],
    env=[
        EnvVariable(name="SYNTH_SESSION_ID", value=self._session_id),
        EnvVariable(name="SYNTH_DB_PATH", value=str(self._db_path)),
        EnvVariable(name="SYNTH_AGENT_ID", value=agent_id),
        EnvVariable(name="SYNTH_COMMUNICATION_MODE", value=self._config.settings.communication_mode.value),
        EnvVariable(name="SYNTH_MAX_AGENTS", value=os.environ.get("SYNTH_MAX_AGENTS", "10")),
        # + AgentConfig.env entries forwarded as additional EnvVariables
    ],
)]
```

### 5.4 Topology Support

SYNTH does not enforce a topology. Topology emerges from how agents are prompted:
- **Human-dispatch**: Human prompts individual agents. Agents don't message each other.
- **Orchestrator**: One coordinator uses `send_message` to delegate to workers. Workers can be dynamically spawned via `launch_agent`.
- **Peer-to-peer**: Any agent messages any other. Broker delivers to idle agents.

Communication modes (`SYNTH_COMMUNICATION_MODE`) scope visibility:
- **MESH** (default): every agent sees every other active agent
- **LOCAL**: agents see only their family (parent, children, siblings) — enables isolated team hierarchies within a single session

---

## 6. Layer Boundary Contracts

### 6.1 Events (Broker → Frontend)

```python
class BrokerEvent(BaseModel, frozen=True):
    timestamp: datetime
    agent_id: str

class AgentStateChanged(BrokerEvent):
    old_state: AgentState
    new_state: AgentState

class MessageChunkReceived(BrokerEvent):
    chunk: str

class ToolCallUpdated(BrokerEvent):
    tool_call_id: str
    title: str
    kind: str
    status: str

class PermissionRequested(BrokerEvent):
    request_id: str
    title: str
    kind: str
    options: list[PermissionOption]  # SDK type from acp.schema

class PermissionAutoResolved(BrokerEvent):
    request_id: str
    decision: PermissionDecision

class TurnComplete(BrokerEvent):
    stop_reason: str

class McpMessageDelivered(BrokerEvent):
    from_agent: str
    to_agent: str
    preview: str = ""

class BrokerError(BrokerEvent):
    message: str
    severity: Literal["warning", "error"] = "error"

class AgentThoughtReceived(BrokerEvent):
    chunk: str

class UsageUpdated(BrokerEvent):
    size: int           # context window size in tokens
    used: int           # tokens currently in context
    cost_amount: float | None = None
    cost_currency: str | None = None
```

Note: The design doc v0.2 showed `_future: asyncio.Future` on `PermissionRequested` and `message_id` on `MessageChunkReceived`. The implementation correctly diverged: the Future lives on `ACPSession._permission_future` (not on the event), and `MessageChunkReceived` has no `message_id` field. The `McpMessagePending` event from v0.2 was replaced by `McpMessageDelivered` with a `preview` field.

### 6.2 Commands (Frontend → Broker)

```python
class BrokerCommand(BaseModel, frozen=True): ...
class LaunchAgent(BrokerCommand):    agent_id: str
class TerminateAgent(BrokerCommand): agent_id: str
class SendPrompt(BrokerCommand):     agent_id: str; text: str
class RespondPermission(BrokerCommand): agent_id: str; option_id: str
class CancelTurn(BrokerCommand):     agent_id: str
```

Note: `RespondPermission` has no `request_id` field. Since ACP enforces one pending permission per agent at a time, `agent_id` is sufficient to find the right Future.

### 6.3 Broker Public API

```python
class ACPBroker:
    async def handle(self, command: BrokerCommand) -> None: ...
    async def events(self) -> AsyncIterator[BrokerEvent]: ...
    def get_agent_states(self) -> dict[str, AgentState]: ...
    def get_agent_configs(self) -> list[AgentConfig]: ...
    async def shutdown(self) -> None: ...
```

`events()` is an infinite async iterator backed by an unbounded `asyncio.Queue`. It terminates (`StopAsyncIteration`) when `shutdown()` is called. Single-consumer design — one frontend at a time.

### 6.4 Textual Bridge

The TUI wraps broker events in a single `BrokerEventMessage` Textual message class. Widgets inspect `event` type via `isinstance`. This avoids a parallel message hierarchy that must stay in sync with `models/events.py`.

```python
class BrokerEventMessage(Message):
    def __init__(self, event: BrokerEvent) -> None:
        self.event = event
        super().__init__()
```

The app runs a named worker (`"broker-consumer"`) that consumes `broker.events()` and posts `BrokerEventMessage` to the Textual message tree. Worker errors are caught and surfaced via `on_worker_state_changed` with auto-restart.

---

## 7. Permission Handling

### 7.1 Flow

```
Agent sends request_permission
  → ACPSession creates Future, stores as _permission_future
  → ACPSession transitions BUSY → AWAITING_PERMISSION
  → ACPSession emits PermissionRequested event to broker
  → Broker's _sink() intercepts:
      → Checks PermissionEngine for persisted rule
      → If rule exists: calls session.resolve_permission(option_id), emits PermissionAutoResolved
      → If no rule: forwards event to UI via event queue
  → UI renders PermissionRequest widget with 4 buttons
  → User clicks button → UI sends RespondPermission command
  → Broker calls session.resolve_permission(option_id)
  → If option is allow_always/reject_always: broker persists rule
  → ACPSession.request_permission() awaits Future, gets option_id
  → ACPSession transitions AWAITING_PERMISSION → BUSY
  → Agent resumes
```

### 7.2 Decision Options

| Kind | Label | Behavior |
|---|---|---|
| `allow_once` | Allow once | Respond with approval, no persistence |
| `allow_always` | Always allow | Respond with approval, persist rule |
| `reject_once` | Reject | Respond with rejection, no persistence |
| `reject_always` | Always reject | Respond with rejection, persist rule |

### 7.3 Graceful Shutdown

Enforced ordering:

1. Stop accepting new commands (`_shutting_down = True`)
2. Cancel all active prompts
3. Stop the message poller (await current cycle)
4. Persist session IDs to `~/.synth/sessions.json`
5. Terminate all sessions (kills subprocesses)
6. Cancel all asyncio tasks

---

## 8. Launch Model & Configuration

### 8.1 Config Resolution Order

1. `--harness` present → build transient config from flags, skip file discovery
2. `--config PATH` present → load that specific file
3. Auto-discover `.synth.toml` then `.synth.json` in CWD
4. First-run interactive picker (TUI mode only; headless prints error and exits)

### 8.2 Config Format — `.synth.toml`

```toml
project = "my-api"

[settings]
communication_mode = "LOCAL"    # or "MESH" (default)

[[agents]]
id      = "orchestrator"
cmd     = ["kiro-cli", "acp", "--agent", "orchestrator"]
label   = "Orchestrator"
profile = "orchestrator"
cwd     = "."

[[agents]]
id      = "reviewer"
cmd     = ["claude", "mcp", "--agent", "reviewer"]
label   = "Code Reviewer"
profile = "reviewer"
env     = { ANTHROPIC_MODEL = "claude-opus-4-6" }
```

### 8.3 AgentConfig Model

```python
class AgentConfig(BaseModel, frozen=True):
    id: str                          # CSS-safe: [a-zA-Z0-9][a-zA-Z0-9_-]*
    cmd: list[str]                   # full argv
    label: str | None = None         # display name; defaults to id
    profile: str | None = None       # harness agent name
    cwd: str = "."
    env: dict[str, str] = {}

    @property
    def display_name(self) -> str: return self.label or self.id
    @property
    def binary(self) -> str: return self.cmd[0]       # backward compat
    @property
    def args(self) -> list[str]: return self.cmd[1:]   # backward compat
```

Legacy coercion: a `model_validator(mode="before")` converts old `{"binary": "kiro-cli", "args": ["acp"]}` to `cmd=["kiro-cli", "acp"]`.

### 8.4 SessionConfig Model

```python
class CommunicationMode(StrEnum):
    MESH = "MESH"
    LOCAL = "LOCAL"

class SettingsConfig(BaseModel, frozen=True):
    communication_mode: CommunicationMode = CommunicationMode.MESH

class SessionConfig(BaseModel, frozen=True):
    project: str                     # was: session
    agents: list[AgentConfig]
    ui: UIConfig = UIConfig()
    settings: SettingsConfig = SettingsConfig()
```

Legacy coercion: `session` accepted as synonym for `project`. `autostart` removed — all configured agents launch on startup.

### 8.5 Known Harnesses Registry

Data-driven registry in `src/synth_acp/data/harnesses/*.toml`. Powers PATH probing, `--harness` flag resolution, dynamic agent harness resolution, and install hints. Adding a new harness is a TOML file addition with no Python changes.

`HarnessEntry` model lives in `models/config.py`. `load_harness_registry()` lives in `synth_acp/harnesses.py`. Both CLI and broker import from these shared locations (broker must not import from CLI layer).

### 8.6 CLI Flags

| Flag | Description |
|---|---|
| `--harness NAME` | Harness to launch (kiro, claude, opencode, gemini). Bypasses config file. |
| `--agent NAME` | Agent within the harness. Only with `--harness`. |
| `--config PATH` / `-c` | Explicit config file path. |
| `--headless` | Run without TUI (stdin/stdout mode). |
| `--verbose` / `-v` | Enable debug logging. |

### 8.7 Data Directory (`~/.synth/`)

```
~/.synth/
├── rules.json          # Persisted permission rules
├── sessions.json       # Session resume IDs
└── synth.db            # SQLite database (inter-agent messages)
```

---

## 9. Textual UI

### 9.1 Theme and Styling

Theme: `catppuccin-mocha`. All styles in external `ui/css/app.tcss`. Only Textual design tokens (`$primary`, `$surface`, `$text-muted`, etc.) — no hardcoded colors in CSS. Agent colors assigned from a rotating palette of 8-10 visually distinct colors by config index.

### 9.2 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  SYNTH             [session: dev-project]  32k tok  $0.14   │
├──────────────────┬──────────────────────────────────────────┤
│  AGENTS          │  ▸ Thinking…                             │
│  ────────────    │  ──────────────────────────────────────  │
│  ● coordinator   │  [10:42] You → agent-1                   │
│  ● kiro-auth     │  Refactor the auth module to use JWT.    │
│  ○ kiro-api      │                                          │
│                  │  [10:42] agent-1                         │
│  [+ Launch]      │  ▶ read  reading src/auth.py     ✓       │
│                  │  ▶ edit  modifying src/auth.py   ●       │
│                  │  I'll start by extracting the session... │
│                  │  ▌                                       │
│                  ├──────────────────────────────────────────┤
│  MCP MESSAGES    │  ⚠ PERMISSION REQUEST                    │
│  ────────────    │  agent-1 wants to execute:               │
│  3 pending       │  $ pip install pyjwt                     │
│                  │                                          │
│                  │  [ Allow once ]  [ Always allow ]        │
│                  │  [ Reject ]      [ Always reject ]       │
│                  ├──────────────────────────────────────────┤
│                  │  > Send message to agent-1...            │
└──────────────────┴──────────────────────────────────────────┘
```

### 9.3 Widget Hierarchy

- `AgentList` — sidebar (32 chars). `AgentTile` per agent, `LaunchButton`, `MCPButton`. Each tile shows a colored status dot and state-specific text (initializing…, idle, working…, awaiting permission…, terminated). Dynamically launched agents get tiles added at runtime via `add_agent_tile()` with the next palette color. Tile preview shows `task` if available, or `via {parent}` for dynamic agents.
- `ConversationFeed` — scrollable container, one per agent (lazy-created, stays alive):
  - `LoadingIndicator` — shown during `INITIALIZING`, removed on `IDLE`, re-mounted on re-launch
  - `PromptBubble` — right-aligned, `$primary` border
  - `ThoughtBlock` — `Collapsible` wrapping `MarkdownStream`, collapsed when finalized
  - `AgentMessage` — extends `Markdown`, uses `MarkdownStream` for incremental rendering
  - `ToolCallBlock` — kind icon + color, title, status badge
  - `PermissionRequest` — yellow border, 4 `Button` widgets
- `MessageQueue` — thread list + detail, grouped by sorted agent pairs
- `InputBar` — `Input` widget, disabled when BUSY/AWAITING_PERMISSION, `@agent-id` routing

### 9.4 Streaming Markdown

Agent responses use Textual's native `Markdown` + `MarkdownStream`:

```python
class AgentMessage(Markdown):
    @property
    def stream(self) -> MarkdownStream:
        if self._stream is None:
            self._stream = self.get_stream(self)
        return self._stream

    async def append_chunk(self, chunk: str) -> None:
        await self.stream.write(chunk)
```

### 9.5 Panel Lifecycle

Panels are created lazily on first selection with event buffering:

- The app maintains `dict[str, list[BrokerEvent]]` per agent from broker start
- On first selection: panel is created, drains buffer, renders backlog
- Once created: panel stays in DOM, switching via `ContentSwitcher(id="right")` — `switcher.current = f"feed-{agent_id}"` or `"messages"`
- After creation: events flow directly to the mounted panel

This gives fast startup, no lost events, and instant switching after first view.

### 9.6 Key Bindings

| Key | Action |
|---|---|
| `Tab` | Cycle agent focus |
| `Enter` | Submit prompt |
| `m` | MCP messages panel |
| `l` | Launch agent (modal) |
| `F1` | Help (modal) |
| `Ctrl+P` | Command palette |
| `q` | Quit |

### 9.7 MCP Messages Panel

TUI-side `dict[tuple[str, str], list[McpMessageDelivered]]` keyed by sorted agent pairs, populated from live events. No SQLite access from Layer 3. Thread list on left, metadata detail on right. Message body display deferred to Phase 3 when broker exposes message content API.

---

## 10. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | asyncio native |
| TUI | Textual | Terminal + browser via `textual serve` |
| ACP transport | JSON-RPC 2.0 over stdio | Standard ACP |
| ACP library | `agent-client-protocol` (PyPI) | Official SDK; Pydantic models, `spawn_agent_process`, contrib helpers |
| Models | Pydantic v2 | Already a dependency via ACP SDK; fast Rust core |
| MCP server | `mcp>=1.0.0` (`mcp.server.fastmcp.FastMCP`) | Agent-to-agent messaging |
| IPC | SQLite (WAL mode) | Zero-config cross-process coordination |
| Config | TOML (stdlib `tomllib`) + JSON backward compat | Human-readable, comments, standard |
| Package manager | uv | Fast, modern Python packaging |
| Linting | ruff | Formatting + linting |
| Type checking | ty | Replaces mypy |
| Testing | pytest + pytest-asyncio | `asyncio_mode = "auto"` |

---

## 11. Design Decisions and Rejected Alternatives

### 11.1 Pydantic v2 over dataclasses
ACP SDK depends on Pydantic. Consistency + free JSON serialization outweighs marginal overhead.

### 11.2 In-memory broker state over SQLite for everything
Broker owns all lifecycles. Persisting ephemeral state provides durability against a failure mode that doesn't exist.

### 11.3 SQLite for permission rules (was JSON)
Session-scoped rules keyed on `(agent_id, tool_kind, session_id)`. SQLite is consistent with the existing `messages`/`agents` tables and supports future session resume without subfolder proliferation.

### 11.4 SQLite message bus over direct IPC
MCP servers are separate processes with no direct channel to broker. SQLite is zero-config IPC.

### 11.5 Composition over inheritance for ACPSession
SDK classes aren't designed for inheritance. Composition is stable across version bumps.

### 11.6 Separate widgets over monolithic conversation renderer
Reusable, independently testable, follows Textual's message-driven architecture.

### 11.7 Single-consumer event stream over broadcast
One frontend at a time. Fan-out added when needed.

### 11.8 Permission Future on session, not on event
Keeps Pydantic events serializable. Broker resolves via `_sessions[agent_id]` lookup.

### 11.9 Broadcast expanded at MCP server send time
Poller stays simple (deliver to named agent). Individual rows independently trackable.

### 11.10 MessagePoller accepts DeliverFn callback
Avoids circular reference. Poller holds callable, broker passes bound method.

### 11.10a MessagePoller accepts optional CommandFn callback
Same pattern as DeliverFn. Poller detects `data_version` changes and dispatches to both callbacks. Avoids duplicating the polling mechanism for `agent_commands`.

### 11.10b `agent_commands` table as command queue
SQLite `status='pending'` rows ARE the queue. No separate in-memory queue needed. The poller's existing `PRAGMA data_version` detection picks up writes from MCP tool processes. Commands that can't be processed (at agent limit) stay `pending` and are retried on the next poll cycle.

### 11.10c Communication mode as env var to MCP server instances
Each MCP server is a separate process. Passing `SYNTH_COMMUNICATION_MODE` as an env var lets `_get_visible_agents()` filter locally without querying the broker. The broker computes visibility from in-memory state for join broadcasts.

### 11.11 `pull_messages` removed from MCP server
Broker poller handles all delivery. Agents receive messages between turns automatically.

### 11.12 `cmd` replaces `binary`/`args` in AgentConfig
Cleaner model. Backward-compat properties (`binary`, `args`) preserve existing callsites.

### 11.13 `project` replaces `session` in SessionConfig
Better name. Legacy coercion accepts `session` key.

### 11.14 `autostart` removed
Orchestrator-first model: all configured agents launch on startup.

### 11.15 TOML over JSON for config
Comments, human-readable, stdlib `tomllib`. JSON supported for backward compat.

### 11.16 Lazy panel creation with event buffering over SessionAccumulator
Simpler, no Phase 1 changes needed. Panels created on first view, stay alive, events buffered until then.

### 11.17 `mcp` package, not `fastmcp`
`FastMCP` lives at `mcp.server.fastmcp.FastMCP` inside the official `mcp` package. No separate dependency.

---

## 12. Implementation Phases

### Phase 1 — ACP Core + Headless Broker ✅

Completed. Includes: ACPSession, ACPBroker, PermissionEngine, MessagePoller, synth-mcp, interactive CLI, 29 tests.

### Phase 2 — Textual TUI ✅

Completed. Includes: SynthApp, AgentList, ConversationFeed, PromptBubble, AgentMessage (MarkdownStream), ToolCallBlock, PermissionRequest, MessageQueue, InputBar, external CSS, `--headless` flag.

### Phase 3 — UX Overhaul + Critical Fixes ✅

Completed across spec phases 1-4. Includes:
- **ACP-7:** Permission persistence fix — 4-value `PermissionDecision`, SQLite-backed `PermissionEngine`, session-scoped rules, `persist()` wired
- **Textual-10:** Worker error handling — named `"broker-consumer"` worker, `on_worker_state_changed` with notify + auto-restart
- **Config migration:** `.synth.toml` support, `cmd` field, `project` field, harness registry, first-run picker, typer CLI with `--harness`/`--agent`/`--config` flags
- **ACP-3:** `InitializeResponse` capabilities captured on `ACPSession`
- **ACP-1a:** `agent_thought_chunk` → `AgentThoughtReceived` event, `ThoughtBlock` widget (Collapsible + MarkdownStream)
- **ACP-1b:** `usage_update` → `UsageUpdated` event, topbar context/cost display
- Remaining: Textual-1 (modal screens), Textual-5 (LoadingIndicator), Textual-2+3 (ContentSwitcher + reactive watchers)

### Phase 3.5 — TUI Refactor ✅

Completed. Proper Textual patterns replacing manual stubs:
- **Textual-2:** `ContentSwitcher(id="right")` replaces manual `display` toggling in `select_agent()`/`show_messages()`
- **Textual-3:** `watch_selected_agent` reactive watcher drives panel switching, tile styling, topbar, input bar state; `select_agent()` reduced to lazy creation + reactive set
- **Textual-1:** `LaunchAgentScreen(ModalScreen)` and `HelpScreen(ModalScreen)` replace notification stubs; `push_screen_wait()` pattern
- **Textual-5:** `LoadingIndicator` in `ConversationFeed` during `INITIALIZING`; removed on `IDLE`; re-mounted on re-launch preserving conversation history

### Phase 3.6 — Dynamic Agent Management + Communication Modes ✅

Completed. Includes:
- **Schema:** `agent_commands` table (durable command queue), `parent`/`task` columns on `agents` table
- **MCP tools:** `launch_agent` and `terminate_agent` write to `agent_commands`; `list_agents` returns `parent`/`task`; `_get_visible_agents()` for MESH/LOCAL filtering
- **Broker:** `_process_commands` (CommandFn callback), harness resolution via shared `load_harness_registry()`, parentage enforcement, `SYNTH_MAX_AGENTS` limit with pending-queue retry, join broadcasts, IDLE-watch for initial prompts
- **Config:** `CommunicationMode` StrEnum, `SettingsConfig` model, `[settings]` section in `.synth.toml`
- **Harness refactor:** `HarnessEntry` → `models/config.py`, `load_harness_registry()` → `harnesses.py`
- **TUI:** Dynamic `AgentTile` creation, task/parent preview, `LaunchButton` fix, `LaunchAgentScreen` includes dynamic agents
- **Bug fix:** `opencode.toml` missing `{agent}` placeholder in `run_cmd_with_agent`

### Phase 4 — Session Management + Control Surface

- **ACP-4:** Session resume via `list_sessions` / `load_session`
- **ACP-5:** `set_session_mode` / `set_session_model` via command palette
- **ACP-2:** Adopt `SessionAccumulator` (refactor session.py, unblocks plan/mode/commands)
- **ACP-1c:** Plan panel
- **Textual-4:** Command palette (`SynthCommandProvider`)
- **Textual-8:** AgentTile keyboard focus

### Phase 5 — Advanced Features

- ~~Dynamic agent management (`launch_agent`/`terminate_agent` in synth-mcp)~~ — **Resolved** (Phase 3.6): `launch_agent`/`terminate_agent` MCP tools, `agent_commands` table, broker command processing, parentage tracking, communication modes (MESH/LOCAL), global agent limit, join broadcasts, TUI dynamic agent display
- Topology visualization
- `@file:path` prompt attachments
- Filesystem/terminal client capabilities (configurable)
- Web mode (`textual serve`)
- JSONL audit log

---

## 13. Entrypoints

```toml
[project.scripts]
synth = "synth_acp.cli:main"
synth-mcp = "synth_acp.mcp.server:main"
```

---

## 14. Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "agent-client-protocol",
    "textual",
    "mcp>=1.0.0",
    "aiosqlite>=0.19.0",
    "typer>=0.9",
]

[dependency-groups]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.11",
    "ty>=0.0.1a7",
]
```

---

## 15. Package Layout

```
synth-acp/
├── pyproject.toml
├── src/synth_acp/
│   ├── __init__.py / __main__.py / cli.py
│   ├── harnesses.py          # load_harness_registry()
│   ├── models/
│   │   ├── agent.py          # AgentConfig, AgentState, transitions
│   │   ├── config.py         # SessionConfig, SettingsConfig, CommunicationMode, HarnessEntry, load_config
│   │   ├── events.py         # BrokerEvent subclasses
│   │   ├── commands.py       # BrokerCommand subclasses
│   │   └── permissions.py    # PermissionDecision, PermissionRule
│   ├── acp/
│   │   └── session.py        # ACPSession
│   ├── broker/
│   │   ├── broker.py         # ACPBroker
│   │   ├── permissions.py    # PermissionEngine
│   │   └── poller.py         # MessagePoller
│   ├── mcp/
│   │   └── server.py         # synth-mcp (FastMCP)
│   ├── data/
│   │   └── harnesses/        # Known harness TOML files
│   └── ui/
│       ├── app.py            # SynthApp
│       ├── messages.py       # BrokerEventMessage
│       ├── screens/          # ModalScreens (launch, help, resume)
│       ├── widgets/          # AgentList, ConversationFeed, etc.
│       └── css/app.tcss
├── docs/
│   ├── DESIGN.md
│   └── references/
├── tests/                    # Mirrors src/ structure
└── examples/
    └── echo_agent.py
```

---

## 16. Gap Analysis Summary

Detailed analysis in `docs/references/design_gaps.md`. Key items by priority:

### 🔴 Critical

| ID | Summary |
|---|---|
| ~~ACP-7~~ | ~~`allow_always`/`reject_always` never persisted~~ — **Resolved** (Phase 1): 4-value `PermissionDecision`, SQLite-backed `PermissionEngine`, session-scoped rules |

### 🟠 High

| ID | Summary |
|---|---|
| ~~ACP-1a~~ | ~~`agent_thought_chunk` dropped~~ — **Resolved** (Phase 3/4): `AgentThoughtReceived` event, `ThoughtBlock` widget with streaming markdown |
| ~~ACP-1b~~ | ~~`usage_update` dropped~~ — **Resolved** (Phase 3/4): `UsageUpdated` event, topbar display with context/cost |
| ~~ACP-3~~ | ~~`InitializeResponse` discarded~~ — **Resolved** (Phase 3): `_capabilities` captured on `ACPSession` |
| ACP-4 | No `load_session`/`list_sessions` — session context lost on restart |
| ~~Textual-1~~ | ~~No `ModalScreen` — launch/help are notification stubs~~ — **Resolved** (Phase 3.5): `LaunchAgentScreen` and `HelpScreen` modals with `push_screen_wait()` |
| ~~Textual-10~~ | ~~Worker error swallowed silently~~ — **Resolved** (Phase 4): named `"broker-consumer"` worker, `on_worker_state_changed` with notify + auto-restart |

### 🟡 Medium

| ID | Summary |
|---|---|
| ACP-2 | `SessionAccumulator` unused — manual event dispatch growing fragile |
| ACP-1c | `AgentPlanUpdate` dropped — no plan visibility |
| ACP-5 | No `set_session_mode`/`set_session_model` — can't control agents at runtime |
| ~~Textual-2~~ | ~~Manual display toggling instead of `ContentSwitcher`~~ — **Resolved** (Phase 3.5): `ContentSwitcher(id="right")` with reactive watcher |
| ~~Textual-3~~ | ~~`reactive` declared but never watched~~ — **Resolved** (Phase 3.5): `watch_selected_agent` drives all panel-switch side effects |
| Textual-4 | No command palette |
| ~~Textual-5~~ | ~~No `LoadingIndicator` during INITIALIZING~~ — **Resolved** (Phase 3.5): spinner in `ConversationFeed`, removed on IDLE, re-mounted on re-launch |

### 🟢 Low

ACP-6, ACP-8, ACP-9, ACP-10, ACP-1d, ACP-1e, Textual-7, Textual-8, Textual-9, Textual-11

---

## Changelog

### v0.4 (2026-03-25)
- **Dynamic agent management:** `launch_agent`/`terminate_agent` MCP tools write to `agent_commands` SQLite table. Broker processes commands via `CommandFn` callback on `MessagePoller`. Harness resolution via shared `load_harness_registry()`. Parentage tracking (`parent`/`task` columns on `agents` table). Orphaned children get `parent=NULL`.
- **Communication modes:** `CommunicationMode` StrEnum (`MESH`/`LOCAL`) in `SettingsConfig`. LOCAL mode restricts visibility to parent/children/siblings. `_get_visible_agents()` filters `list_agents`, `send_message`, and broadcast expansion.
- **Global agent limit:** `SYNTH_MAX_AGENTS` env var (default 10). At-capacity launches queue as `pending` and retry on each poll cycle.
- **Join broadcasts:** System messages sent to visible agents when a new agent is registered.
- **Harness refactor:** `HarnessEntry` moved to `models/config.py`, `load_harness_registry()` to `harnesses.py`. Broker imports without CLI dependency.
- **TUI dynamic agents:** `add_agent_tile()` mounts tiles at runtime. Task/parent preview in `AgentTile`. `LaunchButton` fixed to open modal. `LaunchAgentScreen` includes dynamic agents.
- **Bug fix:** `opencode.toml` missing `{agent}` placeholder in `run_cmd_with_agent`.
- **Config:** `[settings]` section with `communication_mode` in `.synth.toml`. `AgentConfig.env` forwarded to MCP server instances.

### v0.3 (2026-03-25)
- **Permission fix (ACP-7):** `PermissionDecision` extended to four values (`allow_once`, `allow_always`, `reject_once`, `reject_always`). `PermissionEngine` migrated from JSON to SQLite with session-scoped rules. `persist()` now called on always-options. Auto-resolve works within same session.
- **Config overhaul:** `AgentConfig` uses `cmd` field (legacy `binary`/`args` coerced). `SessionConfig` renamed `session` → `project`. TOML config support. Harness registry as package data. First-run interactive picker. CLI migrated to typer with `--harness`/`--agent`/`--config` flags.
- **ACP session improvements:** `InitializeResponse` capabilities captured. `agent_thought_chunk` and `usage_update` handled — emit `AgentThoughtReceived` and `UsageUpdated` events. Broker accumulates usage per agent.
- **UI — ThoughtBlock:** `ThoughtBlock(Collapsible)` widget with streaming `MarkdownStream`. Title "Thinking…" while streaming, "Thought" when finalized. Collapsed by default after finalization.
- **UI — Usage display:** Topbar `#tb-right` shows context usage and cost for selected agent (e.g. `32k ctx  $0.14`).
- **UI — Worker error handling:** Broker consumer worker named `"broker-consumer"`. `on_worker_state_changed` surfaces errors via persistent notification and auto-restarts the consumer.

### v0.2 (2026-03-24)
- Added permission Future-based flow, awaited state notifications, combined message delivery, two-phase status, graceful shutdown ordering, config validation, BrokerError event, rendering strategy, async iterator contract

### v0.1 (2026-03-24)
- Initial design document
