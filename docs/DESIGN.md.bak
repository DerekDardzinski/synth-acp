# SYNTH Design Document

**SYNTH** — **SY**nchronized **N**etwork of **T**eamed **H**arnesses over ACP

Version: 0.2-draft
Date: 2026-03-24

---

## 1. Overview

SYNTH is a multi-agent orchestration dashboard that manages teams of AI coding agents through the Agent Client Protocol (ACP). It replaces the tmux-based delivery mechanism of `team-mcp` with structured, bidirectional JSON-RPC communication over stdio, and provides a Textual-based TUI for real-time session management, streaming output, permission handling, and inter-agent messaging.

### Project Identity

| Attribute | Value |
|---|---|
| PyPI package | `synth_acp` |
| CLI command | `synth` |
| MCP server entrypoint | `synth-mcp` |
| Config file | `.synth.json` |
| Data directory | `~/.synth/` |
| Repo name | `synth-acp` |

### Goals

- Manage teams of ACP-compatible agents (Kiro CLI, Claude Code, Gemini CLI, etc.) from a single process
- Surface permission requests to the operator in real time
- Support flexible agent topologies: human-dispatch, orchestrator, peer-to-peer
- Provide a polished TUI (with future web UI support) for session management and observability
- Enable agent-to-agent communication via an MCP server shipped with the project

### Non-Goals

- Building a new AI agent (SYNTH manages existing agents)
- Supporting non-ACP agents (no tmux fallback)
- Multi-user authentication (single operator, single session)

---

## 2. Architecture

### 2.1 System Diagram

```
synth (single process: broker + UI)
┌──────────────────────────────────────────────────┐
│                                                  │
│  CLI (argparse)                                  │
│    └── parses .synth.json, starts broker + UI    │
│                                                  │
│  ACPBroker ─────────────────────────────────┐    │
│    ├── ACPSession (agent-1)                 │    │
│    ├── ACPSession (agent-2)                 │    │
│    ├── ACPSession (agent-N)                 │    │
│    ├── MessageRouter (in-memory routing)    │    │
│    ├── PermissionEngine (persisted rules)   │    │
│    └── MessagePoller (SQLite watcher)       │    │
│              │                              │    │
│              │ AsyncIterator[BrokerEvent]    │    │
│              ▼                              │    │
│  TeamACPApp (Textual) ◄────────────────────┘    │
│    ├── AgentList                                 │
│    ├── ConversationFeed                          │
│    │     ├── PromptBubble                        │
│    │     ├── AgentMessage                        │
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
│          │  │          │  │          │
│ MCP:     │  │ MCP:     │  │ MCP:     │
│ synth-mcp│  │ synth-mcp│  │ synth-mcp│
└──────────┘  └──────────┘  └──────────┘
      │              │              │
      └──────────────┴──────────────┘
                     │
              SQLite (messages table)
                     │
              Broker reads via poller
```

### 2.2 Three-Layer Design

The codebase is organized into three layers with strict dependency rules. Each layer may only depend on layers below it. The `models/` package is shared across all layers.

| Layer | Package | Responsibility | Imports from |
|---|---|---|---|
| 3 — Frontend | `synth_acp.ui` | Textual TUI rendering | `models`, `broker` |
| 2 — Broker | `synth_acp.broker` | Session lifecycle, routing, permissions | `models`, `acp` |
| 1 — ACP | `synth_acp.acp` | ACP SDK wrapper, subprocess management | `models` |
| Shared | `synth_acp.models` | Pydantic models for events, commands, config | (none) |

The critical constraint for future web UI support: Layers 1 and 2 have zero Textual imports. The frontend communicates with the broker exclusively through typed events and commands defined in `models/`.

### 2.3 Why a Single Process

The broker, UI, and event loop all run in one Python process. This is a deliberate choice:

- The broker must be alive for agents to function (it owns their stdio channels). If the broker dies, all agents die. There is no failure mode where the UI survives but the broker doesn't, or vice versa.
- Textual's event loop is an asyncio loop. The broker is async. Running them in the same loop avoids cross-process IPC for the hot path (streaming chunks, permission requests).
- A future web UI would replace the Textual app with a FastAPI/websocket server in the same process, consuming the same `AsyncIterator[BrokerEvent]` interface.

---

## 3. Component Design

### 3.1 ACPSession (Layer 1)

Wraps the `agent-client-protocol` SDK's `spawn_agent_process` and `Client` interface for a single agent subprocess. Responsibilities:

- Spawn the agent subprocess and perform the ACP handshake (`initialize` → `session/new`)
- Track session state via a finite state machine
- Forward `session_update` notifications (message chunks, tool calls) to the broker as typed events
- Forward `request_permission` calls to the broker and await the response
- Handle `session/prompt`, `session/cancel`, and graceful shutdown

The session implements the SDK's `Client` interface:

```python
class ACPSession(Client):
    """Wraps one ACP agent subprocess."""

    agent_id: str
    state: AgentState
    _conn: Connection          # from spawn_agent_process
    _accumulator: SessionAccumulator  # from acp.contrib
    _event_sink: Callable[[BrokerEvent], Awaitable[None]]
    _permission_future: asyncio.Future | None = None

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Called by ACP SDK when agent streams a response."""
        self._accumulator.apply(update)
        # Emit typed events to broker based on update content

    async def request_permission(self, options: Any, session_id: str,
                                  tool_call: Any, **kwargs: Any) -> dict:
        """Called by ACP SDK when agent requests permission. Blocks until resolved."""
        await self._set_state(AgentState.AWAITING_PERMISSION)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._permission_future = future
        await self._event_sink(PermissionRequested(
            agent_id=self.agent_id, ..., _future=future,
        ))
        try:
            option_id = await future
        except asyncio.CancelledError:
            return {"outcome": {"outcome": "cancelled"}}
        finally:
            self._permission_future = None
        return {"outcome": {"optionId": option_id, "outcome": "selected"}}
```

#### Awaited State Notifications

State change notifications are `await`ed, not fire-and-forget. This prevents a class of race conditions where the broker's view of agent state is stale when the first streaming chunk arrives:

```python
async def _set_state(self, new_state: AgentState) -> None:
    old = self.state
    if new_state not in TRANSITIONS[old]:
        raise InvalidTransition(f"{old} → {new_state}")
    self.state = new_state
    await self._event_sink(AgentStateChanged(
        agent_id=self.agent_id, old_state=old, new_state=new_state,
    ))
```

The `await` ensures the broker's registry is updated and the event is enqueued before `_set_state()` returns. Without this, there is a window where the session is BUSY but the broker still thinks it's IDLE — causing the UI to show chunks for an apparently idle agent, the input bar to remain enabled, or the message router to deliver a buffered message to a busy agent.

#### State Machine

```
UNSTARTED → INITIALIZING → IDLE → BUSY → IDLE → ... → TERMINATED
                                    ↑
                              AWAITING_PERMISSION
```

Valid transitions are enforced at runtime:

```python
TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.UNSTARTED:           {AgentState.INITIALIZING, AgentState.TERMINATED},
    AgentState.INITIALIZING:        {AgentState.IDLE, AgentState.TERMINATED},
    AgentState.IDLE:                {AgentState.BUSY, AgentState.TERMINATED},
    AgentState.BUSY:                {AgentState.IDLE, AgentState.AWAITING_PERMISSION,
                                     AgentState.TERMINATED},
    AgentState.AWAITING_PERMISSION: {AgentState.BUSY, AgentState.TERMINATED},
    AgentState.TERMINATED:          set(),
}
```

#### SDK Contrib Usage

SYNTH uses the ACP SDK's contrib modules as the canonical source of truth for session state, rather than reimplementing tool call tracking and message accumulation. This is a deliberate "use the library" decision — Toad (a single-agent ACP TUI) implemented custom tracking and hit edge cases (tool_call_update arriving before tool_call, out-of-order notifications) that the SDK's accumulator already handles.

- `SessionAccumulator` — the single source of truth for per-agent conversation history. Every `SessionNotification` is fed through it. The UI renders from `accumulator.snapshot()` on panel switch and from incremental events for the active agent. See Section 8.5 for the rendering strategy.
- `ToolCallTracker` — tracks tool call lifecycle (start → progress → complete/fail) with canonical update emission.
- `PermissionBroker` — wraps `requestPermission` RPCs with standard option sets.

### 3.2 ACPBroker (Layer 2)

The central orchestration service. Owns all `ACPSession` instances and routes events between agents and the UI.

Responsibilities:

- Parse `.synth.json` and create `ACPSession` instances
- Launch/terminate agent subprocesses
- Route inter-agent messages (read from SQLite, deliver via `session/prompt`)
- Resolve permission requests against persisted rules before forwarding to UI
- Expose an `AsyncIterator[BrokerEvent]` for the frontend to consume
- Accept `BrokerCommand` instances from the frontend

```python
class ACPBroker:
    """Owns all agent sessions and routes events."""

    _sessions: dict[str, ACPSession]
    _event_queue: asyncio.Queue[BrokerEvent]
    _config: SessionConfig
    _permission_engine: PermissionEngine
    _message_poller: MessagePoller

    async def handle(self, command: BrokerCommand) -> None:
        """Dispatch a command from the frontend."""
        match command:
            case LaunchAgent(agent_id=aid):  ...
            case TerminateAgent(agent_id=aid): ...
            case SendPrompt(agent_id=aid, text=text): ...
            case RespondPermission(agent_id=aid, request_id=rid, option_id=oid): ...
            case CancelTurn(agent_id=aid): ...

    async def events(self) -> AsyncIterator[BrokerEvent]:
        """Yield events for the frontend to consume."""
        while True:
            yield await self._event_queue.get()
```

### 3.3 PermissionEngine

Manages persisted allow/reject rules and auto-resolves permission requests when a matching rule exists.

```python
class PermissionEngine:
    """Auto-resolve permissions from persisted rules."""

    _rules: dict[tuple[str, str], PermissionDecision]  # (agent_id, tool_kind) → decision
    _rules_path: Path  # ~/.synth/rules.json

    def check(self, agent_id: str, tool_kind: str) -> PermissionDecision | None:
        """Return persisted decision if one exists, else None (ask the user)."""

    def persist(self, agent_id: str, tool_kind: str, decision: PermissionDecision) -> None:
        """Write a new rule to disk."""
```

Rules are stored as a JSON file at `~/.synth/rules.json`. This is appropriate because:
- The dataset is small (tens to low hundreds of rules)
- Writes are rare (only when user chooses "always allow" or "always reject")
- No concurrent writers (single broker process)
- Human-readable and editable

### 3.4 MessagePoller

Watches the SQLite database for new inter-agent messages and delivers them to the broker for routing.

```python
class MessagePoller:
    """Poll SQLite for new inter-agent messages via PRAGMA data_version."""

    _db_path: Path
    _last_version: int
    _broker: ACPBroker

    async def run(self) -> None:
        """Poll loop — runs as an asyncio task."""
        while not self._stopped:
            version = await self._check_data_version()
            if version != self._last_version:
                self._last_version = version
                await self._deliver_pending_messages()
            await asyncio.sleep(0.1)  # 100ms poll interval

    async def stop(self) -> None:
        self._stopped = True
```

The poll interval of 100ms provides sub-200ms delivery latency while consuming negligible CPU. `PRAGMA data_version` is a single integer comparison — not a query — so the cost per poll is effectively zero.

#### Combined Delivery

ACP enforces sequential turns — a second `session/prompt` cannot be sent while the first is in progress. When multiple messages are pending for an agent, the poller combines them into a single prompt:

```python
async def _deliver_pending(self, agent_id: str, messages: list[PendingMessage]) -> None:
    if not messages:
        return
    session = self._sessions.get(agent_id)
    if not session or session.state != AgentState.IDLE:
        return  # leave in SQLite for next poll cycle

    combined = "\n\n".join(
        f"[Message from {m.from_agent}]: {m.body}" for m in messages
    )
    try:
        await session.prompt(combined)
        await self._mark_delivered([m.id for m in messages])
    except Exception:
        pass  # leave in SQLite for retry
```

#### Two-Phase Status Updates

A message's status changes to `delivered` only after `session/prompt` succeeds. If delivery fails (agent terminated between state check and prompt), messages remain `pending` in SQLite and the next poll cycle retries. This invariant prevents message loss.

For `pull_messages` (agent-initiated pull via MCP): the existing "atomically claim" design is correct because the agent is calling it while idle and will process the messages itself. The two-phase concern applies only to broker-initiated delivery via the poller.

### 3.5 ACP Client Capabilities

The ACP `Client` protocol defines filesystem (`read_text_file`, `write_text_file`) and terminal (`create_terminal`, `terminal_output`, etc.) methods that the client can provide to agents. SYNTH declares these as unsupported:

```python
client_capabilities = ClientCapabilities(
    fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
    terminal=False,
)
```

This is deliberate. SYNTH's target agents (Kiro CLI, Claude Code, Gemini CLI) are full-featured harnesses with their own file editing and shell execution tools. They don't delegate these operations to the ACP client — they handle them internally and report results via `session/update` notifications with `ToolCallContentDiff` (for file edits) and `ToolCallContentTerminal` (for shell commands). SYNTH receives full diff data and terminal output through these notifications without ever touching the filesystem or running commands itself.

If a future lightweight agent requires client-provided filesystem or terminal support, these methods can be implemented at that point. The error mode is clear (JSON-RPC method-not-found) and the implementations are trivial.

### 3.6 Subprocess Crash Handling

When an agent process exits unexpectedly (crash, broken pipe, OOM kill), the `ACPSession` must:
1. Cancel any pending permission Future
2. Transition to TERMINATED
3. Emit a `BrokerError` with details

```python
async def run(self) -> None:
    try:
        await self._set_state(AgentState.INITIALIZING)
        async with spawn_agent_process(self, self._binary, *self._args,
                                        cwd=self._cwd) as (conn, proc):
            self._conn = conn
            await conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                terminal=False,
            ))
            session = await conn.new_session(cwd=self._cwd, mcp_servers=self._mcp_servers)
            self._session_id = session.session_id
            await self._set_state(AgentState.IDLE)
            await proc.wait()
    except Exception as e:
        await self._event_sink(BrokerError(
            agent_id=self.agent_id, message=f"Agent crashed: {e}",
        ))
    finally:
        if self._permission_future and not self._permission_future.done():
            self._permission_future.cancel()
        if self.state != AgentState.TERMINATED:
            await self._set_state(AgentState.TERMINATED)
```

The `spawn_agent_process` context manager handles subprocess cleanup. The `finally` block ensures state is always consistent regardless of how the process exited.

### 3.7 MCP Server (`synth-mcp`)

A FastMCP server shipped as a separate entrypoint. Each agent spawns its own instance via the MCP config injected in `session/new`. All instances share the same SQLite database.

Tools provided:

| Tool | Description |
|---|---|
| `send_message` | Send a message to a teammate (`to_agent="*"` broadcasts) |
| `pull_messages` | Atomically claim and return pending messages |
| `check_delivery` | Check delivery status of a sent message |
| `list_agents` | List all agents in the current session with status |
| `deregister_agent` | Deregister this agent from the session |

Tools removed from `team-mcp`:
- `launch_agent` — ownership moved to the broker
- `terminate_agent` — ownership moved to the broker
- All tmux-specific code

The MCP server is intentionally stateless and simple. It reads/writes SQLite and nothing else. The broker handles all routing intelligence.

---

## 4. Data Strategy

### 4.1 Design Rationale

The original `team-mcp` used SQLite as the sole coordination point because there was no central process — agents ran in independent tmux panes with no shared memory or IPC channel. SYNTH's ACP-based architecture changes this fundamentally: the broker is a mandatory parent process that owns every agent's stdio channel. If the broker dies, all agents die.

This means in-flight state (agent status, conversation history, pending permissions) cannot outlive the broker. Persisting it to disk provides durability against a failure mode that doesn't exist. SYNTH therefore uses in-memory state for everything the broker owns, and persists only what must survive restarts.

### 4.2 What Lives Where

| Concern | Storage | Rationale |
|---|---|---|
| Agent state (IDLE, BUSY, etc.) | Broker memory | Broker owns lifecycles; state dies with broker (and so do agents) |
| Conversation history | Broker memory (Phase 1) | Streaming through broker already; persistence is Phase 2 |
| Inter-agent messages | SQLite (write by MCP server, read by broker) | MCP servers are separate processes with no direct IPC to broker |
| Permission rules | JSON file (`~/.synth/rules.json`) | Small, rarely written, must survive restarts |
| Session resume IDs | JSON file (`~/.synth/sessions.json`) | Written on graceful shutdown, read on restart |
| Audit/replay log | JSONL file (Phase 2) | Append-only, no query requirements in Phase 1 |

### 4.3 SQLite Schema

SQLite's role is narrowed to inter-agent messaging — the only case where separate processes (MCP server instances) need to coordinate without a direct IPC channel to the broker.

```sql
-- Agent registry (written by MCP server on first tool call, read by list_agents)
CREATE TABLE agents (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL  -- unix ms
);

-- Message queue (written by send_message, read by broker's poller)
CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,      -- agent_id or "*" for broadcast
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | delivered | claimed
    created_at  INTEGER NOT NULL,   -- unix ms
    claimed_at  INTEGER             -- unix ms, set on pull_messages
);
```

WAL mode is enabled for concurrent readers (broker poller) and writers (MCP server instances).

### 4.4 Why Not a Database for Everything

If a reviewer asks "why not use SQLite for all state?":

> The broker is a mandatory single parent process that owns every agent's lifecycle via ACP stdio. If the broker dies, all agents die. Therefore, in-flight state (messages, agent status, conversation history) cannot outlive the broker — persisting it to SQLite provides durability against a failure mode that doesn't exist. The only cross-process coordination need is inter-agent messaging, where MCP server instances (separate processes spawned by agents) must write messages that the broker reads. SQLite serves this narrow IPC role. Everything else is either in-memory (ephemeral by nature) or in simple JSON files (small, rarely written, human-editable).

---

## 5. Inter-Agent Messaging

### 5.1 Why MCP (Not ACP) for Agent-to-Agent

ACP is a client↔agent protocol. The client (broker) sends prompts and receives responses. The agent discovers and calls tools from MCP servers that it connects to. ACP does not provide a mechanism for the client to declare tools that agents can call — tool execution flows from agent to MCP server, not from agent to client.

Therefore, agents need an actual MCP server to discover and call `send_message`, `pull_messages`, etc. The broker cannot expose these as ACP-native tools.

### 5.2 Message Flow

```
agent-1 calls send_message(to="agent-2", body="...")
  → agent-1's synth-mcp instance writes to SQLite messages table
  → Broker's MessagePoller detects PRAGMA data_version change (~100ms)
  → Broker reads pending messages for each agent
  → If agent-2 is IDLE: broker sends session/prompt with message content
  → If agent-2 is BUSY: message stays queued, delivered when agent-2 returns to IDLE
```

The "deliver when IDLE" behavior is a significant improvement over `team-mcp`'s tmux `send-keys` approach, which would interrupt a busy agent. With ACP, the broker has full visibility into agent state and can queue intelligently.

### 5.3 MCP Server Config Injection

When the broker launches an agent via `spawn_agent_process`, it passes the MCP server config in the `session/new` call using the SDK's Pydantic models directly:

```python
from acp.schema import McpServerStdio, EnvVariable

mcp_servers = [McpServerStdio(
    name="synth-mcp",
    command="synth-mcp",
    args=[],
    env=[
        EnvVariable(name="SYNTH_SESSION_ID", value=self.session_id),
        EnvVariable(name="SYNTH_DB_PATH", value=str(self.db_path)),
        EnvVariable(name="SYNTH_AGENT_ID", value=agent_id),
    ],
)]

session = await conn.new_session(cwd=agent_config.cwd, mcp_servers=mcp_servers)
```

Note: `env` is a `list[EnvVariable]` (list of `{name, value}` objects), not a dict. Using the SDK's models ensures correct JSON-RPC serialization.

Every agent automatically gets messaging tools without per-agent opt-in.

### 5.4 Broadcast Messages

When an agent sends `send_message(to_agent="*")`, the MCP server writes one row per target agent to SQLite. The broker's poller picks up each row and delivers independently. This keeps the MCP server simple (no need to know the full agent list at write time — it queries the `agents` table).

### 5.5 Topology Support

SYNTH does not enforce a topology. Topology emerges from how agents are prompted and how they use the MCP tools:

- **Human-dispatch**: Human sends prompts to individual agents via the UI. Agents don't message each other.
- **Orchestrator**: One agent is designated coordinator. Human sends top-level task to coordinator, which uses `send_message` to delegate.
- **Peer-to-peer**: Any agent can message any other. The broker's delivery mechanism ensures messages reach idle agents.

The UI provides visibility into message flows regardless of topology.

---

## 6. Layer Boundary Contracts

The broker and frontend communicate through typed Pydantic models. This is the seam where a future web UI would plug in.

### 6.1 Events (Broker → Frontend)

```python
class BrokerEvent(BaseModel, frozen=True):
    """Base for all events the broker emits."""
    timestamp: datetime
    agent_id: str

class AgentStateChanged(BrokerEvent):
    old_state: AgentState
    new_state: AgentState

class MessageChunkReceived(BrokerEvent):
    chunk: str
    message_id: str

class ToolCallUpdated(BrokerEvent):
    tool_call_id: str
    name: str
    kind: str               # read | edit | execute | ...
    status: str             # pending | running | completed | failed
    input_preview: str | None = None

class PermissionRequested(BrokerEvent):
    request_id: str
    title: str
    kind: str
    description: str
    options: list[PermissionOption]
    _future: asyncio.Future  # not serialized; resolved by broker (auto) or UI (manual)

class PermissionAutoResolved(BrokerEvent):
    request_id: str
    rule_decision: PermissionDecision

class McpMessagePending(BrokerEvent):
    from_agent: str
    preview: str

class McpMessageDelivered(BrokerEvent):
    from_agent: str
    to_agent: str

class BrokerError(BrokerEvent):
    """Non-fatal error surfaced to the UI (agent binary not found, handshake timeout, etc.)."""
    message: str
    severity: Literal["warning", "error"] = "error"
```

### 6.2 Commands (Frontend → Broker)

```python
class BrokerCommand(BaseModel, frozen=True):
    """Base for all commands the frontend sends to the broker."""

class LaunchAgent(BrokerCommand):
    agent_id: str

class TerminateAgent(BrokerCommand):
    agent_id: str

class SendPrompt(BrokerCommand):
    agent_id: str
    text: str

class RespondPermission(BrokerCommand):
    agent_id: str
    request_id: str
    option_id: str

class CancelTurn(BrokerCommand):
    agent_id: str
```

### 6.3 Broker Protocol

For type-checking and future extensibility, the broker exposes a protocol that any frontend can implement against:

```python
class BrokerProtocol(Protocol):
    async def handle(self, command: BrokerCommand) -> None: ...
    def events(self) -> AsyncIterator[BrokerEvent]: ...
    def get_agent_states(self) -> dict[str, AgentState]: ...
    def get_agent_config(self, agent_id: str) -> AgentConfig: ...
```

#### `events()` Contract

`broker.events()` is an infinite async iterator that yields `BrokerEvent` instances in order of emission. It terminates (raises `StopAsyncIteration`) when `broker.shutdown()` is called. The backing queue is unbounded — events are never dropped. A slow consumer will cause the queue to grow; this is acceptable because the consumer (TUI) runs on the same event loop and processes events at rendering speed, which is always faster than event emission speed.

Known limitation: the single-queue design supports one consumer. If a future requirement demands simultaneous TUI + web UI, the queue must be replaced with a fan-out/pub-sub mechanism.

### 6.4 Textual Bridge

The Textual app bridges these typed interfaces into Textual's message system:

```python
class TeamACPApp(App):
    def __init__(self, broker: ACPBroker) -> None:
        super().__init__()
        self.broker = broker

    async def on_mount(self) -> None:
        self.run_worker(self._consume_broker_events())

    async def _consume_broker_events(self) -> None:
        async for event in self.broker.events():
            self.post_message(BrokerEventMessage(event))

    async def on_input_bar_submitted(self, msg: InputBar.Submitted) -> None:
        await self.broker.handle(SendPrompt(agent_id=self.active_agent, text=msg.value))
```

A future web backend would replace `TeamACPApp` with a FastAPI application that consumes the same `broker.events()` iterator and exposes the same `broker.handle()` over websocket/HTTP.

---

## 7. Permission Handling

### 7.1 Flow

When an agent needs to perform a sensitive operation (file write, shell exec, network fetch), it sends a `session/request_permission` request via ACP. This is a blocking JSON-RPC call — the agent subprocess waits indefinitely.

```
Agent subprocess sends request_permission
  → ACP SDK calls ACPSession.request_permission()
  → ACPSession creates an asyncio.Future, attaches it to a PermissionRequested event
  → ACPSession transitions to AWAITING_PERMISSION (awaited)
  → ACPSession emits PermissionRequested event to broker via _event_sink
  → Broker checks PermissionEngine for persisted rule
  → If rule exists:
      → Broker calls event._future.set_result(option_id) directly
      → Broker emits PermissionAutoResolved to UI (informational)
  → If no rule:
      → Broker forwards event (with Future) to UI
      → Terminal bell rings, agent sidebar gets ⚠ badge
      → User responds via PermissionRequest widget
      → Widget calls event._future.set_result(option_id)
  → ACPSession.request_permission() awaits the Future, gets option_id
  → ACPSession transitions back to BUSY
  → Agent resumes
```

The Future lives on the event, not as separate state on the session. This eliminates split-state bugs: the session holds `_permission_future: asyncio.Future | None` as a single field, and termination calls `future.cancel()` if it's pending. The broker and UI both resolve the same Future — whoever gets there first wins, and `future.done()` guards against double-resolution.

### 7.2 Decision Options

| Option ID | Label | Behavior |
|---|---|---|
| `allow_once` | Allow once | Respond with approval, no persistence |
| `allow_always` | Always allow | Respond with approval, persist rule |
| `reject_once` | Reject | Respond with rejection, no persistence |
| `reject_always` | Always reject | Respond with rejection, persist rule |

### 7.3 Alerting

When a permission request arrives and no auto-resolve rule matches:
- Terminal bell (`\a`) is rung
- The agent's entry in the sidebar gets a flashing `⚠` badge
- The permission widget appears inline in the conversation feed (not a blocking modal)
- The alert clears when the user responds

### 7.4 Graceful Shutdown

When the user quits, the following sequence is enforced in order:

```python
async def shutdown(self) -> None:
    # 1. Stop accepting new commands
    self._shutting_down = True

    # 2. Cancel all active prompts (so agents can clean up)
    for s in self._sessions.values():
        if s.state == AgentState.BUSY:
            await s.cancel()

    # 3. Stop the message poller (before terminating sessions)
    await self._message_poller.stop()

    # 4. Persist session IDs for resume (while sessions still exist)
    self._persist_session_ids()

    # 5. Terminate all sessions (kills subprocesses, waits for exit)
    for s in list(self._sessions.values()):
        if s.state != AgentState.TERMINATED:
            await s.terminate()

    # 6. Close SQLite
    await self._db.close()
```

The ordering matters: session IDs must be persisted before termination (step 4 before 5), and the poller must stop before sessions are terminated (step 3 before 5) to prevent delivery attempts to dead sessions. The Textual app calls `await self.broker.shutdown()` in `action_quit()` before calling `self.exit()`.

---

## 8. Textual UI

### 8.1 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  SYNTH             [session: dev-project]      F1:Help Q:Quit│
├──────────────────┬──────────────────────────────────────────┤
│  AGENTS          │  agent-1 ● IDLE                          │
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

### 8.2 Widget Hierarchy

Each visual element is a separate, reusable Textual widget:

- `AgentList` — left sidebar showing agent cards with live status indicators. Selecting an agent switches the conversation feed.
- `ConversationFeed` — scrollable container composing child widgets per message:
  - `PromptBubble` — user prompt (right-aligned or labelled "You →")
  - `AgentMessage` — agent response text (streaming, appended in real time)
  - `ToolCallBlock` — collapsible block with icon by kind, status badge, diff/terminal output
  - `PermissionRequest` — inline widget with operation details and four decision buttons
- `MessageQueue` — summary of pending inter-agent MCP messages
- `InputBar` — bottom prompt input, disabled when agent is BUSY or AWAITING_PERMISSION

### 8.3 Key Bindings

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle agent focus |
| `Enter` | Submit prompt |
| `Ctrl+C` | Cancel active prompt turn (`session/cancel`) |
| `l` | Launch new agent |
| `t` | Terminate selected agent |
| `m` | Open message queue panel |
| `F1` | Help overlay |
| `q` / `Ctrl+Q` | Quit |

### 8.4 Styling

All styles live in an external `app.tcss` file. No inline styles in Python code. This enables:
- Fast layout iteration without touching Python
- Theming support (dark/light)
- Consistent styling across widgets

### 8.5 Rendering Strategy

The conversation feed uses two rendering modes depending on whether the agent is the active (visible) panel:

**Active agent (incremental):** Widgets are mounted one at a time as broker events arrive. This is efficient — mount once, update in place:

```
MessageChunkReceived → AgentMessage.append(chunk)
ToolCallUpdated      → mount(ToolCallBlock) or update existing
PermissionRequested  → mount(PermissionRequest)
TurnComplete         → AgentMessage.finalize()
```

**Panel switch (snapshot replay):** When switching to a different agent, the conversation feed is rebuilt from `SessionAccumulator.snapshot()`:

```
snapshot = session.accumulator.snapshot()
ConversationFeed.replay(snapshot.user_messages, snapshot.tool_calls, ...)
```

If the target agent is currently streaming (BUSY), the snapshot contains finalized messages but the in-progress text may be incomplete. The next `MessageChunkReceived` event resumes incremental rendering.

---

## 9. Package Layout

```
synth-acp/
├── pyproject.toml
├── src/
│   └── synth_acp/
│       ├── __init__.py
│       ├── __main__.py                # python -m synth_acp
│       ├── cli.py                     # argparse CLI: synth [--web] [--port N]
│       │
│       ├── models/                    # Pydantic v2 models — shared across all layers
│       │   ├── __init__.py
│       │   ├── agent.py              # AgentConfig, AgentState, SessionInfo
│       │   ├── events.py             # BrokerEvent and all subclasses
│       │   ├── commands.py           # BrokerCommand and all subclasses
│       │   ├── permissions.py        # PermissionRequest, PermissionRule, PermissionDecision
│       │   └── config.py             # SessionConfig (parsed from .synth.json)
│       │
│       ├── acp/                      # Layer 1: ACP protocol wrapper
│       │   ├── __init__.py
│       │   └── session.py            # ACPSession — wraps acp SDK Client
│       │
│       ├── broker/                   # Layer 2: Orchestration (zero Textual imports)
│       │   ├── __init__.py
│       │   ├── broker.py             # ACPBroker — session lifecycle, event routing
│       │   ├── permissions.py        # PermissionEngine — rule persistence + auto-resolve
│       │   └── poller.py             # MessagePoller — SQLite PRAGMA data_version watcher
│       │
│       ├── mcp/                      # MCP server (shipped as synth-mcp entrypoint)
│       │   ├── __init__.py
│       │   └── server.py             # FastMCP: send_message, pull_messages, etc.
│       │
│       └── ui/                       # Layer 3: Textual frontend
│           ├── __init__.py
│           ├── app.py                # SynthApp — bridges broker ↔ Textual messages
│           ├── messages.py           # Textual Message subclasses wrapping BrokerEvent
│           ├── screens/
│           │   ├── __init__.py
│           │   └── dashboard.py      # Main dashboard screen
│           ├── widgets/
│           │   ├── __init__.py
│           │   ├── agent_list.py     # Left sidebar agent cards
│           │   ├── conversation.py   # ConversationFeed container
│           │   ├── prompt_bubble.py  # User prompt display
│           │   ├── agent_message.py  # Streaming agent response
│           │   ├── tool_call.py      # Collapsible tool call block
│           │   ├── permission.py     # Inline permission request
│           │   ├── message_queue.py  # MCP message summary
│           │   └── input_bar.py      # Bottom prompt input
│           └── css/
│               └── app.tcss          # All Textual CSS
│
└── tests/
    ├── conftest.py
    ├── test_models/
    ├── test_acp/
    ├── test_broker/
    └── test_mcp/
```

---

## 10. Configuration

### 10.1 `.synth.json`

Placed at the project root. Defines the agent team for a session.

```json
{
  "session": "my-project",
  "agents": [
    {
      "id": "coordinator",
      "binary": "kiro-cli",
      "args": ["acp"],
      "cwd": ".",
      "autostart": true
    },
    {
      "id": "kiro-auth",
      "binary": "kiro-cli",
      "args": ["acp"],
      "cwd": "./src/auth",
      "autostart": false
    },
    {
      "id": "researcher",
      "binary": "claude",
      "args": ["mcp"],
      "cwd": ".",
      "autostart": false
    }
  ],
  "ui": {
    "web_port": 8000,
    "theme": "dark"
  }
}
```

Agent identity is the `id` field — the same binary can appear multiple times with different IDs and they are managed independently.

### 10.2 Pydantic Config Model

```python
class AgentConfig(BaseModel, frozen=True):
    id: str
    binary: str
    args: list[str] = []
    cwd: str = "."
    autostart: bool = False

    @field_validator("id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                f"Agent ID '{v}' must match [a-zA-Z0-9][a-zA-Z0-9_-]* "
                f"(no dots, spaces, or special chars)"
            )
        return v

class UIConfig(BaseModel, frozen=True):
    web_port: int = 8000
    theme: str = "dark"

class SessionConfig(BaseModel, frozen=True):
    session: str
    agents: list[AgentConfig]
    ui: UIConfig = UIConfig()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SessionConfig":
        ids = [a.id for a in self.agents]
        dupes = [x for x in ids if ids.count(x) > 1]
        if dupes:
            raise ValueError(f"Duplicate agent IDs: {set(dupes)}")
        return self
```

Agent ID format is restricted because IDs are used as Textual CSS identifiers (e.g. `query_one(f"#tile-{agent_id}")`). A dot in the ID makes `.agent` parse as a CSS class selector, producing confusing `NoMatches` exceptions at runtime.

Relative `cwd` paths are resolved against the config file's parent directory at load time, not the process CWD. This ensures `synth` works correctly regardless of where it's invoked from.

### 10.3 Data Directory (`~/.synth/`)

```
~/.synth/
├── rules.json          # Persisted permission rules
├── sessions.json       # Session resume IDs (written on graceful shutdown)
└── synth.db            # SQLite database (inter-agent messages)
```

---

## 11. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | asyncio native, consistent with team-mcp |
| TUI + future Web | Textual | Single codebase serves terminal and browser via `textual serve` |
| ACP transport | JSON-RPC 2.0 over stdio | Standard ACP; each agent is a subprocess |
| ACP library | `agent-client-protocol` (PyPI) | Official Zed/JetBrains SDK; Pydantic models, async stdio, `spawn_agent_process`, contrib helpers |
| Models | Pydantic v2 | Already a dependency via ACP SDK; fast Rust core; free JSON serialization |
| MCP server | FastMCP | Retained from team-mcp for agent-to-agent messaging |
| IPC (MCP↔broker) | SQLite (WAL mode) | Zero-config cross-process coordination; proven in team-mcp |
| Package manager | uv | Fast, modern Python packaging |
| CLI | argparse (stdlib) | Minimal subcommands; no external dependency needed |

---

## 12. Design Decisions and Rejected Alternatives

### 12.1 Pydantic v2 over dataclasses

**Chosen**: Pydantic v2 `BaseModel` for all cross-layer types.

**Why**: The ACP SDK already depends on Pydantic, so it's not a new dependency. Pydantic v2's Rust core makes model instantiation nearly as fast as dataclasses. The consistency benefit (ACP types are Pydantic, SYNTH types are Pydantic, JSON serialization is free) outweighs marginal overhead. Frozen models (`frozen=True`) provide immutability.

**Rejected**: `@dataclass(frozen=True, slots=True)` — would work but loses free JSON serialization, validation, and consistency with the ACP SDK's type system.

### 12.2 In-memory broker state over SQLite for everything

**Chosen**: Agent state, conversation history, and session metadata live in broker memory. SQLite is used only for inter-agent messaging (cross-process IPC).

**Why**: The broker owns all agent lifecycles. If it dies, agents die. Persisting ephemeral state to SQLite provides durability against a failure mode that doesn't exist. See Section 4.4 for the full defense.

**Rejected**: SQLite for all state — adds write overhead, schema migrations, and complexity for zero durability benefit.

### 12.3 JSON file over SQLite for permission rules

**Chosen**: `~/.synth/rules.json` for persisted allow/reject rules.

**Why**: The dataset is small (tens of rules), writes are rare (only on "always allow/reject"), there's a single writer (the broker), and the file is human-readable/editable.

**Rejected**: SQLite table — overkill for a single small collection with no concurrent writers and no query requirements beyond key lookup.

### 12.4 SQLite message bus over direct IPC

**Chosen**: MCP server instances write to SQLite; broker polls via `PRAGMA data_version`.

**Why**: MCP servers are separate processes spawned by agents, not by the broker. They have no direct IPC channel to the broker. SQLite serves as a zero-configuration IPC mechanism — every process just opens the same file. The alternatives (Unix sockets, HTTP, named pipes) all add connection management complexity.

**Rejected**: Unix socket between MCP server and broker — requires the broker to run a socket server, MCP servers to have connection/retry logic, and a new IPC protocol alongside ACP. More complex for marginal latency improvement (100ms poll vs instant).

**Rejected**: Broker-declared ACP tools — ACP does not support the client declaring tools for agents to call. Tool discovery flows from agent to MCP server, not from agent to client.

### 12.5 Composition over inheritance for ACPSession

**Chosen**: `ACPSession` wraps the ACP SDK's `Client` interface via composition.

**Why**: The SDK's classes aren't designed for deep inheritance. Composition is more stable across SDK version bumps and makes the boundary between SYNTH code and SDK code explicit.

**Rejected**: Subclassing SDK internals — brittle across version updates, unclear ownership of overridden methods.

### 12.6 Separate widgets over monolithic conversation renderer

**Chosen**: Each visual element (prompt bubble, agent message, tool call block, permission request) is a separate Textual widget composed inside a `ConversationFeed` container.

**Why**: Reusable, independently testable, follows Textual's message-driven architecture. Each widget handles its own rendering and responds to its own subset of events.

**Rejected**: Single `ConversationFeed` widget that renders everything internally — simpler initially but becomes a monolith as features grow, harder to test individual elements.

### 12.7 argparse over click/typer

**Chosen**: stdlib `argparse` for the CLI.

**Why**: SYNTH has a handful of subcommands (`synth`, `synth init-token`, `synth rotate-token`). Adding a dependency for this is unjustified.

**Rejected**: Typer/Click — good libraries but unnecessary for the command surface area.

### 12.8 Single-consumer event stream over broadcast

**Chosen**: `broker.events()` returns a single `AsyncIterator[BrokerEvent]` backed by an `asyncio.Queue`.

**Why**: SYNTH runs one frontend at a time (TUI or web, not both simultaneously). A single-consumer queue is simpler and avoids fan-out complexity.

**Rejected**: Broadcast channel — needed only if TUI and web UI run simultaneously, which is not a goal. Can be added later if requirements change.

---

## 13. Implementation Phases

### Phase 1 — ACP Core (no UI)

- `ACPSession`: subprocess management via `spawn_agent_process`, state machine, event emission using `SessionAccumulator`
- `ACPBroker`: session lifecycle, permission auto-resolution, event queue
- `PermissionEngine`: JSON-backed rule persistence
- `MessagePoller`: SQLite `PRAGMA data_version` watcher, message delivery via `session/prompt`
- `synth-mcp`: FastMCP server (send_message, pull_messages, check_delivery, list_agents, deregister_agent)
- SQLite schema (agents, messages tables)
- Pydantic models for events, commands, config
- CLI entrypoint (`synth`) that starts the broker headlessly for testing
- Tests for broker, session state machine, permission engine, message routing

### Phase 2 — Textual TUI

- `SynthApp` with broker event bridge
- `AgentList` widget with live status and `⚠` AWAITING_PERMISSION highlight
- `ConversationFeed` with streaming chunks and tool call blocks
- `PermissionRequest` widget with all four decision options
- Terminal bell on permission request
- `InputBar` with send/cancel
- `MessageQueue` widget
- Keyboard navigation
- External CSS (`app.tcss`)

### Phase 3 — Web Mode + Polish

- `textual serve` integration
- Web authentication (shared-secret token)
- Session resume UI
- Permission rules management screen
- Configuration file editor
- JSONL audit log (optional persistence of broker events)

### Phase 4 — Advanced Features

- Topology visualization (message flow graph)
- Broadcast message support in UI
- Orchestrator mode: auto-forward task outputs as inputs to next agent
- Session templates (pre-defined agent team configs)
- Dynamic agent management: add `launch_agent` / `terminate_agent` to `synth-mcp` (routed through SQLite to broker, with parentage tracking and rate limiting) to enable agents to spawn/terminate other agents for autonomous orchestration

---

## 14. Entrypoints

```toml
[project.scripts]
synth = "synth_acp.cli:main"
synth-mcp = "synth_acp.mcp.server:main"
```

---

## 15. Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "agent-client-protocol",   # ACP SDK (brings pydantic v2)
    "textual",                 # TUI framework
    "fastmcp",                 # MCP server for agent-to-agent messaging
    "aiosqlite",               # Async SQLite for message poller
]
```

All other needs (argparse, asyncio, sqlite3, json, pathlib) are stdlib.
