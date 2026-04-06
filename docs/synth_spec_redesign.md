# synth-acp: Architecture Redesign Spec

**For:** opus-4.6 coding agent  
**Repo:** `synth-acp` (`src/synth_acp/`)  
**Toolchain:** `uv run pytest -q --tb=short --no-header -rF` | `ruff check --fix --output-format concise` | `ty check --output-format concise src/ tests/`

Read `AGENTS.md` before starting. All changes must pass lint, type-check, and the full test suite **after every phase**. Do not proceed to the next phase until the current one is green.

**Key constraints from `AGENTS.md` that interact with this redesign:**

- **Layer dependency rules:** `acp/` may only import from `models/`. `broker/` may import from `acp/` and `models/`. `ui/` may import from any layer. `mcp/` is conceptually part of the broker layer. The new `state_machine.py` in `acp/` must NOT import from `broker/`. The new `message_bus.py` in `broker/` CAN import from `acp/`.
- **One test file per source module.** New modules get new test files: `test_state_machine.py`, `test_registry.py`, `test_lifecycle.py`, `test_message_bus.py`, `test_notifier.py`. Use test classes within the file to organize by feature.
- **Max 5 tests per source function.** Each test must answer: "What real bug does this catch that would otherwise fail silently?"
- **`from __future__ import annotations`** in all new files.
- **Google-style docstrings.**

**Import conventions:** All `__init__.py` files in this project are empty. Use direct module imports (e.g., `from synth_acp.acp.state_machine import AgentStateMachine`). Do not add re-exports to `__init__.py`.

---

## Overview

This spec decomposes three over-coupled files — `broker.py` (820 lines), `session.py` (743 lines), and `server.py` (287 lines) — into focused components with explicit interfaces. The redesign fixes every known production bug by making each one structurally impossible in the new architecture rather than patching them individually.

The redesign has seven phases, each producing a shippable, fully tested intermediate state:

| Phase | Summary |
|-------|---------|
| 1 | Typed event unions + annotation fix — gives `ty` a clean baseline |
| 2 | Extract `AgentStateMachine` from session — fixes state transition bugs |
| 3 | MCP server dependency injection — fixes connection leaks and startup crashes |
| 4 | Extract `AgentRegistry` from broker — separates data from orchestration |
| 5 | Replace poller with `MessageBus` + notification channel — fixes deadlocks and message loss |
| 6 | Extract `AgentLifecycle` from broker — fixes task accumulation |
| 7 | Slim broker coordinator + structured shutdown — fixes shutdown ordering and memory growth |

---

## Target Architecture

### File map after all phases

```
src/synth_acp/
  acp/
    session.py           # slimmed — delegates state to state_machine
    state_machine.py     # NEW — AgentStateMachine
  broker/
    broker.py            # slimmed to ~250 lines — thin coordinator
    registry.py          # NEW — AgentRegistry (data + queries)
    lifecycle.py         # NEW — AgentLifecycle (launch/terminate/prompt)
    message_bus.py       # NEW — replaces poller.py
    poller.py            # DELETED
    permissions.py       # unchanged
  mcp/
    server.py            # refactored — factory + DI, no module globals
    notifier.py          # NEW — BrokerNotifier (socket client)
  models/
    agent.py             # state machine classes moved to acp/state_machine.py
    events.py            # adds typed union aliases
    ...                  # other model files unchanged
  ui/
    app.py               # annotation fix only
    ...                  # other UI files unchanged
```

### Public API contract (unchanged)

The `ACPBroker` class retains the same public methods consumed by `app.py` and `input_bar.py`. All UI call sites remain unmodified:

- `broker.handle(command: BrokerCommand) -> None`
- `broker.events() -> AsyncIterator[BrokerEvent]`
- `broker.shutdown() -> None`
- `broker.get_agent_parent(agent_id) -> str | None`
- `broker.get_agent_harness(agent_id) -> str`
- `broker.is_permission_pending(agent_id) -> bool`
- `broker.get_usage(agent_id) -> UsageUpdated | None`
- `broker.get_agent_states() -> dict[str, AgentState]`
- `broker.get_agent_configs() -> list[AgentConfig]`
- `broker.get_agent_modes(agent_id) -> list[AgentMode]`
- `broker.get_current_mode(agent_id) -> str | None`
- `broker.get_agent_models(agent_id) -> list[AgentModel]`
- `broker.get_current_model(agent_id) -> str | None`

These become thin delegations to internal components. No UI file changes are required except the annotation fix in Phase 1.

---

## Phase 1 — Typed Event Unions + Annotation Fix

**Goal:** Give `ty` a clean baseline before any structural changes. This is entirely independent — no code moves, no class structure changes — and it unlocks type-level checking for every subsequent phase.

### Why this is needed

In `src/synth_acp/ui/app.py`, the handler at line 439 annotates its parameter as `SynthApp.WorkerStateChanged` — a type that does not exist anywhere in Textual or in the codebase. The correct type is `Worker.StateChanged` from `textual.worker`. The handler works at runtime because Python does not enforce annotations, but `ty` flags it.

### Modify `src/synth_acp/models/events.py`

Add union type aliases at the bottom of the file:

```python
from typing import TypeAlias

AgentEvent: TypeAlias = (
    AgentStateChanged | MessageChunkReceived | ToolCallUpdated | TurnComplete
    | AgentThoughtReceived | PlanReceived | AvailableCommandsReceived | TerminalCreated
)

ConfigEvent: TypeAlias = (
    AgentModesReceived | AgentModeChanged | AgentModelsReceived | AgentModelChanged
)

SystemEvent: TypeAlias = (
    BrokerError | PermissionRequested | PermissionAutoResolved
    | UsageUpdated | McpMessageDelivered
)

BrokerEventUnion: TypeAlias = AgentEvent | ConfigEvent | SystemEvent
```

Keep the `BrokerEvent` base class as-is for runtime Pydantic serialization. The type aliases are for static analysis.

### Fix `src/synth_acp/ui/app.py`

Update the import at line 14:
```python
from textual.worker import Worker, WorkerState
```

Update the handler annotation at line 439:
```python
def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
```

Body unchanged.

### Tests

No new test files. `uv run ty check --output-format concise src/` validates the annotation fix.

---

## Phase 2 — Extract `AgentStateMachine`

**Goal:** Pull the state transition logic into a first-class object that eliminates two classes of bugs: `InvalidTransitionError` crashes in cleanup paths, and missing state guards on configuration RPCs.

### Why this is needed

`session.py` currently manages state via a direct `self.state` attribute and an `_set_state()` method. This has three problems:

1. The `run()` method's `except Exception` catches `InvalidTransitionError` alongside genuine agent errors, producing confusing error messages with no actionable context.

2. The `finally` block in `run()` calls `_set_state(AgentState.TERMINATED)` unconditionally. If the session is already TERMINATED (via a concurrent path), this raises `InvalidTransitionError` inside the `finally` block, which Python silently discards — hiding the real failure.

3. `set_model()` does not transition through CONFIGURING before its RPC, unlike `set_mode()` which does. This allows a prompt to be dispatched concurrently with a model switch — undefined behavior for ACP agents.

A state machine object with a safe `force_terminal()` method and validated `transition()` eliminates all three.

Additionally, `SessionAccumulator.subscribe()` returns an unsubscribe callable, but session.py discards it at line 173. This leaks the subscription — the accumulator holds a reference to the session's `_on_snapshot` callback, preventing garbage collection after termination.

### Create `src/synth_acp/acp/state_machine.py`

```python
"""Agent state machine — single source of truth for lifecycle state."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from synth_acp.models.agent import TRANSITIONS, AgentState, InvalidTransitionError

log = logging.getLogger(__name__)

TransitionCallback = Callable[[AgentState, AgentState], Awaitable[None]]


class AgentStateMachine:
    """Encapsulates validated state transitions with an async notification hook.

    Args:
        agent_id: Identifier used in log messages and errors.
        on_transition: Async callback invoked after every state change with (old, new).
    """

    def __init__(self, agent_id: str, on_transition: TransitionCallback) -> None:
        self._agent_id = agent_id
        self._state = AgentState.UNSTARTED
        self._on_transition = on_transition

    @property
    def state(self) -> AgentState:
        return self._state

    async def transition(self, new_state: AgentState) -> None:
        """Validated transition. Raises InvalidTransitionError if disallowed."""
        if new_state not in TRANSITIONS[self._state]:
            raise InvalidTransitionError(f"{self._agent_id}: {self._state} → {new_state}")
        old = self._state
        self._state = new_state
        await self._on_transition(old, new_state)

    async def force_terminal(self) -> None:
        """Unconditional transition to TERMINATED. Idempotent.

        Use for cleanup paths (finally blocks, voluntary exit) where
        raising InvalidTransitionError would be harmful.
        """
        if self._state == AgentState.TERMINATED:
            return
        old = self._state
        self._state = AgentState.TERMINATED
        await self._on_transition(old, AgentState.TERMINATED)
```

### Modify `src/synth_acp/acp/session.py`

1. **Replace direct state management with the state machine.** In `__init__`:
   - Remove `self.state = AgentState.UNSTARTED`
   - Add `self._sm = AgentStateMachine(agent_id, self._on_state_transition)`
   - Add a read-only state property: `@property` `def state(self) -> AgentState: return self._sm.state`

2. **Add public accessors** so other components never reach into private attributes:
   ```python
   @property
   def session_id(self) -> str | None:
       """The ACP session ID, or None if not yet initialized."""
       return self._session_id

   async def force_terminate(self) -> None:
       """Force transition to TERMINATED. Safe for cleanup paths and voluntary exit.

       Other components (broker, lifecycle) call this instead of accessing
       _sm directly to maintain encapsulation.
       """
       await self._sm.force_terminal()
   ```

3. **Add the transition callback:**
   ```python
   async def _on_state_transition(self, old: AgentState, new: AgentState) -> None:
       await self._event_sink(
           AgentStateChanged(agent_id=self.agent_id, old_state=old, new_state=new)
       )
   ```

4. **Replace all `await self._set_state(X)` calls** with `await self._sm.transition(X)`. Remove the `_set_state` method entirely.

5. **Fix `run()` exception handling.** The current single `except Exception` block becomes three:
   ```python
   except InvalidTransitionError as e:
       log.error("Invalid state transition in session %s: %s", self.agent_id, e, exc_info=True)
       await self._event_sink(
           BrokerError(agent_id=self.agent_id, message=f"Internal state error: {e}", severity="error")
       )
   except asyncio.CancelledError:
       log.debug("Session %s cancelled", self.agent_id)
       raise
   except Exception as e:
       log.error("Session %s raised unexpectedly", self.agent_id, exc_info=True)
       await self._event_sink(BrokerError(agent_id=self.agent_id, message=f"Agent error: {e}"))
   ```

6. **Fix `run()` finally block.** Replace the bare `await self._set_state(AgentState.TERMINATED)` with:
   ```python
   await self._sm.force_terminal()
   ```

7. **Store the unsubscribe callable.** In `__init__`, change:
   ```python
   self._accumulator.subscribe(self._on_snapshot)
   ```
   to:
   ```python
   self._unsubscribe: Callable[[], None] = self._accumulator.subscribe(self._on_snapshot)
   ```
   `Callable` is already imported from `collections.abc` at line 11. In the `finally` block of `run()`, call `self._unsubscribe()` before `force_terminal()`. This ordering is safe: the accumulator subscription only fires during `session_update` notifications from the ACP agent process, which have already stopped by the time the `finally` block runs (the process has exited or the context manager has closed the connection). The `force_terminal()` call emits `AgentStateChanged` directly through `_event_sink`, bypassing the accumulator entirely.

8. **Add CONFIGURING guard to `set_model()`.** Rewrite to match `set_mode()`'s pattern:
   ```python
   async def set_model(self, model_id: str) -> None:
       if self._conn and self._session_id and self.state == AgentState.IDLE:
           await self._sm.transition(AgentState.CONFIGURING)
           try:
               await self._conn.set_session_model(model_id=model_id, session_id=self._session_id)
               self._current_model_id = model_id
               await self._event_sink(AgentModelChanged(agent_id=self.agent_id, model_id=model_id))
           finally:
               if self.state == AgentState.CONFIGURING:
                   await self._sm.transition(AgentState.IDLE)
   ```

### Tests

**`tests/acp/test_state_machine.py`** (new file):
- `test_valid_transition_updates_state`: Create machine, transition `UNSTARTED → INITIALIZING`, assert state is `INITIALIZING`.
- `test_invalid_transition_raises`: Assert `UNSTARTED → IDLE` raises `InvalidTransitionError`.
- `test_callback_receives_old_and_new`: Assert the callback is called with `(UNSTARTED, INITIALIZING)`.
- `test_force_terminal_from_any_state`: For each state in `AgentState`, call `force_terminal()`, assert state is `TERMINATED`.
- `test_force_terminal_is_idempotent`: Call `force_terminal()` twice, assert no exception and callback called only once.

**`tests/acp/test_session.py`** — add to existing or new class:
- `test_set_model_transitions_through_configuring`: Mock `_conn.set_session_model`. Call `session.set_model("new-model")`. Assert the sink received `AgentStateChanged(CONFIGURING)` then `AgentStateChanged(IDLE)` and `AgentModelChanged`.
- `test_set_model_restores_idle_on_rpc_failure`: Make `set_session_model` raise. Assert state is `IDLE` after.
- `test_cancelled_error_not_emitted_as_broker_error`: Cancel the `run()` task. Assert no `BrokerError` emitted.
- `test_finally_uses_force_terminal`: Put session in TERMINATED via an error path, trigger finally. Assert no exception propagates.
- `test_unsubscribe_called_on_session_exit`: Mock the accumulator. Assert `_unsubscribe` was called in the finally block.
- `test_session_id_property`: Assert `session.session_id` returns `None` before init and the correct value after.
- `test_force_terminate_public_method`: Call `session.force_terminate()`. Assert state is `TERMINATED`.

---

## Phase 3 — MCP Server Dependency Injection

**Goal:** Eliminate module-level mutable state from the MCP server. This fixes two bugs: DB connections leak when exceptions occur between `_get_db()` and `conn.close()` (there is no `try/finally`), and running `synth-mcp` standalone with empty env vars creates a corrupt database file.

### Why this is needed

Every tool function in `server.py` manually opens a connection via `conn = await _get_db()` and manually closes it at each return path. If any intermediate `await` raises (e.g., `_get_visible_agents_async`, `_ensure_registered`), `conn.close()` is never called — a file descriptor leak that compounds over the session lifetime.

The module reads `DB_PATH`, `SESSION_ID`, `AGENT_ID` as module-level globals at import time via `os.environ.get(..., "")`. If any are empty, `aiosqlite.connect("")` creates a file literally named `""` or fails with a confusing OS error. The `main()` function has no validation.

Additionally, `deregister_agent()` does not filter by `session_id` in its WHERE clause, and it only updates the `agents` table without notifying the broker — so the TUI never learns the agent has deregistered. This phase fixes the WHERE clause and adds a `self_terminate` command insertion; the broker-side handling comes in Phase 5. Note: the SQLite database is ephemeral — created fresh per session under `~/.synth/` — so no data migration is needed for the WHERE clause change.

### Refactor `src/synth_acp/mcp/server.py`

Replace the module-level globals and tool functions with a factory that returns a configured `FastMCP` instance:

```python
"""synth-mcp — FastMCP server for inter-agent messaging via SQLite."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import aiosqlite
from mcp.server.fastmcp import FastMCP

from synth_acp.db import ensure_schema_async


NotifyFn = Callable[[], Awaitable[None]]


async def _noop_notify() -> None:
    """Default no-op notifier used until the notification channel is wired in Phase 5."""


def create_mcp_server(
    db_path: str,
    session_id: str,
    agent_id: str,
    communication_mode: str = "MESH",
    notify: NotifyFn = _noop_notify,
) -> FastMCP:
    """Create a configured synth-mcp server instance.

    All tool functions close over the provided parameters instead of
    reading module-level globals. The notify callback is called after
    every SQLite commit that creates data the broker needs to see.
    """
    mcp = FastMCP("synth-mcp")

    # Schema is ensured once at server creation, not on every connection.
    _schema_ensured = False

    @asynccontextmanager
    async def _db_conn() -> AsyncIterator[aiosqlite.Connection]:
        nonlocal _schema_ensured
        if not db_path:
            raise RuntimeError("SYNTH_DB_PATH is not set — synth-mcp must be launched by synth")
        conn = await aiosqlite.connect(db_path)
        try:
            await conn.execute("PRAGMA journal_mode=WAL")
            if not _schema_ensured:
                await ensure_schema_async(conn)
                _schema_ensured = True
            yield conn
        finally:
            await conn.close()

    # Define all @mcp.tool() functions here, closing over db_path, session_id,
    # agent_id, communication_mode, and notify.
    #
    # Rules:
    # - Every tool uses `async with _db_conn() as conn:`. No manual conn.close().
    # - Every tool that commits data the broker should see calls `await notify()`
    #   after conn.commit() (send_message, launch_agent, terminate_agent, deregister_agent).
    # - Move _ensure_registered and _get_visible_agents_async inside the factory.

    return mcp
```

**`deregister_agent()` gets two fixes:**
```python
@mcp.tool()
async def deregister_agent() -> str:
    async with _db_conn() as conn:
        await conn.execute(
            "UPDATE agents SET status = 'inactive' WHERE agent_id = ? AND session_id = ?",
            (agent_id, session_id),
        )
        now = int(time.time() * 1000)
        await conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, ?, 'self_terminate', '{}', 'pending', ?)",
            (session_id, agent_id, now),
        )
        await conn.commit()
    await notify()
    return json.dumps({"status": "inactive", "agent_id": agent_id})
```

The `AND session_id = ?` addition is a defense-in-depth fix. The SQLite database is ephemeral — created fresh for each synth session and not persisted across runs — so there is no existing production data to migrate.

**Remove all module-level globals:** `SESSION_ID`, `DB_PATH`, `AGENT_ID`, `COMMUNICATION_MODE`, and the `_get_db` function.

**Update `main()` with startup validation:**

```python
def main() -> None:
    """Entry point for the synth-mcp CLI."""
    db_path = os.environ.get("SYNTH_DB_PATH", "")
    session_id = os.environ.get("SYNTH_SESSION_ID", "")
    agent_id = os.environ.get("SYNTH_AGENT_ID", "")
    communication_mode = os.environ.get("SYNTH_COMMUNICATION_MODE", "MESH")

    missing = [
        name for name, val in [
            ("SYNTH_SESSION_ID", session_id),
            ("SYNTH_DB_PATH", db_path),
            ("SYNTH_AGENT_ID", agent_id),
        ]
        if not val
    ]
    if missing:
        print(
            f"synth-mcp: missing required environment variables: {', '.join(missing)}\n"
            "This tool is launched automatically by synth. Do not run it directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    server = create_mcp_server(db_path, session_id, agent_id, communication_mode)
    server.run(transport="stdio")
```

Note: `BrokerNotifier.close()` is intentionally not called in `main()`. The `synth-mcp` process is a subprocess that gets killed by `_spawn_isolated_agent`'s cleanup — it never reaches a graceful shutdown path. The socket's OS-level close on process exit is sufficient cleanup.

### Update tests — `tests/mcp/test_server.py`

The existing `_env` fixture that patches module globals is no longer needed. Replace it with a fixture that calls `create_mcp_server` directly. Access tool functions through the server or by restructuring the factory to expose them for testing.

**New tests:**

`TestMcpConnectionSafety`:
- `test_send_message_closes_conn_on_error`: Inject a failing visibility function. Call `send_message(...)`. Verify a fresh `aiosqlite.connect(db_path)` succeeds afterward.
- `test_list_agents_closes_conn_on_register_error`: Similar, with a failing `_ensure_registered`.

`TestMcpStartupValidation`:
- `test_main_exits_with_missing_env_vars`: Use `monkeypatch` to clear the env vars, call `main()`, assert `SystemExit(1)`.
- `test_main_exits_with_empty_db_path`: Set only `SYNTH_DB_PATH=""`, assert `SystemExit(1)`.

`TestDeregisterAgent`:
- `test_deregister_inserts_self_terminate_command`: Call `deregister_agent()`. Query `agent_commands`. Assert a row with `command = 'self_terminate'` exists.
- `test_deregister_filters_by_session_id`: Assert the UPDATE includes `AND session_id = ?`.

---

## Phase 4 — Extract `AgentRegistry`

**Goal:** Pull all agent metadata storage and query methods out of the broker into a focused data object.

### Create `src/synth_acp/broker/registry.py`

```python
"""AgentRegistry — owns agent sessions and metadata."""

from __future__ import annotations

import logging

from synth_acp.acp.session import ACPSession
from synth_acp.models.agent import AgentConfig, AgentMode, AgentModel, AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import UsageUpdated

log = logging.getLogger(__name__)


class AgentRegistry:
    """Central store for agent sessions, parentage, harness info, and usage.

    Pure data object — no I/O, no async, no tasks. The get_modes/get_models
    methods delegate to session properties which are currently synchronous
    list copies. If those properties ever become async or acquire side effects,
    this contract needs revisiting.
    """

    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._sessions: dict[str, ACPSession] = {}
        self._parents: dict[str, str | None] = {a.agent_id: None for a in config.agents}
        self._harnesses: dict[str, str] = {a.agent_id: a.harness for a in config.agents}
        self._usage: dict[str, UsageUpdated] = {}

    def register(self, agent_id: str, session: ACPSession) -> None:
        self._sessions[agent_id] = session

    def unregister(self, agent_id: str) -> ACPSession | None:
        return self._sessions.pop(agent_id, None)

    def get_session(self, agent_id: str) -> ACPSession | None:
        return self._sessions.get(agent_id)

    def has_session(self, agent_id: str) -> bool:
        return agent_id in self._sessions

    def all_sessions(self) -> dict[str, ACPSession]:
        return dict(self._sessions)

    def set_parent(self, agent_id: str, parent: str | None) -> None:
        self._parents[agent_id] = parent

    def get_parent(self, agent_id: str) -> str | None:
        return self._parents.get(agent_id)

    def set_harness(self, agent_id: str, harness: str) -> None:
        self._harnesses[agent_id] = harness

    def get_harness(self, agent_id: str) -> str:
        return self._harnesses.get(agent_id, "")

    def orphan_children(self, parent_id: str) -> None:
        for aid, p in self._parents.items():
            if p == parent_id:
                self._parents[aid] = None

    def update_usage(self, event: UsageUpdated) -> None:
        prev = self._usage.get(event.agent_id)
        if prev is not None and (
            event.cost_currency is not None
            and prev.cost_currency is not None
            and event.cost_currency != prev.cost_currency
        ):
            log.warning(
                "cost_currency changed for %s: %s → %s",
                event.agent_id, prev.cost_currency, event.cost_currency,
            )
        self._usage[event.agent_id] = event

    def get_usage(self, agent_id: str) -> UsageUpdated | None:
        return self._usage.get(agent_id)

    def get_states(self) -> dict[str, AgentState]:
        return {aid: s.state for aid, s in self._sessions.items()}

    def get_configs(self) -> list[AgentConfig]:
        return list(self._config.agents)

    def get_modes(self, agent_id: str) -> list[AgentMode]:
        s = self._sessions.get(agent_id)
        return s.available_modes if s else []

    def get_current_mode(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        return s.current_mode_id if s else None

    def get_models(self, agent_id: str) -> list[AgentModel]:
        s = self._sessions.get(agent_id)
        return s.available_models if s else []

    def get_current_model(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        return s.current_model_id if s else None

    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.state != AgentState.TERMINATED)
```

### Modify `src/synth_acp/broker/broker.py`

1. Create `self._registry = AgentRegistry(config)` in `__init__`.
2. Remove `_sessions`, `_agent_parents`, `_agent_harnesses`, `_usage` from broker.
3. Remove `_accumulate_usage` and all `get_*` query methods.
4. Add one-line delegation methods for each public query to preserve the UI contract:
   ```python
   def get_agent_parent(self, agent_id: str) -> str | None:
       return self._registry.get_parent(agent_id)
   # ... etc
   ```
5. Update all internal references to use the registry.

### Tests

**`tests/broker/test_registry.py`** (new file):
- `test_register_and_get_session`
- `test_unregister_returns_session`
- `test_orphan_children`
- `test_usage_tracking_keeps_latest`
- `test_usage_warns_on_currency_change` (use `caplog`)
- `test_active_count`

Update existing `tests/broker/test_broker.py` as needed — the public API is unchanged.

---

## Phase 5 — Notification Channel + Message Bus

**Goal:** Replace the 100ms SQLite polling loop with edge-triggered socket notifications and a 2-second fallback poll.

### Why this is needed

**Deadlock:** `MessagePoller.stop()` sets `self._stopped = True` then unconditionally `await self._task` without cancelling it. If `_deliver_pending` is mid-flight awaiting an agent response, `stop()` blocks indefinitely.

**Self-deregistration:** When an agent calls `deregister_agent()`, it marks itself inactive in SQLite. The poller only watches for new messages and commands, not status changes. The TUI never learns the agent deregistered. Phase 3 added a `self_terminate` command insertion; this phase adds the broker-side handler.

**Dropped messages:** `_pending_initial_prompts` is `dict[str, str]`. A second message before the agent reaches IDLE silently overwrites the first.

**Latency:** The 100ms poll interval imposes a floor on inter-agent message delivery. Multi-agent coordination workflows pay 100ms per round-trip.

### Create `src/synth_acp/mcp/notifier.py`

```python
"""BrokerNotifier — socket client for MCP→broker wake-up signals."""

from __future__ import annotations

import asyncio
import contextlib
import logging

log = logging.getLogger(__name__)


class BrokerNotifier:
    """Persistent connection to the broker's notification socket.

    Sends a 1-byte wake-up signal after each SQLite commit. If the
    connection fails, the signal is silently dropped — the broker's
    fallback poll catches the change within 2 seconds.

    Lifecycle: created in synth-mcp's main() and never explicitly closed.
    The synth-mcp process is a subprocess killed by _spawn_isolated_agent's
    cleanup, so OS-level socket close on process exit is the cleanup path.
    """

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._writer: asyncio.StreamWriter | None = None

    async def notify(self) -> None:
        try:
            if self._writer is None or self._writer.is_closing():
                _, self._writer = await asyncio.open_unix_connection(self._socket_path)
            self._writer.write(b"\x01")
            await self._writer.drain()
        except (OSError, ConnectionError):
            self._writer = None

    async def close(self) -> None:
        if self._writer and not self._writer.is_closing():
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
```

### Create `src/synth_acp/broker/message_bus.py`

```python
"""MessageBus — notification-driven inter-agent message delivery."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiosqlite

from synth_acp.db import ensure_schema_async

log = logging.getLogger(__name__)

DeliverFn = Callable[[str, str, list[str]], Awaitable[bool]]
CommandFn = Callable[[list[tuple[int, str, str, str]]], Awaitable[None]]


class MessageBus:
    """Notification-driven message delivery with fallback polling.

    - Listens on a Unix domain socket for 1-byte wake-up signals from MCP servers.
    - On each signal, reads pending messages/commands from SQLite and delivers them.
    - The delivery loop uses `wait_for(wake_event, timeout=fallback_interval)` which
      provides built-in fallback polling — no separate poller task needed.
    - Manages per-agent pending message queues for agents that haven't reached IDLE.
    """

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        deliver: DeliverFn,
        process_commands: CommandFn | None = None,
        fallback_interval: float = 2.0,
    ) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._deliver = deliver
        self._process_commands = process_commands
        self._fallback_interval = fallback_interval
        self._pending: dict[str, list[tuple[str, str]]] = {}  # agent_id → [(from_agent, body)]
        self._stopped = False
        self._tasks: list[asyncio.Task[None]] = []
        self._server: asyncio.Server | None = None
        self._socket_path = str(Path(tempfile.gettempdir()) / f"synth-{session_id}.sock")
        self._wake_event = asyncio.Event()

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def start(self) -> None:
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=self._socket_path)
        self._tasks.append(asyncio.create_task(self._delivery_loop(), name="msg-bus-delivery"))

    async def stop(self, timeout: float = 2.0) -> None:
        """Stop all listeners and cancel tasks. Uses a shared deadline — never exceeds timeout."""
        self._stopped = True
        self._wake_event.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # Cancel all tasks and wait with a shared deadline (not per-task)
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()

    def enqueue_pending(self, agent_id: str, from_agent: str, body: str) -> None:
        """Queue a message for an agent that isn't IDLE yet."""
        self._pending.setdefault(agent_id, []).append((from_agent, body))

    def pop_pending(self, agent_id: str) -> str | None:
        """Pop and return combined pending messages, or None if empty.

        Formats each message as ``[Message from X]: body`` — the same format
        used by ``_deliver_pending`` — so agents see a consistent prompt
        structure regardless of whether the message arrived via the poller
        path or the pending queue path.
        """
        messages = self._pending.pop(agent_id, None)
        if not messages:
            return None
        return "\n\n".join(f"[Message from {sender}]: {body}" for sender, body in messages)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not self._stopped:
                data = await reader.read(64)
                if not data:
                    break
                self._wake_event.set()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def _delivery_loop(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await ensure_schema_async(db)
                await db.commit()
                await self._deliver_pending(db)
                await self._process_pending_commands(db)
                while not self._stopped:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=self._fallback_interval)
                    except TimeoutError:
                        pass
                    except asyncio.CancelledError:
                        raise
                    if self._stopped:
                        break
                    try:
                        await self._deliver_pending(db)
                        await self._process_pending_commands(db)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception("MessageBus delivery error")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MessageBus connection error")

    async def _deliver_pending(self, db: aiosqlite.Connection) -> None:
        """Migrate from poller.py _deliver_pending — identical SQL and delivery logic.

        Query pending messages grouped by recipient, mark delivered before
        calling the deliver callback, revert to pending on failure.
        """
        ...

    async def _process_pending_commands(self, db: aiosqlite.Connection) -> None:
        """Migrate from poller.py _process_pending_commands — identical logic."""
        ...
```

### Wire the notification channel

In `server.py`, update `main()`:
```python
notify_socket = os.environ.get("SYNTH_NOTIFY_SOCKET", "")
notify: NotifyFn = _noop_notify
if notify_socket:
    from synth_acp.mcp.notifier import BrokerNotifier
    notifier = BrokerNotifier(notify_socket)
    notify = notifier.notify
server = create_mcp_server(db_path, session_id, agent_id, communication_mode, notify=notify)
server.run(transport="stdio")
```

In `broker.py`, update `_build_mcp_env` to pass the socket path:
```python
EnvVariable(name="SYNTH_NOTIFY_SOCKET", value=self._message_bus.socket_path if self._message_bus else ""),
```

### Update broker

1. Replace `self._poller` with `self._message_bus`. Replace `_start_poller()` with `_start_message_bus()`.
2. Remove `self._pending_initial_prompts`. Use `self._message_bus.enqueue_pending()` / `pop_pending()`.
3. In `_sink`, replace the pending-prompts dispatch:
   ```python
   if isinstance(event, AgentStateChanged) and event.new_state == AgentState.IDLE:
       if self._message_bus:
           pending = self._message_bus.pop_pending(event.agent_id)
           if pending:
               session = self._registry.get_session(event.agent_id)
               if session:
                   self._tasks[f"prompt-{event.agent_id}"] = asyncio.create_task(session.prompt(pending))
   ```
   **Note:** This interim code references the broker's `self._tasks` directly. In Phase 6, when `_tasks` moves to `AgentLifecycle`, replace the `self._tasks[...] = asyncio.create_task(...)` line with `await self._lifecycle.prompt(event.agent_id, pending)`.
4. Add `self_terminate` command handling — uses the public `force_terminate()` method from Phase 2, never reaches into `session._sm`:
   ```python
   async def _handle_self_terminate_command(self, cmd_id: int, from_agent: str) -> None:
       session = self._registry.get_session(from_agent)
       if session and session.state != AgentState.TERMINATED:
           await session.force_terminate()
       await self._update_command_status(cmd_id, "processed")
   ```

### Ordering constraint

**The `MessageBus` must be created before `AgentLifecycle` (Phase 6)** because the lifecycle needs `message_bus.socket_path` to pass as `SYNTH_NOTIFY_SOCKET` in agent env vars. In the broker's `__init__` or lazy startup, ensure the bus exists before any agent is launched.

### Delete `src/synth_acp/broker/poller.py`

### Tests

**`tests/broker/test_message_bus.py`** (new file):
- `test_stop_does_not_hang_when_delivery_is_slow`: deliver callback sleeps 10s. `stop(timeout=0.5)` returns in <1s.
- `test_stop_is_idempotent`: Two `stop()` calls, no exception.
- `test_notification_triggers_immediate_delivery`: Insert message, send socket byte, assert delivery within 100ms.
- `test_fallback_poll_delivers_without_notification`: Insert message without notification, assert delivery within 3s.
- `test_enqueue_pending_stores_multiple_messages`: Enqueue two, pop, assert both in result.
- `test_pop_pending_returns_none_when_empty`
- `test_socket_cleaned_up_on_stop`

**`tests/mcp/test_notifier.py`** (new file):
- `test_notify_sends_byte_to_socket`
- `test_notify_reconnects_after_disconnect`
- `test_notify_silently_fails_if_no_server`

**`tests/broker/test_broker.py`** — add:
- `test_self_terminate_emits_terminated_event`
- `test_self_terminate_does_not_call_session_terminate`
- `test_two_pending_messages_before_idle_both_delivered`: Enqueue `"msg1"` and `"msg2"` via message bus. Trigger IDLE. Assert the prompt argument is exactly `"msg1\n\nmsg2"` — verifying both messages survive and the join format matches `_deliver_pending`'s convention.

**Delete `tests/broker/test_poller.py`.**

---

## Phase 6 — Extract `AgentLifecycle`

**Goal:** Pull agent launch/terminate/prompt/cancel/set_mode/set_model and all task management out of the broker.

### Why this is needed

When an agent is launched, its `session.run()` task is stored in `self._tasks[agent_id]`. When the agent process exits naturally (without going through `_terminate`), the completed task stays in the dict forever — only `_terminate()` cleans it up. For dynamically launched agents that exit on their own, this is a memory leak proportional to session length. Prompt tasks under `f"prompt-{agent_id}"` have the same issue.

### Create `src/synth_acp/broker/lifecycle.py`

```python
"""AgentLifecycle — agent launch, termination, prompting, and task management."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiosqlite
from acp.schema import EnvVariable, McpServerStdio

from synth_acp.acp.session import ACPSession
from synth_acp.broker.registry import AgentRegistry
from synth_acp.db import ensure_schema_async
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError, BrokerEvent

log = logging.getLogger(__name__)

EventSink = Callable[[BrokerEvent], Awaitable[None]]


class AgentLifecycle:
    """Manages agent launch, termination, prompting, and background tasks.

    Every asyncio.Task created by this class has a done-callback that
    removes it from _tasks on completion, preventing accumulation.
    """

    def __init__(
        self,
        config: SessionConfig,
        registry: AgentRegistry,
        event_sink: EventSink,
        db_path: Path,
        session_id: str,
        notify_socket_path: str = "",
    ) -> None:
        self._config = config
        self._registry = registry
        self._sink = event_sink
        self._db_path = db_path
        self._session_id = session_id
        self._notify_socket_path = notify_socket_path
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._harness_registry = load_harness_registry()
        self._db: aiosqlite.Connection | None = None

    def _make_run_task(self, agent_id: str, session: ACPSession) -> asyncio.Task[None]:
        task = asyncio.create_task(session.run(), name=f"run-{agent_id}")
        def _on_done(t: asyncio.Task[None]) -> None:
            self._tasks.pop(agent_id, None)
            if not t.cancelled() and (exc := t.exception()):
                log.error("session.run() for %s raised", agent_id, exc_info=exc)
        task.add_done_callback(_on_done)
        return task

    def _make_prompt_task(self, agent_id: str, coro: Awaitable[None]) -> asyncio.Task[None]:
        key = f"prompt-{agent_id}"
        task = asyncio.create_task(coro, name=key)
        task.add_done_callback(lambda _: self._tasks.pop(key, None))
        return task
```

**Migration instructions for each method:**

`launch(agent_id, *, adhoc_config=None)` — migrate from broker's `_launch()`:
- Identical logic for resolving harness, building MCP env, creating `ACPSession`.
- Replace `self._sessions[agent_id] = session` with `self._registry.register(agent_id, session)`.
- Replace `self._agent_harnesses[agent_id] = ...` with `self._registry.set_harness(agent_id, ...)`.
- Replace `asyncio.create_task(session.run())` with `self._make_run_task(agent_id, session)`.
- Replace `self._sessions[agent_id]` lookups with `self._registry.get_session(agent_id)`.
- The `if agent_id in self._sessions` existing-session check becomes `if self._registry.has_session(agent_id)`.

`terminate(agent_id)` — migrate from broker's `_terminate()`:
- Identical logic for cancelling tasks, updating SQLite (inactive, orphan children, expire messages).
- Replace `self._sessions.get(agent_id)` with `self._registry.get_session(agent_id)`.
- Replace the parentage loop with `self._registry.orphan_children(agent_id)`.
- Wrap `session.terminate()` in a timeout to prevent blocking on unresponsive agents:
  ```python
  try:
      await asyncio.wait_for(session.terminate(), timeout=5.0)
  except TimeoutError:
      log.warning("session.terminate() timed out for %s", agent_id)
  ```

`prompt(agent_id, text)` — as shown below:
```python
async def prompt(self, agent_id: str, text: str) -> None:
    session = self._registry.get_session(agent_id)
    if not session:
        await self._sink(BrokerError(agent_id=agent_id, message=f"No session for '{agent_id}'"))
        return
    if session.state != AgentState.IDLE:
        await self._sink(BrokerError(agent_id=agent_id, message=f"Agent '{agent_id}' is {session.state}, cannot prompt", severity="warning"))
        return
    self._tasks[f"prompt-{agent_id}"] = self._make_prompt_task(agent_id, session.prompt(text))
```

`cancel(agent_id)` — migrate from broker's `_cancel()`:
- Identical: look up session, call `session.cancel()`. Replace registry access.

`set_mode(agent_id, mode_id)` — migrate from broker's `_set_mode()`:
- Identical: check state is IDLE, call `session.set_mode(mode_id)`. Replace registry access.

`set_model(agent_id, model_id)` — migrate from broker's `_set_model()`:
- Identical: check state is IDLE, call `session.set_model(model_id)`. Replace registry access.

`handle_launch_command(cmd_id, from_agent, data)` — migrate from broker's `_handle_launch_command()`:
- Identical logic for validation, harness resolution, agent config creation, SQLite registration.
- Replace `self._sessions` with `self._registry`.
- Replace `asyncio.create_task(session.run())` with `self._make_run_task(...)`.
- Replace `self._pending_initial_prompts[agent_id] = message`: the lifecycle does not own the message bus. Add an `enqueue_pending: Callable[[str, str, str], None] | None` parameter to `AgentLifecycle.__init__` that the broker binds to `self._message_bus.enqueue_pending`. The lifecycle calls `self._enqueue_pending(agent_id, from_agent, message)` where the old code assigned to the dict. If `_enqueue_pending` is None (no bus started), log a warning and drop the message.
- Move `_update_command_status` to lifecycle as `update_command_status`.

`handle_terminate_command(cmd_id, from_agent, data)` — migrate from broker's `_handle_terminate_command()`:
- Identical: check parentage, call `terminate()`, update command status. Replace registry access.

**Shutdown with terminate timeout:**
```python
async def shutdown(self) -> None:
    for session in self._registry.all_sessions().values():
        if session.state == AgentState.BUSY:
            await session.cancel()
        elif session.state == AgentState.AWAITING_PERMISSION:
            try:
                await asyncio.wait_for(session.terminate(), timeout=5.0)
            except TimeoutError:
                log.warning("session.terminate() timed out for %s", session.agent_id)

    for session in self._registry.all_sessions().values():
        if session.state != AgentState.TERMINATED:
            try:
                await asyncio.wait_for(session.terminate(), timeout=5.0)
            except TimeoutError:
                log.warning("session.terminate() timed out for %s", session.agent_id)

    for task in self._tasks.values():
        if not task.done():
            task.cancel()
    if self._tasks:
        await asyncio.wait(self._tasks.values(), timeout=2.0)
```

**DB and helper methods** — migrate from broker:
- `_ensure_db()` — identical lazy open pattern.
- `close_db()` — close `self._db` if not None, set to None.
- `_build_mcp_env()` — identical, but include `SYNTH_NOTIFY_SOCKET` from `self._notify_socket_path`.
- `register_agents()` — identical pre-registration of config agents in SQLite.
- `_get_visible_agents_for()` — identical `asyncio.to_thread` sync query.
- `_send_join_broadcast()` — identical system message insertion.
- `update_command_status()` — identical SQLite update.

### Update `src/synth_acp/broker/broker.py`

1. Create `self._lifecycle = AgentLifecycle(...)` in `__init__`. **The `MessageBus` must be created first** so its `socket_path` can be passed to the lifecycle as `notify_socket_path`.
2. Remove all migrated methods and their state (`_tasks`, `_harness_registry`, `_db`).
3. `handle()` delegates to lifecycle.
4. `_process_commands()` delegates launch/terminate to lifecycle.
5. `_sink` pending prompt delivery uses `self._lifecycle.prompt()`.

### Tests

**`tests/broker/test_lifecycle.py`** (new file):
- `test_run_task_removed_after_agent_exits`
- `test_prompt_task_removed_after_completion`
- `test_prompt_rejects_non_idle_agent`
- `test_shutdown_terminates_all_then_cancels_tasks`
- `test_terminate_times_out_on_unresponsive_agent`: Mock `session.terminate()` to hang. Assert lifecycle completes shutdown within timeout.

---

## Phase 7 — Slim Broker Coordinator + Structured Shutdown

**Goal:** The broker becomes a thin coordinator. Fix shutdown ordering and add event queue backpressure.

### Why this is needed

**Shutdown ordering:** The current `shutdown()` has two steps labeled `# 3.` (duplicate numbering). The shared `aiosqlite.Connection` is closed before session termination runs, but `_terminate()` writes to SQLite — causing `_ensure_db()` to silently reopen a zombie connection.

**Unbounded event queue:** `asyncio.Queue()` has no `maxsize`. During fast streaming with many `MessageChunkReceived` events, memory grows indefinitely.

### Final broker structure

After phases 4-6, the broker contains only: `__init__`, `handle()`, `_sink()`, `_resolve_permission()`, `_process_commands()`, `_handle_self_terminate_command()`, `_deliver_message()`, `_ensure_message_bus()`, `events()`, `shutdown()`, and one-line delegation methods. Target: **under 300 lines**.

### Event queue backpressure

```python
# ~2 seconds of chunks at 1000 events/s. Tune up if agents stream faster.
# Passed as a constructor parameter (default 2000) so tests can use a small value.

# In ACPBroker.__init__ signature, add: event_queue_maxsize: int = 2000
self._event_queue: asyncio.Queue[BrokerEvent] = asyncio.Queue(maxsize=event_queue_maxsize)
```

In `_sink`:
```python
if isinstance(event, MessageChunkReceived):
    try:
        self._event_queue.put_nowait(event)
    except asyncio.QueueFull:
        log.debug("Event queue full, dropping chunk for %s", event.agent_id)
else:
    await self._event_queue.put(event)
```

Dropping `MessageChunkReceived` under pressure is acceptable — the `SessionAccumulator` holds the authoritative full text.

### Structured shutdown

```python
async def shutdown(self) -> None:
    self._shutting_down = True

    # Phase 1: Stop all agent activity
    await self._lifecycle.shutdown()

    # Phase 2: Stop message bus
    if self._message_bus:
        await self._message_bus.stop()

    # Phase 3: Close DB (all writes complete)
    await self._lifecycle.close_db()

    # Phase 4: Persist session IDs using public property (not _session_id)
    sessions_path = Path.home() / ".synth" / "sessions.json"
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    session_ids = {
        aid: s.session_id
        for aid, s in self._registry.all_sessions().items()
        if s.session_id and s.state == AgentState.TERMINATED
    }
    sessions_path.write_text(json.dumps(session_ids))

    self._shutdown_event.set()
```

Ordering bugs are structurally impossible: lifecycle needs DB and shuts down first. Message bus needs no DB, shuts down second. DB closes third. Persist is pure file I/O, goes last. Session IDs are accessed via the public `session.session_id` property (added in Phase 2), not the private `_session_id` attribute.

### Tests

- `test_shutdown_phases_in_order`: Mock lifecycle, message_bus. Assert call order.
- `test_shutdown_completes_within_timeout`: With slow mocks, assert completion within 5s.
- `test_queue_full_drops_chunk_events_not_state_events`: Create broker with `event_queue_maxsize=5`. Fill queue, verify chunks dropped and state events preserved.
- `test_sink_pending_prompts_delivered_on_idle`: Enqueue via bus, fire IDLE, assert prompt called.

---

## Acceptance Criteria

All phases complete. The following commands pass:

```bash
uv run pytest -q --tb=short --no-header -rF
uv run ruff check --output-format concise src/ tests/
uv run ty check --output-format concise src/ tests/
```

### Structural checks

- `src/synth_acp/broker/poller.py` does not exist.
- `tests/broker/test_poller.py` does not exist.
- `src/synth_acp/broker/broker.py` is under 300 lines.
- `src/synth_acp/mcp/server.py` has zero module-level mutable state.
- `grep -r "_get_db\b" src/` returns zero results.
- `grep -r "self\.state = " src/synth_acp/acp/session.py` returns zero results.
- `grep -r "except Exception" src/synth_acp/acp/session.py` does NOT match a block that catches `InvalidTransitionError`.
- No cross-module access to `session._sm` or `session._session_id` — use `session.force_terminate()` and `session.session_id` instead.

### Behavioral verification

| Bug | Test |
|-----|------|
| Poller deadlock on shutdown | `test_stop_does_not_hang_when_delivery_is_slow` |
| DB connection leak in MCP tools | `test_send_message_closes_conn_on_error` |
| Task dict memory leak | `test_run_task_removed_after_agent_exits` |
| Self-deregistration invisible to TUI | `test_self_terminate_emits_terminated_event` |
| Shutdown ordering | `test_shutdown_phases_in_order` |
| InvalidTransitionError swallowed | `test_finally_uses_force_terminal` |
| Pending prompts dropped | `test_two_pending_messages_before_idle_both_delivered` |
| Unbounded event queue | `test_queue_full_drops_chunk_events_not_state_events` |
| synth-mcp crashes with empty env | `test_main_exits_with_missing_env_vars` |
| set_model missing guard | `test_set_model_transitions_through_configuring` |
| Wrong type annotation | `ty check` passes |

## Implementation Order

Execute phases 1 through 7 in sequence. Each phase must leave all tests green before proceeding. If a phase introduces a test failure in an existing test, fix it within that phase — do not defer.

Within each phase, the recommended order is:
1. Create new files and new classes
2. Write tests for the new code
3. Migrate logic from old files to new
4. Update old file imports and delegations
5. Run full test suite
6. Delete dead code
