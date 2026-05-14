# AGENTS.md

## Project Overview

SYNTH (Synchronized Network of Teamed Harnesses over ACP) is a multi-agent orchestration dashboard that manages teams of AI coding agents through the Agent Client Protocol (ACP). A single process runs the broker (session lifecycle, message routing, permissions) and a Textual TUI.

Package: `synth-acp`
Source: `src/synth_acp/`
Python: 3.12+, async-first

## Architecture

Three layers with strict dependency rules — each layer may only import from layers below it:

| Layer | Package | Responsibility |
|-------|---------|----------------|
| 3 — Frontend | `synth_acp.ui` | Textual TUI rendering |
| 2 — Broker | `synth_acp.broker` | Session lifecycle, routing, permissions |
| 1 — ACP | `synth_acp.acp` | ACP SDK wrapper, subprocess management |
| Shared | `synth_acp.models` | Pydantic v2 models for events, commands, config |

Layers 1 and 2 have zero Textual imports. The frontend communicates with the broker through typed events and commands in `models/`.

### Package Structure

```
src/synth_acp/
├── cli.py              # typer CLI, entry point
├── db.py               # Shared SQLite schema and helpers
├── discovery.py        # Filesystem-based agent discovery for harnesses
├── embeddings.py       # Standalone embedding module for semantic session search
├── harnesses.py        # Harness registry loader (TOML → HarnessEntry)
├── data/
│   └── harnesses/      # Harness TOML definitions (kiro.toml, claude.toml, opencode.toml, gemini.toml)
├── models/
│   ├── agent.py        # AgentState enum, AgentConfig
│   ├── config.py       # SessionConfig, HooksConfig (parsed from .synth.json)
│   ├── events.py       # BrokerEvent and subclasses (broker → frontend)
│   ├── commands.py     # BrokerCommand and subclasses (frontend → broker)
│   ├── visibility.py   # Agent visibility rules (MESH/LOCAL)
│   └── permissions.py  # PermissionRule, PermissionDecision
├── acp/
│   ├── session.py      # ACPSession — wraps acp SDK Client interface
│   └── state_machine.py # AgentStateMachine — typed state transitions
├── broker/
│   ├── broker.py       # ACPBroker — thin coordinator, event sink, command dispatch
│   ├── lifecycle.py    # AgentLifecycle — launch, terminate, prompt, hooks
│   ├── registry.py     # AgentRegistry — sessions, parentage, metadata
│   ├── message_bus.py  # MessageBus — notification-driven message delivery
│   └── permissions.py  # PermissionEngine — rule persistence + auto-resolve
├── mcp/
│   ├── server.py       # synth-mcp entrypoint (FastMCP, agent-to-agent messaging)
│   └── notifier.py     # BrokerNotifier — Unix socket notification to message bus
├── terminal/
│   ├── manager.py      # PTY terminal process management
│   └── shell_read.py   # Buffered async stream reader for PTY output
└── ui/
    ├── app.py          # SynthApp — bridges broker ↔ Textual messages
    ├── file_discovery.py # File discovery and fuzzy scoring for @ file references
    ├── messages.py     # Textual Message subclasses wrapping BrokerEvent
    ├── ansi/           # Vendored ANSI terminal state parser (from toad)
    ├── screens/
    │   ├── launch.py
    │   ├── permission.py
    │   ├── session_picker.py
    │   └── help.py
    ├── widgets/
    │   ├── agent_list.py
    │   ├── conversation.py
    │   ├── expandable_section.py
    │   ├── prompt_bubble.py
    │   ├── prompt_queue.py
    │   ├── agent_message.py
    │   ├── tool_call.py
    │   ├── message_queue.py
    │   ├── input_bar.py
    │   ├── thought_block.py
    │   ├── copy_button.py
    │   ├── shell_result.py
    │   ├── terminal.py
    │   ├── plan_block.py
    │   ├── diff_view.py
    │   └── gradient_bar.py
    └── css/
        └── app.tcss
```

### Key Dependencies

- `agent-client-protocol==0.9.0` — ACP Python SDK (Pydantic models, `spawn_agent_process`, `SessionAccumulator`)
- `mcp>=1.0.0,<2` — MCP server via `mcp.server.fastmcp.FastMCP` (agent-to-agent messaging)
- `textual[syntax]>=8.2.1` — TUI framework with syntax highlighting
- `textual-speedups>=0.2.1` — Cython-accelerated Textual internals
- `typer>=0.9` — CLI framework
- `InquirerPy==0.3.4` — Interactive fuzzy picker for `--select-agent`
- `PyYAML>=6.0` — YAML parsing for agent discovery

Optional (`pip install synth-acp[search]`):
- `onnxruntime>=1.17.0` — ONNX inference for semantic session search
- `tokenizers>=0.15.0` — Tokenization for embedding model

### Reference Docs

- `README.md` — configuration reference, lifecycle hooks, MCP tools
- `examples/synth.example.json` — complete config with all available options

### Harness-Specific Notes

**Claude Code** (`claude.toml`):
- Uses `npx @agentclientprotocol/claude-agent-acp` binary
- `agent_mode_target = "meta_agent"` — passes `agent_mode` as `_meta.claudeCode.options.agent`
- `executable_env_var = "CLAUDE_CODE_EXECUTABLE"` — injects detected binary path
- `clear_env_vars = ["CLAUDECODE"]` — clears stale env vars in subprocess
- Returns `config_options` with mode (permission), model, and effort (model-dependent)
- `set_config_option()` response returns full updated config_options (effort may disappear on model switch)
- Agent names for plugins: `<plugin-name>:<agent-name>` (e.g., `local-SHScienceAgentKit-all:code-planner`)

**Kiro** (`kiro.toml`):
- `mode_arg = "--agent"` — passes `agent_mode` as CLI flag
- Returns `modes` (all agents) + `models`, NO `config_options`
- Synth synthesizes `config_options` from legacy modes/models
- `set_session_mode()` / `set_session_model()` used (no `set_config_option` endpoint)

**OpenCode** (`opencode.toml`):
- `run_cmd = "opencode acp"` — simple ACP mode
- No agent_mode support

**Gemini CLI** (`gemini.toml`):
- `run_cmd = "gemini --experimental-acp"` — experimental ACP flag
- No agent_mode support

## Build System

This project uses uv as the package manager (standalone, not PeruHatch/Brazil).

### Setup

```bash
uv sync            # Install all dependencies (creates .venv)
```

### Running

```bash
uv run synth                       # Run the TUI (requires .synth.json)
uv run pytest                      # Run tests
uv run pytest tests/acp/           # Run specific test directory
uv run pytest -k "test_foo"        # Run matching tests
```

### Publishing

Releases are published to PyPI via GitHub Actions on version tags. The workflow
uses PyPI Trusted Publishing (OIDC) — no API tokens needed.

```bash
# 1. Bump version in pyproject.toml
# 2. Commit the bump
git add pyproject.toml
git commit -m "release: v0.3.1"

# 3. Tag and push (triggers CI → test → publish)
git tag v0.3.1
git push origin main --tags
```

The tag must match the version in `pyproject.toml` exactly (without the `v` prefix).
The CI workflow verifies this before publishing.

## Testing

### Conventions

- **File structure**: Test files mirror the source tree. `src/synth_acp/acp/session.py` →
  `tests/acp/test_session.py`. One test file per source module — don't split a module's
  tests across multiple files. Use test classes within the file to organize by feature.
  A test file may only import from one source module — crossing into another module's
  territory is a structure violation.
- **Async**: `pytest-asyncio` with `asyncio_mode = "auto"`. All async tests are plain
  `async def` — no decorator needed.
- **Fixtures**: Shared helpers used across 3+ test files belong in `tests/conftest.py`,
  not duplicated per file.

### Textual UI tests

Two modes — choose the right one:

- **Live widget tree** (`app.run_test(headless=True, size=(120, 40))`): required when
  the test needs to query the DOM, check CSS classes, simulate clicks or keypresses,
  or mount widgets. Use `pilot.click(selector)`, `pilot.press(key)`, and
  `pilot.pause()` to let pending messages settle before asserting.
- **Direct method calls with mocks**: sufficient for pure logic and routing tests that
  don't need a rendered widget tree. Prefer this — it's faster and less brittle.

`run_test` is expensive. Don't reach for it to test something that can be verified
by calling a method directly. Do reach for it when the contract is "this event causes
this DOM change" — that's exactly what it's for.

What does not earn a Textual test: confirming that a Textual widget property works
(`Collapsible.collapsed` toggles, `Static.content` stores text). Test your logic,
not the framework.


### Running Tests

```bash
uv run pytest -q --tb=short --no-header -rF     # Quick summary
uv run pytest --co                                # List collected tests (dry run)
```

### SQLite best practices

All async DB access uses `asyncio.to_thread` + `contextlib.closing(sqlite3.connect(...))`.
Each call opens a fresh connection, runs the operation, and closes it — no persistent
connections, no background threads to manage at shutdown.

**Why this is safe:** `asyncio.to_thread` dispatches work to the
`concurrent.futures.ThreadPoolExecutor` default pool. `concurrent.futures.thread`
registers `_python_exit` via `threading._register_atexit`, which sends a shutdown
sentinel to every pool worker at interpreter exit. The pool threads are cleaned up
automatically — no explicit close is needed.

**Pattern:**

```python
async def _db_op(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    return await asyncio.to_thread(self._run_db, fn)

def _run_db(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    with contextlib.closing(sqlite3.connect(self._db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        return fn(conn)
```

**Test cleanup:** Any test that triggers `broker.handle(LaunchAgent(...))` or calls
`broker._start_message_bus()` **must** stop the bus in a `finally` block —
otherwise the Unix socket server keeps the event loop alive and the test hangs:

```python
try:
    await broker.handle(LaunchAgent(agent_id="agent-1"))
    # ... assertions ...
finally:
    if broker._message_bus:
        await broker._message_bus.stop()
```

No DB close is needed because no persistent connection exists.

## Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| ruff | Linting and formatting | `ruff check --fix --output-format concise` |
| ty | Type checking | `ty check --output-format concise src/ tests/` |
| pytest | Testing | `uv run pytest -q --tb=short --no-header -rF` |

## Style

- `from __future__ import annotations` in all files.
- Google-style docstrings.
- Pydantic v2 `BaseModel` with `frozen=True` for all cross-layer types.
- Use the `agent-client-protocol` SDK's Pydantic models directly (e.g. `McpServerStdio`, `EnvVariable`) — don't hand-build dicts for ACP payloads.
- `SessionAccumulator` from `acp.contrib` is the canonical source of per-agent conversation history. Don't reimplement tool call tracking.

## Async Concurrency Rules

### Broker layer

Any method that mutates registry/session state across `await` points **must** hold
`self._registry.agent_lock(agent_id)`. This includes: `prompt`, `set_mode`,
`set_model`, `terminate`, `resurrect`, `handle_launch_command`.

Three patterns that introduce races:

1. **Pop-after-await**: Never `dict.pop()` after an `await` if concurrent code can
   write to the same key during the await. Snapshot before, consume only the snapshot.
2. **Unguarded slot in drain loops**: If a loop processes a queue with `await` points
   inside, hold any shared "active" slot for the full iteration — not just the terminal
   path.
3. **Read-await-mutate without lock**: If you read shared state, await, then mutate
   based on the read — you need a lock. The await is a yield point where any other
   coroutine can invalidate your read.

### UI layer (Textual workers)

`run_worker` / `@work` create asyncio tasks **concurrent** with the message pump.
Any worker that modifies shared app state must be guarded:

- **`exclusive=True`** (last-writer-wins): Cancels ALL prior workers with the same
  `(node, group)` — NOT same name. Always specify `group=` to isolate cancellation
  pools. Current groups:
  - `group="modal"` — `action_launch`, `action_restore`, `_do_restore` (mutually
    exclusive modal flows that should cancel each other)
  - `group="broker"` — `broker-consumer` (long-lived, must never be cancelled by
    user actions)
- **`dict[str, asyncio.Task]`** (first-writer-wins): Use when the operation is
  idempotent and the second caller wants the same result — `_selecting` for
  `select_agent` with the same agent_id.
- **Bare `run_worker()`**: Only safe for fire-and-forget operations that don't touch
  shared state (e.g. `broker.handle(CancelTurn(...))`, `select_agent` from tile
  clicks since `_selecting` already guards, `show_messages` since it's idempotent).

When adding `_draining`-style guards, use `dict[str, asyncio.Event]` over `set[str]`
— matches the repo convention and allows future callers to await completion.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
