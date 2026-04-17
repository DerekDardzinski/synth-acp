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
├── harnesses.py        # Harness registry loader (TOML → HarnessEntry)
├── data/
│   └── harnesses/      # Harness TOML definitions (kiro.toml, claude.toml, etc.)
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
    │   ├── prompt_bubble.py
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

- `agent-client-protocol` — ACP Python SDK (Pydantic models, `spawn_agent_process`, `SessionAccumulator`)
- `mcp>=1.0.0` — MCP server via `mcp.server.fastmcp.FastMCP` (agent-to-agent messaging)
- `textual` — TUI framework
- `typer` — CLI framework
- `aiosqlite` — async SQLite for message bus

### Reference Docs

- `README.md` — configuration reference, lifecycle hooks, MCP tools
- `examples/synth.example.json` — complete config with all available options

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
git commit -m "release: v0.2.0"

# 3. Tag and push (triggers CI → test → publish)
git tag v0.2.0
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

### aiosqlite / SQLite best practices

`aiosqlite` creates a dedicated **non-daemon thread** per connection. An unclosed
connection keeps the process alive after the event loop exits — the user sees a
hang requiring Ctrl-C. Sync `sqlite3` has no background threads, so a leaked
connection is just a file descriptor, not a hung process.

**Choose the right tool for the pattern:**

| Pattern | Use | Why |
|---------|-----|-----|
| Long-lived connection (lifecycle DB, message bus delivery loop) | `aiosqlite` | Dedicated thread amortises setup; `async with` or explicit `close_db()` ensures cleanup |
| Open-close per call (permission writes, one-off queries) | `asyncio.to_thread` + `sqlite3` | Borrows a daemon pool thread briefly; no shutdown hang risk |
| Sync init before event loop starts (`__init__`, CLI setup) | `sqlite3` directly | No event loop yet; sync is fine and has zero thread overhead |
| Agent subprocess (MCP server) | `aiosqlite` persistent conn | Subprocess gets killed anyway; `close_db()` hook exists for tests |

**Rules:**

- Every `aiosqlite.connect()` in the main process **must** be inside `async with`
  or have a guaranteed `close()` in a `finally` block. A bare
  `conn = await aiosqlite.connect(...)` without `try/finally` is a shutdown hang
  waiting to happen.
- Never open `aiosqlite` connections for short-lived one-off operations. Use
  `asyncio.to_thread` with sync `sqlite3` instead.
- Sync `sqlite3` with WAL mode can deadlock against `aiosqlite` connections to the
  same database when both are in the same process. Keep sync `sqlite3` usage limited
  to init-time schema creation (before `aiosqlite` connections are opened) or
  offloaded to `asyncio.to_thread`.

**Test cleanup:** Any test that triggers `broker.handle(LaunchAgent(...))` or calls
`broker._start_message_bus()` **must** stop the bus and close the lifecycle DB
in a `finally` block — otherwise the aiosqlite background thread and the Unix
socket server keep the event loop alive and the test hangs indefinitely:

```python
try:
    await broker.handle(LaunchAgent(agent_id="agent-1"))
    # ... assertions ...
finally:
    if broker._message_bus:
        await broker._message_bus.stop()
    if broker._lifecycle:
        await broker._lifecycle.close_db()
```

Tests that create MCP servers with `create_mcp_server()` must call
`await server.close_db()` after the test (or use the `mcp_factory` fixture
in `tests/mcp/test_server.py` which handles this automatically).

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
