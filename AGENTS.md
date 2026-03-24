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
├── cli.py              # argparse CLI, entry point
├── models/
│   ├── agent.py        # AgentState enum, AgentConfig
│   ├── config.py       # SessionConfig (parsed from .synth.json)
│   ├── events.py       # BrokerEvent and subclasses (broker → frontend)
│   ├── commands.py     # BrokerCommand and subclasses (frontend → broker)
│   └── permissions.py  # PermissionRule, PermissionDecision
├── acp/
│   └── session.py      # ACPSession — wraps acp SDK Client interface
├── broker/
│   ├── broker.py       # ACPBroker — owns sessions, routes events
│   ├── permissions.py  # PermissionEngine — rule persistence + auto-resolve
│   └── poller.py       # MessagePoller — SQLite PRAGMA data_version watcher
├── mcp/
│   └── server.py       # synth-mcp entrypoint (FastMCP, agent-to-agent messaging)
└── ui/
    ├── app.py          # SynthApp — bridges broker ↔ Textual messages
    ├── messages.py     # Textual Message subclasses wrapping BrokerEvent
    ├── screens/
    │   └── dashboard.py
    ├── widgets/
    │   ├── agent_list.py
    │   ├── conversation.py
    │   ├── prompt_bubble.py
    │   ├── agent_message.py
    │   ├── tool_call.py
    │   ├── permission.py
    │   ├── message_queue.py
    │   └── input_bar.py
    └── css/
        └── app.tcss
```

### Key Dependencies

- `agent-client-protocol` — ACP Python SDK (Pydantic models, `spawn_agent_process`, `SessionAccumulator`)
- `mcp>=1.0.0` — MCP server via `mcp.server.fastmcp.FastMCP` (agent-to-agent messaging)
- `textual` — TUI framework
- `aiosqlite` — async SQLite for message poller

### Reference Docs

- `docs/DESIGN.md` — full design document with architectural decisions and rationale
- `docs/references/acp_sdk.md` — ACP SDK imports, Client interface, spawn_agent_process
- `docs/references/acp_protocol.md` — ACP types quick reference
- `docs/references/toad_agent.md` — Toad's ACP client patterns (annotated)

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

## Testing

### Conventions

- **File structure**: Test files mirror the source tree. `src/synth_acp/acp/session.py` → `tests/acp/test_session.py`.
- **Async**: Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. All async tests are plain `async def` functions.
- **Fixtures**: Shared fixtures in `tests/conftest.py`.
- **Max 5 tests per source function** unless explicitly justified.
- **Durable tests**: Test behavioral contracts and invariants, not implementation details. Tests should survive refactors — if the function signature and semantics don't change, the tests shouldn't break.
- **Priority**: error handling > boundary conditions > invalid inputs > happy path.

### Running Tests

```bash
uv run pytest -q --tb=short --no-header -rF     # Quick summary
uv run pytest --co                                # List collected tests (dry run)
```

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
