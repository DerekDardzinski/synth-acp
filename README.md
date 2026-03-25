# SYNTH

**SY**nchronized **N**etwork of **T**eamed **H**arnesses over ACP

A multi-agent orchestration dashboard that manages teams of AI coding agents through the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/). Agent-agnostic — works with any ACP-compatible agent (Kiro CLI, Claude Code, Gemini CLI, etc.).

## What It Does

- Launches and manages multiple ACP agent subprocesses from a single terminal
- Streams agent responses with rendered markdown in a Textual TUI
- Surfaces permission requests inline with one-click approve/reject
- Enables agent-to-agent messaging via a bundled MCP server
- Supports flexible topologies: human-dispatch, orchestrator, peer-to-peer

## Quick Start

```bash
# Install
uv sync

# Create a config file
cat > .synth.json << 'EOF'
{
  "session": "my-project",
  "agents": [
    {
      "id": "kiro",
      "binary": "kiro-cli",
      "args": ["acp"],
      "cwd": ".",
      "autostart": true
    }
  ]
}
EOF

# Launch the TUI
uv run synth

# Or run headless (CLI mode)
uv run synth --headless
```

## Configuration

Create a `.synth.json` in your project root:

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
      "id": "researcher",
      "binary": "claude",
      "args": ["mcp"],
      "cwd": ".",
      "autostart": false
    }
  ]
}
```

Agent identity is the `id` field — the same binary can appear multiple times with different IDs and they are managed independently.

## Architecture

```
synth (single process: broker + TUI)
┌──────────────────────────────────────────┐
│  ACPBroker                               │
│    ├── ACPSession per agent (ACP SDK)    │
│    ├── PermissionEngine (rules.json)     │
│    └── MessagePoller (SQLite watcher)    │
│              │                           │
│              ▼                           │
│  SynthApp (Textual TUI)                  │
└──────────────────────────────────────────┘
     │ spawns N agent subprocesses via ACP
     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ kiro-cli │  │ kiro-cli │  │ claude   │
│   acp    │  │   acp    │  │   mcp    │
│ MCP:     │  │ MCP:     │  │ MCP:     │
│ synth-mcp│  │ synth-mcp│  │ synth-mcp│
└──────────┘  └──────────┘  └──────────┘
```

Three layers with strict dependency rules:

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Frontend | `synth_acp.ui` | Textual TUI |
| Broker | `synth_acp.broker` | Session lifecycle, permissions, message routing |
| ACP | `synth_acp.acp` | ACP SDK wrapper, subprocess management |
| Shared | `synth_acp.models` | Pydantic v2 models |

The broker has zero UI imports. A future web frontend can consume the same `broker.events()` / `broker.handle()` interface.

## Key Bindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle agent focus |
| `Enter` | Submit prompt |
| `m` | MCP messages panel |
| `l` | Launch agent |
| `F1` | Help |
| `q` | Quit |

## Inter-Agent Messaging

Agents communicate via a bundled MCP server (`synth-mcp`) that provides `send_message`, `check_delivery`, `list_agents`, and `deregister_agent` tools. The broker automatically injects this MCP server into every agent session. Messages are delivered to idle agents between turns.

## Development

```bash
uv sync                                          # Install dependencies
uv run pytest -q --tb=short --no-header -rF      # Run tests
uv run ruff check --fix --output-format concise   # Lint
uv run ruff format                                # Format
uv run ty check --output-format concise src/      # Type check
```

## License

See LICENSE file.
