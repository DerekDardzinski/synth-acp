# SYNTH

**SY**nchronized **N**etwork of **T**eamed **H**arnesses over ACP

A multi-agent orchestration dashboard that manages teams of AI coding agents through the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/). Agent-agnostic — works with any ACP-compatible harness (Kiro CLI, Claude Code, Gemini CLI, OpenCode, etc.).

## What It Does

- Launches and manages multiple ACP agent subprocesses from a single terminal
- Streams agent responses with rendered markdown in a Textual TUI
- Surfaces permission requests inline with one-click approve/reject
- Enables agent-to-agent messaging via a bundled MCP server
- Supports flexible topologies: human-dispatch, orchestrator, peer-to-peer
- Configurable lifecycle hooks for agent join, exit, and prompt injection

## Quick Start

```bash
# Install
uv sync

# Initialize a config file (interactive)
uv run synth init

# Or create one manually
cat > .synth.json << 'EOF'
{
  "project": "my-project",
  "agents": [
    { "agent_id": "kiro", "harness": "kiro" }
  ]
}
EOF

# Launch the TUI
uv run synth

# Or run headless (CLI mode)
uv run synth --headless
```

## Configuration

Create a `.synth.json` in your project root. A minimal config:

```json
{
  "project": "my-project",
  "agents": [
    { "agent_id": "lead", "harness": "kiro" },
    { "agent_id": "worker", "harness": "claude" }
  ]
}
```

### Full Config Reference

See [`examples/synth.example.json`](examples/synth.example.json) for a complete config with all available options.

```json
{
  "project": "my-project",

  "agents": [
    {
      "agent_id": "lead",
      "harness": "kiro",
      "agent_mode": "coder",
      "cwd": ".",
      "env": { "CUSTOM_VAR": "value" }
    }
  ],

  "settings": {
    "communication_mode": "MESH",
    "auto_approve_tools": ["synth-mcp/send_message", "synth-mcp/list_agents"],

    "hooks": {
      "on_agent_startup": {
        "prepend": "<orchestration_context>\nagent_id: {agent_id}\nsession: You are in a multi-agent session.\n</orchestration_context>\n\n"
      },
      "on_agent_prompt": {
        "prepend": "<orchestration_context>\nagent_id: {agent_id}\nparent_agent: {parent_id}\nreply_tool: send_message(to_agent='{parent_id}', kind='response')\n</orchestration_context>\n\n"
      },
      "on_agent_join": {
        "recipients": "none",
        "template": "Agent \"{agent_id}\" is now active. Task: \"{task}\".",
        "kind": "system"
      },
      "on_agent_exit": {
        "recipients": "none",
        "template": "Agent \"{agent_id}\" has left the session.",
        "kind": "system"
      }
    }
  },

  "ui": {
    "web_port": 8000,
    "theme": "dark"
  }
}
```

### Agent Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `agent_id` | yes | — | Unique identifier, shown to other agents |
| `harness` | yes | — | Short name: `kiro`, `claude`, `opencode`, `gemini` |
| `agent_mode` | no | — | ACP mode ID applied after session creation |
| `cwd` | no | `"."` | Working directory (relative to config file) |
| `env` | no | `{}` | Extra environment variables passed to the harness |

### Settings

| Field | Default | Description |
|-------|---------|-------------|
| `communication_mode` | `"MESH"` | `MESH` (all agents visible) or `LOCAL` (family only) |
| `auto_approve_tools` | `[]` | Tool name patterns to auto-approve without prompting |

### Lifecycle Hooks

Hooks fire at specific moments in an agent's lifecycle. They are configurable in `settings.hooks`.

#### `on_agent_startup`

Fires on the first prompt to any config-defined or TUI-launched agent. Prepends context so the agent knows it's in a multi-agent session.

| Field | Default | Description |
|-------|---------|-------------|
| `prepend` | *(identity + session awareness block)* | Text prepended to the agent's first prompt |

**Template slots:** `{agent_id}`

#### `on_agent_prompt`

Fires when a dynamically launched child agent receives its initial message from the parent. Prepends routing context so the child knows who it is and how to reply.

| Field | Default | Description |
|-------|---------|-------------|
| `prepend` | *(routing context with agent_id, parent, reply_tool)* | Text prepended to the initial prompt |

**Template slots:** `{agent_id}`, `{parent_id}`, `{task}`, `{message}`

#### `on_agent_join`

Fires when a dynamically launched agent is registered. Sends a templated message to the configured recipients.

| Field | Default | Description |
|-------|---------|-------------|
| `recipients` | `"none"` | `none`, `parent`, `family`, or `mesh` |
| `template` | `""` | Message body template |
| `kind` | `"system"` | Message kind: `system` or `chat` |

**Template slots:** `{agent_id}`, `{task}`, `{parent_id}`, `{sibling_ids}`

**Recipient modes:**

| Mode | Recipients |
|------|-----------|
| `none` | No message sent (default) |
| `parent` | Direct parent only |
| `family` | Parent + siblings (agents sharing the same parent) |
| `mesh` | All visible agents |

#### `on_agent_exit`

Fires when an agent is terminated. Same fields and recipient modes as `on_agent_join`.

#### Environment Variable Overrides

For quick experimentation without editing the config file:

| Env var | Overrides |
|---------|-----------|
| `SYNTH_JOIN_RECIPIENTS` | `settings.hooks.on_agent_join.recipients` |
| `SYNTH_JOIN_TEMPLATE` | `settings.hooks.on_agent_join.template` |
| `SYNTH_ROUTING_TEMPLATE` | `settings.hooks.on_agent_prompt.prepend` |

## Architecture

```
synth (single process: broker + TUI)
┌──────────────────────────────────────────┐
│  ACPBroker                               │
│    ├── AgentLifecycle (launch/terminate)  │
│    ├── AgentRegistry (sessions/metadata) │
│    ├── MessageBus (notification-driven)   │
│    └── PermissionEngine (rule persistence)│
│              │                           │
│              ▼                           │
│  SynthApp (Textual TUI)                  │
└──────────────────────────────────────────┘
     │ spawns N agent subprocesses via ACP
     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ kiro-cli │  │ kiro-cli │  │ claude   │
│   acp    │  │   acp    │  │   acp    │
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

## Inter-Agent Messaging

Agents communicate via a bundled MCP server (`synth-mcp`) injected into every agent session. Available tools:

| Tool | Description |
|------|-------------|
| `send_message` | Send a message to another agent (the only inter-agent communication channel) |
| `check_delivery` | Poll whether a sent message has been delivered |
| `list_agents` | List all visible agents with their status, parent, and task |
| `launch_agent` | Launch a new child agent |
| `terminate_agent` | Terminate a child agent you previously launched |
| `get_my_context` | Get your identity, parent, task, and communication rules |

Messages are delivered to idle agents between turns. Message kinds (`chat`, `request`, `response`) are formatted distinctly on delivery so agents can distinguish requests from responses.

## Key Bindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle agent focus |
| `Enter` | Submit prompt |
| `m` | MCP messages panel |
| `l` | Launch agent |
| `F1` | Help |
| `q` | Quit |

## Development

```bash
uv sync                                          # Install dependencies
uv run pytest -q --tb=short --no-header -rF      # Run tests
uv run ruff check --fix --output-format concise   # Lint
uv run ruff format                                # Format
uv run ty check --output-format concise src/      # Type check
```

## Security & Trust Model

Synth is a single-user desktop tool. All agents run as child processes of the synth process and inherit the user's OS-level privileges.

**Trust boundary**: Agents have the same access as the user. Each agent subprocess receives environment variables (`SYNTH_DB_PATH`, `SYNTH_SESSION_ID`) that grant direct access to the shared SQLite database. The MCP server provides structured access, but does not enforce an authentication boundary — any child process can read or write the database directly.

**Implications**:
- Do not expose synth over a network or use it in a multi-user/multi-tenant context without additional sandboxing.
- The permission system (approve/reject tool calls) is a UX convenience for human oversight, not a security boundary against a malicious agent.
- The `!` shell escape in the input bar executes commands with the full privileges of the synth process.

**File permissions**: The synth database directory (`~/.synth/`) is created with mode `0o700` and the database file with mode `0o600` to prevent access by other local users.

## License

See LICENSE file.
