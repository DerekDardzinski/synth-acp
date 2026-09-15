# SYNTH

**SY**nchronized **N**etwork of **T**eamed **H**arnesses over ACP

A multi-agent orchestration dashboard that manages teams of AI coding agents through the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/). Agent-agnostic — works with any ACP-compatible harness (Kiro CLI, Claude Code, Gemini CLI, OpenCode, etc.).

## What It Does

- Launches an ACP agent and manages dynamically spawned child agents from a single terminal
- Streams agent responses with rendered markdown in a Textual TUI
- Surfaces permission requests inline with one-click approve/reject
- Enables agent-to-agent messaging via a bundled MCP server
- Supports flexible topologies: orchestrator spawns children via `launch_agent`
- Configurable lifecycle hooks for agent startup, join, and exit
- Session restore with optional semantic search (`synth --restore`)
- Agent discovery and interactive picker (`synth --list-agents`, `synth --select-agent`)

## Install

```bash
# Install as a global CLI tool (run `synth` from anywhere)
uv tool install synth-acp

# With semantic session search support (recommended)
uv tool install "synth-acp[search]"
```

The `[search]` extra enables semantic search when restoring previous sessions (`synth --restore`). Without it, session restore falls back to recency-based listing only.

## Quick Start

```bash
# Navigate to your project
cd my-project

# Launch — auto-detects your harness if only one is installed
synth

# Or specify a harness explicitly
synth --harness kiro

# Launch with a specific agent mode
synth --harness kiro --agent-mode plan

# Launch Claude Code with a specific agent (full qualified name for plugins)
synth --harness claude --agent-mode local-SHScienceAgentKit-all:code-planner

# Restore a previous session
synth --restore

# List available agents for a harness
synth --list-agents --harness kiro

# Interactive fuzzy agent picker
synth --select-agent

# Set a default so you can just run `synth` anywhere
synth config set default_harness kiro
```

No config file required. Synth auto-detects installed harnesses and launches immediately.

## Global Config (`~/.synth/config.json`)

Created automatically on first run. Stores personal defaults:

```json
{
  "default_harness": "kiro",
  "default_agent_id": null,
  "default_agent_mode": null,
  "communication_mode": "LOCAL",
  "auto_approve_tools": ["synth-mcp"],
  "messages_interrupt": true,
  "handoff_nudge": true,
  "handoff_nudge_threshold": 0.5,
  "hooks": {
    "on_agent_startup": { "active": true },
    "on_agent_join": { "active": false, "recipients": "parent", "template": "Agent \"{agent_id}\" is now active. Task: \"{task}\".", "kind": "system" },
    "on_agent_exit": { "active": false, "recipients": "parent", "template": "Agent \"{agent_id}\" has exited.", "kind": "system" },
    "on_mcp_message": { "active": true, "template": "[{message_type} from {from_agent}]: ", "system_template": "[System notification — no action required]: " }
  }
}
```

### `synth config` Commands

```bash
synth config list                    # Show all settings with descriptions
synth config set default_harness kiro  # Set a default
synth config set communication_mode MESH
synth config set auto_approve_tools "synth-mcp/send_message,synth-mcp/list_agents"
synth config path                    # Print config file path
```

Settable keys: `default_harness`, `default_agent_id`, `default_agent_mode`, `communication_mode`, `auto_approve_tools`, `messages_interrupt`, `handoff_nudge`, `handoff_nudge_threshold`, `handoff_nudge_template`. Hooks are edited directly in the JSON file.

## Startup Context (`~/.synth/context.md`)

Created alongside the global config on first run. This file is prepended to every agent's first prompt, giving it awareness of the Synth session:

- Agent identity and parent
- The harness the agent is running in
- Visibility rules (text output vs inter-agent messaging)
- Available MCP tools
- Guidance to prefer `launch_agent` over the harness's native subagent feature
- The agent configurations available in the current harness (valid `agent_mode` values for `launch_agent`)
- Message delivery semantics

Edit `~/.synth/context.md` to customize what agents know about your workflow. Set `on_agent_startup.active: false` in config to disable injection entirely.

**Template slots in context.md:** `{agent_id}`, `{parent_id}`, `{task}`, `{harness}`, `{available_agents}`

For the root agent, `{parent_id}` and `{task}` render empty; they carry values for dynamically launched child agents. `{available_agents}` renders the agent configurations discovered for the current harness (the valid `agent_mode` values), or a "no named agent configurations" line when the harness exposes none.

## Project Config (`.synth.json`)

Optional. Create one to override global settings for a specific project:

```json
{
  "project": "my-project",
  "settings": {
    "communication_mode": "MESH",
    "auto_approve_tools": ["synth-mcp/send_message", "synth-mcp/list_agents"],
    "hooks": {
      "on_agent_join": { "active": true, "recipients": "mesh", "template": "Agent \"{agent_id}\" joined. Task: \"{task}\".", "kind": "system" }
    }
  }
}
```

When `.synth.json` exists, its settings override global config. Fields not set in `.synth.json` fall through to global defaults.

### Settings

| Field | Global Default | Description |
|-------|---------------|-------------|
| `communication_mode` | `"LOCAL"` | `MESH` (all agents visible) or `LOCAL` (family only) |
| `auto_approve_tools` | `["synth-mcp"]` | Tool name patterns to auto-approve without prompting |
| `messages_interrupt` | `true` | Deliver inter-agent messages into a running turn instead of waiting for it to end. Kiro only; other harnesses always wait for the turn to end |
| `handoff_nudge` | `true` | Send an agent a one-time message suggesting it call `handoff` once its context window passes `handoff_nudge_threshold` |
| `handoff_nudge_threshold` | `0.5` | Fraction of the context window that triggers the nudge. Must be greater than 0 and at most 1 |
| `handoff_nudge_template` | see below | Nudge body. Slots: `{agent_id}`, `{used}`, `{nudge_threshold}`. `{used}` and `{nudge_threshold}` render as whole percents (e.g. `62%`) |
| `harness_env` | `{}` | Per-harness environment policy, keyed by harness short name. See [Harness Environment](#harness-environment) |

### Harness Environment

Agent subprocesses inherit your environment. The ACP SDK trims it to six variables
(`HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`), which silently drops anything an
internally-configured harness needs, so synth forwards the parent environment instead
and subtracts a small denylist.

This applies to **every** harness, not only claude. A harness is a program you already
run interactively in this same shell, so handing it a smaller environment than a direct
invocation gets is a difference you did not ask for and cannot see.

Set `harness_env` in `~/.synth/config.json` at the top level, or in `.synth.json` under
`settings`. Keys are harness short names (`claude`, `kiro`, `opencode`, `gemini`):

```json
{
  "harness_env": {
    "claude": {
      "env": { "CLAUDE_CODE_USE_BEDROCK": "1", "MCP_SERVER_CONNECTION_BATCH_SIZE": "10" },
      "inherit": "all"
    }
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `env` | `{}` | Literal name/value pairs set on the subprocess. Applied last, so an explicit setting beats an inherited one of the same name |
| `inherit` | `"all"` | `"all"` forwards the whole parent environment minus the denylist. A list forwards the names and glob patterns it contains, and nothing else |

Two knobs rather than one because they do different jobs. `env` sets a value that is not
in your shell to begin with. `inherit` forwards one that is, without you having to write
a rotating or machine-specific value (a credential, a session path) into a config file.

**Always dropped from the inherited set**, whatever `inherit` says: names beginning
`npm_config_`, `npm_package_`, `npm_lifecycle_`, `npm_command`, `npm_execpath`,
`npm_node_execpath`, `init_cwd`, `brazil_`, `envroot`, `canonical_envroot`, plus
`LD_LIBRARY_PATH`, plus any value that begins `()` (an exported shell function).
The prefixes are narrow on purpose: a broad `npm_` match would also take `NPM_TOKEN`
and `NPM_AUTH_TOKEN`, which is what a private registry needs, turning an install into a
silent 401. A name you set explicitly in `env` is honored even if a prefix denies it.

Project `harness_env` merges over global **per variable**, so a project overriding one
value does not drop the rest of that harness's block. Two `inherit` lists concatenate,
so a project can widen; a list facing `"all"` replaces it, so a project can narrow to
the strict posture.

**Security note.** `inherit: "all"` means the agent subprocess sees every variable in
your shell, including secrets. This matches synth's stated trust model: agents already
run with your full OS privileges, so withholding a variable from an agent that can read
the file it came from buys nothing. On a shared host, in CI, or with an agent you trust
less, set `inherit` to an explicit list.

`inherit` cannot be used to withhold `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM` or
`USER`. The ACP SDK adds those six unconditionally and a harness needs them to run at
all, so `inherit: []` yields a child with exactly those six, not an empty environment.

### Context-Window Nudge

When an agent's context window passes `handoff_nudge_threshold`, synth sends that agent
one message suggesting it call the `handoff` MCP tool and continue in a fresh session.
The nudge fires at most once per agent per synth run. An agent that hands off gets a
successor with an empty context, and the successor is nudgeable again in its turn.

The nudge is delivered as an ordinary inter-agent message from the sender `synth`, so it
appears in the conversation transcript exactly like any other message — mid-turn on Kiro
(where in-turn steering is supported), or at the next idle point on other harnesses. It is
a suggestion: the agent decides whether to act on it.

Usage data comes from whatever the harness reports. Claude Code reports token counts.
Kiro CLI reports a percentage over a proprietary notification and reports no token
counts, so the usage bar and this threshold are accurate to one percent for Kiro agents.
Harnesses that report no usage at all never trigger the nudge.

Set `handoff_nudge` to `false` to disable it.

### Lifecycle Hooks

All hooks have an `active` field that controls whether they fire.

#### `on_agent_startup`

Fires on the first prompt to every agent (root and dynamically launched children). Prepends the startup context from `~/.synth/context.md`.

| Field | Default | Description |
|-------|---------|-------------|
| `active` | `true` | Whether to inject startup context |

**Template slots in context.md:** `{agent_id}`, `{parent_id}`, `{task}`

#### `on_agent_join`

Fires when a dynamically launched agent is registered. Sends a templated message to other agents.

| Field | Default | Description |
|-------|---------|-------------|
| `active` | `false` | Whether to send the notification |
| `recipients` | `"parent"` | `parent`, `family`, or `mesh` |
| `template` | `""` | Message body template |
| `kind` | `"system"` | Message kind: `system` or `chat` |

**Template slots:** `{agent_id}`, `{task}`, `{parent_id}`, `{sibling_ids}`

#### `on_agent_exit`

Fires when an agent is terminated. Same fields as `on_agent_join`.

#### `on_mcp_message`

Fires when an MCP-delivered inter-agent message reaches a recipient. Prefixes the message body with sender attribution so the recipient knows who sent it and whom to reply to. The body is always appended verbatim — there is no `{body}` slot.

| Field | Default | Description |
|-------|---------|-------------|
| `active` | `true` | Whether to prefix delivered messages (active by default so attribution works without migration) |
| `template` | `"[{message_type} from {from_agent}]: "` | Prefix for `chat`/`request`/`response` messages |
| `system_template` | `"[System notification — no action required]: "` | Prefix for `system` messages (join/exit notifications) |

**Allowed template slots:** `{from_agent}`, `{to_agent}`, `{kind}`, `{message_type}`. `message_type` maps `chat` → `Message`, `request` → `Request`, `response` → `Response`. Messages with `kind: "system"` use `system_template`; all other kinds use `template`. Both templates are validated at config load time — an invalid template (unknown slot, format spec, conversion, attribute/index access, or malformed braces) fails loudly rather than silently.

## Config Resolution

When you run `synth`, configuration is resolved in this order:

| Priority | Source | What it provides |
|----------|--------|-----------------|
| 1 | `--harness` / `--agent-mode` / `--agent-id` flags | Agent to launch |
| 2 | `.synth.json` in CWD | Project settings |
| 3 | `~/.synth/config.json` `default_harness` | Agent to launch (if no flags) |
| 4 | Single harness in PATH | Auto-detect agent |

Settings resolution: project `.synth.json` fields override global config. Unset fields fall through to global defaults.

## Harness-Specific Behavior

### Agent Mode

The `--agent-mode` flag has different semantics per harness:

| Harness | `agent_mode` meaning | Example |
|---------|---------------------|---------|
| Kiro | Agent config name (passed as `--agent` CLI flag + `set_session_mode`) | `plan`, `code-planner` |
| Claude Code | Agent name (passed as `_meta.claudeCode.options.agent` on session creation) | `local-SHScienceAgentKit-all:code-planner` |
| OpenCode | Not supported | — |
| Gemini CLI | Not supported | — |

For Claude Code, plugin agents require the full qualified name: `<plugin-name>:<agent-name>`. User/project agents (in `~/.claude/agents/` or `.claude/agents/`) use just the agent name.

### Config Options

Synth dynamically renders configuration pickers based on what the harness advertises:

| Harness | Available pickers |
|---------|------------------|
| Kiro | Mode (agents), Model |
| Claude Code | Mode (permission), Model, Effort |
| OpenCode | None |
| Gemini CLI | None |

The effort level picker appears only for models that support it (e.g., Opus, Sonnet). Switching to Haiku removes the effort picker automatically.

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
     │ spawns agent subprocesses via ACP
     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ kiro-cli │  │ claude   │  │ opencode │
│   acp    │  │   acp    │  │   acp    │
│ MCP:     │  │ MCP:     │  │ MCP:     │
│ synth-mcp│  │ synth-mcp│  │ synth-mcp│
└──────────┘  └──────────┘  └──────────┘
```

One agent is launched on startup. Additional agents are spawned dynamically via `launch_agent`.

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
| `send_message` | Send a message to another agent |
| `list_agents` | List all visible agents with their status, parent, and task |
| `launch_agent` | Launch a new child agent |
| `terminate_agent` | Terminate a child agent you previously launched |
| `resurrect_agent` | Resurrect a previously terminated agent |
| `get_my_context` | Get your identity, parent, task, and communication rules |
| `handoff` | Retire yourself and hand your work to a fresh session that keeps your agent id |

Messages are delivered to idle agents between turns. MCP-delivered inter-agent messages are prefixed with sender attribution via the configurable `on_mcp_message` hook (active by default), so recipients can see who sent a message and whom to reply to; `system` messages (join/exit notifications) use a no-action wrapper instead. Message bodies are always preserved unchanged.

## Key Bindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle agent focus |
| `m` | MCP messages panel |
| `l` | Launch agent |
| `Ctrl+r` | Restore session |
| `F1` | Help |
| `q` | Quit |

## Development

The repo pins CPython 3.12 via `.python-version`, so `uv sync` creates a 3.12
virtualenv. `requires-python` stays `>=3.12` — synth runs on newer interpreters — but
development and performance work should use the pinned version. CPython 3.13 replaced
the three-generation garbage collector with an incremental one, which changes both GC
pause timings and the generation numbers reported to `gc.callbacks`, so an unpinned
checkout produces performance numbers that cannot be compared against another.

```bash
uv sync                                          # Install dependencies
uv run pytest -q --tb=short --no-header -rF      # Run tests
uv run ruff check --fix --output-format concise   # Lint
uv run ruff format                                # Format
uv run ty check --output-format concise src/      # Type check
```

## Publishing

Releases are published to PyPI via GitHub Actions when a version tag is pushed.

```bash
# 1. Bump version in pyproject.toml
# 2. Commit and tag
git add pyproject.toml
git commit -m "release: v0.3.1"
git tag v0.3.1
git push origin main --tags
```

The tag must match the version in `pyproject.toml` (without the `v` prefix). CI runs
tests and lint before publishing — if either fails, nothing is uploaded.

Install from PyPI:

```bash
uv pip install synth-acp

# With semantic session search support
uv pip install "synth-acp[search]"
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
