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

# With Mermaid diagram and LaTeX math drawing
uv tool install "synth-acp[search,diagrams,math]"
```

The `[search]` extra enables semantic search when restoring previous sessions (`synth --restore`). Without it, session restore falls back to recency-based listing only.

The `[diagrams]` extra draws ```` ```mermaid ```` blocks in agent replies as text diagrams, using [termaid](https://github.com/fasouto/termaid). A diagram is drawn when the reply finishes streaming; until then, and for anything termaid cannot draw, the block shows its source. Wide diagrams scroll sideways.

The `[math]` extra draws LaTeX math in agent replies as Unicode text, using [TeXicode](https://github.com/dxddxx/TeXicode): `$…$` inline, and `$$…$$` or ```` ```math ```` blocks on their own lines. Inline math is drawn only when it fits on one line; a stacked fraction inside a sentence stays as source. Dollar signs count as math only in the strict form: Pandoc's rule (no space just inside them, no digit right after the closing one) plus no letter or digit touching either dollar from outside. Prices (`$5`, `US$ 500`) and shell variables (`$HOME`) stay as written. ```` ```latex ```` blocks are left as code, since they usually hold LaTeX source. As with diagrams, math is drawn when the reply finishes, and anything TeXicode cannot draw shows its source.

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

# Launch Claude Code with a specific agent (full qualified name for plugins;
# `synth --list-agents --harness claude` prints the exact names on your machine)
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

Settable keys: `default_harness`, `default_agent_id`, `default_agent_mode`, `communication_mode`, `auto_approve_tools`, `messages_interrupt`, `handoff_nudge`, `handoff_nudge_threshold`, `handoff_nudge_template`, `handoff_switch_template`. Hooks are edited directly in the JSON file.

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
| `auto_approve_tools` | `["synth-mcp"]` | Tool name patterns to auto-approve without prompting. When the harness reports the tool's identity (`<server>/<tool>`, Kiro v3 MCP tools), a pattern must equal the server name (`synth-mcp`) or the full identity (`synth-mcp/send_message`); a prefix such as `synth` no longer matches. Otherwise the pattern is matched as a substring of the permission title |
| `messages_interrupt` | `true` | Deliver inter-agent messages into a running turn instead of waiting for it to end. Claude, Kiro and Codex only; other harnesses always wait for the turn to end. Applies to agent messages only; a user steer (Shift+Enter) works either way |
| `handoff_nudge` | `true` | Send an agent a one-time message suggesting it call `handoff` once its context window passes `handoff_nudge_threshold` |
| `handoff_nudge_threshold` | `0.5` | Fraction of the context window that triggers the nudge. Must be greater than 0 and at most 1 |
| `handoff_nudge_template` | see below | Nudge body. Slots: `{agent_id}`, `{used}`, `{nudge_threshold}`. `{used}` and `{nudge_threshold}` render as whole percents (e.g. `62%`) |
| `handoff_switch_template` | built in | Body of the message that asks an agent to hand off to the profile picked in the agent picker (see [Agent Mode](#agent-mode)). Slots: `{agent_id}`, `{agent_mode}` (the value to pass to `handoff`, `""` for the default agent), `{profile}` (`the agent profile "<name>"` or `the default agent`), `{instructions}` (empty, or a header followed by the user's instructions verbatim) |
| `harness_env` | `{}` | Per-harness environment policy, keyed by harness short name. See [Harness Environment](#harness-environment) |
| `harness_variant` | `{}` | Engine variant per harness short name, e.g. `{"kiro": "v2"}`. A harness without an entry uses its TOML's `default_variant`. JSON-only (not settable with `synth config set`). A project value overrides the global one per harness. An unknown variant name stops synth with an error that lists the valid ones. See [Adding a Harness](#adding-a-harness) |

### Harness Environment

Agent subprocesses inherit your environment. The ACP SDK trims it to six variables
(`HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`), which silently drops anything an
internally-configured harness needs, so synth forwards the parent environment instead
and subtracts a small denylist.

This applies to **every** harness, not only claude. A harness is a program you already
run interactively in this same shell, so handing it a smaller environment than a direct
invocation gets is a difference you did not ask for and cannot see.

Set `harness_env` in `~/.synth/config.json` at the top level, or in `.synth.json` under
`settings`. Keys are harness short names (`claude`, `kiro`, `codex`, `opencode`, `gemini`):

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
appears in the conversation transcript exactly like any other message — mid-turn on Claude,
Kiro and Codex (where in-turn steering is supported), or at the next idle point on other harnesses. It is
a suggestion: the agent decides whether to act on it.

Usage data comes from whatever the harness reports. Claude Code reports token counts.
Kiro CLI reports a percentage and no token counts (the v2 engine over a proprietary
notification, the v3 engine in `session_info_update` notifications), so the usage bar
and this threshold are accurate to one percent for Kiro agents.
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
| Kiro | Agent config name. v2 engine: passed as the `--agent` CLI flag + `set_session_mode`. v3 engine: passed as `_meta.kiro.modeId` on session creation + `set_session_mode` (v3 rejects `--agent`) | `plan`, `code-planner` |
| Claude Code | Agent name (passed by the `claude` extension as `_meta.claudeCode.options.extraArgs.agent`, which the adaptor turns into the Claude CLI's `--agent` flag; the agent picker switches agents by handoff, see below) | `local-SHScienceAgentKit-all:code-planner` |
| Codex | Codex mode id, applied with `set_session_mode` | `read-only`, `workspace-write`, `agent`, `agent-full-access` |
| OpenCode | OpenCode primary agent id (built-in `build`, `plan`, or a custom agent from OpenCode's config), applied with `set_config_option` on the `mode` option after the session starts | `plan`, `review` |
| Gemini CLI | Not supported | — |

For Claude Code, plugin agents require the full qualified name: `<plugin-name>:<agent-name>`. User/project agents (in `~/.claude/agents/` or `.claude/agents/`) use just the agent name.

Claude agent selection is verified on claude-agent-acp 0.70.0 and 0.81.2 (the adaptor's
first-class `options.agent` was removed in 0.77.0; `extraArgs.agent` works on both). Notes:

- The adaptor accepts an unknown agent name silently and runs the default agent, so synth
  checks the name against the user, project and plugin agents it discovers and shows a
  warning when none matches. The launch continues.
- The Claude CLI fixes a session's agent when the session is created. Resume and fork keep
  that agent, so restore cannot change a session's agent.
- The agent picker therefore switches a Claude agent by handoff. After you confirm, synth
  sends the agent a message (`handoff_switch_template`) asking it to call `handoff` with
  `agent_mode` set to the chosen agent and a briefing for it. The successor starts as the
  chosen agent, and any instructions you added are appended verbatim to its opening
  prompt. The picker shows the switch as pending until the handoff happens. If the agent
  ends its turn without handing off, or hands off without `agent_mode`, the switch is
  dropped with a warning; pick the agent again to retry. Kiro switches in place with
  `set_session_mode`.
- **In-turn steering:** Shift+Enter, the Steer button and eligible inter-agent
  messages use `_session/steering`. Synth asks the adaptor to leave a turn-end race
  unconsumed so it can deliver the text as a normal prompt. While a permission request
  is open, synth queues the steer for compatibility with adaptor 0.70.0; 0.71.0 fixed
  the adaptor to defer it safely.
- Sessions created on claude-agent-acp 0.77.0 or later by an older synth, which sent
  `options.agent`, ran as the default agent, and restoring them keeps the default agent.

### Config Options

Synth dynamically renders configuration pickers based on what the harness advertises:

| Harness | Available pickers |
|---------|------------------|
| Kiro (v2 engine) | Mode (agents), Model |
| Kiro (v3 engine) | Whatever v3 advertises: Mode, Model, Effort, and v3's own options (e.g. thinking, autopilot) |
| Claude Code | Mode (permission), Model, Effort |
| Codex | Mode, Collaboration mode, Model, Reasoning effort |
| OpenCode | Model, Effort, Mode (primary agents) |
| Gemini CLI | None |

Only options in the ACP-reserved categories (`mode`, `model`, `thought_level`) and the agent-profile picker appear inline. Any other option the harness returns, such as Kiro v3's Autopilot or Memory reflection, sits behind a `⋯ N more` label that opens a modal.

The effort level picker appears only for models that support it (e.g., Opus, Sonnet). Switching to Haiku removes the effort picker automatically.

### Kiro CLI v3 Engine

The `kiro` harness has two engine variants. `v2` (`kiro-cli acp`) is the default. `v3`
runs `kiro-cli acp --agent-engine=v3 --auth-method=cli` (kiro-cli 2.24.0 or later).
Enable v3 in `~/.synth/config.json`, or under `settings` in `.synth.json`:

```json
{ "harness_variant": { "kiro": "v3" } }
```

Differences on v3:

- **Agent profiles that use `allowedTools` or `toolsSettings` are rejected by v3**
  (reason code `cli_only_agent`), which includes the SHScienceAgentKit profiles. v3 then
  runs the agent in its default mode `vibe`, and synth shows a warning naming the
  requested profile, the fallback and the reason. The launch continues.
- Context usage is reported as a percentage, as on v2.
- A mode switch keeps the MCP servers connected, so synth does not reload the session.
- v3 finishes starting a session before its MCP servers connect, so synth keeps a v3
  agent initializing until v3 reports every MCP server connected or failed, for at most
  30 seconds. Prompts sent meanwhile are queued, so the first prompt always sees the MCP
  tools. A server that fails to start, and a server still connecting after 30 seconds,
  each produce a warning; the agent starts without that server's tools.
- v3 shows an agent only the MCP tools its profile's `tools` list names. So that every
  profile gets the synth-mcp tools, synth appends `synth-mcp` to the
  `ASBX_KIRO_MANDATORY_MCPS` environment variable of the v3 subprocess, keeping any
  servers already listed there.
- Restore loads the session without replaying its history (see below).
- v3 starts your global `~/.kiro` MCP servers and steering files in every session, so N
  agents run N copies of each global MCP server.

### Codex

The `codex` harness runs the `codex-acp` adaptor (verified on 2.0.0). Install it with:

```bash
npm install -g @agentclientprotocol/codex-acp
```

- **Auth** comes from your existing Codex login and `~/.codex/config.toml`. synth never
  calls ACP `authenticate`; if Codex needs a login, run `codex login` outside synth.
- The adaptor runs the Codex binary bundled with it, not a `codex` on your PATH.
- `agent_mode` is a Codex mode id: `read-only`, `workspace-write`, `agent` (Codex's
  default) or `agent-full-access`. Codex has no agent profiles.
- **In-turn steering**: an inter-agent message sent while the agent is busy is steered
  into the running turn. When the turn has already ended, Codex starts a turn of its own
  with the message instead ("startedNewTurn"), and synth gets no reply for it. synth
  keeps the agent busy through such a turn, following Codex's thread-status updates,
  and delivers queued prompts only after it ends. If Codex reports no activity within
  5 seconds of starting the turn, synth treats it as ended. Cancel works on these turns
  as on any other.
- **synth-mcp auto-approval**: Codex's MCP permission requests carry no tool name, so
  synth takes the tool identity (`synth-mcp/<tool>`) from the earlier tool call with the
  same id. Codex asks for MCP permission in `read-only` and `workspace-write`; in
  `agent` mode it runs its own review model call instead of asking.
- Codex starts your global `~/.codex` MCP servers in every session, so N agents run N
  copies of each.
- Restore uses `session/resume`, which keeps the conversation. Codex resets mode, model
  and reasoning effort on resume; synth re-applies `agent_mode` but not picker changes.

### OpenCode

The `opencode` harness runs `opencode acp` (verified on OpenCode 2.0.28).

- `agent_mode` is an OpenCode primary agent: `build` (default), `plan`, or a custom agent
  defined in OpenCode's config files or `agent/*.md` files. synth lists them for the launch
  picker by reading the same files OpenCode reads; agents added by organization config or
  plugins appear only in the live Mode picker.
- An `agent_mode` that OpenCode does not offer produces one warning, and the agent keeps its
  current agent: OpenCode's default on launch, the resumed session's agent on restore.
- **No in-turn steering**: OpenCode's ACP mode refuses a second prompt while a turn runs, so
  synth does not offer Steer for OpenCode agents; prompts and inter-agent messages sent while
  the agent is busy are queued for the next turn.
- Restore uses `session/resume`, which keeps the conversation; synth re-applies `agent_mode`.

### Session Restore

`synth --restore` reconnects each agent to its previous ACP session without replaying
the conversation into synth. For every harness, synth uses standard `session/resume`
when the agent advertises it (Claude Code), otherwise `session/load`, where Kiro v3 is
asked not to replay history. Any history an agent does replay during a load is
suppressed. If the resume or load fails, the agent starts a fresh session with a warning.

synth records the engine variant that created each agent's session, and restores each
agent under that recorded variant rather than the configured one, because a saved
session can only be loaded by the engine that created it. A session created under Kiro
v2 is restored by Kiro v2 even after you switch `harness_variant` to v3; agents launched
after the restore use the configured variant. Only when the recorded variant is one the
harness no longer declares does the agent start a fresh session under the configured
variant, without its previous conversation, with a warning naming the agent and both
variants.

### Adding a Harness

A harness is one TOML file in `src/synth_acp/data/harnesses/`. Fields:

| Field | Required | Description |
|-------|----------|-------------|
| `identity`, `name`, `short_name` | yes | Unique key, display name, and the value used with `--harness` and in `harness_env` / `harness_variant` |
| `binary_names` | yes | Executables searched in PATH for auto-detect |
| `run_cmd` | yes | Command that starts the harness in ACP mode |
| `mode_arg` | no | CLI flag that carries `agent_mode` (e.g. `--agent`) |
| `extension` | no | ACP extension name; defaults to `"standard"` (plain ACP) |
| `executable_env_var`, `clear_env_vars`, `install_hint` | no | Subprocess environment tweaks and the install command shown when `run_cmd` is missing |

A harness with more than one engine declares variants instead of top-level `run_cmd` / `mode_arg` / `extension`, and `harness_variant` picks one:

```toml
default_variant = "v2"

[variants.v2]
run_cmd = "kiro-cli acp"
mode_arg = "--agent"
extension = "kiro_v2"
```

Standard ACP (sessions, prompts, permissions, modes, models, config options, usage updates) works with no extension. Write an extension module only for proprietary behavior: extra `_meta` on session requests, proprietary notifications or agent-to-client requests, in-turn steering, or agent-profile discovery. Subclass `AcpExtension` in `src/synth_acp/acp/extensions/`, override only the hooks the engine needs, and register the class in `src/synth_acp/acp/extensions/__init__.py`. ACPSession, the broker and the UI need no change: they read only the extension's `ExtensionCapabilities` and the normalized values its hooks return.

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
| `handoff` | Retire yourself and hand your work to a fresh session that keeps your agent id. Optional `agent_mode` starts the successor as another agent profile (`""` for the harness default; omitted keeps the current one). An unknown profile, or one other than the profile the user picked, is returned as an error and nothing retires |

Messages are delivered to idle agents between turns. MCP-delivered inter-agent messages are prefixed with sender attribution via the configurable `on_mcp_message` hook (active by default), so recipients can see who sent a message and whom to reply to; `system` messages (join/exit notifications) use a no-action wrapper instead. Message bodies are always preserved unchanged.

## Key Bindings

| Key | Action |
|-----|--------|
| `Enter` | Send the prompt; while the agent is busy, queue it for the next turn |
| `Shift+Enter` | While the agent is busy, steer: inject the text into the running turn, ahead of queued prompts (Claude, Kiro and Codex) |
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
