# SYNTH — UX & CLI Design

**Version:** 0.1-draft  
**Date:** 2026-03-25  
**Relates to:** `DESIGN.md`, `CONFIG_ALTERNATIVES.md`, `GAPS.md`

---

## 1. Design Principles

**Feel like the underlying tool, not a wrapper.** Running `synth` in a project folder should feel as natural as running `kiro-cli` or `claude` directly. The multi-agent capability is additive — it should not create overhead or friction for the single-agent case.

**Config file is an output, not a prerequisite.** A user should be able to run `synth` in a new project with no prior setup and be guided through configuration interactively. The resulting `.synth.toml` is a project artifact worth committing to git, but it is never required to have been hand-edited.

**The orchestrator is primary.** The design assumes the primary workflow is one orchestrator that the user talks to directly, which dynamically spawns worker agents as needed. Static team configs are supported but secondary. This shapes every layout and interaction decision.

---

## 2. Launch Model

### 2.1 Happy Path — Project with Existing Config

```
$ cd ~/projects/my-api
$ synth
```

SYNTH finds `.synth.toml` (or `.synth.json` for backward compat), loads it, launches the configured agent(s), and opens the TUI. No flags needed. This mirrors running `kiro-cli` or `claude` in a project folder directly.

### 2.2 First-Run — No Config File

```
$ cd ~/projects/new-project
$ synth

No .synth.toml found. Let's set one up.

Which harness?
  1) kiro    (/home/user/.local/bin/kiro-cli)
  2) claude  (/usr/local/bin/claude)

  › 1

Agent? (leave blank for kiro's default)
  The agent list is discovered live at session start via the ACP protocol.
  Examples: orchestrator, backend-specialist, reviewer

  › orchestrator

Agent ID for the TUI sidebar (default: orchestrator)

  › [Enter]

Project name (default: new-project)

  › [Enter]

Saved .synth.toml — launching...
```

The config file is written, then SYNTH launches immediately. No second invocation required.

**Key behaviour:**
- Probes `PATH` (plus `~/.local/bin`, `~/.cargo/bin`, npm global paths) for known ACP-capable binaries.
- Shows only installed harnesses, with their resolved path for disambiguation.
- Agent input is free-text because the full list is discovered live from the ACP `modes` field in `NewSessionResponse` — there's no reliable way to enumerate agents before starting a session. The input line explains this.
- If no harnesses are found, prints install instructions for common ones and exits.
- First-run is CLI-based (plain `input()` prompts), not TUI — consistent with how `git init`, `npm init`, and similar tools work. The TUI is for running, not setup.

### 2.3 One-Off — CLI Flags, No Config File

```
$ synth --harness kiro --agent orchestrator
$ synth --harness claude
$ synth --harness kiro --agent orchestrator --headless
```

`--harness` and `--agent` bypass config file discovery entirely and construct a transient `SessionConfig` in memory. Nothing is written to disk. This is the escape hatch for:
- Trying a new harness or agent without modifying project config
- CI/CD pipelines where config is constructed dynamically
- Scripting one-off sessions

**Flag naming rationale:**
- `--harness` replaces `--provider`. "Harness" is the term already used throughout SYNTH's architecture (`DESIGN.md` section 3.1). It's specific to SYNTH's vocabulary and less likely to conflict with existing tooling.
- `--agent` is the established convention across the harnesses SYNTH targets. Kiro uses `kiro-cli acp --agent <name>` and Claude Code uses `claude --agent <name>`. Reusing the same flag keeps the mental model consistent — users arriving from those tools already know what `--agent` means. The `--harness` flag present alongside it provides enough context to make clear that `--agent` refers to an agent profile within the harness, not a SYNTH agent ID.

### 2.4 Explicit Config Path

```
$ synth --config /path/to/team.synth.toml
$ synth --config ~/.synth/shared-team.toml
```

Supports shared configs stored outside the project directory.

### 2.5 Resolution Order

When `main()` runs, config is resolved in this priority order:

1. `--harness` present → build transient config from flags, skip all file discovery
2. `--config PATH` present → load that specific file
3. Auto-discover `.synth.toml` then `.synth.json` in CWD
4. First-run interactive picker (TUI mode only; headless prints an error and exits)

---

## 3. Config File Format

### 3.1 New Format — `.synth.toml`

```toml
# .synth.toml
project = "my-api"

[[agents]]
id      = "orchestrator"
cmd     = ["kiro-cli", "acp", "--agent", "orchestrator"]
label   = "Orchestrator"        # display name in TUI sidebar (optional, defaults to id)
profile = "orchestrator"        # harness agent name, preserved for display/tracking (optional)
cwd     = "."

[[agents]]
id      = "reviewer"
cmd     = ["claude", "mcp", "--agent", "reviewer"]
label   = "Code Reviewer"
profile = "reviewer"
env     = { ANTHROPIC_MODEL = "claude-opus-4-6" }
```

### 3.2 Field Reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `project` | `str` | Yes | Replaces `session`. Shown in TUI topbar and used in session ID construction. |
| `[[agents]]` | array | Yes | One or more agent definitions. |
| `id` | `str` | Yes | Machine identifier. Must match `[a-zA-Z0-9][a-zA-Z0-9_-]*`. Used as CSS ID in TUI. |
| `cmd` | `list[str]` | Yes | Full argv to spawn the agent subprocess. Replaces `binary` + `args`. |
| `label` | `str` | No | Human-readable name in the TUI sidebar. Defaults to `id`. |
| `profile` | `str` | No | Records which harness agent was selected (e.g. `"orchestrator"`). Preserved for display and session tracking; not used at spawn time since it is already baked into `cmd`. |
| `cwd` | `str` | No | Working directory, relative to the config file's location. Defaults to `"."`. |
| `env` | `dict` | No | Per-agent environment variables merged on top of the inherited process environment. Does not replace the inherited environment. |

### 3.3 Removed Fields

| Field | Disposition |
|---|---|
| `binary` | Removed. Merged into `cmd[0]`. |
| `args` | Removed. Merged into `cmd[1:]`. |
| `session` | Renamed to `project`. Supported as a deprecated alias through the migration period — old `.synth.json` files load without modification. |
| `autostart` | Removed. With an orchestrator-first model, SYNTH always starts all configured agents. The `autostart` boolean was a static-team concept. |

### 3.4 Backward Compatibility

The `load_config` function reads both `.toml` and `.json`. The JSON parser accepts the old `session` key as a synonym for `project`, and the old `binary`/`args` fields are coerced to `cmd` at parse time. Existing `.synth.json` files load without modification.

File discovery checks `.synth.toml` first, then `.synth.json`. If both exist, `.synth.toml` wins with a warning.

### 3.5 Generated Config Example

When the first-run picker creates a config, it writes this minimal form:

```toml
project = "my-api"

[[agents]]
id = "orchestrator"
cmd = ["kiro-cli", "acp", "--agent", "orchestrator"]
label = "orchestrator"
profile = "orchestrator"
```

---

## 4. Known Harnesses Registry

SYNTH ships a small built-in registry of known ACP-capable binaries. This powers:
- PATH probing in the first-run picker
- `--harness` flag resolution
- Display names and install hints when a harness is not found

### 4.1 Registry Entries (initial set)

| Short Name | Binary Names | ACP Command | With `--agent` |
|---|---|---|---|
| `kiro` | `kiro-cli` | `kiro-cli acp` | `kiro-cli acp --agent {agent}` |
| `claude` | `claude` | `claude mcp` | `claude mcp --agent {agent}` |
| `opencode` | `opencode` | `opencode acp` | `opencode acp` (agent via config) |
| `gemini` | `gemini` | `gemini --experimental-acp` | `gemini --experimental-acp -e {agent}` |

The registry is data, not code — stored as TOML files in `src/synth_acp/data/harnesses/`. Each file is one harness definition. Adding support for a new harness is a TOML file addition with no Python changes required.

### 4.2 Harness File Schema

```toml
# src/synth_acp/data/harnesses/kiro.dev.toml

identity    = "kiro.dev"           # unique reverse-domain key
name        = "Kiro CLI"
short_name  = "kiro"               # used with --harness flag
binary_names = ["kiro-cli"]        # searched in PATH

run_cmd."*"              = "kiro-cli acp"
run_cmd_with_agent."*"  = "kiro-cli acp --agent {agent}"

[actions."*".install]
command     = "curl -fsSL https://cli.kiro.dev/install | bash"
description = "Install Kiro CLI"

[actions."*".login]
command     = "kiro-cli login"
description = "Login (run once after install)"
```

The `run_cmd` and `run_cmd_with_agent` fields use an OS matrix (`"*"`, `"macos"`, `"linux"`, `"windows"`) identical to Toad's pattern, allowing platform-specific command differences without code changes.

---

## 5. `AgentConfig` Model Changes

### 5.1 New Shape

```python
class AgentConfig(BaseModel, frozen=True):
    id: str                          # machine identifier, CSS-safe
    cmd: list[str]                   # full argv, always resolved
    label: str | None = None         # display name; defaults to id
    profile: str | None = None       # harness agent name, preserved for display
    cwd: str = "."
    env: dict[str, str] = {}
```

### 5.2 Backward-Compatible Properties

To avoid churn in existing callsites while the codebase migrates:

```python
@property
def display_name(self) -> str:
    return self.label or self.id

@property
def binary(self) -> str:
    """cmd[0] — for backward compatibility."""
    return self.cmd[0]

@property
def args(self) -> list[str]:
    """cmd[1:] — for backward compatibility."""
    return self.cmd[1:]
```

The broker's `_launch()` call currently reads `agent_cfg.binary` and `agent_cfg.args` separately. These properties mean it continues to work without changes until we do a cleanup pass.

### 5.3 Legacy Coercion

A `model_validator(mode="before")` converts the old shape automatically:

```python
# Old .synth.json:  {"binary": "kiro-cli", "args": ["acp"]}
# Loaded as:        AgentConfig(cmd=["kiro-cli", "acp"])
```

---

## 6. `SessionConfig` Model Changes

### 6.1 Rename `session` → `project`

```python
class SessionConfig(BaseModel, frozen=True):
    project: str          # was: session
    agents: list[AgentConfig]
    ui: UIConfig = UIConfig()
```

A `model_validator(mode="before")` accepts `session` as a synonym during the migration period. The broker currently uses `config.session` in one place (session ID construction); that reference updates to `config.project`.

### 6.2 Remove `autostart`

`AgentConfig.autostart` is removed. SYNTH always launches all configured agents on startup. The field no longer carries meaning with an orchestrator-first model — there is no concept of a configured agent that you don't want to start.

If users want an agent to be available-but-not-started, they simply don't add it to `.synth.toml` — they can launch it from the TUI later or add it to the config when they're ready to use it consistently.

---

## 7. CLI Flag Summary

| Flag | Short | Description |
|---|---|---|
| `--harness NAME` | | Harness to launch. Known values: `kiro`, `claude`, `opencode`, `gemini`. Bypasses config file. |
| `--agent NAME` | | Agent within the harness (e.g. `orchestrator`). Only meaningful with `--harness`. Matches the `--agent` flag convention used by Kiro and Claude Code. |
| `--config PATH` | `-c` | Explicit path to a `.synth.toml` or `.synth.json`. |
| `--headless` | | Run without TUI (stdin/stdout mode). |
| `--verbose` | `-v` | Enable debug logging. |

### 7.1 Usage Examples

```bash
# Standard usage — loads .synth.toml from CWD
synth

# One-off session with a specific harness and agent, no config file
synth --harness kiro --agent orchestrator

# One-off with just a harness — uses the harness's default agent
synth --harness claude

# Use a specific config file
synth --config ~/configs/backend-team.toml

# Headless one-off for scripting
synth --harness kiro --agent orchestrator --headless

# Verbose for debugging ACP traffic
synth --verbose
```

---

## 8. What Does NOT Change

This document covers changes to the launch model, config format, and CLI surface. The following are explicitly out of scope and unchanged:

- **Broker architecture** — `ACPBroker`, `ACPSession`, `PermissionEngine`, `MessagePoller` are all unchanged.
- **TUI layout** — `SynthApp`, `ConversationFeed`, `AgentList`, all widgets are unchanged.
- **ACP session layer** — The handshake, streaming, permission flow, and tool callbacks are unchanged.
- **MCP server** — `synth-mcp`, the SQLite message bus, and inter-agent messaging are unchanged.
- **Models/events/commands** — `BrokerEvent`, `BrokerCommand` subclasses are unchanged.
- **The headless mode** — The stdin/stdout interactive loop is preserved as-is.

The broker still receives an `AgentConfig` with a resolved `cmd` list — the resolution of harness + agent into a `cmd` happens at config load time, not in the broker. From the broker's perspective, nothing changes.

---

## 9. Implementation Order

1. **`models/agent.py`** — Replace `binary`/`args`/`autostart` with `cmd`/`label`/`profile`/`env`. Add backward-compat coercion and `binary`/`args` properties.
2. **`models/config.py`** — Rename `session`→`project`, add `find_config()`, TOML support via `tomllib`, `write_toml_config()`. Add legacy coercion for `session` key.
3. **`cli.py`** — Add `--harness`/`--agent` flags, `_KNOWN_HARNESSES` registry, `_first_run_picker()`, `_resolve_config()`. Remove the hard exit on missing config.
4. **`src/synth_acp/data/harnesses/*.toml`** — Ship the initial harness TOML files as package data.
5. **`.synth.json` → `.synth.toml`** — Update the repo's own config file.
6. **Broker callsite** — Update `broker.py` to use `cmd[0]`/`cmd[1:]` directly instead of `binary`/`args`. Update `config.session` reference to `config.project`.
7. **Tests** — Update `test_config.py`, `test_agent.py`, and broker/session test fixtures to use the new field names.
8. **`DESIGN.md`** — Update section 3.4 (SessionConfig), section 14 (Entrypoints), and the config examples throughout.
