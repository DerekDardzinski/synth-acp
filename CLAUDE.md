# AGENTS.md

## Multi-Agent Orchestration

When working on this project within a Synth session, **always use `synth-mcp` tools (`launch_agent`, `send_message`, `terminate_agent`) to spawn and coordinate child agents** — never use native Claude Code subagents (the `Agent` tool). Native subagents bypass the Synth broker, so they won't appear in the TUI, won't be managed by the session lifecycle, and their messages won't route through the message bus. Using synth-mcp ensures agents are visible, manageable, and testable through the same system you're developing.

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
├── discovery.py        # Agent discovery: dispatches to the harness extension's discover_profiles
├── embeddings.py       # Standalone embedding module for semantic session search
├── harnesses.py        # Harness registry loader (TOML + harness_variant → HarnessEntry)
├── data/
│   └── harnesses/      # Harness TOML definitions (kiro.toml, claude.toml, codex.toml, opencode.toml, gemini.toml)
├── models/
│   ├── agent.py        # AgentState enum, AgentConfig, DiscoveredAgent
│   ├── config.py       # SessionConfig, HooksConfig (parsed from .synth.json)
│   ├── extensions.py   # ExtensionCapabilities, signals, UpdateHints, SteerOutcome (extension → synth)
│   ├── events.py       # BrokerEvent and subclasses (broker → frontend)
│   ├── commands.py     # BrokerCommand and subclasses (frontend → broker)
│   ├── visibility.py   # Agent visibility rules (MESH/LOCAL)
│   └── permissions.py  # PermissionRule, PermissionDecision
├── acp/
│   ├── session.py      # ACPSession — wraps acp SDK Client interface; standard ACP for every harness
│   ├── state_machine.py # AgentStateMachine — typed state transitions
│   └── extensions/     # Per-engine proprietary protocol, one stateless AcpExtension each
│       ├── __init__.py # Explicit registry: get_extension, register_extension
│       ├── base.py     # AcpExtension — plain standard ACP (the "standard" extension)
│       ├── claude.py   # ClaudeExtension — claudeCode _meta, handoff to switch agent, discovery
│       ├── codex.py    # CodexExtension — _session/steering, threadStatus → TurnActivity, MCP identity
│       ├── kiro_v2.py  # KiroV2Extension — kiro.dev/metadata usage, session/steer, discovery
│       ├── kiro_v3.py  # KiroV3Extension — _meta.kiro modeId/noReplay, context_usage, shell_type
│       └── opencode.py # OpenCodeExtension — agent_mode via mode config option, offline agent discovery
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

- `agent-client-protocol==0.9.0` — ACP Python SDK (Pydantic models, `spawn_agent_process`)
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
- Runs the `claude-agent-acp` binary directly, resolved from PATH. Install it with
  `npm install -g @agentclientprotocol/claude-agent-acp`. No `npx`: a package manager
  in the spawn path resolves over the network on every launch and, with an unreachable
  registry, was measured hanging with no output at all, which synth could only show as
  a permanent INITIALIZING
- `install_hint` in the TOML names that install command, reported when the adaptor is
  absent from PATH. Auto-detect also skips a harness whose `run_cmd` program is missing,
  so claude is not offered on a machine that has Claude Code but not the adaptor
- `extension = "claude"` — `ClaudeExtension` passes `agent_mode` as `_meta.claudeCode.options.extraArgs.agent`
  on every session op (the adaptor passes it to the Claude CLI as `--agent`). Never `options.agent`:
  claude-agent-acp >= 0.77.0 deletes it, and sessions created on those versions with it ran as the
  default agent. Verified on 0.70.0 and 0.81.2, including `<plugin>:<agent>` names
- `ClaudeExtension.check_profile` warns (one BrokerError, launch continues) when `agent_mode` matches
  no discovered agent, because the adaptor accepts unknown names silently
- The CLI fixes the agent at session creation: resume and fork keep it, so restore cannot change a
  session's agent. `profile_switch = "handoff"` makes the picker send `RequestProfileSwitch`: the broker
  asks the agent (template `handoff_switch_template`) to call the `handoff` MCP tool with `agent_mode`,
  and the successor starts as that agent with the user's instructions appended verbatim. The switch is
  pending until the handoff, and is dropped with a warning when the agent's turn ends without one
- ACPSession logs `InitializeResponse.agentInfo` name and version at INFO for every harness, so a
  selection regression can be traced to an adaptor bump
- `executable_env_var = "CLAUDE_CODE_EXECUTABLE"` — injects detected binary path
- `clear_env_vars = ["CLAUDECODE"]` — clears stale env vars in subprocess
- Returns `config_options` with mode (permission), model, and effort (model-dependent)
- In-turn steering uses `_session/steering` with `idleBehavior: promptRequired`, so a
  turn-end race leaves the text unconsumed for synth to deliver as a prompt rather than
  starting an untracked detached turn. Live-verified on claude-agent-acp 0.70.0 and 0.90.0
- While a Claude permission request is open, synth queues steering messages instead of
  injecting them. This keeps compatibility with adaptor 0.70.0, whose immediate steer
  can cancel the permission card and leave the prompt hanging; 0.71.0 fixed the adaptor
- `set_config_option()` response returns full updated config_options (effort may disappear on model switch)
- Agent names for plugins: `<plugin-name>:<agent-name>` (e.g., `local-SHScienceAgentKit-all:code-planner`)

**Kiro** (`kiro.toml`):
- `default_variant = "v2"`; `[variants.v2]` has `run_cmd = "kiro-cli acp"`, `mode_arg = "--agent"`
  (passes `agent_mode` as CLI flag) and `extension = "kiro_v2"`. Pick a variant with the
  `harness_variant` setting
- `KiroV2Extension`: in-turn steering via `session/steer` (`capabilities.steer`), context usage
  from `kiro.dev/metadata`, and an MCP reload via `session/load` after a mode switch
  (`reload_after_mode_switch`)
- v2 returns `modes` (all agents) + `models`, NO `config_options`
- Synth synthesizes `config_options` from legacy modes/models
- v2: `set_session_mode()` / `set_session_model()` used (no `set_config_option` endpoint)
- `[variants.v3]` has `run_cmd = "kiro-cli acp --agent-engine=v3 --auth-method=cli"`, no
  `mode_arg` (v3 exits 2 on `--agent`) and `extension = "kiro_v3"`. Enable with
  `{"harness_variant": {"kiro": "v3"}}`
- `KiroV3Extension`: answers the `_kiro/terminal/shell_type` request (v3 session/new fails
  without it), sends `agent_mode` as `_meta.kiro.modeId` on session/new and
  `_meta.kiro.noReplay` on session/load (`no_replay_load`), reads `_meta.kiro.replay` and
  `context_usage` session_info_updates, maps `_kiro/customAgent/not_found` / `config_error`
  to a warning, and maps steer `queued: false` to `dropped` so the broker queues the message.
  No MCP reload after a mode switch (v3 keeps the servers)
- v3 rejects profiles that use `allowedTools` / `toolsSettings` (reason `cli_only_agent`,
  all SHScienceAgentKit profiles) and falls back to mode `vibe`; synth shows a warning
- v3 returns native `configOptions` and no `models`; `session/set_model` fails (-32603), so
  only the native `set_config_option` path is used
- Restore: `session/resume` when advertised (Claude), else `session/load` with the
  extension's load `_meta`. `agents.harness_variant` in synth.db records the variant that
  created each session; restore starts fresh with a warning when it differs from the
  configured variant (existing kiro rows are backfilled as `v2`)

**Codex** (`codex.toml`):
- `run_cmd = "codex-acp"` (npm `@agentclientprotocol/codex-acp`, verified on 2.0.0; it runs its
  bundled Codex, not one on PATH), `extension = "codex"`. Auth comes from the user's Codex login;
  synth never calls `authenticate`
- `agent_mode` = Codex mode id (`read-only`, `workspace-write`, `agent`, `agent-full-access`)
  through the standard `set_session_mode` path; no profiles to discover
- `CodexExtension`: steering via `ext_method("session/steering", {sessionId, prompt: [text block]})`,
  outcome `injected` → `delivered`, `startedNewTurn` → `started_turn` (accepted, not re-queued),
  anything else → `failed`; `_meta.codex.threadStatus` on `session_info_update` → `TurnActivity`
  (`active` → True; `idle` / `systemError` / `notLoaded` → False); MCP permission identity from
  rawInput `{server, tool}`, or from an `mcp.<server>.<tool>` title only when the request is
  flagged `is_mcp_tool_approval`
- Adopted turns (generic, in ACPSession): a `TurnActivity(True)` on an IDLE agent with no own
  prompt moves it BUSY until `TurnActivity(False)`, which emits TurnComplete and goes IDLE; a
  finishing prompt waits (bounded by `_STEER_ACCEPTANCE_TIMEOUT`) for an in-flight steer answer;
  a `started_turn` steer keeps the agent BUSY until active-then-idle, or IDLE with no
  TurnComplete if no active arrives within `_ADOPTED_TURN_START_TIMEOUT` (5 s). Activity during
  synth's own prompt is ignored (Codex sends idle just before the PromptResponse). Without
  this a queued prompt sent into a Codex-started turn merges into it and is never answered
- Codex MCP permission requests carry only `{toolCallId, kind, status}`; ACPSession fills title
  and rawInput from the earlier tool_call with that id (generic cache, dropped at
  completed/failed). The filled title reaches `PermissionRequested.title` only when the
  extension found an identity, so the title substring fallback never sees a cached shell command
- Restore: `session/resume` (advertised), zero replay; Codex resets mode/model/effort on open
- Global `~/.codex` MCP servers load in every session

**OpenCode** (`opencode.toml`):
- `run_cmd = "opencode acp"`, `extension = "opencode"`. Verified on OpenCode 2.0.28 (product v2;
  the wire is ACP protocol v1, standard methods only)
- Returns native `configOptions` only (model, effort = `thought_level`, mode); no legacy `modes`
  or `models`, and session/new ignores `_meta`
- `OpenCodeExtension.capabilities.profile_switch = "config_option"`: ACPSession applies
  `agent_mode` after session/new and after restore with `set_config_option` on the
  `mode`-category option (`_apply_agent_mode_option`), only when the current value differs; a
  value the option does not offer is one BrokerError warning and the session runs in its current mode
- Mode values are OpenCode primary agents: built-ins `build`, `plan`, plus custom agents.
  `discover_profiles` reads them offline in OpenCode's order: global dir (`$OPENCODE_CONFIG_DIR`, else
  `$XDG_CONFIG_HOME|~/.config` + `/opencode`) json, jsonc, then `{agent,agents}/**/*.md` and
  `{mode,modes}/*.md`; `$OPENCODE_CONFIG`; `opencode.json[c]` from filesystem root down to cwd;
  each `.opencode` dir on that path; `$OPENCODE_CONFIG_CONTENT`. Later wins; `disabled`/`disable`
  removes; subagent or hidden agents are not selectable; `default_agent` goes first in
  `discover_profiles` (`discovery.discover_agents` re-sorts by source and name for the picker); markdown
  frontmatter is retried with unquoted-colon values as block scalars and empty files are skipped, as
  OpenCode's `ConfigMarkdown.parse` does. Organization
  (wellknown) config and plugin-added agents are not visible offline
- No steering: a second session/prompt during a turn is rejected (-32603 "Session already has an
  active ACP prompt") and no steer method exists over ACP, so messages queue until the turn ends.
  Each `opencode acp` runs its own private server (random port, password in its env only); do not
  build on the background `opencode service`
- Not handled (no current need): the `opencode/child-session-updates` opt-in (without it child
  updates are projected into the parent stream with `_meta["opencode/child-session"]` and toolCallIds
  prefixed `<childId>:`), `_meta["opencode/retry"]` and `_meta["opencode/compaction"]` markers on
  `session_info_update`, and MCP permission identity (default permissions never ask for MCP tools,
  and Code Mode wraps MCP calls in an `execute` tool)
- Restore: `session/resume` (advertised), zero replay; resume requires the session's original cwd

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
  A UNIT test file may only import from one source module — crossing into another module's
  territory is a structure violation. An INTEGRATION test that DRIVES one layer and ASSERTS
  on another is permitted, and belongs in the mirrored file of the module under ASSERTION:
  a test that drives an `ACPSession` and asserts on the rows the broker journals is a
  broker test.
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
- Render agent output incrementally from the raw `session/update` notification stream. Synth deliberately does NOT use `SessionAccumulator`: its `apply()` deep-copies the entire accumulated history into a validated snapshot on every notification (166.7 ms per notification at real scale) that nothing reads, and it retains every chunk for the agent's whole lifetime. `ACPSession.session_update` dispatches straight to a `BrokerEvent`.

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
