# Changelog

## 0.3.0 — 2026-04-30

### Features

- **resurrect_agent tool** — recover terminated agents, restoring their conversation history
- **Copy button on tool call blocks** — one-click copy of tool call content
- **Launch modal on startup** — skip CLI picker and show the launch modal when no `.synth.json` is found

### Performance

- **Viewport-aware turn visibility** — only render visible turns in long conversations, dramatically reducing DOM size
- **Batch replay optimization** — faster session restore by batching message replay
- **Optimized TUI rendering** — reduced layout thrash for long sessions with many agents

### Bug Fixes

- **Shutdown hang eliminated** — complete rewrite of shutdown path: removed `conn.close()` (which had no timeout), switched to single-phase process kill, null executor trick to skip 300s `shutdown_default_executor` wait, and watchdog timer as safety net
- **Headless shutdown hang** — fixed restorable status transition that blocked headless mode exit
- **MarkupError crashes** — escape unescaped content in Static widgets
- **TUI crash vectors** — fix agent switching in ContentSwitcher when agents are terminated
- **Shell output duplication** — deduplicate shell output in tool call blocks, fix RichLog overflow/reflow
- **Blank scroll-back** — remove redundant turn visibility toggling
- **Scroll position after restore** — scroll to bottom after session restore replay
- **Dots in agent IDs** — allow dotted identifiers (e.g., `bd-a1b2c3.1`) in launch commands and throughout
- **Descriptive launch errors** — return actionable error messages from `launch_agent` MCP tool
- **RichLog reflow** — only reflow on width changes to prevent layout collapse; skip redundant reflow on first mount

### Internal

- **Terminated agent cleanup** — remove session state for terminated agents to prevent memory leaks
- **Shutdown ordering** — crash-resilient shutdown sequence with proper task cancellation
- **Sync sqlite3 for session creation** — prevent shutdown hang from aiosqlite thread in `_on_acp_session_created`

## 0.2.0

Initial public release.
