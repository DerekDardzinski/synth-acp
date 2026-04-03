#!/usr/bin/env python3
"""
probe_config_options.py — Test ACP agent capabilities for mode switching.

Tests two things:
  1. configOptions — whether set_config_option returns model info (eliminating
     the need for the load_session probe).
  2. session/resume — whether the agent supports resume_session with mcp_servers,
     which re-establishes MCP tool connections after a mode switch without
     triggering a history replay.

Usage:
    python probe_config_options.py <command> [args...]

Examples:
    python probe_config_options.py claude mcp
    python probe_config_options.py kiro-cli acp
    python probe_config_options.py opencode acp
    python probe_config_options.py gemini --experimental-acp
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import os
import sys
import time
from typing import Any

from acp.client.connection import ClientSideConnection
from acp.schema import (
    ClientCapabilities,
    FileSystemCapabilities,
    Implementation,
    McpServerStdio,
)
from acp.transports import default_environment

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {CYAN}·{RESET} {msg}")


def section(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")


def kv(key: str, val: Any) -> None:
    print(f"    {DIM}{key:<24}{RESET} {val}")


# ── Minimal ACP client — tracks session_update call counts ────────────────────


class MinimalClient:
    """Duck-typed ACP Client. Counts session_update calls per session ID so we
    can detect whether an operation is triggering a history replay."""

    def __init__(self) -> None:
        self.update_counts: dict[str, int] = {}

    def reset_counts(self) -> None:
        self.update_counts.clear()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.update_counts[session_id] = self.update_counts.get(session_id, 0) + 1

    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
        from acp.schema import DeniedOutcome, RequestPermissionResponse

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    def on_connect(self, conn: Any) -> None:
        pass


# ── Timing helper ──────────────────────────────────────────────────────────────


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)


# ── Fake MCP server spec (used to test mcp_servers re-passing) ────────────────


def _fake_mcp_server() -> McpServerStdio:
    """A placeholder MCP server spec for testing. Uses 'true' (always exits 0)
    so the agent attempts to connect but gets an immediate EOF — enough to verify
    the agent accepts the parameter without erroring on the RPC itself."""
    return McpServerStdio(
        name="probe-mcp",
        command="true",
        args=[],
        env=[],
    )


# ── Core probe ─────────────────────────────────────────────────────────────────


async def probe(command: str, args: list[str], cwd: str) -> None:
    section(f"Spawning agent: {command} {' '.join(args)}")

    env = dict(default_environment())
    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
        stderr=aio_subprocess.PIPE,
        env=env,
        cwd=cwd,
        process_group=0,
    )

    if not process.stdout or not process.stdin:
        err("Failed to open stdin/stdout pipes")
        return

    client = MinimalClient()
    conn = ClientSideConnection(client, process.stdin, process.stdout)

    try:
        # ── 1. Initialize ──────────────────────────────────────────────────────
        section("1 · initialize")
        t = Timer()
        init = await conn.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                terminal=False,
            ),
            client_info=Implementation(name="probe_config_options", version="0.1.0"),
        )
        info(f"Completed in {t.elapsed_ms()} ms")

        caps = getattr(init, "agent_capabilities", None)
        if caps:
            kv("loadSession", getattr(caps, "load_session", "?"))
            kv("agentCapabilities", str(caps)[:100])

        # ── 2. session/new ─────────────────────────────────────────────────────
        section("2 · session/new")
        t = Timer()
        session = await conn.new_session(cwd=cwd)
        info(f"Completed in {t.elapsed_ms()} ms")
        kv("sessionId", session.session_id)
        session_id = session.session_id

        # ── 3. Inspect configOptions / legacy modes ────────────────────────────
        section("3 · Mode API support")
        config_options = getattr(session, "config_options", None) or []
        mode_ids: list[str] = []

        if config_options:
            ok(f"configOptions: {len(config_options)} option(s) returned")
            mode_option = None
            for opt in config_options:
                category = getattr(opt, "category", None)
                opt_id = getattr(opt, "id", "?")
                current = getattr(opt, "current_value", "?")
                option_vals = [
                    getattr(o, "value", "?") for o in (getattr(opt, "options", []) or [])
                ]
                kv(f"[{opt_id}] category", category or "(none)")
                kv(f"[{opt_id}] current", current)
                kv(f"[{opt_id}] options", ", ".join(option_vals))
                if category == "mode":
                    mode_option = opt
                    mode_ids = option_vals

            if mode_option:
                ok(f"Mode config option found: id='{getattr(mode_option, 'id', '?')}'")
                await _test_set_config_option(conn, session_id, mode_option)
            else:
                warn("No config option with category='mode' found")
        else:
            warn("No configOptions — using legacy modes API")
            modes = getattr(session, "modes", None)
            if modes:
                current = getattr(modes, "current_mode_id", "?")
                available = getattr(modes, "available_modes", []) or []
                mode_ids = [getattr(m, "id", "?") for m in available]
                kv("current mode", current)
                kv("available", ", ".join(mode_ids))
            else:
                err("No modes field — agent exposes no mode information")

        # ── 4. session/resume test ─────────────────────────────────────────────
        if mode_ids:
            await _test_resume_session(conn, client, session_id, cwd, mode_ids)
        else:
            warn("Skipping resume test — no modes available to switch to")

    finally:
        await conn.close()
        if process.stdin and not process.stdin.is_closing():
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except (TimeoutError, asyncio.CancelledError):
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                process.kill()


# ── set_config_option sub-test ─────────────────────────────────────────────────


async def _test_set_config_option(conn: Any, session_id: str, mode_option: Any) -> None:
    section("3a · session/set_config_option — mode switching")

    mode_opt_id = getattr(mode_option, "id", "mode")
    mode_options = getattr(mode_option, "options", []) or []
    current_mode = getattr(mode_option, "current_value", None)

    print(f"\n  Starting mode: {BOLD}{current_mode}{RESET}")
    results: list[dict[str, Any]] = []

    for mode_opt in mode_options:
        target_mode = getattr(mode_opt, "value", "?")
        print(f"  {CYAN}→ set_config_option('{mode_opt_id}', '{target_mode}'){RESET}")
        t = Timer()
        try:
            response = await conn.set_config_option(
                config_id=mode_opt_id, session_id=session_id, value=target_mode
            )
            elapsed = t.elapsed_ms()
            resp_opts = getattr(response, "config_options", []) or []
            new_mode = next(
                (
                    getattr(o, "current_value", None)
                    for o in resp_opts
                    if getattr(o, "category", None) == "mode"
                ),
                None,
            )
            new_model = next(
                (
                    getattr(o, "current_value", None)
                    for o in resp_opts
                    if getattr(o, "category", None) == "model"
                ),
                None,
            )
            model_str = (
                f"{BOLD}{new_model}{RESET}" if new_model else f"{DIM}(not in response){RESET}"
            )
            mode_str = (
                f"{GREEN}{new_mode}{RESET}"
                if new_mode == target_mode
                else f"{RED}{new_mode}{RESET}"
            )
            print(f"    elapsed: {elapsed} ms  |  mode: {mode_str}  |  model: {model_str}")
            results.append(
                {
                    "mode": target_mode,
                    "elapsed": elapsed,
                    "model": new_model,
                    "success": new_mode == target_mode,
                }
            )
        except Exception as exc:
            elapsed = t.elapsed_ms()
            err(f"Failed after {elapsed} ms: {exc}")
            results.append(
                {"mode": target_mode, "elapsed": elapsed, "model": None, "success": False}
            )
        print()

    any_model = any(r["model"] for r in results)
    avg_elapsed = int(sum(r["elapsed"] for r in results) / len(results)) if results else 0
    if any_model:
        ok(f"Model returned in response — probe eliminated (~{avg_elapsed} ms avg)")
    else:
        warn("Model NOT returned — probe still needed for model discovery")


# ── resume_session sub-test ────────────────────────────────────────────────────


async def _test_resume_session(
    conn: Any,
    client: MinimalClient,
    session_id: str,
    cwd: str,
    mode_ids: list[str],
) -> None:
    section("4 · session/resume — MCP server re-passing after mode switch")

    print("""
  Context: Kiro drops MCP server connections when set_session_mode is called.
  session/resume re-passes mcp_servers to an existing session. Unlike
  session/load it should NOT replay conversation history.

  We measure:
    • Does the call succeed (no JSON-RPC error)?
    • How many session_update notifications fire?
        0 updates = no history replay  (ideal for resume)
        N updates = history replayed   (load_session behaviour)
""")

    # Switch to a non-default mode so there's a real mode change to recover from
    target_mode = mode_ids[1] if len(mode_ids) > 1 else mode_ids[0]
    fake_mcp = _fake_mcp_server()

    print(f"  Switching to mode '{target_mode}' via set_session_mode first...")
    try:
        t = Timer()
        await conn.set_session_mode(mode_id=target_mode, session_id=session_id)
        info(f"set_session_mode completed in {t.elapsed_ms()} ms")
    except Exception as exc:
        err(f"set_session_mode failed: {exc}")
        warn("Cannot proceed without a successful mode switch")
        return

    print()

    # ── Test session/resume ────────────────────────────────────────────────────
    print(f"  {CYAN}→ session/resume (mcp_servers=['{fake_mcp.name}']){RESET}")
    client.reset_counts()
    t = Timer()
    resume_ok = False
    resume_elapsed = 0
    resume_updates = 0
    resume_error = ""

    try:
        response = await conn.resume_session(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=[fake_mcp],
        )
        resume_elapsed = t.elapsed_ms()
        resume_updates = client.update_counts.get(session_id, 0)
        resume_ok = True

        kv("elapsed", f"{resume_elapsed} ms")
        kv("session_updates", resume_updates)
        kv("modes returned", bool(getattr(response, "modes", None)))
        kv("models returned", bool(getattr(response, "models", None)))
        kv("configOptions", bool(getattr(response, "config_options", None)))

        if resume_updates == 0:
            ok("No history replay — session/resume is safe to use after mode switch")
        else:
            warn(f"{resume_updates} session_update(s) fired — agent may be replaying history")

    except Exception as exc:
        resume_elapsed = t.elapsed_ms()
        resume_error = str(exc)
        err(f"session/resume failed after {resume_elapsed} ms")
        kv("error", resume_error[:120])

    print()

    # ── Test session/load for comparison ──────────────────────────────────────
    print(f"  {CYAN}→ session/load (mcp_servers=['{fake_mcp.name}']) — comparison{RESET}")
    client.reset_counts()
    t = Timer()
    load_ok = False
    load_elapsed = 0
    load_updates = 0
    load_error = ""

    try:
        await conn.load_session(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=[fake_mcp],
        )
        load_elapsed = t.elapsed_ms()
        load_updates = client.update_counts.get(session_id, 0)
        load_ok = True
        kv("elapsed", f"{load_elapsed} ms")
        kv("session_updates", load_updates)
        if load_updates > 0:
            info(f"{load_updates} update(s) — this is the history replay session/resume avoids")
        else:
            info("No updates fired (session has no conversation history yet)")

    except Exception as exc:
        load_elapsed = t.elapsed_ms()
        load_error = str(exc)
        err(f"session/load failed after {load_elapsed} ms")
        kv("error", load_error[:120])

    # ── Results table ──────────────────────────────────────────────────────────
    section("5 · Results")
    print()
    print(f"  {'Method':<20} {'Result':<12} {'ms':>6}  {'Updates':>8}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 6}  {'-' * 8}")

    def result_str(ok_flag: bool, error: str) -> str:
        return (GREEN + "ok" + RESET) if ok_flag else (RED + "error" + RESET)

    print(
        f"  {'session/resume':<20} {result_str(resume_ok, resume_error):<20} {resume_elapsed:>6}  {resume_updates:>8}"
    )
    print(
        f"  {'session/load':<20}  {result_str(load_ok, load_error):<20} {load_elapsed:>6}  {load_updates:>8}"
    )
    print()

    # ── Recommendation ─────────────────────────────────────────────────────────
    section("6 · Recommendation for synth-acp set_mode implementation")
    print()

    if resume_ok and resume_updates == 0:
        ok("session/resume works cleanly — use it to restore MCP servers after set_session_mode")
        ok("No history replay, no suppression flag needed")
        print("""
    Sequence:
      set_session_mode(new_mode)
      set_session_model(current_model)   # Option A: preserve model
      resume_session(session_id, cwd, mcp_servers)
""")
    elif resume_ok and resume_updates > 0:
        warn("session/resume works but triggers updates — behaves like session/load")
        warn("Use _suppress_history_replay flag in session_update for both methods")
        print("""
    Sequence:
      set_session_mode(new_mode)
      set_session_model(current_model)
      self._suppress_history_replay = True
      try:
          resume_session(session_id, cwd, mcp_servers)
      finally:
          self._suppress_history_replay = False
""")
    elif not resume_ok and load_ok:
        warn("session/resume not supported — fall back to session/load")
        warn("Use _suppress_history_replay flag to block the history replay in session_update")
        print(f"""
    Sequence:
      set_session_mode(new_mode)
      set_session_model(current_model)
      self._suppress_history_replay = True
      try:
          load_session(session_id, cwd, mcp_servers)
      finally:
          self._suppress_history_replay = False

    resume_error: {resume_error[:80]}
""")
    else:
        err("Neither method succeeded — MCP servers cannot be restored after mode switch")
        if resume_error:
            kv("resume error", resume_error[:100])
        if load_error:
            kv("load error", load_error[:100])


# ── Entry point ────────────────────────────────────────────────────────────────


async def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]
    cwd = os.getcwd()

    print(f"{BOLD}ACP capability probe{RESET}")
    print(f"Command: {command} {' '.join(args)}")
    print(f"CWD:     {cwd}")

    try:
        await probe(command, args, cwd)
    except FileNotFoundError:
        err(f"Command not found: '{command}'")
        err("Make sure the agent binary is installed and on your PATH")
        sys.exit(1)
    except Exception as exc:
        err(f"Unexpected error: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
