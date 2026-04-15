"""CLI entry point for SYNTH."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

from synth_acp.broker.broker import ACPBroker
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.commands import LaunchAgent, RespondPermission, SendPrompt
from synth_acp.models.config import (
    HarnessEntry,
    SessionConfig,
    find_config,
    load_config,
    write_json_config,
)
from synth_acp.models.events import (
    AgentStateChanged,
    BrokerError,
    BrokerEvent,
    McpMessageDelivered,
    MessageChunkReceived,
    PermissionAutoResolved,
    PermissionRequested,
    ToolCallUpdated,
    TurnComplete,
)

log = logging.getLogger(__name__)

app = typer.Typer(invoke_without_command=True)


# ------------------------------------------------------------------
# Input parsing (preserved from original)
# ------------------------------------------------------------------


def parse_input(text: str, default_agent: str | None) -> tuple[str, str] | None:
    """Parse user input into (agent_id, message) or None for commands.

    Args:
        text: Raw user input.
        default_agent: Currently selected default agent, or None.

    Returns:
        Tuple of (agent_id, message) for prompt commands, None for /select.

    Raises:
        ValueError: If no agent can be determined for bare text.
    """
    text = text.strip()
    if text.startswith("@"):
        parts = text[1:].split(None, 1)
        agent_id = parts[0]
        message = parts[1] if len(parts) > 1 else ""
        return (agent_id, message)
    if text.startswith("/select "):
        return None
    if default_agent:
        return (default_agent, text)
    raise ValueError("No default agent set. Use @agent-id or /select agent-id")


def parse_permission_response(text: str, options: list) -> str | None:
    """Parse numeric input as permission option selection.

    Args:
        text: Raw user input.
        options: List of PermissionOption from the SDK.

    Returns:
        The option_id if text is a valid 1-based index, else None.
    """
    try:
        idx = int(text) - 1
        if 0 <= idx < len(options):
            return options[idx].option_id
    except ValueError:
        pass
    return None


# ------------------------------------------------------------------
# Event printing (headless mode)
# ------------------------------------------------------------------


def _print_event(
    event: BrokerEvent,
    pending_permissions: list[dict[str, Any]],
) -> None:
    """Print a broker event to stdout.

    Args:
        event: The broker event to print.
        pending_permissions: FIFO queue of pending permission requests.
    """
    match event:
        case AgentStateChanged(agent_id=aid, old_state=old, new_state=new):
            print(f"[state] {aid}: {old} → {new}")
        case MessageChunkReceived(chunk=chunk):
            print(chunk, end="", flush=True)
        case ToolCallUpdated(agent_id=aid, title=title, kind=kind, status=status):
            print(f"\n[tool] {aid}: {kind} — {title} [{status}]")
        case BrokerError(agent_id=aid, message=msg, severity=sev):
            print(f"[{sev}] {aid}: {msg}", file=sys.stderr)
        case PermissionRequested(agent_id=aid, request_id=rid, title=title, kind=kind, options=opts):
            pending_permissions.append({"agent_id": aid, "request_id": rid, "options": opts})
            print(f"\n[permission] {aid} requests permission:")
            print(f"  Title: {title}")
            print(f"  Kind: {kind}")
            print("  Options:")
            for i, opt in enumerate(opts, 1):
                print(f"    {i}) {opt.name}")
            if len(pending_permissions) == 1:
                print(f"Enter option number for {aid}: ", end="", flush=True)
            else:
                print(f"  (queued — respond to {pending_permissions[0]['agent_id']} first)")
        case PermissionAutoResolved(agent_id=aid, request_id=rid, decision=dec):
            print(f"[auto-resolved] {aid}: {rid} → {dec}")
        case TurnComplete(agent_id=aid, stop_reason=reason):
            print(f"\n[turn complete] {aid}: {reason}")
        case McpMessageDelivered(from_agent=src, to_agent=dst):
            print(f"[message] {src} → {dst}")


# ------------------------------------------------------------------
# Async stdin reader
# ------------------------------------------------------------------


async def _read_input() -> str:
    """Read a line from stdin via event loop reader."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def _on_stdin_ready() -> None:
        loop.remove_reader(sys.stdin)
        line = sys.stdin.readline()
        if not line:
            future.set_exception(EOFError())
        else:
            future.set_result(line.rstrip("\n"))

    loop.add_reader(sys.stdin, _on_stdin_ready)
    try:
        return await future
    except asyncio.CancelledError:
        loop.remove_reader(sys.stdin)
        raise


# ------------------------------------------------------------------
# Config resolution
# ------------------------------------------------------------------


def _build_transient_config(
    harness_name: str, agent_id: str | None, agent_mode: str | None,
) -> SessionConfig:
    """Build a transient SessionConfig from a harness name.

    Args:
        harness_name: The ``--harness`` value (short_name).
        agent_id: Optional ``--agent-id`` value.
        agent_mode: Optional ``--agent-mode`` value.

    Returns:
        A SessionConfig with a single agent.

    Raises:
        typer.Exit: If the harness is not found.
    """
    registry = load_harness_registry()
    harness = next((h for h in registry if h.short_name == harness_name), None)
    if not harness:
        known = ", ".join(sorted(h.short_name for h in registry))
        print(f"Unknown harness '{harness_name}'. Known: {known}", file=sys.stderr)
        raise typer.Exit(1)

    aid = agent_id or agent_mode or harness.short_name
    agent_dict: dict[str, Any] = {"agent_id": aid, "harness": harness_name}
    if agent_mode:
        agent_dict["agent_mode"] = agent_mode

    return SessionConfig(
        project=Path.cwd().name,
        agents=[agent_dict],
    )


def _resolve_config(
    harness: str | None,
    agent_id: str | None,
    agent_mode: str | None,
    config_path: Path | None,
    headless: bool,
) -> SessionConfig:
    """Resolve configuration using the priority order.

    Args:
        harness: ``--harness`` flag value.
        agent_id: ``--agent-id`` flag value.
        agent_mode: ``--agent-mode`` flag value.
        config_path: ``--config`` flag value.
        headless: Whether running in headless mode.

    Returns:
        Resolved SessionConfig.

    Raises:
        typer.Exit: If no config can be resolved.
    """
    # 1. --harness → transient config
    if harness:
        return _build_transient_config(harness, agent_id, agent_mode)

    # 2. --config → load file
    if config_path:
        if not config_path.exists():
            print(f"Config not found: {config_path}", file=sys.stderr)
            raise typer.Exit(1)
        return load_config(config_path)

    # 3. Auto-discover
    found = find_config(Path.cwd())
    if found:
        return load_config(found)

    # 4. First-run picker (TUI only)
    if headless:
        print("No config found. Run without --headless for interactive setup.", file=sys.stderr)
        raise typer.Exit(1)

    return _first_run_picker()


# ------------------------------------------------------------------
# First-run picker
# ------------------------------------------------------------------


def _first_run_picker() -> SessionConfig:
    """Interactive first-run setup that probes PATH for harnesses.

    Detects installed harnesses, shows a numbered list, prompts for
    agent name and project name, writes ``.synth.toml``, and returns
    the config.

    Returns:
        The newly created SessionConfig.

    Raises:
        typer.Exit: If no harnesses are found.
    """
    registry = load_harness_registry()

    # Probe PATH for installed harnesses
    installed: list[tuple[HarnessEntry, str]] = []
    for harness in registry:
        for binary in harness.binary_names:
            path = shutil.which(binary)
            if path:
                installed.append((harness, path))
                break

    if not installed:
        print("\nNo supported harnesses found in PATH.\n", file=sys.stderr)
        print("Install one of:", file=sys.stderr)
        for h in registry:
            print(f"  - {h.name} ({', '.join(h.binary_names)})", file=sys.stderr)
        raise typer.Exit(1)

    print("\nNo .synth.toml found. Let's set one up.\n")
    print("Which harness?")
    for i, (harness, path) in enumerate(installed, 1):
        print(f"  {i}) {harness.short_name}    ({path})")

    # Get harness selection
    while True:
        choice = input("\n  > ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(installed):
                selected_harness, _ = installed[idx]
                break
        except ValueError:
            pass
        print(f"  Enter a number 1-{len(installed)}")

    # Get agent name
    default_agent = selected_harness.short_name
    agent_input = input(f"\nAgent name [{default_agent}]: ").strip()
    agent_name = agent_input or default_agent

    # Get project name
    default_project = Path.cwd().name
    project_input = input(f"Project name [{default_project}]: ").strip()
    project_name = project_input or default_project

    # Build config
    config = SessionConfig(
        project=project_name,
        agents=[{"agent_id": agent_name, "harness": selected_harness.short_name}],
    )

    config_path = Path.cwd() / ".synth.json"
    write_json_config(config_path, config)
    print(f"\nWrote {config_path}\n")

    return load_config(config_path)


# ------------------------------------------------------------------
# Headless run
# ------------------------------------------------------------------


async def _run(config: SessionConfig) -> None:
    """Run in headless mode with the given config.

    Args:
        config: Resolved session configuration.
    """
    broker = ACPBroker(config)

    if not config.agents:
        print("No agents configured", file=sys.stderr)
        return

    # Launch all agents
    for agent in config.agents:
        print(f"[synth] Launching {agent.agent_id}...")
        await broker.handle(LaunchAgent(agent_id=agent.agent_id))

    # Wait for all agents to reach IDLE
    idle_agents: set[str] = set()
    expected = {a.agent_id for a in config.agents}
    async for event in broker.events():
        _print_event(event, [])
        if isinstance(event, AgentStateChanged) and event.new_state == "idle":
            idle_agents.add(event.agent_id)
            if idle_agents >= expected:
                break
        if isinstance(event, BrokerError):
            await broker.shutdown()
            return

    # Auto-select default agent if only one
    default_agent: str | None = config.agents[0].agent_id if len(config.agents) == 1 else None
    pending_permissions: list[dict[str, Any]] = []

    # Start event consumer task
    async def _consume_events() -> None:
        async for event in broker.events():
            _print_event(event, pending_permissions)

    event_task = asyncio.create_task(_consume_events())

    # Interactive stdin read loop
    try:
        while True:
            try:
                raw = await _read_input()
            except (EOFError, asyncio.CancelledError):
                break

            text = raw.strip()
            if not text:
                continue

            # Check for permission response
            if pending_permissions:
                option_id = parse_permission_response(text, pending_permissions[0]["options"])
                if option_id is not None:
                    perm = pending_permissions.pop(0)
                    await broker.handle(
                        RespondPermission(
                            agent_id=perm["agent_id"],
                            request_id=perm["request_id"],
                            option_id=option_id,
                        )
                    )
                    # Prompt for next queued permission if any
                    if pending_permissions:
                        nxt = pending_permissions[0]
                        print(
                            f"Enter option number for {nxt['agent_id']}: ",
                            end="",
                            flush=True,
                        )
                    continue

            # Handle /select command
            if text.startswith("/select "):
                default_agent = text.split(None, 1)[1]
                print(f"[synth] Default agent set to: {default_agent}")
                continue

            # Parse and route input
            try:
                result = parse_input(text, default_agent)
            except ValueError as e:
                print(f"[synth] {e}", file=sys.stderr)
                continue

            if result is not None:
                agent_id, message = result
                await broker.handle(SendPrompt(agent_id=agent_id, text=message))

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\n[synth] Shutting down...")
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
        try:
            await broker.shutdown()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass


# ------------------------------------------------------------------
# TUI run
# ------------------------------------------------------------------


_STYLE_CSS = {
    "default": "css/app.tcss",
}


def _run_tui(config: SessionConfig, style: str = "default", restore: bool = False) -> None:
    """Launch the Textual TUI.

    Args:
        config: Resolved session configuration.
        style: CSS style variant name.
        restore: Whether to show the session picker on startup.
    """
    from synth_acp.ui.app import SynthApp

    broker = ACPBroker(config)
    css_path = _STYLE_CSS.get(style, _STYLE_CSS["default"])
    SynthApp(broker, config, css_path=css_path, restore=restore).run()


# ------------------------------------------------------------------
# Typer command
# ------------------------------------------------------------------


@app.command()
def cli(
    harness: str | None = typer.Option(None, help="Harness to launch (e.g. kiro, claude)"),
    agent_id: str | None = typer.Option(None, "--agent-id", help="Agent identifier"),
    agent_mode: str | None = typer.Option(None, "--agent-mode", help="Agent mode (e.g. code, plan, chat)"),
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config file"),
    headless: bool = typer.Option(False, help="Run without TUI (stdin/stdout mode)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
    restore: bool = typer.Option(False, "-r", "--restore", help="Restore a previous session"),
    style: str = typer.Option(
        "default",
        "-s",
        "--style",
        help="TUI style variant",
    ),
) -> None:
    """SYNTH — multi-agent ACP orchestrator."""
    if verbose:
        log_file = Path.home() / ".synth" / "synth.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=str(log_file),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        logging.getLogger("aiosqlite").setLevel(logging.WARNING)
        logging.getLogger("markdown_it").setLevel(logging.WARNING)

    resolved = _resolve_config(harness, agent_id, agent_mode, config, headless)

    if headless:
        if restore:
            print("Session restore requires TUI mode.", file=sys.stderr)
            raise typer.Exit(1)
        asyncio.run(_run(resolved))
    else:
        _run_tui(resolved, style=style, restore=restore)


main = app
