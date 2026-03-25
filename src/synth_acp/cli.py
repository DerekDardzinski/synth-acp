"""CLI entry point for SYNTH."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.commands import LaunchAgent, RespondPermission, SendPrompt
from synth_acp.models.config import load_config
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
        case PermissionRequested(agent_id=aid, title=title, kind=kind, options=opts):
            pending_permissions.append({"agent_id": aid, "options": opts})
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


async def _read_input() -> str:
    """Read a line from stdin via event loop reader.

    Uses add_reader instead of run_in_executor so no background thread
    blocks shutdown when Ctrl+C fires.
    """
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


async def _run(config_path: Path) -> None:
    """Load config, launch autostart agents, run interactive loop.

    Args:
        config_path: Path to .synth.json config file.
    """
    config = load_config(config_path)
    broker = ACPBroker(config)

    autostart = [a for a in config.agents if a.autostart]
    if not autostart:
        print(f"No autostart agents in {config_path}", file=sys.stderr)
        return

    # Launch all autostart agents
    for agent in autostart:
        print(f"[synth] Launching {agent.id}...")
        await broker.handle(LaunchAgent(agent_id=agent.id))

    # Wait for all autostart agents to reach IDLE
    idle_agents: set[str] = set()
    expected = {a.id for a in autostart}
    async for event in broker.events():
        _print_event(event, {})
        if isinstance(event, AgentStateChanged) and event.new_state == "idle":
            idle_agents.add(event.agent_id)
            if idle_agents >= expected:
                break
        if isinstance(event, BrokerError):
            await broker.shutdown()
            return

    # Auto-select default agent if only one
    default_agent: str | None = autostart[0].id if len(config.agents) == 1 else None
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


def _run_tui(config_path: Path) -> None:
    """Launch the Textual TUI.

    Args:
        config_path: Path to .synth.json config file.
    """
    from synth_acp.ui.app import SynthApp

    config = load_config(config_path)
    broker = ACPBroker(config)
    SynthApp(broker, config).run()


def main() -> None:
    """Entry point for `synth` CLI."""
    parser = argparse.ArgumentParser(description="SYNTH — multi-agent ACP orchestrator")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(".synth.json"),
        help="Path to .synth.json (default: .synth.json in CWD)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without TUI (stdin/stdout mode)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.headless:
        asyncio.run(_run(args.config))
    else:
        _run_tui(args.config)
