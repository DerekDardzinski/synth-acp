"""CLI entry point for SYNTH."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from synth_acp.broker.broker import ACPBroker
from synth_acp.models.config import load_config
from synth_acp.models.events import (
    AgentStateChanged,
    BrokerError,
    MessageChunkReceived,
    ToolCallUpdated,
)


async def _run(config_path: Path) -> None:
    """Load config, launch first autostart agent, send a test prompt, print events."""
    config = load_config(config_path)
    broker = ACPBroker(config)

    autostart = [a for a in config.agents if a.autostart]
    if not autostart:
        print(f"No autostart agents in {config_path}", file=sys.stderr)
        return

    agent_id = autostart[0].id
    print(f"[synth] Launching {agent_id}...")
    await broker.launch(agent_id)

    # Wait for agent to become IDLE before prompting
    idle = False
    async for event in broker.events():
        _print_event(event)
        if isinstance(event, AgentStateChanged) and event.new_state == "idle":
            idle = True
            break
        if isinstance(event, BrokerError):
            await broker.shutdown()
            return

    if not idle:
        print("[synth] Agent never reached IDLE", file=sys.stderr)
        await broker.shutdown()
        return

    print("[synth] Sending test prompt...")
    prompt_task = asyncio.create_task(broker.prompt(agent_id, "Hello! What can you do?"))

    async for event in broker.events():
        _print_event(event)
        if isinstance(event, AgentStateChanged) and event.new_state == "idle":
            break
        if isinstance(event, AgentStateChanged) and event.new_state == "terminated":
            break

    await prompt_task
    print("\n[synth] Turn complete. Shutting down...")
    await broker.shutdown()


def _print_event(event: object) -> None:
    """Print a broker event to stdout."""
    match event:
        case AgentStateChanged(agent_id=aid, old_state=old, new_state=new):
            print(f"[state] {aid}: {old} → {new}")
        case MessageChunkReceived(agent_id=aid, chunk=chunk):
            print(chunk, end="", flush=True)
        case ToolCallUpdated(agent_id=aid, title=title, kind=kind, status=status):
            print(f"\n[tool] {aid}: {kind} — {title} [{status}]")
        case BrokerError(agent_id=aid, message=msg, severity=sev):
            print(f"[{sev}] {aid}: {msg}", file=sys.stderr)


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
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(args.config))
