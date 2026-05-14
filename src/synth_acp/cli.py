"""CLI entry point for SYNTH."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import threading
from pathlib import Path

import typer

from synth_acp.discovery import discover_agents
from synth_acp.harnesses import load_harness_registry
from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import (
    GLOBAL_CONFIG_PATH,
    CommunicationMode,
    GlobalConfig,
    HarnessEntry,
    HooksConfig,
    MessageHook,
    RawSessionConfig,
    SessionConfig,
    SettingsConfig,
    StartupHookConfig,
    ensure_synth_dir,
    find_config,
    load_config,
    load_global_config,
    save_global_config,
)

log = logging.getLogger(__name__)


def _force_exit_if_threads_linger() -> None:
    """Last-resort exit if non-daemon threads keep the process alive."""
    non_daemon = [
        t for t in threading.enumerate()
        if t.is_alive() and not t.daemon and t is not threading.main_thread()
    ]
    if non_daemon:
        os._exit(0)


atexit.register(_force_exit_if_threads_linger)

app = typer.Typer(invoke_without_command=True)

SETTABLE_KEYS: dict[str, str] = {
    "default_harness": "Default harness for new sessions",
    "default_agent_id": "Default agent ID when launching with a harness",
    "default_agent_mode": "Default agent mode (e.g. code, plan, chat)",
    "communication_mode": "Agent visibility mode (MESH or LOCAL)",
    "auto_approve_tools": "Comma-separated tool patterns to auto-approve",
}

config_app = typer.Typer()
app.add_typer(config_app, name="config")


@config_app.command("path")
def config_path() -> None:
    """Print the path to the global config file."""
    print(GLOBAL_CONFIG_PATH)


@config_app.command("list")
def config_list() -> None:
    """Show all config keys with current values."""
    cfg = load_global_config()
    print("Global config:")
    for key, desc in SETTABLE_KEYS.items():
        val = getattr(cfg, key)
        if val is None:
            display = "(not set)"
        elif isinstance(val, list):
            display = ", ".join(val) if val else "(empty)"
        else:
            display = str(val)
        print(f"  {key}: {display}  — {desc}")
    print("\nHooks:")
    print(f"  on_agent_startup: active={cfg.hooks.on_agent_startup.active}")
    print(f"  on_agent_join: active={cfg.hooks.on_agent_join.active}")
    print(f"  on_agent_exit: active={cfg.hooks.on_agent_exit.active}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key to set"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a global config value."""
    if key not in SETTABLE_KEYS:
        valid = ", ".join(SETTABLE_KEYS)
        print(f"Unknown key '{key}'. Valid keys: {valid}", file=sys.stderr)
        raise typer.Exit(1)

    ensure_synth_dir()
    cfg = load_global_config()

    new_value: str | list[str] | CommunicationMode | None
    if key in ("default_harness", "default_agent_id", "default_agent_mode"):
        new_value = None if value in ("none", "null", "") else value
    elif key == "communication_mode":
        try:
            new_value = CommunicationMode(value)
        except ValueError:
            valid_modes = ", ".join(m.value for m in CommunicationMode)
            print(f"Invalid communication_mode '{value}'. Valid: {valid_modes}", file=sys.stderr)
            raise typer.Exit(1) from None
    else:  # auto_approve_tools
        new_value = [item.strip() for item in value.split(",")]

    updated = cfg.model_copy(update={key: new_value})
    save_global_config(updated)
    display = new_value if new_value is not None else "(cleared)"
    print(f"Set {key} = {display}")


# ------------------------------------------------------------------
# Config resolution
# ------------------------------------------------------------------


def _build_transient_config(
    harness_name: str, agent_id: str | None, agent_mode: str | None,
    global_cfg: GlobalConfig,
) -> tuple[SessionConfig, AgentConfig]:
    """Build a transient SessionConfig and AgentConfig from a harness name.

    Args:
        harness_name: The ``--harness`` value (short_name).
        agent_id: Optional ``--agent-id`` value.
        agent_mode: Optional ``--agent-mode`` value.
        global_cfg: Global config providing settings values.

    Returns:
        Tuple of (SessionConfig, AgentConfig).

    Raises:
        typer.Exit: If the harness is not found.
    """
    registry = load_harness_registry()
    harness = next((h for h in registry if h.short_name == harness_name), None)
    if not harness:
        known = ", ".join(sorted(h.short_name for h in registry))
        print(f"Unknown harness '{harness_name}'. Known: {known}", file=sys.stderr)
        raise typer.Exit(1)

    # agent_mode may contain colons (e.g. plugin:agent-name) which are invalid in agent_id
    if agent_id:
        aid = agent_id
    elif agent_mode:
        aid = agent_mode.split(":")[-1] if ":" in agent_mode else agent_mode
    else:
        aid = harness.short_name
    agent = AgentConfig(
        agent_id=aid,
        harness=harness_name,
        cwd=str(Path.cwd().resolve()),
        agent_mode=agent_mode,
    )

    settings = SettingsConfig(
        communication_mode=global_cfg.communication_mode,
        auto_approve_tools=global_cfg.auto_approve_tools,
        hooks=HooksConfig(
            on_agent_startup=global_cfg.hooks.on_agent_startup,
            on_agent_join=global_cfg.hooks.on_agent_join,
            on_agent_exit=global_cfg.hooks.on_agent_exit,
        ),
    )
    config = SessionConfig(
        project=Path.cwd().name,
        settings=settings.model_dump(),
    )
    return config, agent


def _apply_global_settings(raw: RawSessionConfig, global_cfg: GlobalConfig) -> SessionConfig:
    """Resolve RawSessionConfig into SessionConfig by filling None fields from global config.

    Args:
        raw: Parsed project config with potentially unset fields.
        global_cfg: Global config providing fallback values.

    Returns:
        Fully-resolved SessionConfig with no None values in settings.
    """
    comm_mode = (
        raw.settings.communication_mode
        if raw.settings.communication_mode is not None
        else global_cfg.communication_mode
    )
    auto_approve = (
        raw.settings.auto_approve_tools
        if raw.settings.auto_approve_tools is not None
        else global_cfg.auto_approve_tools
    )

    # Hooks merge: project default → use global, otherwise project wins
    on_startup = (
        global_cfg.hooks.on_agent_startup
        if raw.settings.hooks.on_agent_startup == StartupHookConfig()
        else raw.settings.hooks.on_agent_startup
    )
    on_join = (
        global_cfg.hooks.on_agent_join
        if raw.settings.hooks.on_agent_join == MessageHook()
        else raw.settings.hooks.on_agent_join
    )
    on_exit = (
        global_cfg.hooks.on_agent_exit
        if raw.settings.hooks.on_agent_exit == MessageHook()
        else raw.settings.hooks.on_agent_exit
    )

    settings = SettingsConfig(
        communication_mode=comm_mode,
        auto_approve_tools=auto_approve,
        hooks=HooksConfig(
            on_agent_startup=on_startup,
            on_agent_join=on_join,
            on_agent_exit=on_exit,
        ),
    )
    return SessionConfig(
        project=raw.project,
        settings=settings.model_dump(),
    )


def _load_config_with_agent(
    path: Path, global_cfg: GlobalConfig,
) -> tuple[SessionConfig, AgentConfig | None]:
    """Load .synth.json and extract the initial agent from the legacy agents array.

    Args:
        path: Path to .synth.json.
        global_cfg: Global config for settings resolution.

    Returns:
        Tuple of (SessionConfig, AgentConfig or None if no agents in config).
    """
    raw_data = json.loads(path.read_text())

    # Extract agent from raw JSON before model parsing strips it
    agents_raw = raw_data.get("agents")
    if not agents_raw or not isinstance(agents_raw, list) or len(agents_raw) == 0:
        # No agents in .synth.json — load settings only, resolve agent from global/auto-detect
        raw = load_config(path)
        session_config = _apply_global_settings(raw, global_cfg)
        return session_config, None

    if len(agents_raw) > 1:
        log.warning(
            "Multiple agents in .synth.json is deprecated. Using first agent '%s'.",
            agents_raw[0].get("agent_id", "unknown"),
        )

    first = agents_raw[0]
    config_dir = path.parent.resolve()
    cwd = str((config_dir / first.get("cwd", ".")).resolve())
    agent = AgentConfig(
        agent_id=first["agent_id"],
        harness=first["harness"],
        agent_mode=first.get("agent_mode"),
        cwd=cwd,
        env=first.get("env", {}),
    )

    raw = load_config(path)
    session_config = _apply_global_settings(raw, global_cfg)
    return session_config, agent


def _resolve_agent(
    agent_id: str | None,
    agent_mode: str | None,
    global_cfg: GlobalConfig,
) -> AgentConfig:
    """Resolve the initial agent from global config or auto-detect.

    Used when .synth.json provides settings but no agents.
    """
    # Try global config default_harness
    if global_cfg.default_harness:
        aid = agent_id or global_cfg.default_agent_id or agent_mode or global_cfg.default_harness
        return AgentConfig(
            agent_id=aid,
            harness=global_cfg.default_harness,
            agent_mode=agent_mode or global_cfg.default_agent_mode,
            cwd=str(Path.cwd().resolve()),
        )

    # Try auto-detect
    installed = _detect_installed_harnesses()
    if len(installed) == 1:
        entry, _ = installed[0]
        print(
            f"[synth] Using {entry.name} (only harness in PATH). "
            f"Make permanent: synth config set default_harness {entry.short_name}",
            file=sys.stderr,
        )
        aid = agent_id or agent_mode or entry.short_name
        return AgentConfig(
            agent_id=aid,
            harness=entry.short_name,
            agent_mode=agent_mode,
            cwd=str(Path.cwd().resolve()),
        )

    if len(installed) > 1:
        print("Multiple harnesses found in PATH:", file=sys.stderr)
        for entry, path in installed:
            print(f"  - {entry.short_name} ({path})", file=sys.stderr)
        print(
            "\nSet a default: synth config set default_harness <name>",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    registry = load_harness_registry()
    print("No ACP harnesses found in PATH.", file=sys.stderr)
    print("Install one of:", file=sys.stderr)
    for h in registry:
        print(f"  - {h.name} ({', '.join(h.binary_names)})", file=sys.stderr)
    raise typer.Exit(1)


def _resolve_config(
    harness: str | None,
    agent_id: str | None,
    agent_mode: str | None,
    config_path: Path | None,
) -> tuple[SessionConfig, AgentConfig]:
    """4-level resolution: CLI flags > .synth.json > global config > auto-detect.

    Always returns fully-resolved (SessionConfig, AgentConfig).

    Args:
        harness: ``--harness`` flag value.
        agent_id: ``--agent-id`` flag value.
        agent_mode: ``--agent-mode`` flag value.
        config_path: ``--config`` flag value.

    Returns:
        Tuple of (SessionConfig, AgentConfig).

    Raises:
        typer.Exit: If no config can be resolved.
    """
    global_cfg = load_global_config()

    # 1. --harness → transient config
    if harness:
        return _build_transient_config(harness, agent_id, agent_mode, global_cfg)

    # 2. --config → load file
    if config_path:
        if not config_path.exists():
            print(f"Config not found: {config_path}", file=sys.stderr)
            raise typer.Exit(1)
        session_config, agent = _load_config_with_agent(config_path, global_cfg)
        if agent:
            return session_config, agent
        # No agents in config — resolve agent from global/auto-detect below
        return session_config, _resolve_agent(agent_id, agent_mode, global_cfg)

    # 3. Auto-discover .synth.json
    found = find_config(Path.cwd())
    if found:
        session_config, agent = _load_config_with_agent(found, global_cfg)
        if agent:
            return session_config, agent
        # No agents in config — resolve agent from global/auto-detect below
        return session_config, _resolve_agent(agent_id, agent_mode, global_cfg)

    # 4. Global config default_harness
    if global_cfg.default_harness:
        return _build_transient_config(
            global_cfg.default_harness,
            agent_id or global_cfg.default_agent_id,
            agent_mode or global_cfg.default_agent_mode,
            global_cfg,
        )

    # 5. Auto-detect installed harnesses
    installed = _detect_installed_harnesses()
    if len(installed) == 1:
        entry, _ = installed[0]
        print(
            f"[synth] Using {entry.name} (only harness in PATH). "
            f"Make permanent: synth config set default_harness {entry.short_name}",
            file=sys.stderr,
        )
        return _build_transient_config(entry.short_name, agent_id, agent_mode, global_cfg)

    if len(installed) > 1:
        print("Multiple harnesses found in PATH:", file=sys.stderr)
        for entry, path in installed:
            print(f"  - {entry.short_name} ({path})", file=sys.stderr)
        print(
            "\nSet a default: synth config set default_harness <name>",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    # No harnesses found
    registry = load_harness_registry()
    print("No ACP harnesses found in PATH.", file=sys.stderr)
    print("Install one of:", file=sys.stderr)
    for h in registry:
        print(f"  - {h.name} ({', '.join(h.binary_names)})", file=sys.stderr)
    raise typer.Exit(1)


def _resolve_harness_for_discovery(harness: str | None) -> HarnessEntry:
    """Resolve a HarnessEntry for discovery flags (--list-agents, --select-agent)."""
    registry = load_harness_registry()

    if harness:
        entry = next((h for h in registry if h.short_name == harness), None)
        if not entry:
            known = ", ".join(sorted(h.short_name for h in registry))
            print(f"Unknown harness '{harness}'. Known: {known}", file=sys.stderr)
            raise typer.Exit(1)
        return entry

    global_cfg = load_global_config()
    if global_cfg.default_harness:
        entry = next((h for h in registry if h.short_name == global_cfg.default_harness), None)
        if entry:
            return entry

    installed = _detect_installed_harnesses()
    if len(installed) == 1:
        return installed[0][0]

    if len(installed) > 1:
        print("Multiple harnesses found. Use --harness to specify:", file=sys.stderr)
        for entry, path in installed:
            print(f"  - {entry.short_name} ({path})", file=sys.stderr)
    else:
        print("No ACP harnesses found in PATH.", file=sys.stderr)
        print("Install one or set: synth config set default_harness <name>", file=sys.stderr)
    raise typer.Exit(1)


def _detect_installed_harnesses() -> list[tuple[HarnessEntry, str]]:
    """Probe PATH for installed ACP harness binaries.

    Returns:
        List of (HarnessEntry, binary_path) pairs for each harness found.
    """
    registry = load_harness_registry()
    installed: list[tuple[HarnessEntry, str]] = []
    for harness in registry:
        for binary in harness.binary_names:
            path = shutil.which(binary)
            if path:
                installed.append((harness, path))
                break
    return installed


# ------------------------------------------------------------------
# TUI run
# ------------------------------------------------------------------


_STYLE_CSS = {
    "default": "css/app.tcss",
}


def _run_tui(
    config: SessionConfig, initial_agent: AgentConfig, style: str = "default", restore: bool = False,
) -> None:
    """Launch the Textual TUI.

    Args:
        config: Resolved session configuration.
        initial_agent: The agent to launch on startup.
        style: CSS style variant name.
        restore: Whether to show the session picker on startup.
    """
    from textual.geometry import Region

    from synth_acp.broker.broker import ACPBroker
    from synth_acp.ui.app import SynthApp
    if type(Region).__module__ != "builtins":
        logging.getLogger("synth_acp").warning(
            "textual-speedups not active — install textual-speedups for better performance"
        )

    broker = ACPBroker(config, initial_agent)
    css_path = _STYLE_CSS.get(style, _STYLE_CSS["default"])
    SynthApp(broker, config, initial_agent=initial_agent, css_path=css_path, restore=restore).run()


# ------------------------------------------------------------------
# Discovery flag handlers
# ------------------------------------------------------------------


def _handle_list_agents(harness: str | None) -> None:
    """Handle --list-agents: print table of discovered agents and exit."""
    from rich.console import Console
    from rich.table import Table

    entry = _resolve_harness_for_discovery(harness)
    agents = discover_agents(entry, Path.cwd())

    if not agents:
        print(f"No agents found for {entry.name}.")
        raise typer.Exit(0)

    table = Table(title=f"Agents for {entry.name}")
    table.add_column("Name")
    table.add_column("Qualified Name")
    table.add_column("Source")
    table.add_column("Description")

    for a in agents:
        table.add_row(a.name, a.qualified_name, a.source, a.description)

    Console().print(table)
    raise typer.Exit(0)


def _handle_select_agent(harness: str | None) -> str:
    """Handle --select-agent: present fuzzy picker and return selected qualified_name."""
    from InquirerPy import inquirer

    entry = _resolve_harness_for_discovery(harness)
    agents = discover_agents(entry, Path.cwd())

    if not agents:
        print(f"No agents found for {entry.name}.")
        raise typer.Exit(0)

    choices = [
        {"name": f"{a.qualified_name} — {a.description}", "value": a.qualified_name}
        for a in agents
    ]

    try:
        result = inquirer.fuzzy(message="Select agent:", choices=choices).execute()
    except (KeyboardInterrupt, EOFError):
        raise typer.Exit(0) from None

    if result is None:
        raise typer.Exit(0)

    return result


# ------------------------------------------------------------------
# Typer command
# ------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    harness: str | None = typer.Option(None, help="Harness to launch (e.g. kiro, claude)"),
    agent_id: str | None = typer.Option(None, "--agent-id", help="Agent identifier"),
    agent_mode: str | None = typer.Option(None, "--agent-mode", help="Agent mode (e.g. code, plan, chat)"),
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
    restore: bool = typer.Option(False, "-r", "--restore", help="Restore a previous session"),
    style: str = typer.Option(
        "default",
        "-s",
        "--style",
        help="TUI style variant",
    ),
    list_agents: bool = typer.Option(False, "--list-agents", help="List available agents and exit"),
    select_agent: bool = typer.Option(False, "--select-agent", help="Interactive agent picker"),
) -> None:
    """SYNTH — multi-agent ACP orchestrator."""
    if ctx.invoked_subcommand is not None:
        return

    ensure_synth_dir()

    if verbose:
        log_file = Path.home() / ".synth" / "synth.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=str(log_file),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        logging.getLogger("markdown_it").setLevel(logging.WARNING)

    if list_agents:
        _handle_list_agents(harness)
        return

    if select_agent:
        agent_mode = _handle_select_agent(harness)

    resolved_config, initial_agent = _resolve_config(harness, agent_id, agent_mode, config)
    _run_tui(resolved_config, initial_agent, style=style, restore=restore)


main = app
