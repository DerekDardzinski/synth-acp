"""Filesystem-based agent discovery for ACP harnesses."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

from synth_acp.models.config import HarnessEntry

log = logging.getLogger(__name__)


class DiscoveredAgent(BaseModel, frozen=True):
    """A discovered agent configuration."""

    qualified_name: str  # The value to pass as agent_mode
    name: str  # Human-readable short name (from file's name field)
    description: str  # From file metadata, empty string if absent
    source: str  # e.g. "user", "project", "plugin:local-SHScienceAgentKit-all"


def discover_agents(harness: HarnessEntry, cwd: Path) -> list[DiscoveredAgent]:
    """Discover available agents for a harness via filesystem.

    This function does not raise. Malformed files, missing directories,
    and unknown harness identities result in an empty or partial list
    with warnings logged via the standard logging module.

    Args:
        harness: The harness entry to discover agents for.
        cwd: Working directory (for project-level agent discovery).

    Returns:
        List of discovered agents, sorted by source then name.
        Empty list if harness has no discoverable agents or if
        harness.identity is not recognized.
    """
    if harness.identity == "claude":
        agents = _discover_claude(cwd)
    elif harness.identity == "kiro":
        agents = _discover_kiro()
    else:
        return []
    return sorted(agents, key=lambda a: (a.source, a.name))


def _parse_frontmatter(path: Path) -> dict | None:
    """Parse YAML frontmatter from a markdown file.

    Returns the parsed dict or None if parsing fails.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        log.warning("Could not read agent file: %s", path)
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        log.warning("No YAML frontmatter found in: %s", path)
        return None

    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        log.warning("Malformed YAML frontmatter in: %s", path)
        return None

    if not isinstance(data, dict):
        log.warning("Frontmatter is not a mapping in: %s", path)
        return None
    return data


def _discover_claude(cwd: Path) -> list[DiscoveredAgent]:
    """Discover Claude Code agents from filesystem."""
    agents: list[DiscoveredAgent] = []
    home = Path.home()

    # User agents
    _collect_md_agents(home / ".claude" / "agents", "user", None, agents)

    # Project agents
    _collect_md_agents(cwd / ".claude" / "agents", "project", None, agents)

    # Managed agents
    managed_path = Path("/Library/Application Support/ClaudeCode/managed-settings/.claude/agents")
    _collect_md_agents(managed_path, "managed", None, agents)

    # Plugin agents (marketplace + extraKnownMarketplaces)
    settings = _load_claude_settings(home)
    enabled_plugins = settings.get("enabledPlugins") if settings else None

    # Official marketplace: <root>/<marketplace-name>/plugins/<plugin-dir>/agents/*.md
    official_root = home / ".claude" / "plugins" / "marketplaces"
    if official_root.is_dir():
        for marketplace_dir in official_root.iterdir():
            if not marketplace_dir.is_dir():
                continue
            plugins_dir = marketplace_dir / "plugins"
            if plugins_dir.is_dir():
                _collect_plugins_from_dir(
                    plugins_dir, marketplace_dir.name, enabled_plugins, agents,
                )

    # Extra marketplaces: <path>/<plugin-dir>/agents/*.md (flat structure)
    if settings:
        for marketplace_name, marketplace in settings.get("extraKnownMarketplaces", {}).items():
            source = marketplace.get("source", {})
            if path_str := source.get("path"):
                marketplace_path = Path(path_str)
                if marketplace_path.is_dir():
                    _collect_plugins_from_dir(
                        marketplace_path, marketplace_name, enabled_plugins, agents,
                    )

    return agents


def _collect_md_agents(
    agents_dir: Path,
    source: str,
    plugin_name: str | None,
    agents: list[DiscoveredAgent],
) -> None:
    """Collect agents from a directory of .md files with YAML frontmatter."""
    if not agents_dir.is_dir():
        return

    for md_file in agents_dir.glob("*.md"):
        data = _parse_frontmatter(md_file)
        if data is None:
            continue

        name = data.get("name")
        if not name:
            log.warning("Missing 'name' field in frontmatter: %s", md_file)
            continue

        description = data.get("description", "")
        if plugin_name:
            qualified_name = f"{plugin_name}:{name}"
            actual_source = f"plugin:{plugin_name}"
        else:
            qualified_name = name
            actual_source = source

        agents.append(
            DiscoveredAgent(
                qualified_name=qualified_name,
                name=name,
                description=description,
                source=actual_source,
            )
        )


def _collect_plugins_from_dir(
    plugins_dir: Path,
    marketplace_name: str,
    enabled_plugins: dict | None,
    agents: list[DiscoveredAgent],
) -> None:
    """Collect agents from plugin directories under a plugins dir.

    Structure: <plugins_dir>/<plugin-dir>/agents/*.md
    Each plugin-dir contains .claude-plugin/plugin.json with the plugin name.
    """
    if not plugins_dir.is_dir():
        return

    for plugin_dir in plugins_dir.iterdir():
        if not plugin_dir.is_dir():
            continue

        # Read plugin name from .claude-plugin/plugin.json
        plugin_json_path = plugin_dir / ".claude-plugin" / "plugin.json"
        plugin_name = _read_plugin_name(plugin_json_path)
        if plugin_name is None:
            continue

        # Filter by enabledPlugins if available
        if enabled_plugins is not None:
            plugin_key = f"{plugin_name}@{marketplace_name}"
            if not enabled_plugins.get(plugin_key, False):
                continue

        _collect_md_agents(plugin_dir / "agents", "plugin", plugin_name, agents)


def _read_plugin_name(plugin_json_path: Path) -> str | None:
    """Read the plugin name from .claude-plugin/plugin.json."""
    if not plugin_json_path.is_file():
        return None
    try:
        data = json.loads(plugin_json_path.read_text(encoding="utf-8"))
        name = data.get("name")
        if not name:
            log.warning("Missing 'name' in plugin.json: %s", plugin_json_path)
            return None
        return name
    except (OSError, json.JSONDecodeError):
        log.warning("Malformed plugin.json: %s", plugin_json_path)
        return None


def _load_claude_settings(home: Path) -> dict | None:
    """Load ~/.claude/settings.json, returning None on failure."""
    settings_path = home / ".claude" / "settings.json"
    if not settings_path.is_file():
        return None
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Could not read Claude settings: %s", settings_path)
        return None


def _discover_kiro() -> list[DiscoveredAgent]:
    """Discover Kiro agents from ~/.kiro/agents/*.json."""
    agents: list[DiscoveredAgent] = []
    agents_dir = Path.home() / ".kiro" / "agents"

    if not agents_dir.is_dir():
        return agents

    for json_file in agents_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Malformed agent JSON: %s", json_file)
            continue

        name = data.get("name")
        if not name:
            log.warning("Missing 'name' field in agent JSON: %s", json_file)
            continue

        agents.append(
            DiscoveredAgent(
                qualified_name=name,
                name=name,
                description=data.get("description", ""),
                source="user",
            )
        )

    return agents
