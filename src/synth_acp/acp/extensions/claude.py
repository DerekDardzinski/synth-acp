"""Claude Code (claude-agent-acp) extension.

Agent selection rides on ``_meta.claudeCode.options.extraArgs.agent`` on every session
op, which the adaptor passes to the Claude CLI as ``--agent <name>``; switching agents
forks the session.  ``options.agent`` is not used: claude-agent-acp 0.77.0 and later
delete it.  The adaptor accepts an unknown agent name silently, so ``check_profile``
validates the name against the discovered agents.  Tool calls made by a subagent
carry ``_meta.claudeCode.parentToolUseId``.  Profiles are discovered from the user,
project, managed and plugin agent directories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from synth_acp.acp.extensions.base import AcpExtension, SessionOp
from synth_acp.models.agent import DiscoveredAgent
from synth_acp.models.extensions import ExtensionCapabilities, ProfileUnavailable, UpdateHints

log = logging.getLogger(__name__)

class ClaudeExtension(AcpExtension):
    """claude-agent-acp: agent selection via _meta, agent switch by fork."""

    name = "claude"
    capabilities = ExtensionCapabilities(profile_switch="handoff")

    def session_meta(self, op: SessionOp, agent_mode: str | None) -> dict[str, Any]:
        """``{"claudeCode": {"options": {"extraArgs": {"agent": agent_mode}}}}`` for every op.

        Never also sends ``options.agent``: adaptors before 0.77.0 would then pass
        ``--agent`` twice.  Sent on resume and load too, where the Claude CLI ignores it
        (measured on 0.70.0 and 0.81.2: a session keeps the agent it was created with).
        """
        if agent_mode:
            return {"claudeCode": {"options": {"extraArgs": {"agent": agent_mode}}}}
        return {}

    def interpret_update(self, update: Any) -> UpdateHints:
        """Read ``field_meta.claudeCode.parentToolUseId`` (set on subagent tool calls)."""
        field_meta = getattr(update, "field_meta", None)
        if not isinstance(field_meta, dict):
            return UpdateHints()
        claude_meta = field_meta.get("claudeCode")
        if not isinstance(claude_meta, dict):
            return UpdateHints()
        parent = claude_meta.get("parentToolUseId")
        return UpdateHints(parent_tool_call_id=parent if isinstance(parent, str) else None)

    def discover_profiles(self, cwd: Path) -> list[DiscoveredAgent]:
        """Claude Code agents from the user, project, managed and plugin directories."""
        return _discover_claude(cwd)

    def check_profile(self, agent_mode: str, cwd: Path) -> ProfileUnavailable | None:
        """Reject an agent_mode that names no discovered agent; the adaptor would not warn."""
        if any(a.qualified_name == agent_mode for a in self.discover_profiles(cwd)):
            return None
        return ProfileUnavailable(
            requested=agent_mode,
            fallback="default",
            reason=f"no Claude agent named '{agent_mode}' found in user, project or plugin agents",
        )


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
