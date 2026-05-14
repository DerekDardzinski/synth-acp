"""Tests for synth_acp.discovery module."""

from __future__ import annotations

import json
from pathlib import Path

from synth_acp.discovery import DiscoveredAgent, discover_agents
from synth_acp.models.config import HarnessEntry


def _claude_harness() -> HarnessEntry:
    return HarnessEntry(
        identity="claude",
        name="Claude Code",
        short_name="claude",
        binary_names=["claude"],
        run_cmd="claude-agent-acp",
    )


def _kiro_harness() -> HarnessEntry:
    return HarnessEntry(
        identity="kiro",
        name="Kiro CLI",
        short_name="kiro",
        binary_names=["kiro-cli"],
        run_cmd="kiro-cli acp",
    )


def _write_md_agent(path: Path, name: str, description: str = "") -> None:
    """Write a minimal .md agent file with YAML frontmatter."""
    desc_line = f"\ndescription: {description}" if description else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}{desc_line}\n---\n# Agent\n")


class TestDiscoverClaudeUserAgents:
    def test_discovers_user_agents(self, tmp_path: Path, monkeypatch):
        """Guards correct parsing of user agents from ~/.claude/agents/*.md."""
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        _write_md_agent(fake_home / ".claude" / "agents" / "planner.md", "code-planner", "Plans code")

        result = discover_agents(_claude_harness(), tmp_path)

        assert len(result) == 1
        assert result[0] == DiscoveredAgent(
            qualified_name="code-planner",
            name="code-planner",
            description="Plans code",
            source="user",
        )

    def test_discovers_project_agents(self, tmp_path: Path, monkeypatch):
        """Guards project-level agent discovery from <cwd>/.claude/agents/*.md."""
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        cwd = tmp_path / "project"
        _write_md_agent(cwd / ".claude" / "agents" / "reviewer.md", "reviewer", "Reviews code")

        result = discover_agents(_claude_harness(), cwd)

        assert len(result) == 1
        assert result[0].qualified_name == "reviewer"
        assert result[0].source == "project"


class TestDiscoverClaudePluginAgents:
    def test_plugin_agents_filtered_by_enabled(self, tmp_path: Path, monkeypatch):
        """Guards plugin filtering by enabledPlugins — disabled plugins excluded."""
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Create settings with one enabled, one disabled plugin
        settings = {
            "enabledPlugins": {
                "my-plugin@official": True,
                "disabled-plugin@official": False,
            }
        }
        settings_path = fake_home / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings))

        # Create enabled plugin
        plugin_dir = fake_home / ".claude" / "plugins" / "marketplaces" / "official" / "plugins" / "my-plugin-dir"
        plugin_json = plugin_dir / ".claude-plugin" / "plugin.json"
        plugin_json.parent.mkdir(parents=True, exist_ok=True)
        plugin_json.write_text(json.dumps({"name": "my-plugin"}))
        _write_md_agent(plugin_dir / "agents" / "helper.md", "helper-agent", "Helps")

        # Create disabled plugin
        disabled_dir = fake_home / ".claude" / "plugins" / "marketplaces" / "official" / "plugins" / "disabled-dir"
        disabled_json = disabled_dir / ".claude-plugin" / "plugin.json"
        disabled_json.parent.mkdir(parents=True, exist_ok=True)
        disabled_json.write_text(json.dumps({"name": "disabled-plugin"}))
        _write_md_agent(disabled_dir / "agents" / "bad.md", "bad-agent")

        result = discover_agents(_claude_harness(), tmp_path)

        assert len(result) == 1
        assert result[0] == DiscoveredAgent(
            qualified_name="my-plugin:helper-agent",
            name="helper-agent",
            description="Helps",
            source="plugin:my-plugin",
        )


class TestDiscoverClaudeSkipsMalformed:
    def test_skips_malformed_frontmatter(self, tmp_path: Path, monkeypatch):
        """Guards graceful skip — malformed files don't crash, valid ones still returned."""
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        agents_dir = fake_home / ".claude" / "agents"
        agents_dir.mkdir(parents=True)

        # Valid agent
        _write_md_agent(agents_dir / "good.md", "good-agent")

        # Malformed: no frontmatter delimiters
        (agents_dir / "bad.md").write_text("no frontmatter here")

        # Malformed: missing name field
        (agents_dir / "noname.md").write_text("---\ndescription: no name\n---\n")

        result = discover_agents(_claude_harness(), tmp_path)

        assert len(result) == 1
        assert result[0].qualified_name == "good-agent"


class TestDiscoverKiroAgents:
    def test_discovers_kiro_agents(self, tmp_path: Path, monkeypatch):
        """Guards Kiro JSON parsing — correct field extraction."""
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        agents_dir = fake_home / ".kiro" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "planner.json").write_text(
            json.dumps({"name": "plan", "description": "Planning agent"})
        )

        result = discover_agents(_kiro_harness(), tmp_path)

        assert len(result) == 1
        assert result[0] == DiscoveredAgent(
            qualified_name="plan",
            name="plan",
            description="Planning agent",
            source="user",
        )


class TestDiscoverUnknownHarness:
    def test_returns_empty_for_unknown_identity(self, tmp_path: Path):
        """Guards empty return for unknown harness identity — no crash."""
        harness = HarnessEntry(
            identity="unknown",
            name="Unknown",
            short_name="unknown",
            binary_names=["unknown"],
            run_cmd="unknown",
        )
        result = discover_agents(harness, tmp_path)
        assert result == []
