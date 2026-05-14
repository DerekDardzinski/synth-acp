"""Tests for CLI config resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_acp.cli import (
    _apply_global_settings,
    _build_transient_config,
    _detect_installed_harnesses,
    _load_config_with_agent,
    _resolve_config,
)


class TestBuildTransientConfig:
    """Tests for _build_transient_config."""

    def test_build_transient_config_returns_tuple_with_correct_agent(self) -> None:
        """Transient config must return (SessionConfig, AgentConfig) with correct harness and absolute cwd."""
        from synth_acp.models.agent import AgentConfig
        from synth_acp.models.config import GlobalConfig, SessionConfig

        config, agent = _build_transient_config("kiro", None, None, GlobalConfig())
        assert isinstance(config, SessionConfig)
        assert isinstance(agent, AgentConfig)
        assert agent.harness == "kiro"
        assert agent.cwd != "."
        assert Path(agent.cwd).is_absolute()


class TestLoadConfigWithAgent:
    """Tests for _load_config_with_agent."""

    def test_load_config_with_agent_extracts_first_agent(self, tmp_path: Path) -> None:
        """Legacy .synth.json with agents array must produce correct AgentConfig from first agent."""
        from synth_acp.models.agent import AgentConfig
        from synth_acp.models.config import GlobalConfig, SessionConfig

        config_file = tmp_path / ".synth.json"
        config_file.write_text(json.dumps({
            "project": "test",
            "agents": [
                {"agent_id": "lead", "harness": "kiro", "cwd": "./src"},
                {"agent_id": "worker", "harness": "claude"},
            ],
        }))

        config, agent = _load_config_with_agent(config_file, GlobalConfig())
        assert isinstance(config, SessionConfig)
        assert isinstance(agent, AgentConfig)
        assert agent.agent_id == "lead"
        assert agent.harness == "kiro"
        assert Path(agent.cwd).is_absolute()
        assert agent.cwd == str((tmp_path / "src").resolve())

    def test_load_config_with_agent_no_agents_exits(self, tmp_path: Path) -> None:
        """.synth.json without agents key returns None agent."""
        from synth_acp.models.config import GlobalConfig

        config_file = tmp_path / ".synth.json"
        config_file.write_text(json.dumps({"project": "test"}))

        session_config, agent = _load_config_with_agent(config_file, GlobalConfig())
        assert agent is None
        assert session_config.project == "test"


class TestConfigPath:
    """Tests for config path command."""

    def test_config_path_prints_global_config_path(self) -> None:
        from typer.testing import CliRunner

        from synth_acp.cli import app
        from synth_acp.models.config import GLOBAL_CONFIG_PATH

        runner = CliRunner()
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert str(GLOBAL_CONFIG_PATH) in result.output


class TestConfigList:
    """Tests for config list command."""

    def test_config_list_shows_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app

        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "config.json")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        assert "default_harness" in result.output
        assert "default_agent_id" in result.output
        assert "default_agent_mode" in result.output
        assert "communication_mode" in result.output
        assert "auto_approve_tools" in result.output
        assert "on_agent_startup" in result.output
        assert "on_agent_join" in result.output
        assert "on_agent_exit" in result.output


class TestConfigSet:
    """Tests for config set command."""

    def test_config_set_scalar_persists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app

        config_file = tmp_path / "config.json"
        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", config_file)
        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "default_harness", "kiro"])
        assert result.exit_code == 0
        cfg = config_mod.load_global_config()
        assert cfg.default_harness == "kiro"

    def test_config_set_clears_with_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app

        config_file = tmp_path / "config.json"
        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", config_file)
        runner = CliRunner()
        runner.invoke(app, ["config", "set", "default_harness", "kiro"])
        result = runner.invoke(app, ["config", "set", "default_harness", "none"])
        assert result.exit_code == 0
        cfg = config_mod.load_global_config()
        assert cfg.default_harness is None

    def test_config_set_communication_mode_validates(self) -> None:
        from typer.testing import CliRunner

        from synth_acp.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "communication_mode", "INVALID"])
        assert result.exit_code == 1
        assert "MESH" in result.output or "MESH" in (result.stderr or "")

    def test_config_set_auto_approve_tools_splits_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app

        config_file = tmp_path / "config.json"
        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", config_file)
        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "auto_approve_tools", "synth-mcp/send_message,synth-mcp/list_agents"])
        assert result.exit_code == 0
        cfg = config_mod.load_global_config()
        assert cfg.auto_approve_tools == ["synth-mcp/send_message", "synth-mcp/list_agents"]

    def test_config_set_unknown_key_exits(self) -> None:
        from typer.testing import CliRunner

        from synth_acp.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "unknown_key", "val"])
        assert result.exit_code == 1


class TestSynthNoSubcommand:
    """Tests that bare 'synth' still invokes main cli function."""

    def test_synth_no_subcommand_still_invokes_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import patch

        from typer.testing import CliRunner

        from synth_acp.cli import app

        runner = CliRunner()
        with patch("synth_acp.cli._resolve_config") as mock_resolve:
            mock_resolve.side_effect = SystemExit(1)
            runner.invoke(app, [])
        # The main cli() was invoked (it called _resolve_config)
        mock_resolve.assert_called_once()


class TestDetectInstalledHarnesses:
    """Tests for _detect_installed_harnesses."""

    def test_detect_installed_harnesses_finds_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from synth_acp.models.config import HarnessEntry

        entry = HarnessEntry(
            identity="test", name="Test", short_name="test",
            binary_names=["test-bin"], run_cmd="test-bin",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/test-bin" if b == "test-bin" else None)
        result = _detect_installed_harnesses()
        assert result == [(entry, "/usr/bin/test-bin")]

    def test_detect_installed_harnesses_empty_when_none_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from synth_acp.models.config import HarnessEntry

        entry = HarnessEntry(
            identity="test", name="Test", short_name="test",
            binary_names=["test-bin"], run_cmd="test-bin",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        monkeypatch.setattr("shutil.which", lambda _b: None)
        result = _detect_installed_harnesses()
        assert result == []


class TestApplyGlobalSettings:
    """Tests for _apply_global_settings."""

    def test_apply_global_settings_uses_raw_communication_mode(self) -> None:
        from synth_acp.models.config import (
            CommunicationMode,
            GlobalConfig,
            RawSessionConfig,
            RawSettingsConfig,
        )

        raw = RawSessionConfig(
            project="test",
            settings=RawSettingsConfig(communication_mode=CommunicationMode.MESH),
        )
        global_cfg = GlobalConfig(communication_mode=CommunicationMode.LOCAL)
        result = _apply_global_settings(raw, global_cfg)
        assert result.settings.communication_mode == CommunicationMode.MESH

    def test_apply_global_settings_falls_through_to_global(self) -> None:
        from synth_acp.models.config import (
            CommunicationMode,
            GlobalConfig,
            RawSessionConfig,
            RawSettingsConfig,
        )

        raw = RawSessionConfig(
            project="test",
            settings=RawSettingsConfig(),  # communication_mode=None
        )
        global_cfg = GlobalConfig(communication_mode=CommunicationMode.LOCAL)
        result = _apply_global_settings(raw, global_cfg)
        assert result.settings.communication_mode == CommunicationMode.LOCAL

    def test_apply_global_settings_hooks_merge_default_uses_global(self) -> None:
        from synth_acp.models.config import (
            GlobalConfig,
            GlobalHooksConfig,
            MessageHook,
            RawSessionConfig,
            RawSettingsConfig,
        )

        global_join = MessageHook(active=True, recipients="mesh", template="Hello {agent_id}")
        global_cfg = GlobalConfig(
            hooks=GlobalHooksConfig(on_agent_join=global_join),
        )
        raw = RawSessionConfig(
            project="test",
            settings=RawSettingsConfig(),  # default hooks
        )
        result = _apply_global_settings(raw, global_cfg)
        assert result.settings.hooks.on_agent_join == global_join


class TestResolveConfig:
    """Tests for _resolve_config."""

    def test_resolve_config_harness_flag_returns_tuple(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--harness flag returns (SessionConfig, AgentConfig) tuple with correct harness."""
        import synth_acp.models.config as config_mod
        from synth_acp.models.agent import AgentConfig
        from synth_acp.models.config import SessionConfig

        # Create a .synth.json that would be discovered
        config_file = tmp_path / ".synth.json"
        config_file.write_text(json.dumps({"project": "p", "agents": [{"agent_id": "from-file", "harness": "claude"}]}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "global.json")

        config, agent = _resolve_config("kiro", None, None, None)
        assert isinstance(config, SessionConfig)
        assert isinstance(agent, AgentConfig)
        assert agent.harness == "kiro"

    def test_resolve_config_global_default_harness(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Global config default_harness used when no .synth.json exists."""
        import synth_acp.models.config as config_mod
        from synth_acp.models.config import GlobalConfig

        monkeypatch.chdir(tmp_path)  # No .synth.json here
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "global.json")
        monkeypatch.setattr(
            "synth_acp.cli.load_global_config",
            lambda: GlobalConfig(default_harness="kiro"),
        )

        config, agent = _resolve_config(None, None, None, None)
        assert agent.harness == "kiro"

    def test_resolve_config_auto_detect_single(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Single harness in PATH → session launches with stderr hint."""
        import synth_acp.models.config as config_mod
        from synth_acp.models.config import GlobalConfig, HarnessEntry

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "global.json")
        monkeypatch.setattr("synth_acp.cli.load_global_config", GlobalConfig)

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr(
            "synth_acp.cli._detect_installed_harnesses",
            lambda: [(entry, "/usr/bin/kiro-cli")],
        )

        config, agent = _resolve_config(None, None, None, None)
        assert agent.harness == "kiro"
        captured = capsys.readouterr()
        assert "Using" in captured.err
        assert "synth config set default_harness" in captured.err

    def test_resolve_config_auto_detect_multiple_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Multiple harnesses → exit 1 with listing."""
        from click.exceptions import Exit as ClickExit

        import synth_acp.models.config as config_mod
        from synth_acp.models.config import GlobalConfig, HarnessEntry

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "global.json")
        monkeypatch.setattr("synth_acp.cli.load_global_config", GlobalConfig)

        entries = [
            (HarnessEntry(identity="kiro", name="Kiro", short_name="kiro", binary_names=["kiro-cli"], run_cmd="kiro-cli"), "/usr/bin/kiro-cli"),
            (HarnessEntry(identity="claude", name="Claude", short_name="claude", binary_names=["claude"], run_cmd="claude"), "/usr/bin/claude"),
        ]
        monkeypatch.setattr("synth_acp.cli._detect_installed_harnesses", lambda: entries)

        with pytest.raises(ClickExit) as exc_info:
            _resolve_config(None, None, None, None)
        assert exc_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "kiro" in captured.err
        assert "claude" in captured.err
        assert "synth config set default_harness" in captured.err

    def test_resolve_config_no_harnesses_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """No harnesses → exit 1 with install instructions."""
        from click.exceptions import Exit as ClickExit

        import synth_acp.models.config as config_mod
        from synth_acp.models.config import GlobalConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "global.json")
        monkeypatch.setattr("synth_acp.cli.load_global_config", GlobalConfig)
        monkeypatch.setattr("synth_acp.cli._detect_installed_harnesses", list)

        with pytest.raises(ClickExit) as exc_info:
            _resolve_config(None, None, None, None)
        assert exc_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "Install" in captured.err


class TestResolveHarnessForDiscovery:
    """Tests for _resolve_harness_for_discovery."""

    def test_explicit_harness_returns_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit --harness flag resolves to matching HarnessEntry."""
        from synth_acp.cli import _resolve_harness_for_discovery
        from synth_acp.models.config import HarnessEntry

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])

        result = _resolve_harness_for_discovery("kiro")
        assert result.short_name == "kiro"

    def test_auto_detect_single_returns_entry(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Single installed harness auto-detected when no --harness and no default."""
        import synth_acp.models.config as config_mod
        from synth_acp.cli import _resolve_harness_for_discovery
        from synth_acp.models.config import GlobalConfig, HarnessEntry

        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "config.json")

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        monkeypatch.setattr("synth_acp.cli.load_global_config", GlobalConfig)
        monkeypatch.setattr(
            "synth_acp.cli._detect_installed_harnesses",
            lambda: [(entry, "/usr/bin/kiro-cli")],
        )

        result = _resolve_harness_for_discovery(None)
        assert result.short_name == "kiro"


class TestListAgents:
    """Tests for --list-agents flag."""

    def test_list_agents_prints_table(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--list-agents prints agent names in output and exits 0."""
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app
        from synth_acp.discovery import DiscoveredAgent
        from synth_acp.models.config import HarnessEntry

        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "config.json")

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        from synth_acp.models.config import GlobalConfig

        monkeypatch.setattr("synth_acp.cli.load_global_config", lambda: GlobalConfig(default_harness="kiro"))

        agents = [
            DiscoveredAgent(qualified_name="planner", name="planner", description="Plans things", source="user"),
            DiscoveredAgent(qualified_name="coder", name="coder", description="Writes code", source="user"),
        ]
        monkeypatch.setattr("synth_acp.cli.discover_agents", lambda _h, _c: agents)

        runner = CliRunner()
        result = runner.invoke(app, ["--list-agents"])
        assert result.exit_code == 0
        assert "planner" in result.output
        assert "coder" in result.output
        assert "Plans things" in result.output

    def test_list_agents_no_agents(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--list-agents with no agents prints message and exits 0."""
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app
        from synth_acp.models.config import HarnessEntry

        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "config.json")

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        from synth_acp.models.config import GlobalConfig

        monkeypatch.setattr("synth_acp.cli.load_global_config", lambda: GlobalConfig(default_harness="kiro"))
        monkeypatch.setattr("synth_acp.cli.discover_agents", lambda _h, _c: [])

        runner = CliRunner()
        result = runner.invoke(app, ["--list-agents"])
        assert result.exit_code == 0
        assert "No agents found" in result.output


class TestSelectAgent:
    """Tests for --select-agent flag."""

    def test_select_agent_uses_selection(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--select-agent passes selected qualified_name as agent_mode to _resolve_config."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app
        from synth_acp.discovery import DiscoveredAgent
        from synth_acp.models.config import HarnessEntry

        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "config.json")

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        from synth_acp.models.config import GlobalConfig

        monkeypatch.setattr("synth_acp.cli.load_global_config", lambda: GlobalConfig(default_harness="kiro"))

        agents = [
            DiscoveredAgent(qualified_name="planner", name="planner", description="Plans", source="user"),
        ]
        monkeypatch.setattr("synth_acp.cli.discover_agents", lambda _h, _c: agents)

        # Mock InquirerPy to return "planner"
        mock_prompt = type("MockPrompt", (), {"execute": lambda _self: "planner"})()
        monkeypatch.setattr("InquirerPy.inquirer.fuzzy", lambda **_kw: mock_prompt)

        runner = CliRunner()
        with patch("synth_acp.cli._resolve_config") as mock_resolve, \
             patch("synth_acp.cli._run_tui"):
            mock_resolve.return_value = (None, None)
            runner.invoke(app, ["--select-agent"])

        # agent_mode arg should be "planner"
        assert mock_resolve.call_args[0][2] == "planner"

    def test_select_agent_keyboard_interrupt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--select-agent exits cleanly on KeyboardInterrupt."""
        from typer.testing import CliRunner

        import synth_acp.models.config as config_mod
        from synth_acp.cli import app
        from synth_acp.discovery import DiscoveredAgent
        from synth_acp.models.config import HarnessEntry

        monkeypatch.setattr(config_mod, "SYNTH_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "config.json")

        entry = HarnessEntry(
            identity="kiro", name="Kiro CLI", short_name="kiro",
            binary_names=["kiro-cli"], run_cmd="kiro-cli",
        )
        monkeypatch.setattr("synth_acp.cli.load_harness_registry", lambda: [entry])
        from synth_acp.models.config import GlobalConfig

        monkeypatch.setattr("synth_acp.cli.load_global_config", lambda: GlobalConfig(default_harness="kiro"))

        agents = [
            DiscoveredAgent(qualified_name="planner", name="planner", description="Plans", source="user"),
        ]
        monkeypatch.setattr("synth_acp.cli.discover_agents", lambda _h, _c: agents)

        def raise_interrupt(**_kw):
            return type("M", (), {"execute": lambda _self: (_ for _ in ()).throw(KeyboardInterrupt)})()

        monkeypatch.setattr("InquirerPy.inquirer.fuzzy", raise_interrupt)

        runner = CliRunner()
        result = runner.invoke(app, ["--select-agent"])
        assert result.exit_code == 0
