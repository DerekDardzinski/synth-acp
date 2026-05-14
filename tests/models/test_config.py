"""Tests for config models, load_config, find_config, hooks, and global config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_acp.models.config import (
    CommunicationMode,
    GlobalConfig,
    HooksConfig,
    MessageHook,
    RawSessionConfig,
    SessionConfig,
    find_config,
    load_config,
    render_template,
)


class TestSessionConfigValidation:
    def test_session_config_when_session_key_coerces_to_project(self):
        config = SessionConfig(
            session="test",
        )
        assert config.project == "test"


class TestRawSessionConfigDeprecation:
    def test_raw_session_config_strips_agents_key(self):
        """Old .synth.json with agents key must parse without error."""
        config = RawSessionConfig(
            project="test",
            agents=[{"agent_id": "a", "harness": "kiro"}],
        )
        assert config.project == "test"
        assert not hasattr(config, "agents")

    def test_raw_session_config_strips_ui_key(self):
        """Old .synth.json with ui key must parse without error."""
        config = RawSessionConfig(
            project="test",
            ui={"web_port": 9000, "theme": "light"},
        )
        assert config.project == "test"
        assert not hasattr(config, "ui")


class TestLoadConfig:
    def test_load_config_when_json_file_parses_correctly(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps(
                {"project": "myproject", "agents": [{"agent_id": "kiro", "harness": "kiro"}]}
            )
        )
        config = load_config(config_file)
        assert config.project == "myproject"

    def test_load_config_returns_raw_session_config(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps({"project": "p", "agents": [{"agent_id": "a", "harness": "kiro"}]})
        )
        result = load_config(config_file)
        assert isinstance(result, RawSessionConfig)


class TestFindConfig:
    def test_find_config_when_json_exists_returns_path(self, tmp_path: Path):
        (tmp_path / ".synth.json").write_text(
            '{"project":"t","agents":[{"agent_id":"a","harness":"kiro"}]}'
        )
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.json"

    def test_find_config_when_no_config_returns_none(self, tmp_path: Path):
        assert find_config(tmp_path) is None


class TestSettingsConfig:
    def test_load_config_when_settings_has_local_mode_parses_enum(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps(
                {
                    "project": "p",
                    "settings": {"communication_mode": "LOCAL"},
                    "agents": [{"agent_id": "a", "harness": "kiro"}],
                }
            )
        )
        config = load_config(config_file)
        assert config.settings.communication_mode == CommunicationMode.LOCAL




class TestHooksConfig:

    def test_hooks_from_json(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps(
                {
                    "project": "p",
                    "settings": {
                        "hooks": {
                            "on_agent_join": {
                                "recipients": "parent",
                                "template": "Agent {agent_id} joined.",
                            }
                        }
                    },
                    "agents": [{"agent_id": "a", "harness": "kiro"}],
                }
            )
        )
        config = load_config(config_file)
        assert config.settings.hooks.on_agent_join.recipients == "parent"
        assert config.settings.hooks.on_agent_join.template == "Agent {agent_id} joined."

    def test_env_override_join_recipients(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SYNTH_JOIN_RECIPIENTS", "family")
        config = RawSessionConfig(
            project="test",
        )
        assert config.settings.hooks.on_agent_join.recipients == "family"

    def test_message_hook_recipients_none_backward_compat(self):
        hook = MessageHook.model_validate({"recipients": "none", "template": "hi"})
        assert hook.active is False
        assert hook.recipients == "parent"  # default after removing 'none'

    def test_hooks_config_ignores_on_agent_prompt(self):
        hooks = HooksConfig.model_validate(
            {
                "on_agent_prompt": {"prepend": "some context"},
                "on_agent_join": {"recipients": "parent", "template": "hi"},
            }
        )
        assert not hasattr(hooks, "on_agent_prompt")
        assert hooks.on_agent_join.recipients == "parent"

    def test_hooks_config_ignores_startup_prepend(self):
        hooks = HooksConfig.model_validate(
            {
                "on_agent_startup": {"active": True, "prepend": "old context"},
            }
        )
        assert hooks.on_agent_startup.active is True


class TestRenderTemplate:
    def test_renders_known_slots(self):
        result = render_template(
            "Hello {agent_id}, parent is {parent_id}", {"agent_id": "a", "parent_id": "lead"}
        )
        assert result == "Hello a, parent is lead"

    def test_unknown_slots_become_empty(self):
        result = render_template("Hello {agent_id} {unknown}", {"agent_id": "a"})
        assert result == "Hello a "


class TestFileIO:
    def test_load_global_config_returns_defaults_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", tmp_path / "config.json")
        from synth_acp.models.config import load_global_config

        cfg = load_global_config()
        assert cfg.communication_mode == CommunicationMode.LOCAL
        assert cfg.auto_approve_tools == ["synth-mcp"]

    def test_save_and_load_global_config_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("synth_acp.models.config.SYNTH_DIR", tmp_path)
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", tmp_path / "config.json")
        from synth_acp.models.config import load_global_config, save_global_config

        cfg = GlobalConfig(
            default_harness="kiro",
            communication_mode=CommunicationMode.MESH,
            auto_approve_tools=["synth-mcp/send_message"],
        )
        save_global_config(cfg)
        loaded = load_global_config()
        assert loaded.default_harness == "kiro"
        assert loaded.communication_mode == CommunicationMode.MESH
        assert loaded.auto_approve_tools == ["synth-mcp/send_message"]

    def test_ensure_synth_dir_seeds_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        synth_dir = tmp_path / ".synth"
        monkeypatch.setattr("synth_acp.models.config.SYNTH_DIR", synth_dir)
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", synth_dir / "config.json")
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", synth_dir / "context.md")
        from synth_acp.models.config import ensure_synth_dir

        ensure_synth_dir()
        assert (synth_dir / "config.json").exists()
        assert (synth_dir / "context.md").exists()

    def test_ensure_synth_dir_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        synth_dir = tmp_path / ".synth"
        monkeypatch.setattr("synth_acp.models.config.SYNTH_DIR", synth_dir)
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", synth_dir / "config.json")
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", synth_dir / "context.md")
        from synth_acp.models.config import ensure_synth_dir

        ensure_synth_dir()
        # Modify files
        (synth_dir / "context.md").write_text("custom content")
        ensure_synth_dir()
        # User edits preserved
        assert (synth_dir / "context.md").read_text() == "custom content"

    def test_load_startup_context_reads_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        context_file = tmp_path / "context.md"
        context_file.write_text("my custom context")
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", context_file)
        from synth_acp.models.config import load_startup_context

        assert load_startup_context() == "my custom context"

    def test_load_startup_context_returns_default_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", tmp_path / "nonexistent.md")
        from synth_acp.models.config import DEFAULT_STARTUP_CONTEXT, load_startup_context

        assert load_startup_context() == DEFAULT_STARTUP_CONTEXT
