"""Tests for SessionConfig validation, load_config, find_config, and hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_acp.models.config import (
    CommunicationMode,
    SessionConfig,
    find_config,
    load_config,
    render_template,
)


class TestSessionConfigValidation:
    def test_rejects_duplicate_agent_ids(self):
        with pytest.raises(ValidationError, match="Duplicate agent IDs"):
            SessionConfig(
                project="test",
                agents=[
                    {"agent_id": "agent1", "harness": "kiro"},
                    {"agent_id": "agent1", "harness": "claude"},
                ],
            )

    def test_session_config_when_session_key_coerces_to_project(self):
        config = SessionConfig(
            session="test",
            agents=[{"agent_id": "a1", "harness": "kiro"}],
        )
        assert config.project == "test"


class TestLoadConfig:
    def test_resolves_cwd_relative_to_config_parent(self, tmp_path: Path):
        config_dir = tmp_path / "project"
        config_dir.mkdir()
        config_file = config_dir / ".synth.json"
        config_file.write_text(
            json.dumps(
                {
                    "session": "test",
                    "agents": [{"agent_id": "a1", "harness": "kiro", "cwd": "./src/auth"}],
                }
            )
        )

        config = load_config(config_file)
        assert config.agents[0].cwd == str(config_dir / "src" / "auth")

    def test_load_config_when_json_file_parses_correctly(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps({"project": "myproject", "agents": [{"agent_id": "kiro", "harness": "kiro"}]})
        )
        config = load_config(config_file)
        assert config.project == "myproject"
        assert config.agents[0].agent_id == "kiro"
        assert config.agents[0].harness == "kiro"


class TestFindConfig:
    def test_find_config_when_json_exists_returns_path(self, tmp_path: Path):
        (tmp_path / ".synth.json").write_text('{"project":"t","agents":[{"agent_id":"a","harness":"kiro"}]}')
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.json"

    def test_find_config_when_no_config_returns_none(self, tmp_path: Path):
        assert find_config(tmp_path) is None


class TestSettingsConfig:
    def test_load_config_when_settings_has_local_mode_parses_enum(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps({
                "project": "p",
                "settings": {"communication_mode": "LOCAL"},
                "agents": [{"agent_id": "a", "harness": "kiro"}],
            })
        )
        config = load_config(config_file)
        assert config.settings.communication_mode == CommunicationMode.LOCAL


class TestHooksConfig:
    def test_default_hooks_have_none_recipients(self):
        config = SessionConfig(
            project="test",
            agents=[{"agent_id": "a", "harness": "kiro"}],
        )
        assert config.settings.hooks.on_agent_join.recipients == "none"
        assert config.settings.hooks.on_agent_exit.recipients == "none"

    def test_hooks_from_json(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps({
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
            })
        )
        config = load_config(config_file)
        assert config.settings.hooks.on_agent_join.recipients == "parent"
        assert config.settings.hooks.on_agent_join.template == "Agent {agent_id} joined."

    def test_env_override_join_recipients(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SYNTH_JOIN_RECIPIENTS", "family")
        config = SessionConfig(
            project="test",
            agents=[{"agent_id": "a", "harness": "kiro"}],
        )
        assert config.settings.hooks.on_agent_join.recipients == "family"


class TestRenderTemplate:
    def test_renders_known_slots(self):
        result = render_template("Hello {agent_id}, parent is {parent_id}", {"agent_id": "a", "parent_id": "lead"})
        assert result == "Hello a, parent is lead"

    def test_unknown_slots_become_empty(self):
        result = render_template("Hello {agent_id} {unknown}", {"agent_id": "a"})
        assert result == "Hello a "
