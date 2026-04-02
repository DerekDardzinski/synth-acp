"""Tests for SessionConfig validation, load_config, find_config, and TOML support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_acp.models.config import CommunicationMode, SessionConfig, find_config, load_config


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

    def test_missing_config_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_load_config_when_toml_file_parses_correctly(self, tmp_path: Path):
        config_file = tmp_path / ".synth.toml"
        config_file.write_text(
            'project = "myproject"\n\n[[agents]]\nagent_id = "kiro"\nharness = "kiro"\n'
        )
        config = load_config(config_file)
        assert config.project == "myproject"
        assert config.agents[0].agent_id == "kiro"
        assert config.agents[0].harness == "kiro"


class TestFindConfig:
    def test_find_config_when_both_files_exist_prefers_toml(self, tmp_path: Path):
        (tmp_path / ".synth.toml").write_text('project = "t"\n[[agents]]\nagent_id="a"\nharness="kiro"\n')
        (tmp_path / ".synth.json").write_text('{"project":"t","agents":[{"agent_id":"a","harness":"kiro"}]}')
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.toml"

    def test_find_config_when_only_json_returns_json(self, tmp_path: Path):
        (tmp_path / ".synth.json").write_text('{"project":"t","agents":[{"agent_id":"a","harness":"kiro"}]}')
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.json"

    def test_find_config_when_no_files_returns_none(self, tmp_path: Path):
        assert find_config(tmp_path) is None


class TestSettingsConfig:
    def test_load_config_when_settings_has_local_mode_parses_enum(self, tmp_path: Path):
        config_file = tmp_path / ".synth.toml"
        config_file.write_text(
            'project = "p"\n\n[settings]\ncommunication_mode = "LOCAL"\n\n'
            '[[agents]]\nagent_id = "a"\nharness = "kiro"\n'
        )
        config = load_config(config_file)
        assert config.settings.communication_mode == CommunicationMode.LOCAL

    def test_load_config_when_settings_absent_defaults_mesh(self, tmp_path: Path):
        config_file = tmp_path / ".synth.toml"
        config_file.write_text('project = "p"\n\n[[agents]]\nagent_id = "a"\nharness = "kiro"\n')
        config = load_config(config_file)
        assert config.settings.communication_mode == CommunicationMode.MESH
