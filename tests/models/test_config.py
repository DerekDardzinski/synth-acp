"""Tests for SessionConfig validation, load_config, find_config, and TOML support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_acp.models.config import SessionConfig, find_config, load_config


class TestSessionConfigValidation:
    def test_rejects_duplicate_agent_ids(self):
        with pytest.raises(ValidationError, match="Duplicate agent IDs"):
            SessionConfig(
                project="test",
                agents=[
                    {"id": "agent1", "cmd": ["kiro-cli"]},
                    {"id": "agent1", "cmd": ["claude"]},
                ],
            )

    def test_session_config_when_session_key_coerces_to_project(self):
        config = SessionConfig(
            session="test",
            agents=[{"id": "a1", "cmd": ["kiro-cli"]}],
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
                    "agents": [{"id": "a1", "binary": "kiro-cli", "cwd": "./src/auth"}],
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
            'project = "myproject"\n\n[[agents]]\nid = "kiro"\ncmd = ["kiro-cli", "acp"]\n'
        )
        config = load_config(config_file)
        assert config.project == "myproject"
        assert config.agents[0].id == "kiro"
        assert config.agents[0].cmd == ["kiro-cli", "acp"]


class TestFindConfig:
    def test_find_config_when_both_files_exist_prefers_toml(self, tmp_path: Path):
        (tmp_path / ".synth.toml").write_text('project = "t"\n[[agents]]\nid="a"\ncmd=["x"]\n')
        (tmp_path / ".synth.json").write_text('{"project":"t","agents":[{"id":"a","cmd":["x"]}]}')
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.toml"

    def test_find_config_when_only_json_returns_json(self, tmp_path: Path):
        (tmp_path / ".synth.json").write_text('{"project":"t","agents":[{"id":"a","cmd":["x"]}]}')
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.json"

    def test_find_config_when_no_files_returns_none(self, tmp_path: Path):
        assert find_config(tmp_path) is None
