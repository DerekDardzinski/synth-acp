"""Tests for SessionConfig validation and load_config CWD resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_acp.models.config import SessionConfig, load_config


class TestSessionConfigValidation:
    def test_rejects_duplicate_agent_ids(self):
        with pytest.raises(ValidationError, match="Duplicate agent IDs"):
            SessionConfig(
                session="test",
                agents=[
                    {"id": "agent1", "binary": "kiro-cli"},
                    {"id": "agent1", "binary": "claude"},
                ],
            )


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
