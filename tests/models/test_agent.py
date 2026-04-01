"""Tests for AgentConfig validation and state machine transitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_acp.models.agent import TRANSITIONS, AgentConfig, AgentState


class TestAgentConfigIdValidation:
    def test_rejects_dots_in_id(self):
        with pytest.raises(ValidationError, match="must start with alphanumeric"):
            AgentConfig(id="my.agent", cmd=["kiro-cli"])

    def test_rejects_leading_hyphen(self):
        with pytest.raises(ValidationError, match="must start with alphanumeric"):
            AgentConfig(id="-bad", cmd=["kiro-cli"])

    def test_accepts_valid_id(self):
        cfg = AgentConfig(id="kiro-auth_1", cmd=["kiro-cli"])
        assert cfg.id == "kiro-auth_1"


class TestAgentConfigLegacyCoercion:
    def test_agent_config_when_binary_args_provided_coerces_to_cmd(self):
        cfg = AgentConfig(id="a", binary="bin", args=["arg"])
        assert cfg.cmd == ["bin", "arg"]
        assert cfg.binary == "bin"
        assert cfg.args == ["arg"]

    def test_agent_config_when_cmd_empty_raises(self):
        with pytest.raises(ValidationError, match="cmd must not be empty"):
            AgentConfig(id="a", cmd=[])

    def test_agent_config_when_autostart_present_drops_it(self):
        cfg = AgentConfig(id="a", binary="bin", autostart=True)
        assert cfg.cmd == ["bin"]
        assert not hasattr(cfg, "autostart") or "autostart" not in cfg.model_fields

    def test_display_name_when_label_set_returns_label(self):
        cfg = AgentConfig(id="a", cmd=["bin"], label="My Agent")
        assert cfg.display_name == "My Agent"

    def test_display_name_when_no_label_returns_id(self):
        cfg = AgentConfig(id="a", cmd=["bin"])
        assert cfg.display_name == "a"


class TestStateTransitions:
    def test_invalid_transition_not_in_map(self):
        assert AgentState.INITIALIZING not in TRANSITIONS[AgentState.BUSY]

    def test_terminated_is_terminal(self):
        assert TRANSITIONS[AgentState.TERMINATED] == {AgentState.INITIALIZING}
