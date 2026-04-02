"""Tests for AgentConfig, AgentMode, AgentModel, and state machine transitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_acp.models.agent import (
    TRANSITIONS,
    AgentConfig,
    AgentMode,
    AgentModel,
    AgentState,
)


class TestAgentConfigValidation:
    def test_rejects_dots_in_agent_id(self):
        with pytest.raises(ValidationError, match="must start with alphanumeric"):
            AgentConfig(agent_id="my.agent", harness="kiro")

    def test_rejects_leading_hyphen(self):
        with pytest.raises(ValidationError, match="must start with alphanumeric"):
            AgentConfig(agent_id="-bad", harness="kiro")

    def test_rejects_empty_harness(self):
        with pytest.raises(ValidationError, match="harness must not be empty"):
            AgentConfig(agent_id="alice", harness="")

    def test_accepts_valid_config(self):
        cfg = AgentConfig(agent_id="kiro-auth_1", harness="kiro")
        assert cfg.agent_id == "kiro-auth_1"
        assert cfg.harness == "kiro"
        assert cfg.agent_mode is None

    def test_accepts_agent_mode(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", agent_mode="kiro_planner")
        assert cfg.agent_mode == "kiro_planner"

    def test_display_name_returns_agent_id(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro")
        assert cfg.display_name == "alice"


class TestAgentConfigLegacyCoercion:
    def test_id_coerced_to_agent_id(self):
        cfg = AgentConfig(id="alice", harness="kiro")
        assert cfg.agent_id == "alice"

    def test_profile_coerced_to_agent_mode(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", profile="planner")
        assert cfg.agent_mode == "planner"

    def test_cmd_dropped_silently(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", cmd=["kiro-cli", "acp"])
        assert cfg.harness == "kiro"
        assert not hasattr(cfg, "cmd")

    def test_binary_args_dropped_silently(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", binary="kiro-cli", args=["acp"])
        assert not hasattr(cfg, "binary")

    def test_label_dropped_silently(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", label="Alice")
        assert not hasattr(cfg, "label")

    def test_autostart_dropped_silently(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", autostart=True)
        assert not hasattr(cfg, "autostart")


class TestAgentModeDataclass:
    def test_agent_mode_fields(self):
        mode = AgentMode(id="architect", name="Architect", description="Plan only")
        assert mode.id == "architect"
        assert mode.name == "Architect"
        assert mode.description == "Plan only"

    def test_agent_mode_description_optional(self):
        mode = AgentMode(id="code", name="Code")
        assert mode.description is None


class TestAgentModelDataclass:
    def test_agent_model_fields(self):
        model = AgentModel(id="claude-sonnet-4-5", name="Claude Sonnet 4.5")
        assert model.id == "claude-sonnet-4-5"
        assert model.name == "Claude Sonnet 4.5"
        assert model.description is None


class TestStateTransitions:
    def test_invalid_transition_not_in_map(self):
        assert AgentState.INITIALIZING not in TRANSITIONS[AgentState.BUSY]

    def test_terminated_is_terminal(self):
        assert TRANSITIONS[AgentState.TERMINATED] == {AgentState.INITIALIZING}
