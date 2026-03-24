"""Tests for AgentConfig validation and state machine transitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_acp.models.agent import TRANSITIONS, AgentConfig, AgentState


class TestAgentConfigIdValidation:
    def test_rejects_dots_in_id(self):
        with pytest.raises(ValidationError, match="must start with alphanumeric"):
            AgentConfig(id="my.agent", binary="kiro-cli")

    def test_rejects_leading_hyphen(self):
        with pytest.raises(ValidationError, match="must start with alphanumeric"):
            AgentConfig(id="-bad", binary="kiro-cli")

    def test_accepts_valid_id(self):
        cfg = AgentConfig(id="kiro-auth_1", binary="kiro-cli")
        assert cfg.id == "kiro-auth_1"


class TestStateTransitions:
    def test_invalid_transition_not_in_map(self):
        assert AgentState.INITIALIZING not in TRANSITIONS[AgentState.BUSY]

    def test_terminated_is_terminal(self):
        assert TRANSITIONS[AgentState.TERMINATED] == set()
