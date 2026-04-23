"""Tests for AgentConfig, AgentMode, AgentModel, and state machine transitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_acp.models.agent import (
    TRANSITIONS,
    AgentConfig,
    AgentState,
    css_id,
)


class TestAgentConfigValidation:
    def test_accepts_dots_in_agent_id(self):
        cfg = AgentConfig(agent_id="my.agent", harness="kiro")
        assert cfg.agent_id == "my.agent"

    def test_dots_sanitized_in_css_id(self):
        assert css_id("my.agent") == "my-agent"

    def test_rejects_empty_harness(self):
        with pytest.raises(ValidationError, match="harness must not be empty"):
            AgentConfig(agent_id="alice", harness="")


class TestAgentConfigLegacyCoercion:
    def test_id_coerced_to_agent_id(self):
        cfg = AgentConfig(id="alice", harness="kiro")
        assert cfg.agent_id == "alice"

    def test_profile_coerced_to_agent_mode(self):
        cfg = AgentConfig(agent_id="alice", harness="kiro", profile="planner")
        assert cfg.agent_mode == "planner"


class TestStateTransitions:
    def test_invalid_transition_not_in_map(self):
        assert AgentState.INITIALIZING not in TRANSITIONS[AgentState.BUSY]

    def test_terminated_is_terminal(self):
        assert TRANSITIONS[AgentState.TERMINATED] == {AgentState.INITIALIZING}

    def test_configuring_reachable_from_idle(self):
        """IDLE → CONFIGURING must be a valid transition — it's the entry point
        for mode switching."""
        assert AgentState.CONFIGURING in TRANSITIONS[AgentState.IDLE]

    def test_configuring_exits_to_idle(self):
        """CONFIGURING → IDLE must be valid — it's how mode switching completes."""
        assert AgentState.IDLE in TRANSITIONS[AgentState.CONFIGURING]

    def test_configuring_exits_to_terminated(self):
        """CONFIGURING → TERMINATED must be valid — agent may be killed mid-switch."""
        assert AgentState.TERMINATED in TRANSITIONS[AgentState.CONFIGURING]

    def test_configuring_cannot_go_to_busy(self):
        """CONFIGURING → BUSY must be invalid — a prompt cannot be accepted
        while the agent is mid-switch. This is the core of the race condition fix."""
        assert AgentState.BUSY not in TRANSITIONS[AgentState.CONFIGURING]

    def test_idle_cannot_go_to_configuring_from_busy(self):
        """BUSY → CONFIGURING must be invalid — a mode switch cannot start
        while a prompt is in flight."""
        assert AgentState.CONFIGURING not in TRANSITIONS[AgentState.BUSY]
