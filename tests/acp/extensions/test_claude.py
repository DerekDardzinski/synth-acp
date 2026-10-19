"""Tests for synth_acp.acp.extensions.claude."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from synth_acp.acp.extensions.claude import ClaudeExtension
from synth_acp.models.extensions import ProfileUnavailable, SteerOutcome


class TestSessionMeta:
    def test_session_meta_by_op(self) -> None:
        """Every op carries the agent as extraArgs, never options.agent: claude-agent-acp
        0.77.0+ deletes options.agent, and a restore dropping the agent runs the default."""
        ext = ClaudeExtension()
        expected = {"claudeCode": {"options": {"extraArgs": {"agent": "probe-parrot"}}}}
        for op in ("new", "fork", "resume", "load"):
            assert ext.session_meta(op, "probe-parrot") == expected
        assert ext.session_meta("new", None) == {}


class TestSteer:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ({"outcome": "injected"}, SteerOutcome(status="delivered")),
            ({"outcome": "startedNewTurn"}, SteerOutcome(status="started_turn")),
            (
                {"outcome": "promptRequired", "reason": "noRunningTurn"},
                SteerOutcome(status="dropped"),
            ),
            ({"outcome": "unknown"}, SteerOutcome(status="dropped")),
        ],
    )
    async def test_steer_uses_host_owned_idle_fallback_and_maps_outcome(
        self, answer: dict[str, str], expected: SteerOutcome
    ) -> None:
        """The message must never become an untracked detached turn when the active
        turn ends before the request lands; promptRequired leaves it owned by synth."""
        conn = AsyncMock()
        conn.ext_method.return_value = answer

        outcome = await ClaudeExtension().steer(conn, "session-1", "change course")

        assert outcome == expected
        conn.ext_method.assert_awaited_once_with(
            "session/steering",
            {
                "sessionId": "session-1",
                "prompt": [{"type": "text", "text": "change course"}],
                "_meta": {"steering": {"idleBehavior": "promptRequired"}},
            },
        )

    def test_capabilities_enable_steering_and_keep_handoff_profile_switch(self) -> None:
        caps = ClaudeExtension.capabilities
        assert caps.steer is True
        assert caps.profile_switch == "handoff"


class TestCheckProfile:
    def test_check_profile_matches_discovered_qualified_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adaptor runs an unknown agent name as the default agent without error, so
        this check is synth's only warning; a present name must not warn."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        cwd = tmp_path / "ws"
        agents_dir = cwd / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "probe-parrot.md").write_text(
            "---\nname: probe-parrot\ndescription: test\n---\nSay PARROT.\n", encoding="utf-8"
        )
        ext = ClaudeExtension()

        assert ext.check_profile("probe-parrot", cwd) is None
        assert ext.check_profile("probe-parot", cwd) == ProfileUnavailable(
            requested="probe-parot",
            fallback="default",
            reason="no Claude agent named 'probe-parot' found in user, project or plugin agents",
        )
