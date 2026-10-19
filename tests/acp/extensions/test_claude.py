"""Tests for synth_acp.acp.extensions.claude."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_acp.acp.extensions.claude import ClaudeExtension
from synth_acp.models.extensions import ProfileUnavailable


class TestSessionMeta:
    def test_session_meta_by_op(self) -> None:
        """Every op carries the agent as extraArgs, never options.agent: claude-agent-acp
        0.77.0+ deletes options.agent, and a restore dropping the agent runs the default."""
        ext = ClaudeExtension()
        expected = {"claudeCode": {"options": {"extraArgs": {"agent": "probe-parrot"}}}}
        for op in ("new", "fork", "resume", "load"):
            assert ext.session_meta(op, "probe-parrot") == expected
        assert ext.session_meta("new", None) == {}


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
