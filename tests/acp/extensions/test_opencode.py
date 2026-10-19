"""Tests for OpenCodeExtension: offline discovery of OpenCode's selectable agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_acp.acp.extensions.opencode import OpenCodeExtension


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated global config dir and a project tree; no OpenCode env leaks in."""
    for var in (
        "OPENCODE_CONFIG",
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        "OPENCODE_CONFIG_PROJECT_DISABLE",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "global"))
    (tmp_path / "global").mkdir()
    (tmp_path / "repo" / "pkg").mkdir(parents=True)
    return tmp_path


def _ids(cwd: Path) -> list[tuple[str, str, str]]:
    return [
        (a.qualified_name, a.description, a.source)
        for a in OpenCodeExtension().discover_profiles(cwd)
    ]


class TestDiscoverProfiles:
    def test_no_config_lists_build_then_plan(self, tree: Path) -> None:
        """Silent failure: the picker offers hidden or subagent built-ins (general,
        compaction), which the live mode option never accepts."""
        assert _ids(tree / "repo" / "pkg") == [
            ("build", "", "builtin"),
            ("plan", "", "builtin"),
        ]

    def test_markdown_and_jsonc_agents_follow_opencode_rules(self, tree: Path) -> None:
        """Nested ids, missing mode = primary, subagent/hidden dropped, V1 ``disable``
        removes a built-in, JSONC comments and trailing commas, default_agent first,
        unquoted-colon values retried as OpenCode does, empty files skipped."""
        g = tree / "global"
        _write(g / "agent" / "team" / "review.md", "---\ndescription: Reviews code\n---\nBody\n")
        _write(g / "agent" / "helper.md", "---\nmode: subagent\n---\n")
        _write(g / "agents" / "secret.md", "---\nhidden: true\n---\n")
        _write(g / "mode" / "writer.md", "---\nmode: subagent\nprompt: forced primary\n---\n")
        _write(g / "agent" / "broken.md", "---\nmode: [unclosed\n---\n")
        _write(g / "agent" / "colon.md", "---\ndescription: Use when: reviewing\n---\n")
        _write(g / "agent" / "empty.md", "")
        _write(
            tree / "repo" / ".opencode" / "opencode.jsonc",
            """{
              // project overrides
              "agent": { "plan": { "disable": true, }, },
              /* block */ "default_agent": "team/review",
            }""",
        )
        assert _ids(tree / "repo" / "pkg") == [
            ("team/review", "Reviews code", "user"),
            ("build", "", "builtin"),
            ("colon", "Use when: reviewing", "user"),
            ("writer", "", "user"),
        ]

    def test_nearest_project_dir_wins_and_dot_opencode_beats_direct_files(self, tree: Path) -> None:
        """Silent failure: precedence inverted, so a repo's override of a global agent
        is ignored or a removed agent is still offered."""
        _write(tree / "global" / "agent" / "x.md", "---\nmode: subagent\n---\n")
        _write(tree / "repo" / ".opencode" / "agent" / "x.md", "---\nmode: subagent\n---\n")
        _write(tree / "repo" / "pkg" / ".opencode" / "agent" / "x.md", "---\nmode: all\n---\n")
        # A direct file in cwd applies before every .opencode dir, so it cannot remove x.
        _write(tree / "repo" / "pkg" / "opencode.json", '{"agent": {"x": {"disable": true}}}')
        assert [i for i, _, _ in _ids(tree / "repo" / "pkg")] == ["build", "plan", "x"]

    def test_project_config_disabled_by_env(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OPENCODE_DISABLE_PROJECT_CONFIG makes OpenCode skip the walk; offering a
        project agent then would launch into a mode the engine does not have."""
        _write(tree / "repo" / ".opencode" / "agent" / "local.md", "---\n---\n")
        monkeypatch.setenv("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
        assert [i for i, _, _ in _ids(tree / "repo" / "pkg")] == ["build", "plan"]
