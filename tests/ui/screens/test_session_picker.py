"""Tests for SessionPickerScreen search logic."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from synth_acp.ui.screens.session_picker import SessionPickerScreen


def _make_session(
    session_id: str = "sess-1",
    agents: list[str] | None = None,
    last_active: int | None = None,
    cwd: str = "/home/user/project",
    tasks: list[str] | None = None,
    first_messages: list[str] | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "agents": agents or ["agent-1"],
        "last_active": last_active or int(time.time() * 1000),
        "agent_count": len(agents) if agents else 1,
        "cwd": cwd,
        "tasks": tasks or ["fix the bug"],
        "first_messages": first_messages or ["hello"],
    }


class TestSubstringFilter:
    def test_case_insensitive_match(self) -> None:
        sessions = [_make_session(agents=["foo-agent"])]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        result = picker._substring_filter("FOO", sessions)
        assert len(result) == 1
        assert result[0]["session_id"] == "sess-1"

    def test_no_match_returns_empty(self) -> None:
        sessions = [_make_session(agents=["bar-agent"])]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        result = picker._substring_filter("zzz", sessions)
        assert result == []

    def test_matches_against_tasks(self) -> None:
        sessions = [_make_session(tasks=["implement semantic search"])]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        result = picker._substring_filter("semantic", sessions)
        assert len(result) == 1

    def test_matches_against_first_messages(self) -> None:
        sessions = [_make_session(first_messages=["fix the auth bug"])]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        result = picker._substring_filter("auth", sessions)
        assert len(result) == 1


class TestDoSearch:
    def test_empty_query_returns_all_sorted_by_recency(self) -> None:
        now = int(time.time() * 1000)
        sessions = [
            _make_session(session_id="old", last_active=now - 100000),
            _make_session(session_id="new", last_active=now),
            _make_session(session_id="mid", last_active=now - 50000),
        ]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        table = MagicMock()
        picker.query_one = MagicMock(return_value=table)  # type: ignore[method-assign]
        picker._do_search("")
        assert picker._row_keys == ["new", "mid", "old"]

    def test_substring_search_filters(self) -> None:
        sessions = [
            _make_session(session_id="s1", agents=["planner"]),
            _make_session(session_id="s2", agents=["builder"]),
        ]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        table = MagicMock()
        picker.query_one = MagicMock(return_value=table)  # type: ignore[method-assign]
        picker._do_search("builder")
        assert picker._row_keys == ["s2"]
