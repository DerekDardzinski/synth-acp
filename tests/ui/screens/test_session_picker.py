"""Tests for SessionPickerScreen search logic."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from synth_acp.ui.screens.session_picker import (
    SessionPickerScreen,
    _build_bm25_text,
    _rrf_fuse,
    _tokenize_for_bm25,
)


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


class TestTokenizeForBm25:
    def test_splits_on_dashes(self) -> None:
        assert _tokenize_for_bm25("researcher-shap") == ["researcher", "shap"]

    def test_splits_on_underscores(self) -> None:
        assert _tokenize_for_bm25("builder_auth") == ["builder", "auth"]

    def test_splits_camel_case(self) -> None:
        assert _tokenize_for_bm25("SHScienceEngram") == ["sh", "science", "engram"]

    def test_splits_mixed_separators_and_camel(self) -> None:
        assert _tokenize_for_bm25("builder-SHScience.v2") == ["builder", "sh", "science", "v2"]

    def test_empty_string(self) -> None:
        assert _tokenize_for_bm25("") == []

    def test_single_word(self) -> None:
        assert _tokenize_for_bm25("planner") == ["planner"]


class TestBuildBm25Text:
    def test_includes_tokenized_agents(self) -> None:
        session = _make_session(agents=["researcher-shap", "code-planner"])
        text = _build_bm25_text(session)
        assert "researcher shap" in text
        assert "code planner" in text

    def test_includes_cwd_basename(self) -> None:
        session = _make_session(cwd="/workspace/SHScienceEngram")
        text = _build_bm25_text(session)
        assert "sh science engram" in text

    def test_includes_tasks_and_messages(self) -> None:
        session = _make_session(tasks=["Build auth"], first_messages=["Add login"])
        text = _build_bm25_text(session)
        assert "Build auth" in text
        assert "Add login" in text


class TestRrfFuse:
    def test_basic_fusion(self) -> None:
        semantic = {"s1": 0, "s2": 1}
        bm25 = {"s1": 1, "s2": 0}
        scores = _rrf_fuse(semantic, bm25, n_sessions=5)
        # s1 and s2 should have equal scores (symmetric ranks)
        assert abs(scores["s1"] - scores["s2"]) < 1e-9

    def test_missing_from_one_ranker_gets_worst_rank(self) -> None:
        semantic = {"s1": 0}
        bm25 = {"s2": 0}
        scores = _rrf_fuse(semantic, bm25, n_sessions=5, k=60)
        # s1: 1/(60+0) + 1/(60+5) = 1/60 + 1/65
        # s2: 1/(60+5) + 1/(60+0) = 1/65 + 1/60
        assert abs(scores["s1"] - scores["s2"]) < 1e-9

    def test_top_ranked_in_both_wins(self) -> None:
        semantic = {"s1": 0, "s2": 2}
        bm25 = {"s1": 0, "s2": 1}
        scores = _rrf_fuse(semantic, bm25, n_sessions=5)
        assert scores["s1"] > scores["s2"]


class TestHybridRankFallback:
    def test_falls_back_to_substring_when_no_search_deps(self) -> None:
        sessions = [
            _make_session(session_id="s1", agents=["planner"]),
            _make_session(session_id="s2", agents=["builder"]),
        ]
        picker = SessionPickerScreen(sessions, Path("/tmp/db"), None, False)
        table = MagicMock()
        picker.query_one = MagicMock(return_value=table)  # type: ignore[method-assign]
        # Force bm25 unavailable
        picker._ensure_bm25 = lambda: False  # type: ignore[method-assign]
        result = picker._hybrid_rank("builder")
        assert result[0]["session_id"] == "s2"

    def test_empty_sessions_does_not_crash(self) -> None:
        picker = SessionPickerScreen([], Path("/tmp/db"), None, False)
        table = MagicMock()
        picker.query_one = MagicMock(return_value=table)  # type: ignore[method-assign]
        picker._do_search("anything")
        assert picker._row_keys == []


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
        # Matching session ranked first, non-matching appended after
        assert picker._row_keys[0] == "s2"
