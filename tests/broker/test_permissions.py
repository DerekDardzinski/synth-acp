"""Tests for PermissionEngine rule loading, lookup, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from synth_acp.broker.permissions import PermissionEngine
from synth_acp.models.permissions import PermissionDecision, PermissionRule


class TestPermissionEngine:
    def test_check_when_rule_exists_returns_decision(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(
            json.dumps([{"agent_id": "agent-1", "tool_kind": "execute", "decision": "allow"}])
        )
        engine = PermissionEngine(rules_file)

        assert engine.check("agent-1", "execute") == PermissionDecision.allow

    def test_check_when_no_rule_returns_none(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps([]))
        engine = PermissionEngine(rules_file)

        assert engine.check("agent-1", "execute") is None

    def test_persist_when_called_writes_to_disk_and_updates_cache(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        engine = PermissionEngine(rules_file)

        rule = PermissionRule(
            agent_id="agent-1", tool_kind="edit", decision=PermissionDecision.reject
        )
        engine.persist(rule)

        # In-memory cache updated
        assert engine.check("agent-1", "edit") == PermissionDecision.reject
        # Disk written — reload from scratch to verify
        engine2 = PermissionEngine(rules_file)
        assert engine2.check("agent-1", "edit") == PermissionDecision.reject

    def test_init_when_file_missing_starts_empty(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "nonexistent" / "rules.json"
        engine = PermissionEngine(rules_file)

        assert engine.check("agent-1", "execute") is None

    def test_init_when_file_corrupt_starts_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        rules_file = tmp_path / "rules.json"
        rules_file.write_text("{invalid json")

        with caplog.at_level(logging.WARNING):
            engine = PermissionEngine(rules_file)

        assert engine.check("agent-1", "execute") is None
        assert "Corrupt rules file" in caplog.text
