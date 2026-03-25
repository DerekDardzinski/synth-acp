"""Tests for CLI harness registry and config resolution."""

from __future__ import annotations

from synth_acp.cli import load_harness_registry


class TestHarnessRegistry:
    def test_load_harness_registry_when_called_returns_all_harnesses(self) -> None:
        entries = load_harness_registry()
        identities = {e.identity for e in entries}
        assert identities == {"kiro", "claude", "opencode", "gemini"}

    def test_load_harness_registry_when_called_entries_have_required_fields(self) -> None:
        entries = load_harness_registry()
        for entry in entries:
            assert entry.short_name
            assert entry.binary_names
            assert entry.run_cmd
            assert entry.run_cmd_with_agent
