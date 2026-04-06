"""Tests for harness registry loader."""

from __future__ import annotations

from synth_acp.harnesses import load_harness_registry


class TestLoadHarnessRegistry:
    def test_load_harness_registry_returns_entries_with_required_fields(self) -> None:
        entries = load_harness_registry()
        assert len(entries) >= 1
        for entry in entries:
            assert entry.identity
            assert entry.short_name
            assert entry.binary_names
            assert entry.run_cmd
