"""Harness registry loader."""

from __future__ import annotations

import importlib.resources
import tomllib

from synth_acp.models.config import HarnessEntry


def load_harness_registry() -> list[HarnessEntry]:
    """Load all harness definitions from package data.

    Returns:
        List of HarnessEntry objects parsed from TOML files.
    """
    entries: list[HarnessEntry] = []
    harness_dir = importlib.resources.files("synth_acp.data.harnesses")
    for item in harness_dir.iterdir():
        if hasattr(item, "name") and item.name.endswith(".toml"):
            data = tomllib.loads(item.read_text())
            entries.append(HarnessEntry.model_validate(data))
    return entries
