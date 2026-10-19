"""Filesystem-based agent discovery for ACP harnesses."""

from __future__ import annotations

from pathlib import Path

from synth_acp.acp.extensions import get_extension
from synth_acp.models.agent import DiscoveredAgent
from synth_acp.models.config import HarnessEntry


def discover_agents(harness: HarnessEntry, cwd: Path) -> list[DiscoveredAgent]:
    """Discover available agents for a harness via filesystem.

    Dispatches to the discover_profiles hook of the harness's ACP extension.
    Malformed files and missing directories result in an empty or partial list
    with warnings logged via the standard logging module.

    Args:
        harness: The harness entry to discover agents for.
        cwd: Working directory (for project-level agent discovery).

    Returns:
        List of discovered agents, sorted by source then name.
        Empty list if the harness's extension discovers no profiles.

    Raises:
        UnknownExtensionError: ``harness.extension`` names no registered extension.
    """
    agents = get_extension(harness.extension).discover_profiles(cwd)
    return sorted(agents, key=lambda a: (a.source, a.name))
