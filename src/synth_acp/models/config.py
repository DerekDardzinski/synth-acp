"""Session configuration parsed from .synth.toml or .synth.json."""

from __future__ import annotations

import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from synth_acp.models.agent import AgentConfig


class CommunicationMode(StrEnum):
    """Communication scoping mode for inter-agent visibility."""

    MESH = "MESH"
    LOCAL = "LOCAL"


class SettingsConfig(BaseModel, frozen=True):
    """Global session settings."""

    communication_mode: CommunicationMode = CommunicationMode.MESH


class HarnessEntry(BaseModel, frozen=True):
    """A known ACP-capable harness from the registry.

    Attributes:
        identity: Unique key for the harness.
        name: Human-readable display name.
        short_name: Used with the ``harness`` config field.
        binary_names: Executables searched in PATH.
        run_cmd: Command string to launch the harness (no agent flag).
    """

    identity: str
    name: str
    short_name: str
    binary_names: list[str]
    run_cmd: str


class UIConfig(BaseModel, frozen=True):
    """UI settings."""

    web_port: int = 8000
    theme: str = "dark"


class SessionConfig(BaseModel, frozen=True):
    """Top-level session configuration.

    Supports both ``project`` (new) and ``session`` (legacy) keys.
    """

    project: str
    agents: list[AgentConfig]
    ui: UIConfig = UIConfig()
    settings: SettingsConfig = SettingsConfig()

    @model_validator(mode="before")
    @classmethod
    def _coerce_session_to_project(cls, data: Any) -> Any:
        """Rename legacy ``session`` key to ``project``."""
        if isinstance(data, dict):
            data = dict(data)
            if "session" in data and "project" not in data:
                data["project"] = data.pop("session")
        return data

    @model_validator(mode="after")
    def validate_unique_ids(self) -> SessionConfig:
        """Ensure no duplicate agent IDs."""
        ids = [a.agent_id for a in self.agents]
        dupes = [x for x in ids if ids.count(x) > 1]
        if dupes:
            raise ValueError(f"Duplicate agent IDs: {set(dupes)}")
        return self


def find_config(cwd: Path) -> Path | None:
    """Find a config file in the given directory.

    Checks ``.synth.toml`` first, then ``.synth.json``.

    Args:
        cwd: Directory to search in.

    Returns:
        Path to the config file, or None if not found.
    """
    toml_path = cwd / ".synth.toml"
    if toml_path.exists():
        return toml_path
    json_path = cwd / ".synth.json"
    if json_path.exists():
        return json_path
    return None


def load_config(path: Path) -> SessionConfig:
    """Load and validate a config file (.toml or .json).

    Relative agent CWD paths are resolved against the config file's parent directory.

    Args:
        path: Path to the config file.

    Returns:
        Validated SessionConfig.
    """
    if path.suffix == ".toml":
        raw = tomllib.loads(path.read_text())
    else:
        raw = json.loads(path.read_text())

    config = SessionConfig.model_validate(raw)

    config_dir = path.parent.resolve()
    resolved_agents = []
    for agent in config.agents:
        cwd = (config_dir / agent.cwd).resolve()
        resolved_agents.append(agent.model_copy(update={"cwd": str(cwd)}))

    return config.model_copy(update={"agents": resolved_agents})


def write_toml_config(path: Path, config: SessionConfig) -> None:
    """Write a SessionConfig as TOML.

    Args:
        path: Destination file path.
        config: The configuration to write.
    """
    lines = [f'project = "{config.project}"', ""]
    for agent in config.agents:
        lines.append("[[agents]]")
        lines.append(f'agent_id = "{agent.agent_id}"')
        lines.append(f'harness = "{agent.harness}"')
        if agent.agent_mode:
            lines.append(f'agent_mode = "{agent.agent_mode}"')
        if agent.cwd != ".":
            lines.append(f'cwd = "{agent.cwd}"')
        lines.append("")
    path.write_text("\n".join(lines))
