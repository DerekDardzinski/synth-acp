"""Session configuration parsed from .synth.json."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, model_validator

from synth_acp.models.agent import AgentConfig


class UIConfig(BaseModel, frozen=True):
    """UI settings."""

    web_port: int = 8000
    theme: str = "dark"


class SessionConfig(BaseModel, frozen=True):
    """Top-level configuration from .synth.json."""

    session: str
    agents: list[AgentConfig]
    ui: UIConfig = UIConfig()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> SessionConfig:
        ids = [a.id for a in self.agents]
        dupes = [x for x in ids if ids.count(x) > 1]
        if dupes:
            raise ValueError(f"Duplicate agent IDs: {set(dupes)}")
        return self


def load_config(path: Path) -> SessionConfig:
    """Load and validate a .synth.json file.

    Relative agent CWD paths are resolved against the config file's parent directory.
    """
    raw = json.loads(path.read_text())
    config = SessionConfig.model_validate(raw)

    config_dir = path.parent.resolve()
    resolved_agents = []
    for agent in config.agents:
        cwd = (config_dir / agent.cwd).resolve()
        resolved_agents.append(agent.model_copy(update={"cwd": str(cwd)}))

    return config.model_copy(update={"agents": resolved_agents})
