"""Session configuration parsed from .synth.json."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from synth_acp.models.agent import AgentConfig

DEFAULT_ROUTING_CONTEXT = (
    "<orchestration_context>\n"
    "agent_id: {agent_id}\n"
    "parent_agent: {parent_id}\n"
    "reply_tool: send_message(to_agent='{parent_id}', kind='response')\n"
    "visibility: Your text output goes to the UI only. Other agents cannot see it.\n"
    "recovery: Call get_my_context() if you lose track of this information.\n"
    "</orchestration_context>\n\n"
)

DEFAULT_STARTUP_CONTEXT = (
    "<orchestration_context>\n"
    "agent_id: {agent_id}\n"
    "session: You are in a multi-agent session. Use list_agents() to see other agents.\n"
    "communication: Use send_message() to talk to other agents. Your text output goes to the user only.\n"
    "</orchestration_context>\n\n"
)


class CommunicationMode(StrEnum):
    """Communication scoping mode for inter-agent visibility."""

    MESH = "MESH"
    LOCAL = "LOCAL"


class MessageHook(BaseModel, frozen=True):
    """Hook that sends a templated message to a set of recipients."""

    recipients: Literal["none", "parent", "family", "mesh"] = "none"
    template: str = ""
    kind: Literal["system", "chat"] = "system"


class PromptHook(BaseModel, frozen=True):
    """Hook that prepends context to a launched agent's initial prompt."""

    prepend: str = DEFAULT_ROUTING_CONTEXT


class HooksConfig(BaseModel, frozen=True):
    """Lifecycle hooks for agent events."""

    on_agent_join: MessageHook = MessageHook()
    on_agent_exit: MessageHook = MessageHook()
    on_agent_prompt: PromptHook = PromptHook()
    on_agent_startup: PromptHook = PromptHook(prepend=DEFAULT_STARTUP_CONTEXT)


class SettingsConfig(BaseModel, frozen=True):
    """Global session settings."""

    communication_mode: CommunicationMode = CommunicationMode.MESH
    auto_approve_tools: list[str] = []
    hooks: HooksConfig = HooksConfig()


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
    """Top-level session configuration."""

    project: str
    agents: list[AgentConfig]
    ui: UIConfig = UIConfig()
    settings: SettingsConfig = SettingsConfig()

    @model_validator(mode="before")
    @classmethod
    def _coerce_session_to_project(cls, data: Any) -> Any:
        """Rename legacy ``session`` key to ``project`` and apply env overrides."""
        if isinstance(data, dict):
            data = dict(data)
            if "session" in data and "project" not in data:
                data["project"] = data.pop("session")
            # Apply env var overrides into settings.hooks
            settings = dict(data.get("settings") or {})
            hooks = dict(settings.get("hooks") or {})

            if val := os.environ.get("SYNTH_JOIN_RECIPIENTS"):
                join = dict(hooks.get("on_agent_join") or {})
                join["recipients"] = val
                hooks["on_agent_join"] = join

            if val := os.environ.get("SYNTH_JOIN_TEMPLATE"):
                join = dict(hooks.get("on_agent_join") or {})
                join["template"] = val
                hooks["on_agent_join"] = join

            if val := os.environ.get("SYNTH_ROUTING_TEMPLATE"):
                prompt = dict(hooks.get("on_agent_prompt") or {})
                prompt["prepend"] = val
                hooks["on_agent_prompt"] = prompt

            if hooks:
                settings["hooks"] = hooks
                data["settings"] = settings
        return data

    @model_validator(mode="after")
    def validate_unique_ids(self) -> SessionConfig:
        """Ensure no duplicate agent IDs."""
        ids = [a.agent_id for a in self.agents]
        dupes = [x for x in ids if ids.count(x) > 1]
        if dupes:
            raise ValueError(f"Duplicate agent IDs: {set(dupes)}")
        return self


def render_template(template: str, slots: dict[str, str]) -> str:
    """Render a template string with named slots.

    Unknown slots are left as empty strings rather than raising KeyError.
    """
    return template.format_map(defaultdict(str, slots))


def find_config(cwd: Path) -> Path | None:
    """Find a .synth.json config file in the given directory."""
    path = cwd / ".synth.json"
    return path if path.exists() else None


def load_config(path: Path) -> SessionConfig:
    """Load and validate a .synth.json config file.

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


def write_json_config(path: Path, config: SessionConfig) -> None:
    """Write a SessionConfig as JSON.

    Only writes non-default fields for agents to keep the output minimal.
    """
    agents = []
    for agent in config.agents:
        entry: dict[str, Any] = {"agent_id": agent.agent_id, "harness": agent.harness}
        if agent.agent_mode:
            entry["agent_mode"] = agent.agent_mode
        if agent.cwd != ".":
            entry["cwd"] = agent.cwd
        agents.append(entry)

    data: dict[str, Any] = {"project": config.project, "agents": agents}
    path.write_text(json.dumps(data, indent=2) + "\n")
