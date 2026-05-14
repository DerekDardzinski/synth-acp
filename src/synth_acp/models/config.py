"""Session configuration parsed from .synth.json."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

log = logging.getLogger(__name__)


class CommunicationMode(StrEnum):
    """Communication scoping mode for inter-agent visibility."""

    MESH = "MESH"
    LOCAL = "LOCAL"


class StartupHookConfig(BaseModel, frozen=True):
    """Controls whether startup context is injected. Content comes from context.md file, not config."""

    active: bool = True


class MessageHook(BaseModel, frozen=True):
    """Hook that sends a templated message to a set of recipients."""

    active: bool = True
    recipients: Literal["parent", "family", "mesh"] = "parent"
    template: str = ""
    kind: Literal["system", "chat"] = "system"

    @model_validator(mode="before")
    @classmethod
    def _handle_recipients_none(cls, data: Any) -> Any:
        """Backward compat: recipients='none' maps to active=False."""
        if isinstance(data, dict) and data.get("recipients") == "none":
            data = dict(data)
            log.warning("MessageHook recipients='none' is deprecated. Use active=false instead.")
            data.pop("recipients")
            data.setdefault("active", False)
        return data


class HooksConfig(BaseModel, frozen=True):
    """Lifecycle hooks for agent events."""

    on_agent_startup: StartupHookConfig = StartupHookConfig()
    on_agent_join: MessageHook = MessageHook()
    on_agent_exit: MessageHook = MessageHook()

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_fields(cls, data: Any) -> Any:
        """Backward compat: ignore on_agent_prompt and startup prepend fields."""
        if isinstance(data, dict):
            data = dict(data)
            if "on_agent_prompt" in data:
                log.warning(
                    "on_agent_prompt hook is deprecated and will be ignored. "
                    "Startup context is now managed via ~/.synth/context.md."
                )
                data.pop("on_agent_prompt")
            startup = data.get("on_agent_startup")
            if isinstance(startup, dict) and "prepend" in startup:
                log.warning(
                    "on_agent_startup.prepend is deprecated and will be ignored. "
                    "Startup context is now managed via ~/.synth/context.md."
                )
                startup = dict(startup)
                startup.pop("prepend")
                data["on_agent_startup"] = startup
        return data


class GlobalHooksConfig(BaseModel, frozen=True):
    """Global hooks config with inactive join/exit defaults and visible templates."""

    on_agent_startup: StartupHookConfig = StartupHookConfig()
    on_agent_join: MessageHook = MessageHook(
        active=False, template='Agent "{agent_id}" is now active. Task: "{task}".'
    )
    on_agent_exit: MessageHook = MessageHook(
        active=False, template='Agent "{agent_id}" has exited.'
    )


class GlobalConfig(BaseModel, frozen=True):
    """Global configuration stored at ~/.synth/config.json."""

    default_harness: str | None = None
    default_agent_id: str | None = None
    default_agent_mode: str | None = None
    communication_mode: CommunicationMode = CommunicationMode.LOCAL
    auto_approve_tools: list[str] = ["synth-mcp"]
    hooks: GlobalHooksConfig = GlobalHooksConfig()


class SettingsConfig(BaseModel, frozen=True):
    """Fully resolved session settings. No None values. Used by broker."""

    communication_mode: CommunicationMode = CommunicationMode.MESH
    auto_approve_tools: list[str] = []
    hooks: HooksConfig = HooksConfig()


class RawSettingsConfig(BaseModel, frozen=True):
    """Parsed from .synth.json. None = not set, inherit from global config."""

    communication_mode: CommunicationMode | None = None
    auto_approve_tools: list[str] | None = None
    hooks: HooksConfig = HooksConfig()


class HarnessEntry(BaseModel, frozen=True):
    """A known ACP-capable harness from the registry.

    Attributes:
        identity: Unique key for the harness.
        name: Human-readable display name.
        short_name: Used with the ``harness`` config field.
        binary_names: Executables searched in PATH.
        run_cmd: Command string to launch the harness (no agent flag).
        mode_arg: CLI flag to pass agent_mode directly (e.g. ``--agent``).
        executable_env_var: If set, the env var name to inject with the
            detected binary path (e.g. ``"CLAUDE_CODE_EXECUTABLE"``).
            Resolved by searching ``binary_names`` in PATH in order;
            the first match wins. No-op if none found.
        clear_env_vars: Env var names to explicitly clear (set to ``""``)
            in the agent subprocess environment.
    """

    identity: str
    name: str
    short_name: str
    binary_names: list[str]
    run_cmd: str
    mode_arg: str | None = None
    executable_env_var: str | None = None
    clear_env_vars: list[str] = []
    agent_mode_target: Literal["acp_mode", "meta_agent"] | None = None


class RawSessionConfig(BaseModel, frozen=True):
    """Parsed from .synth.json. Settings may have unresolved None fields."""

    project: str
    settings: RawSettingsConfig = RawSettingsConfig()

    @model_validator(mode="before")
    @classmethod
    def _coerce_session_to_project(cls, data: Any) -> Any:
        """Rename legacy ``session`` key to ``project``, strip deprecated fields, apply env overrides."""
        if isinstance(data, dict):
            data = dict(data)
            if "session" in data and "project" not in data:
                data["project"] = data.pop("session")

            # Strip deprecated fields
            if "agents" in data:
                log.warning("'agents' in .synth.json is deprecated and will be ignored.")
                data.pop("agents")
            if "ui" in data:
                log.warning("'ui' in .synth.json is deprecated and will be ignored.")
                data.pop("ui")

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

            if hooks:
                settings["hooks"] = hooks
                data["settings"] = settings
        return data


class SessionConfig(BaseModel, frozen=True):
    """Fully resolved config. Passed to broker. No None values in settings."""

    project: str
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

            if hooks:
                settings["hooks"] = hooks
                data["settings"] = settings
        return data


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

SYNTH_DIR: Path = Path.home() / ".synth"
GLOBAL_CONFIG_PATH: Path = SYNTH_DIR / "config.json"
CONTEXT_MD_PATH: Path = SYNTH_DIR / "context.md"

# ---------------------------------------------------------------------------
# Default startup context (rich block with 5 rules)
# ---------------------------------------------------------------------------

DEFAULT_STARTUP_CONTEXT = (
    "<orchestration_context>\n"
    "agent_id: {agent_id}\n"
    "parent_agent: {parent_id}\n"
    "task: {task}\n"
    "session: You are in a multi-agent orchestration session managed by Synth.\n"
    "\n"
    "Rules:\n"
    "1. Visibility: Your text output goes to the orchestration UI only — other agents cannot see it.\n"
    "2. Communication: Use send_message() to talk to other agents. Use list_agents() to discover peers.\n"
    "3. Spawning: Use launch_agent() to create child agents for subtasks. You are their parent.\n"
    "4. Reply: When completing work for a parent, call send_message(to_agent='{parent_id}', kind='response') with your results.\n"
    "5. Message delivery: Messages arrive only between turns. After sending a message, finish your current turn — the response will be delivered as your next input. Do not poll or loop waiting for replies.\n"
    "\n"
    "How Synth works:\n"
    "- Synth runs multiple AI agents as parallel subprocesses, each with their own context window.\n"
    "- Agents cannot see each other's text output. All inter-agent communication goes through send_message().\n"
    "- Messages are queued and delivered when the recipient is idle (between turns).\n"
    "- Each agent gets a synth-mcp tool server with: send_message, list_agents, launch_agent, terminate_agent, resurrect_agent, get_my_context.\n"
    "- The user sees all agents in a shared dashboard and can send prompts to any agent directly.\n"
    "- If you lose track of your identity or role, call get_my_context() to recover it.\n"
    "</orchestration_context>\n\n"
)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_global_config() -> GlobalConfig:
    """Load global config from ~/.synth/config.json, or return defaults."""
    if GLOBAL_CONFIG_PATH.exists():
        raw = json.loads(GLOBAL_CONFIG_PATH.read_text())
        return GlobalConfig.model_validate(raw)
    return GlobalConfig()


def save_global_config(config: GlobalConfig) -> None:
    """Write global config to ~/.synth/config.json, creating dir if needed."""
    SYNTH_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")


def ensure_synth_dir() -> None:
    """Create ~/.synth/ and seed config.json + context.md if they don't exist."""
    SYNTH_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not GLOBAL_CONFIG_PATH.exists():
        save_global_config(GlobalConfig())
    if not CONTEXT_MD_PATH.exists():
        CONTEXT_MD_PATH.write_text(DEFAULT_STARTUP_CONTEXT)


def load_startup_context() -> str:
    """Load startup context: ~/.synth/context.md if exists, else DEFAULT_STARTUP_CONTEXT."""
    if CONTEXT_MD_PATH.exists():
        return CONTEXT_MD_PATH.read_text()
    return DEFAULT_STARTUP_CONTEXT


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_template(template: str, slots: dict[str, str]) -> str:
    """Render a template string with named slots.

    Unknown slots are left as empty strings rather than raising KeyError.
    """
    return template.format_map(defaultdict(str, slots))


# ---------------------------------------------------------------------------
# Config file discovery and loading
# ---------------------------------------------------------------------------


def find_config(cwd: Path) -> Path | None:
    """Find a .synth.json config file in the given directory."""
    path = cwd / ".synth.json"
    return path if path.exists() else None


def load_config(path: Path) -> RawSessionConfig:
    """Load and validate a .synth.json config file."""
    raw = json.loads(path.read_text())
    return RawSessionConfig.model_validate(raw)
