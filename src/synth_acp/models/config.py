"""Session configuration parsed from .synth.json."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from synth_acp.discovery import DiscoveredAgent

log = logging.getLogger(__name__)


class CommunicationMode(StrEnum):
    """Communication scoping mode for inter-agent visibility."""

    MESH = "MESH"
    LOCAL = "LOCAL"


ENV_DENY_PREFIXES: tuple[str, ...] = (
    "npm_config_",
    "npm_package_",
    "npm_lifecycle_",
    "npm_command",
    "npm_execpath",
    "npm_node_execpath",
    "init_cwd",
    "brazil_",
    "envroot",
    "canonical_envroot",
)
"""Lowercased name prefixes never forwarded to a harness subprocess.

Narrow named prefixes, NOT a broad ``npm_`` glob.  A broad match would also take
``NPM_TOKEN`` and ``NPM_AUTH_TOKEN`` with it -- the credentials a private registry
needs -- turning a child's install into a silent 401.  These are the names the build
and package tooling actually exports, and they are dropped because a harness
subprocess that inherits them resolves paths against the parent's build context
rather than its own.
"""

ENV_DENY_NAMES: frozenset[str] = frozenset({"ld_library_path"})
"""Lowercased exact names never forwarded to a harness subprocess."""


def _is_denied(name: str) -> bool:
    """Return True when ``name`` must not be forwarded from the parent environment."""
    lowered = name.lower()
    return lowered in ENV_DENY_NAMES or lowered.startswith(ENV_DENY_PREFIXES)


class HarnessEnvConfig(BaseModel, frozen=True):
    """Per-harness environment policy.

    Attributes:
        env: Literal name/value pairs set on the subprocess.  Applied after
            anything inherited, so an explicit setting always wins.
        inherit: Either ``"all"``, forwarding the whole parent environment minus
            the denylists, or a list of names and :mod:`fnmatch` patterns naming
            exactly what to forward.
    """

    env: dict[str, str] = {}
    inherit: Literal["all"] | list[str] = "all"


def merge_harness_env(
    base: dict[str, HarnessEnvConfig], override: dict[str, HarnessEnvConfig]
) -> dict[str, HarnessEnvConfig]:
    """Merge a project harness-env policy over a global one, per harness.

    ``env`` merges per variable, so a project that sets one variable does not drop
    the others -- unlike ``auto_approve_tools``, where a project list replaces the
    global one wholesale.  The divergence is deliberate: a project overriding
    ``ANTHROPIC_MODEL`` should not silently drop a global ``CLAUDE_CODE_USE_BEDROCK``
    and send the harness at a different backend.

    ``inherit`` concatenates when both sides are lists, so a project can widen a
    global allowlist.  A list facing ``"all"`` on the other side replaces it, so a
    project can also narrow to the strict posture, and ``"all"`` facing a list
    widens back.

    Args:
        base: Global policy, keyed by harness ``short_name``.
        override: Project policy, keyed by harness ``short_name``.

    Returns:
        A new mapping covering every harness named by either side.
    """
    merged: dict[str, HarnessEnvConfig] = dict(base)
    for harness, over in override.items():
        under = merged.get(harness)
        if under is None:
            merged[harness] = over
            continue
        if isinstance(under.inherit, list) and isinstance(over.inherit, list):
            inherit: Literal["all"] | list[str] = [*under.inherit, *over.inherit]
        elif "inherit" in over.model_fields_set:
            inherit = over.inherit
        else:
            inherit = under.inherit
        merged[harness] = HarnessEnvConfig(env={**under.env, **over.env}, inherit=inherit)
    return merged


def resolve_harness_env(
    entry: HarnessEntry,
    policy: HarnessEnvConfig | None,
    parent_env: Mapping[str, str],
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, str]:
    """Build the environment overrides for one harness subprocess.

    Returned as OVERRIDES rather than a complete environment, because the caller
    merges them over the ACP SDK's ``default_environment()``.  Under the default
    ``inherit="all"`` the result already contains the parent environment, so that
    merge is a no-op; under a strict allowlist the SDK's six variables still reach
    the child, which is the behaviour a harness needs to run at all.

    Order of application, later winning:

    1. the inherited slice of ``parent_env``, per ``policy.inherit``
    2. ``entry.executable_env_var``, resolved from ``entry.binary_names``
    3. ``policy.env``
    4. ``entry.clear_env_vars``, each set to ``""``

    ``policy.env`` outranks the inherited slice so a user can correct a value their
    shell already sets.  ``clear_env_vars`` outranks ``policy.env`` because it exists
    to neutralise a variable the harness must not see, and inheriting the parent
    environment is exactly what makes it load-bearing: before inheritance,
    ``CLAUDECODE`` was never present to clear.

    Args:
        entry: The resolved harness entry.
        policy: This harness's env policy, or None for the default policy.
        parent_env: The environment to inherit from, normally ``os.environ``.
        which: PATH lookup, injected for testing.

    Returns:
        Environment overrides.  Empty when nothing applies.
    """
    policy = policy or HarnessEnvConfig()
    overrides: dict[str, str] = {}

    for name, value in parent_env.items():
        # Exported shell functions, skipped for the same reason the SDK's
        # default_environment skips them: the value is not a value.
        if value.startswith("()") or _is_denied(name):
            continue
        if policy.inherit == "all" or any(fnmatch(name, pattern) for pattern in policy.inherit):
            overrides[name] = value

    if entry.executable_env_var:
        for name in entry.binary_names:
            path = which(name)
            if path:
                overrides[entry.executable_env_var] = path
                break
        else:
            log.warning(
                "Harness '%s': executable_env_var '%s' set but none of %s found in PATH",
                entry.short_name,
                entry.executable_env_var,
                entry.binary_names,
            )

    # Explicit settings are honoured even against the denylist: a user naming a
    # variable is stating intent, where inheriting one is not.
    overrides.update(policy.env)

    for var in entry.clear_env_vars:
        overrides[var] = ""

    return overrides


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


MessageKind = Literal["chat", "request", "response", "system", "notification"]


def normalize_message_kind(raw: str) -> MessageKind:
    """Narrow an unconstrained DB kind string to MessageKind.

    Unknown or legacy values map to ``"chat"``, preserving pre-24274df behavior.
    Uses explicit literal-return branches so the return type narrows soundly.
    Never raises.

    ``"notification"`` MUST have a branch here.  This function is the choke point
    every delivered row passes through (``message_bus._poll_messages``), so a kind
    without a branch is silently rewritten to ``"chat"`` and every downstream
    decision -- envelope wording, steer eligibility -- behaves as if an agent had
    sent an ordinary chat message.  The failure is invisible: no error, no log.
    """
    if raw == "request":
        return "request"
    if raw == "response":
        return "response"
    if raw == "system":
        return "system"
    if raw == "notification":
        return "notification"
    return "chat"


_ALLOWED_SLOTS = frozenset({"from_agent", "to_agent", "kind", "message_type"})
_NUDGE_SLOTS = frozenset({"agent_id", "used", "nudge_threshold"})


def _validate_template(v: str, allowed: frozenset[str], label: str) -> str:
    """Structurally validate a message template via the format grammar.

    Proves ``v.format_map(defaultdict(str, ...))`` cannot raise for any string
    slot values. Rejects malformed braces, positional/empty fields (``{}``,
    ``{0}``), attribute/index traversal (``{x.y}``, ``{x[y]}``), non-empty format
    specs (including nested fields like ``{x:{y}}``), any conversion (``{x!r}``),
    and unknown slot names. An empty test-render probe is insufficient because
    e.g. ``{from_agent:{kind}}`` passes an empty probe but raises at delivery.

    Args:
        v: Template string to validate.
        allowed: Slot names this template may reference.
        label: Config field name, used in error messages.
    """
    try:
        parsed = list(Formatter().parse(v))
    except ValueError as e:
        raise ValueError(f"malformed {label} template {v!r}: {e}") from e
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue  # pure literal segment
        if field_name == "" or field_name.isdigit():
            raise ValueError(f"positional fields not allowed in {label} template {v!r}")
        if "." in field_name or "[" in field_name or "]" in field_name:
            raise ValueError(
                f"attribute/index access not allowed in {label} template {v!r}"
            )
        if field_name not in allowed:
            raise ValueError(
                f"unknown slot {field_name!r} in {label} template {v!r}; "
                f"allowed: {sorted(allowed)}"
            )
        if format_spec:
            raise ValueError(f"format specifiers not allowed in {label} template {v!r}")
        if conversion is not None:
            raise ValueError(f"conversions not allowed in {label} template {v!r}")
    return v


class McpMessageHook(BaseModel, frozen=True):
    """Recipient-facing envelope for MCP-delivered inter-agent messages.

    ``template`` and ``system_template`` are structurally grammar-validated at
    construction (bare approved slots only: from_agent|to_agent|kind|message_type;
    no traversal, format spec, conversion, or positional/malformed fields), so a
    constructed instance is always renderable and ``format_mcp_message`` never
    raises.
    """

    active: bool = True
    template: str = "[{message_type} from {from_agent}]: "
    system_template: str = "[System notification — no action required]: "

    @field_validator("template", "system_template")
    @classmethod
    def _check(cls, v: str) -> str:
        return _validate_template(v, _ALLOWED_SLOTS, "on_mcp_message")


DEFAULT_HANDOFF_NUDGE_TEMPLATE = (
    "Your context window is {used} full, past the {nudge_threshold} nudge threshold. "
    "If you are at a natural stopping point, consider calling the handoff MCP tool: it "
    "retires you and starts a fresh session that keeps your agent id, so your parent and "
    "your children keep addressing you unchanged. Your successor inherits only your "
    "handoff_message and starts with an empty transcript, so write that message as a "
    "complete briefing. Continuing in this session is also fine — this is a suggestion, "
    "not an instruction."
)


def validate_handoff_nudge_template(v: str) -> str:
    """Validate a handoff-nudge template. Raises ValueError on a bad template."""
    return _validate_template(v, _NUDGE_SLOTS, "handoff_nudge_template")


class HooksConfig(BaseModel, frozen=True):
    """Lifecycle hooks for agent events."""

    on_agent_startup: StartupHookConfig = StartupHookConfig()
    on_agent_join: MessageHook = MessageHook()
    on_agent_exit: MessageHook = MessageHook()
    on_mcp_message: McpMessageHook = McpMessageHook()

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
    on_mcp_message: McpMessageHook = McpMessageHook()


class GlobalConfig(BaseModel, frozen=True):
    """Global configuration stored at ~/.synth/config.json."""

    default_harness: str | None = None
    default_agent_id: str | None = None
    default_agent_mode: str | None = None
    communication_mode: CommunicationMode = CommunicationMode.LOCAL
    auto_approve_tools: list[str] = ["synth-mcp"]
    messages_interrupt: bool = True
    handoff_nudge: bool = True
    handoff_nudge_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    handoff_nudge_template: str = DEFAULT_HANDOFF_NUDGE_TEMPLATE
    harness_env: dict[str, HarnessEnvConfig] = {}
    hooks: GlobalHooksConfig = GlobalHooksConfig()

    @field_validator("handoff_nudge_template")
    @classmethod
    def _check_nudge_template(cls, v: str) -> str:
        return validate_handoff_nudge_template(v)


class SettingsConfig(BaseModel, frozen=True):
    """Fully resolved session settings. No None values. Used by broker."""

    communication_mode: CommunicationMode = CommunicationMode.MESH
    auto_approve_tools: list[str] = []
    messages_interrupt: bool = True
    handoff_nudge: bool = True
    handoff_nudge_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    handoff_nudge_template: str = DEFAULT_HANDOFF_NUDGE_TEMPLATE
    harness_env: dict[str, HarnessEnvConfig] = {}
    hooks: HooksConfig = HooksConfig()

    @field_validator("handoff_nudge_template")
    @classmethod
    def _check_nudge_template(cls, v: str) -> str:
        return validate_handoff_nudge_template(v)


class RawSettingsConfig(BaseModel, frozen=True):
    """Parsed from .synth.json. None = not set, inherit from global config."""

    communication_mode: CommunicationMode | None = None
    auto_approve_tools: list[str] | None = None
    messages_interrupt: bool | None = None
    handoff_nudge: bool | None = None
    handoff_nudge_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    handoff_nudge_template: str | None = None
    harness_env: dict[str, HarnessEnvConfig] = {}
    hooks: HooksConfig = HooksConfig()

    @field_validator("handoff_nudge_template")
    @classmethod
    def _check_nudge_template(cls, v: str | None) -> str | None:
        return v if v is None else validate_handoff_nudge_template(v)


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
        install_hint: Command that installs the program named by ``run_cmd``,
            shown when that program is absent from PATH.  Needed only where
            ``run_cmd`` names a DIFFERENT program from ``binary_names`` -- Claude
            Code's ACP adaptor is a separate npm package from Claude Code itself,
            so a user can have the harness installed and still be unable to launch
            it.  Leave None where the two are the same program, since the existing
            "no harnesses found" message already names it.
        steer_protocol: Wire protocol for in-turn steering, or None when the
            harness does not support it. A ``Literal`` rather than a free-form
            method string: Claude's steering takes a different payload shape
            (``prompt`` as an array of content blocks, versus Kiro's ``message``
            string), so a bare method name would let someone enable Claude by
            adding a string and get a silent ``-32602``. Adding a harness must
            require adding a variant here.
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
    steer_protocol: Literal["kiro"] | None = None
    install_hint: str | None = None


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

DEFAULT_STARTUP_CONTEXT = """\
<orchestration_context>
You are one agent in a Synth multi-agent orchestration session. This context
explains your environment and how to operate in it.

<identity>
agent_id: {agent_id}
parent_agent: {parent_id}
task: {task}
harness: {harness}
</identity>

<what_synth_is>
Synth (Synchronized Network of Teamed Harnesses) runs multiple AI coding agents
as parallel subprocesses, each with its own context window, coordinated over the
Agent Client Protocol (ACP). You run inside the "{harness}" harness, but Synth —
not your harness — owns agent spawning, message routing, and the shared dashboard
the user watches.

- Agents cannot see each other's text output. Your replies stream to the user's
  dashboard only.
- All inter-agent communication goes through your synth-mcp tools: send_message,
  list_agents, launch_agent, terminate_agent, resurrect_agent, get_my_context.
- The user sees every agent in the dashboard and can prompt any of them directly.
</what_synth_is>

<spawning_subagents>
This guidance applies to every child agent you create, throughout the whole
session — not only the first one.

Use the synth-mcp launch_agent tool to spawn child agents. A child launched this
way is a first-class participant in the session: the user can see and steer it in
the dashboard, and it can exchange messages with other agents through the bus.
Visibility and coordination are the reason this environment exists.

Your harness also has a built-in subagent/task feature. It works, but its children
run outside Synth: the user cannot see or interact with them, and they cannot
message other agents. Use it only for a self-contained subtask that needs neither
user visibility nor coordination with another agent — for example, a quick parallel
file read whose result you fold straight back into your own work. Whenever a subtask
could plausibly need either, use launch_agent. You are the parent of any agent you
launch, and you coordinate it with send_message.
</spawning_subagents>

<managing_child_agents>
Each agent_id is permanent within the session: once used it cannot be reused by
launch_agent, even after the agent is terminated. You can manage any agent you
launched — terminate_agent ends it, and resurrect_agent is the only way to bring
a terminated agent back, restoring it with its conversation history intact.

To launch a child as a specific agent configuration, set agent_mode to one of the
configurations available in your harness ({harness}):
{available_agents}
Other harnesses have their own configurations. See the launch_agent tool
description for details.
</managing_child_agents>

<working_rules>
1. Visibility: Your text output goes to the dashboard only — other agents cannot
   see it. Reach other agents with send_message.
2. Discovery: Use list_agents to see who is active, with their status, parent, and
   task.
3. Replying to a parent: When you finish work for the agent that launched you, call
   send_message(to_agent="{parent_id}", kind="response") with your results.
4. Message delivery: Messages arrive only between turns. After sending one, finish
   your current turn — the reply is delivered as your next input. Do not poll or
   loop waiting for a reply.
5. Recovering state: If you lose track of your identity or role, call
   get_my_context().
</working_rules>
</orchestration_context>

"""


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


_MESSAGE_TYPE_BY_KIND = {
    "chat": "Message",
    "request": "Request",
    "response": "Response",
    "notification": "Notification",
}


def format_percent(fraction: float) -> str:
    """Render a 0..1 fraction as a whole-percent string, e.g. ``"62%"``."""
    return f"{round(fraction * 100)}%"


def format_handoff_nudge(
    template: str, *, agent_id: str, used_fraction: float, threshold: float
) -> str:
    """Render the handoff-nudge body.

    ``used_fraction`` and ``threshold`` are 0..1 fractions, rendered into the
    ``{used}`` and ``{nudge_threshold}`` slots as whole-percent strings so the
    template never needs a format spec (which the validator forbids).

    Raises:
        Never for a template that passed ``validate_handoff_nudge_template``.
    """
    return render_template(
        template,
        {
            "agent_id": agent_id,
            "used": format_percent(used_fraction),
            "nudge_threshold": format_percent(threshold),
        },
    )


def format_mcp_message(
    hook: McpMessageHook,
    *,
    from_agent: str,
    to_agent: str,
    body: str,
    kind: MessageKind,
) -> str:
    """Render the configured prefix and append ``body`` unchanged.

    ``active=False`` returns ``body`` unchanged. ``kind == "system"`` uses
    ``system_template``; otherwise ``template`` with ``message_type`` mapped
    chat->Message, request->Request, response->Response. Body is always appended
    verbatim (never a slot).

    Raises:
        Never for a McpMessageHook instance. Templates are grammar-validated at
        construction such that format_map over string slots cannot raise; there
        is no other failure mode (pure, side-effect free).
    """
    if not hook.active:
        return body
    slots = {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "kind": kind,
        "message_type": _MESSAGE_TYPE_BY_KIND.get(kind, ""),
    }
    template = hook.system_template if kind == "system" else hook.template
    return render_template(template, slots) + body


def format_available_agents(agents: list[DiscoveredAgent]) -> str:
    """Render the ``{available_agents}`` slot body.

    Args:
        agents: Discovered agent configurations for the current harness.

    Returns:
        A non-empty list renders one line per agent, two-space indented::

            '  - {qualified_name} — {description}'

        When ``description`` is empty, the line is just ``'  - {qualified_name}'``.
        An empty list renders the single line
        ``'  (no named agent configurations are available in this harness)'``.
        Lines are joined with newlines; there is no trailing newline (the
        template supplies the surrounding newlines around the slot).
    """
    if not agents:
        return "  (no named agent configurations are available in this harness)"
    lines: list[str] = []
    for agent in agents:
        if agent.description:
            lines.append(f"  - {agent.qualified_name} — {agent.description}")
        else:
            lines.append(f"  - {agent.qualified_name}")
    return "\n".join(lines)


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
