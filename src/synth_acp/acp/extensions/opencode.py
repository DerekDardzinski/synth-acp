"""OpenCode (``opencode acp``, verified on 2.0.28): agent profiles through the mode config option.

OpenCode speaks standard ACP v1 only, so ACPSession handles sessions, prompts,
permissions, native config options, usage and resume unchanged.  What this extension
adds:

- ``profile_switch="config_option"``: OpenCode advertises its primary agents only as the
  values of the ``mode``-category config option (no legacy ``modes``) and ignores
  ``_meta`` on session/new, so ACPSession applies agent_mode with set_config_option after
  the session exists.
- ``discover_profiles``: the agents that will appear as those values, read offline from
  OpenCode's config files so the launch picker and ``{available_agents}`` work without a
  session.

Steering stays unsupported: OpenCode rejects a second session/prompt during a turn
(-32603 "Session already has an active ACP prompt") and exposes no steer method over ACP,
so synth queues messages until the turn ends.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import yaml

from synth_acp.acp.extensions.base import AcpExtension
from synth_acp.models.agent import DiscoveredAgent
from synth_acp.models.extensions import ExtensionCapabilities

log = logging.getLogger(__name__)

# Frontmatter keys of OpenCode's native agent schema.  A file using any other key is
# decoded with the V1 schema, whose ``disable`` means ``disabled``.
_NATIVE_KEYS = frozenset(
    {
        "variant",
        "model",
        "request",
        "system",
        "description",
        "mode",
        "hidden",
        "color",
        "steps",
        "disabled",
        "permissions",
    }
)
_AGENT_MODES = frozenset({"primary", "subagent", "all"})


class OpenCodeExtension(AcpExtension):
    """OpenCode: agent_mode through the mode config option; offline agent discovery."""

    name: ClassVar[str] = "opencode"
    capabilities: ClassVar[ExtensionCapabilities] = ExtensionCapabilities(
        profile_switch="config_option"
    )

    def discover_profiles(self, cwd: Path) -> list[DiscoveredAgent]:
        """OpenCode's selectable primary agents for ``cwd``, default first.

        Reads the same sources OpenCode merges, in its order (later wins): the global
        config dir, ``$OPENCODE_CONFIG``, ``opencode.json[c]`` in each directory from the
        filesystem root down to ``cwd``, each ``.opencode`` dir on that path, then
        ``$OPENCODE_CONFIG_CONTENT``.  Organization (wellknown) config and plugin-added
        agents are not visible offline; the live ``mode`` option stays authoritative.
        """
        return _discover(cwd, os.environ)


class _Agents:
    """OpenCode's agent map: insertion-ordered, entry-wise merge, ``disabled`` removes."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, str] = {}
        self.default_agent: str | None = None
        # Built-ins in OpenCode's insertion order; only build and plan are selectable.
        for agent_id, fields in (
            ("build", {}),
            ("general", {"mode": "subagent"}),
            ("explore", {"mode": "subagent"}),
            ("compaction", {"hidden": True}),
            ("title", {"hidden": True}),
            ("summary", {"hidden": True}),
            ("plan", {}),
        ):
            self.apply(agent_id, fields, "builtin")

    def apply(self, agent_id: str, fields: Mapping[str, Any], source: str) -> None:
        if fields.get("disabled") is True:
            self.entries.pop(agent_id, None)
            self.sources.pop(agent_id, None)
            return
        entry = self.entries.setdefault(agent_id, {"mode": "primary", "hidden": False})
        for key in ("mode", "hidden", "description"):
            if key in fields:
                entry[key] = fields[key]
        self.sources[agent_id] = source

    def selectable(self) -> list[DiscoveredAgent]:
        ids = [
            agent_id
            for agent_id, entry in self.entries.items()
            if entry["mode"] != "subagent" and entry["hidden"] is not True
        ]
        if not ids:
            return []
        if self.default_agent in ids:
            default = self.default_agent
        elif "build" in ids:
            default = "build"
        else:
            default = ids[0]
        ordered = [default, *(i for i in ids if i != default)]
        return [
            DiscoveredAgent(
                qualified_name=agent_id,
                name=agent_id,
                description=str(self.entries[agent_id].get("description") or ""),
                source=self.sources[agent_id],
            )
            for agent_id in ordered
        ]


def _discover(cwd: Path, env: Mapping[str, str]) -> list[DiscoveredAgent]:
    agents = _Agents()
    global_dir = _global_dir(env)
    _apply_directory(agents, global_dir, "user")
    if config_file := env.get("OPENCODE_CONFIG"):
        _apply_document(agents, Path(config_file).expanduser(), "user")
    cwd = cwd.expanduser().resolve()
    if not _project_disabled(cwd, global_dir, env):
        chain = [*reversed(cwd.parents), cwd]  # filesystem root first, cwd last
        for directory in chain:
            for name in ("opencode.json", "opencode.jsonc"):
                _apply_document(agents, directory / name, "project")
        for directory in chain:
            dot = directory / ".opencode"
            if dot.resolve() != global_dir.resolve():
                _apply_directory(agents, dot, "project")
    if content := env.get("OPENCODE_CONFIG_CONTENT"):
        _apply_config(agents, _parse_jsonc(content, "$OPENCODE_CONFIG_CONTENT"), "env")
    return agents.selectable()


def _global_dir(env: Mapping[str, str]) -> Path:
    if explicit := env.get("OPENCODE_CONFIG_DIR"):
        return Path(explicit).expanduser()
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "opencode"


def _project_disabled(cwd: Path, global_dir: Path, env: Mapping[str, str]) -> bool:
    """OpenCode reads ``OPENCODE_CONFIG_PROJECT_DISABLE ?? OPENCODE_DISABLE_PROJECT_CONFIG``."""
    flag = env.get("OPENCODE_CONFIG_PROJECT_DISABLE")
    if flag is None:
        flag = env.get("OPENCODE_DISABLE_PROJECT_CONFIG", "")
    if flag.strip().lower() in ("1", "true", "yes"):
        return True
    return cwd == global_dir.expanduser().resolve()


def _apply_directory(agents: _Agents, directory: Path, source: str) -> None:
    """A config directory: its opencode.json, opencode.jsonc, then its markdown agents."""
    if not directory.is_dir():
        return
    for name in ("opencode.json", "opencode.jsonc"):
        _apply_document(agents, directory / name, source)
    for agent_id, fields in _markdown_agents(directory):
        agents.apply(agent_id, fields, source)


def _apply_document(agents: _Agents, path: Path, source: str) -> None:
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        log.warning("Unreadable OpenCode config: %s", path)
        return
    _apply_config(agents, _parse_jsonc(text, str(path)), source)


def _apply_config(agents: _Agents, config: Any, source: str) -> None:
    """``agent`` (V1), ``mode`` (V1, forced primary), then ``agents`` (native).

    Within one file a later key replaces a whole entry of the same id.
    """
    if not isinstance(config, dict):
        return
    merged: dict[str, dict[str, Any]] = {}
    for key, v1, force_primary in (
        ("agent", True, False),
        ("mode", True, True),
        ("agents", False, False),
    ):
        section = config.get(key)
        if not isinstance(section, dict):
            continue
        for agent_id, raw in section.items():
            fields = _fields(raw, v1=v1, force_primary=force_primary)
            if fields is not None:
                merged[str(agent_id)] = fields
    for agent_id, fields in merged.items():
        agents.apply(agent_id, fields, source)
    if isinstance(config.get("default_agent"), str):
        agents.default_agent = config["default_agent"]


def _fields(raw: Any, *, v1: bool, force_primary: bool) -> dict[str, Any] | None:
    """The fields that decide mode-list membership, or None for an entry OpenCode rejects."""
    if not isinstance(raw, dict):
        return None
    fields = dict(raw)
    if v1 and "disable" in fields:
        fields["disabled"] = fields.pop("disable")
    if force_primary:
        fields["mode"] = "primary"
    if "mode" in fields and fields["mode"] not in _AGENT_MODES:
        return None
    return fields


def _markdown_agents(directory: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """``{agent,agents}/**/*.md`` (sorted), then ``{mode,modes}/*.md`` (forced primary)."""
    nested = sorted(
        (
            p
            for sub in ("agent", "agents")
            for p in (directory / sub).glob("**/*.md")
            if p.is_file()
        ),
        key=lambda p: p.relative_to(directory).as_posix(),
    )
    flat = sorted(
        (p for sub in ("mode", "modes") for p in (directory / sub).glob("*.md") if p.is_file()),
        key=lambda p: p.relative_to(directory).as_posix(),
    )
    for paths, force_primary in ((nested, False), (flat, True)):
        for path in paths:
            relative = path.relative_to(directory).as_posix()
            agent_id = re.sub(r"^(agents?|modes?)/", "", relative).removesuffix(".md")
            front = _frontmatter(path)
            if front is None:
                continue
            fields = _fields(front, v1=not set(front) <= _NATIVE_KEYS, force_primary=force_primary)
            if fields is not None:
                yield agent_id, fields


def _frontmatter(path: Path) -> dict[str, Any] | None:
    """YAML frontmatter as a dict; ``{}`` when there is none; None when OpenCode skips the file.

    Mirrors OpenCode's ``ConfigMarkdown.parse``: an empty file is skipped, and frontmatter
    that fails to parse is retried once with unquoted-colon values as block scalars.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text:
        return None
    match = re.match(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.DOTALL)
    if match is None:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        try:
            data = yaml.safe_load(_sanitize_frontmatter(match.group(1)))
        except yaml.YAMLError:
            log.warning("Malformed OpenCode agent frontmatter: %s", path)
            return None
    if data is None:
        return {}
    return data if isinstance(data, dict) else None


def _sanitize_frontmatter(frontmatter: str) -> str:
    """OpenCode's ``sanitize``: a top-level ``key: value`` whose value holds a colon becomes ``key: |-``."""
    lines: list[str] = []
    for line in re.split(r"\r?\n", frontmatter):
        stripped = line.strip()
        entry = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$", line)
        if stripped.startswith("#") or not stripped or re.match(r"^\s+", line) or entry is None:
            lines.append(line)
            continue
        value = entry.group(2).strip()
        if value in ("", ">", "|") or value.startswith(('"', "'")) or ":" not in value:
            lines.append(line)
            continue
        lines.extend((f"{entry.group(1)}: |-", f"  {value}"))
    return "\n".join(lines)


def _parse_jsonc(text: str, origin: str) -> Any:
    """JSON with ``//`` and ``/* */`` comments and trailing commas; None on error."""
    try:
        return json.loads(_strip_jsonc(text))
    except json.JSONDecodeError:
        log.warning("Malformed OpenCode config: %s", origin)
        return None


def _strip_jsonc(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(ch)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))
