"""Permission engine with persistent rule storage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from synth_acp.models.permissions import PermissionDecision, PermissionRule

logger = logging.getLogger(__name__)


class PermissionEngine:
    """Loads, caches, and persists per-agent permission rules.

    Rules are keyed on ``(agent_id, tool_kind)`` and stored as JSON at
    *rules_path*.  Missing or corrupt files are treated as an empty ruleset.
    """

    def __init__(self, rules_path: Path) -> None:
        self._rules_path = rules_path
        self._cache: dict[tuple[str, str], PermissionDecision] = {}
        self._load()

    def check(self, agent_id: str, tool_kind: str) -> PermissionDecision | None:
        """Return the stored decision for *(agent_id, tool_kind)*, or ``None``."""
        return self._cache.get((agent_id, tool_kind))

    def persist(self, rule: PermissionRule) -> None:
        """Update the in-memory cache and write all rules to disk atomically."""
        self._cache[(rule.agent_id, rule.tool_kind)] = rule.decision
        self._write()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._rules_path.exists():
            return
        try:
            data = json.loads(self._rules_path.read_text())
            for item in data:
                r = PermissionRule.model_validate(item)
                self._cache[(r.agent_id, r.tool_kind)] = r.decision
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Corrupt rules file %s — starting with empty ruleset", self._rules_path)

    def _write(self) -> None:
        rules = [
            PermissionRule(agent_id=aid, tool_kind=tk, decision=d).model_dump()
            for (aid, tk), d in self._cache.items()
        ]
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._rules_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rules, indent=2))
        tmp.replace(self._rules_path)
