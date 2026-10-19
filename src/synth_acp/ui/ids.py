"""Textual widget ids derived from identifiers chosen by an ACP agent."""

from __future__ import annotations

import re

_INVALID = re.compile(r"[^A-Za-z0-9_-]")


def widget_id(prefix: str, raw: str) -> str:
    """Return a valid Textual id for ``raw``, which an agent chose (a tool-call or request id).

    Textual ids may contain only letters, digits, ``_`` and ``-``, and must not start with a
    digit.  Agents use other characters (codex-acp: ``guardian_assessment:<uuid>``), so each
    invalid character becomes ``_x<hex>_``.  A valid ``raw`` is kept unchanged, so existing
    ids stay byte-identical.  ``prefix`` must be a valid id start such as ``"tool"``.

    Args:
        prefix: Leading id segment, joined to the escaped ``raw`` with ``-``.
        raw: The agent-supplied identifier.

    Returns:
        ``f"{prefix}-{escaped raw}"``.
    """
    escaped = _INVALID.sub(lambda m: f"_x{ord(m.group()):x}_", raw)
    return f"{prefix}-{escaped}"
