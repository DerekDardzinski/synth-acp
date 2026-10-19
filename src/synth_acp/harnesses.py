"""Harness registry loader."""

from __future__ import annotations

import importlib.resources
import logging
import tomllib
from collections.abc import Mapping
from typing import Any

from synth_acp.models.config import HarnessEntry

log = logging.getLogger(__name__)


class UnknownHarnessVariantError(ValueError):
    """harness_variant names a variant the harness TOML does not declare."""


def _resolve_variant(data: dict[str, Any], variants: Mapping[str, str]) -> dict[str, Any]:
    """Flatten one TOML into HarnessEntry fields, choosing its engine variant.

    A TOML without a ``variants`` table is returned unchanged.  Otherwise the variant is
    ``variants[short_name]`` when set, else ``default_variant``, and its table's fields
    (run_cmd, mode_arg, extension) are laid over the top-level ones.

    Raises:
        UnknownHarnessVariantError: The chosen variant is not declared, including any
            requested variant (even an empty string) for a TOML without a ``variants``
            table.
    """
    table = data.get("variants")
    short_name = data["short_name"]
    requested = variants.get(short_name)
    if table is None:
        if requested is not None:
            raise UnknownHarnessVariantError(
                f"Unknown variant '{requested}' for harness '{short_name}': it declares no variants"
            )
        return data
    name = data["default_variant"] if requested is None else requested
    if name not in table:
        valid = ", ".join(sorted(table))
        raise UnknownHarnessVariantError(
            f"Unknown variant '{name}' for harness '{short_name}'. Valid: {valid}"
        )
    base = {k: v for k, v in data.items() if k not in ("variants", "default_variant")}
    return {**base, **table[name], "variant": name}


def load_harness_registry(variants: Mapping[str, str] | None = None) -> list[HarnessEntry]:
    """Load all harness definitions from package data.

    Args:
        variants: The ``harness_variant`` setting, ``{short_name: variant name}``.
            A harness with variants and no entry here uses its ``default_variant``.
            A key naming no known harness is ignored with a warning.

    Returns:
        List of HarnessEntry objects, one per TOML file, each flattened to its variant.

    Raises:
        UnknownHarnessVariantError: A variant name is not declared by its harness; the
            message names the harness, the requested variant and the valid ones.
    """
    variants = variants or {}
    entries: list[HarnessEntry] = []
    harness_dir = importlib.resources.files("synth_acp.data.harnesses")
    for item in harness_dir.iterdir():
        if hasattr(item, "name") and item.name.endswith(".toml"):
            data = tomllib.loads(item.read_text())
            entries.append(HarnessEntry.model_validate(_resolve_variant(data, variants)))
    known = {e.short_name for e in entries}
    for key in variants:
        if key not in known:
            log.warning("harness_variant names unknown harness '%s'; ignored", key)
    return entries
