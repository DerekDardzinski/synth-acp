"""Per-harness-engine ACP extensions and their explicit registry.

A harness TOML names its extension (``extension = "kiro_v2"``); ``get_extension`` maps
that name to the one shared, stateless instance.  The registry is an explicit dict
rather than entry points so a typo in a TOML fails loudly at lookup.
"""

from __future__ import annotations

from synth_acp.acp.extensions.base import AcpExtension
from synth_acp.acp.extensions.claude import ClaudeExtension
from synth_acp.acp.extensions.codex import CodexExtension
from synth_acp.acp.extensions.kiro_v2 import KiroV2Extension
from synth_acp.acp.extensions.kiro_v3 import KiroV3Extension

__all__ = [
    "AcpExtension",
    "UnknownExtensionError",
    "get_extension",
    "register_extension",
]


class UnknownExtensionError(KeyError):
    """No extension is registered under the requested name."""


_REGISTRY: dict[str, AcpExtension] = {}


def register_extension(cls: type[AcpExtension]) -> None:
    """Add cls to the registry under cls.name.

    Raises:
        ValueError: An extension is already registered under that name.
    """
    if cls.name in _REGISTRY:
        raise ValueError(f"ACP extension '{cls.name}' is already registered")
    _REGISTRY[cls.name] = cls()


def get_extension(name: str) -> AcpExtension:
    """Return the shared instance registered under name.

    Raises:
        UnknownExtensionError: Nothing is registered under name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownExtensionError(f"Unknown ACP extension '{name}'. Known: {known}") from None


register_extension(AcpExtension)
register_extension(ClaudeExtension)
register_extension(CodexExtension)
register_extension(KiroV2Extension)
register_extension(KiroV3Extension)
