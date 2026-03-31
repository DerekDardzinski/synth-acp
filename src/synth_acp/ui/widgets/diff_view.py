"""Collapsible unified diff view for file edits."""

from __future__ import annotations

import difflib
from pathlib import PurePath

from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Collapsible, Static


def _unified_diff_markup(
    path: str,
    old_text: str | None,
    new_text: str,
    context: int = 3,
) -> str:
    """Generate Rich-markup unified diff output.

    Args:
        path: File path for the diff header.
        old_text: Original file content, or None for new files.
        new_text: Updated file content.
        context: Number of context lines around changes.

    Returns:
        Rich markup string with color-coded diff lines.
    """
    old = (old_text or "").splitlines(keepends=True)
    new = new_text.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(old, new, fromfile=path, tofile=path, n=context))

    if not diff_lines:
        return "[dim](no changes)[/dim]"

    parts: list[str] = []
    for line in diff_lines:
        text = line.rstrip("\n").replace("[", "\\[")
        if line.startswith("---") or line.startswith("+++"):
            parts.append(f"[bold]{text}[/bold]")
        elif line.startswith("@@"):
            parts.append(f"[$primary]{text}[/$primary]")
        elif line.startswith("+"):
            parts.append(f"[$success]{text}[/$success]")
        elif line.startswith("-"):
            parts.append(f"[$error]{text}[/$error]")
        else:
            parts.append(f"[dim]{text}[/dim]")
    return "\n".join(parts)


class DiffView(Vertical):
    """Collapsible unified diff display for a single file.

    Args:
        path: File path being diffed.
        old_text: Original content, or None for new files.
        new_text: Updated content.
        collapsed: Whether the collapsible starts collapsed.
    """

    def __init__(
        self,
        path: str,
        old_text: str | None,
        new_text: str,
        *,
        collapsed: bool = True,
    ) -> None:
        super().__init__()
        self._path = path
        self._old_text = old_text
        self._new_text = new_text
        self._collapsed = collapsed

    def compose(self):
        """Compose the collapsible diff widget."""
        markup = _unified_diff_markup(self._path, self._old_text, self._new_text)
        with Collapsible(
            title=PurePath(self._path).name, collapsed=self._collapsed
        ), ScrollableContainer(classes="diff-scroll"):
            yield Static(markup)
