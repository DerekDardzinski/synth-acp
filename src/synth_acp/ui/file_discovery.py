"""File discovery and fuzzy scoring for @ file references."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import NamedTuple

_IGNORE_DIRS = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".cache", "target"}
)


class FileEntry(NamedTuple):
    """A discovered project file."""

    rel_path: str  # POSIX-normalized (forward slashes on all platforms)
    size_bytes: int


async def discover_files(cwd: Path) -> list[FileEntry]:
    """List project files respecting gitignore. Thread-safe, runs git subprocess.

    Returns files sorted alphabetically by rel_path.
    rel_path is always POSIX-normalized (forward slashes) regardless of OS.
    Uses git ls-files for git repos, Path.rglob with standard ignores as fallback.
    """
    return await asyncio.to_thread(_discover_files_sync, cwd)


def _discover_files_sync(cwd: Path) -> list[FileEntry]:
    """Synchronous file discovery implementation."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        paths = [p for p in tracked + untracked if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return _fallback_discover(cwd)

    entries: list[FileEntry] = []
    for rel in paths:
        try:
            size = (cwd / rel).stat().st_size
        except OSError:
            continue
        entries.append(FileEntry(rel_path=rel, size_bytes=size))
    entries.sort(key=lambda e: e.rel_path)
    return entries


def _fallback_discover(cwd: Path) -> list[FileEntry]:
    """Fallback discovery using Path.rglob when git is unavailable."""
    entries: list[FileEntry] = []
    for p in cwd.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _IGNORE_DIRS for part in p.relative_to(cwd).parts):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        entries.append(FileEntry(rel_path=p.relative_to(cwd).as_posix(), size_bytes=size))
    entries.sort(key=lambda e: e.rel_path)
    return entries


def _char_class(ch: str) -> int:
    """Classify character: 0=lower, 1=upper, 2=separator, 3=digit."""
    if ch in "/-_. \t":
        return 2
    if ch.isupper():
        return 1
    if ch.isdigit():
        return 3
    return 0


def fuzzy_score(query: str, path: str) -> int | None:
    """Score a path against a query using DP over all match positions.

    Uses fzf-style optimal alignment: considers all valid subsequence
    positions and picks the highest-scoring one. Rewards contiguous runs,
    word boundary matches, and basename matches.

    Returns None if query is not a subsequence of path.
    Higher score = better match.

    Time: O(M*N), Space: O(N) where M=len(query), N=len(path).
    """
    if not query:
        return 0
    m = len(query)
    n = len(path)
    if m > n:
        return None

    q = query.lower()
    p = path.lower()

    # Scoring constants
    score_match = 16
    bonus_boundary = 8
    bonus_camel = 7
    bonus_basename = 14
    bonus_consecutive = 4

    # Precompute per-position bonus
    pos_bonus = [0] * n
    prev_class = 2  # start-of-string treated as separator
    last_sep = path.rfind("/")
    for i, ch in enumerate(path):
        curr_class = _char_class(ch)
        if prev_class == 2:
            pos_bonus[i] = bonus_boundary
        elif prev_class == 0 and curr_class == 1:
            pos_bonus[i] = bonus_camel
        if i > last_sep:
            pos_bonus[i] += bonus_basename
        prev_class = curr_class

    # DP with two rolling rows + run length tracking
    neg_inf = float("-inf")
    prev_d = [neg_inf] * n  # best score ending with match at j
    prev_m = [neg_inf] * n  # best score considering path[0..j]
    prev_run = [0] * n  # consecutive run length ending at j

    for i in range(m):
        curr_d = [neg_inf] * n
        curr_m = [neg_inf] * n
        curr_run = [0] * n
        for j in range(i, n):
            if q[i] != p[j]:
                curr_m[j] = curr_m[j - 1] if j > 0 else neg_inf
                continue

            sc = score_match + pos_bonus[j]
            if i == 0:
                curr_d[j] = sc
                curr_run[j] = 1
            else:
                consec = neg_inf
                if j > 0 and prev_d[j - 1] != neg_inf:
                    run_len = prev_run[j - 1]
                    consec = prev_d[j - 1] + sc + bonus_consecutive * (run_len + 1)
                non_consec = (prev_m[j - 1] + sc) if j > 0 else neg_inf
                if consec >= non_consec:
                    curr_d[j] = consec
                    curr_run[j] = (prev_run[j - 1] + 1) if j > 0 else 1
                else:
                    curr_d[j] = non_consec
                    curr_run[j] = 1

            curr_m[j] = max(curr_m[j - 1] if j > 0 else neg_inf, curr_d[j])

        prev_d = curr_d
        prev_m = curr_m
        prev_run = curr_run

    result = prev_m[n - 1]
    return result if result != neg_inf else None


def filter_files(query: str, files: list[FileEntry], limit: int = 15) -> list[FileEntry]:
    """Filter and rank files by fuzzy score. Returns top limit results.

    Tiebreaker: shorter paths win (closer to root = more likely target).
    """
    scored = []
    for entry in files:
        s = fuzzy_score(query, entry.rel_path)
        if s is not None:
            scored.append((s, entry))
    scored.sort(key=lambda x: (-x[0], len(x[1].rel_path)))
    return [entry for _, entry in scored[:limit]]


def estimate_tokens(size_bytes: int) -> str:
    """Human-readable token estimate. E.g. '~1.2k tok', '~45 tok'."""
    tokens = size_bytes // 4
    if tokens >= 1000:
        k = tokens / 1000
        formatted = f"{k:.1f}".rstrip("0").rstrip(".")
        return f"~{formatted}k tok"
    return f"~{tokens} tok"
