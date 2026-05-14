"""Tests for synth_acp.ui.file_discovery."""

from __future__ import annotations

import subprocess
from pathlib import Path

from synth_acp.ui.file_discovery import (
    FileEntry,
    discover_files,
    estimate_tokens,
    filter_files,
    fuzzy_score,
)


class TestDiscoverFilesGitRepo:
    async def test_returns_tracked_and_untracked_files(self, tmp_path: Path) -> None:
        """Git repo returns both committed and untracked-not-ignored files."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )

        # Tracked file
        (tmp_path / "tracked.py").write_text("x = 1")
        subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True,
        )

        # Untracked file (not ignored)
        (tmp_path / "untracked.txt").write_text("hello")

        # Ignored file
        (tmp_path / ".gitignore").write_text("ignored.log\n")
        (tmp_path / "ignored.log").write_text("secret")

        result = await discover_files(tmp_path)
        rel_paths = [e.rel_path for e in result]

        assert "tracked.py" in rel_paths
        assert "untracked.txt" in rel_paths
        assert ".gitignore" in rel_paths
        assert "ignored.log" not in rel_paths

    async def test_returns_correct_size(self, tmp_path: Path) -> None:
        """FileEntry.size_bytes matches actual file size."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "a.txt").write_text("12345")
        subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, capture_output=True, check=True)

        result = await discover_files(tmp_path)
        entry = next(e for e in result if e.rel_path == "a.txt")
        assert entry.size_bytes == 5


class TestDiscoverFilesFallback:
    async def test_non_git_dir_uses_rglob(self, tmp_path: Path) -> None:
        """Non-git directory falls back to rglob, excluding standard ignore dirs."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("pkg")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")

        result = await discover_files(tmp_path)
        rel_paths = [e.rel_path for e in result]

        assert "src/main.py" in rel_paths
        assert "node_modules/pkg.js" not in rel_paths
        assert "__pycache__/mod.pyc" not in rel_paths


class TestDiscoverFilesSorted:
    async def test_results_sorted_alphabetically(self, tmp_path: Path) -> None:
        """Results are sorted by rel_path regardless of filesystem order."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "m.txt").write_text("m")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)

        result = await discover_files(tmp_path)
        rel_paths = [e.rel_path for e in result]
        assert rel_paths == sorted(rel_paths)


class TestFuzzyScore:
    def test_no_match_returns_none(self) -> None:
        """Non-subsequence query returns None."""
        assert fuzzy_score("xyz", "abc/def.py") is None

    def test_contiguous_bonus(self) -> None:
        """Contiguous matches score higher than scattered."""
        # Both match in basename — contiguous "abc" vs scattered "a_b_c"
        assert fuzzy_score("abc", "src/abc.py") > fuzzy_score("abc", "src/a_b_c.py")  # type: ignore[operator]

    def test_basename_bonus(self) -> None:
        """Basename match scores higher than directory match."""
        assert fuzzy_score("foo", "src/foo.py") > fuzzy_score("foo", "foobar/other.py")  # type: ignore[operator]

    def test_separator_boundary_bonus(self) -> None:
        """Match at path separator boundary scores higher than mid-segment."""
        # 's' at position 0 (start of path) vs 's' at position 2 (mid-segment 'abs')
        assert fuzzy_score("s", "src/file.py") > fuzzy_score("s", "abs/file.py")  # type: ignore[operator]


class TestFilterFiles:
    def test_returns_top_n(self) -> None:
        """Returns at most limit results."""
        files = [FileEntry(f"file{i}.py", 100) for i in range(20)]
        result = filter_files("file", files, limit=5)
        assert len(result) == 5

    def test_sorted_by_score_desc(self) -> None:
        """First result has highest fuzzy_score."""
        files = [
            FileEntry("src/utils/helper.py", 100),
            FileEntry("src/main.py", 100),
            FileEntry("main.txt", 100),
        ]
        result = filter_files("main", files)
        # "main.txt" should rank highest (basename match, contiguous, separator boundary)
        scores = [fuzzy_score("main", e.rel_path) for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_excludes_non_matches(self) -> None:
        """Non-subsequence files are excluded from results."""
        files = [
            FileEntry("abc.py", 100),
            FileEntry("xyz.py", 100),
        ]
        result = filter_files("abc", files)
        assert all(e.rel_path == "abc.py" for e in result)


class TestEstimateTokens:
    def test_small_file(self) -> None:
        """Small files show raw token count."""
        assert estimate_tokens(180) == "~45 tok"

    def test_large_file(self) -> None:
        """Large files show k-suffix."""
        assert estimate_tokens(4800) == "~1.2k tok"

    def test_round_k(self) -> None:
        """Round k values don't show .0 suffix."""
        assert estimate_tokens(4000) == "~1k tok"
