"""SessionPickerScreen — modal for selecting a session to restore."""

from __future__ import annotations

import contextlib
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Resize
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import DataTable, Input, Label

if TYPE_CHECKING:
    from synth_acp.embeddings import EmbeddingEngine

_COL_TIME_W = 12
_COL_DIR_W = 15
_COL_AGENTS_W = 25

# Regex for splitting camelCase: insert boundary before uppercase letter
# that follows a lowercase letter, or before an uppercase followed by lowercase
# after another uppercase (e.g., "SHScience" → "SH", "Science")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Separators: dash, underscore, dot, slash, whitespace
_SEP_RE = re.compile(r"[-_./\s]+")


def _patch_tqdm() -> None:
    """Replace tqdm with a no-op wrapper to avoid multiprocessing RLock crash in Textual.

    tqdm's constructor creates a multiprocessing.RLock even when disabled,
    which fails on Python 3.14 inside Textual's async event loop due to
    bad file descriptors in the resource tracker.
    """
    if "tqdm" in sys.modules and hasattr(sys.modules["tqdm"], "_is_synth_noop"):
        return

    class _NoOpTqdm:
        def __init__(self, iterable=None, *_args, **_kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable) if self._iterable is not None else iter([])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def update(self, _n=1):
            pass

        def close(self):
            pass

        def set_description(self, *_args, **_kwargs):
            pass

    import types

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = _NoOpTqdm  # type: ignore[attr-defined]
    fake_tqdm._is_synth_noop = True  # type: ignore[attr-defined]
    fake_auto = types.ModuleType("tqdm.auto")
    fake_auto.tqdm = _NoOpTqdm  # type: ignore[attr-defined]
    fake_asyncio = types.ModuleType("tqdm.asyncio")
    fake_asyncio.tqdm_asyncio = _NoOpTqdm  # type: ignore[attr-defined]
    fake_asyncio.tqdm = _NoOpTqdm  # type: ignore[attr-defined]

    sys.modules["tqdm"] = fake_tqdm
    sys.modules["tqdm.auto"] = fake_auto
    sys.modules["tqdm.asyncio"] = fake_asyncio
    sys.modules["tqdm.std"] = fake_tqdm


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25: split separators, camelCase, lowercase."""
    parts = _SEP_RE.split(text)
    tokens = []
    for part in parts:
        sub_parts = _CAMEL_RE.split(part)
        for sp in sub_parts:
            if sp:
                tokens.append(sp.lower())
    return tokens


def _build_bm25_text(session: dict) -> str:
    """Build searchable text for a session, with agent IDs tokenized."""
    parts: list[str] = []
    for agent in session.get("agents", []):
        parts.append(" ".join(_tokenize_for_bm25(agent)))
    cwd = session.get("cwd")
    if cwd:
        parts.append(" ".join(_tokenize_for_bm25(Path(cwd).name)))
    for task in session.get("tasks", []):
        parts.append(task)
    for msg in session.get("first_messages", []):
        parts.append(msg)
    return "\n".join(parts)


def _relative_time(ms_timestamp: int) -> str:
    """Convert a millisecond epoch timestamp to a human-readable relative string."""
    delta = int(time.time()) - (ms_timestamp // 1000)
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = delta // 60
        return f"{m}m ago"
    if delta < 86400:
        h = delta // 3600
        return f"{h}h ago"
    d = delta // 86400
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


def _rrf_fuse(
    semantic_ranks: dict[str, int],
    bm25_ranks: dict[str, int],
    n_sessions: int,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over two rank lists."""
    all_ids = set(semantic_ranks) | set(bm25_ranks)
    scores: dict[str, float] = {}
    for sid in all_ids:
        sem_rank = semantic_ranks.get(sid, n_sessions)
        bm25_rank = bm25_ranks.get(sid, n_sessions)
        scores[sid] = 1.0 / (k + sem_rank) + 1.0 / (k + bm25_rank)
    return scores


class SessionPickerScreen(ModalScreen[str | None]):
    """Modal with search input + DataTable for session selection."""

    DEFAULT_CSS = ""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Close")]

    def __init__(
        self,
        sessions: list[dict],
        db_path: Path,
        engine: EmbeddingEngine | None,
        indexing_complete: bool,
    ) -> None:
        super().__init__()
        self._sessions = [s for s in sessions if s.get("first_messages")]
        self._db_path = db_path
        self._engine = engine
        self._indexing_complete = indexing_complete
        self._search_timer: Timer | None = None
        self._embeddings: tuple[list[str], object] | None = None
        self._bm25: object | None = None  # bm25s.BM25 instance
        self._bm25_session_ids: list[str] = []
        self._row_keys: list[str] = []
        self._msg_col_width: int = 40
        self._last_query: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-container"):
            yield Label("Restore Session", id="picker-title")
            yield Input(placeholder="Search sessions...", id="search-input")
            table = DataTable(id="session-table", cursor_type="row", zebra_stripes=True)
            table.add_column("Time", width=_COL_TIME_W)
            table.add_column("Directory", width=_COL_DIR_W)
            table.add_column("Agents", width=_COL_AGENTS_W, key="agents")
            table.add_column("Message", width=self._msg_col_width, key="msg")
            yield table

    def on_mount(self) -> None:
        """Populate the initial table."""
        self._do_search("")

    def on_resize(self, event: Resize) -> None:  # noqa: ARG002
        """Recalculate Message column width when container is sized."""
        container = self.query_one("#picker-container")
        # Available width minus padding (2*2), border (2), other columns, separators
        available = container.size.width - 6 - _COL_TIME_W - _COL_DIR_W - _COL_AGENTS_W - 3
        new_width = max(20, available)
        if new_width != self._msg_col_width:
            self._msg_col_width = new_width
            table = self.query_one("#session-table", DataTable)
            table.columns["msg"].width = new_width
            self._row_keys.clear()
            self._do_search(self._last_query)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Debounce search input (50ms timer)."""
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(0.05, lambda: self._do_search(event.value))

    def _do_search(self, query: str) -> None:
        """Rank sessions by hybrid BM25+semantic or fallback methods."""
        self._last_query = query

        if not query:
            ranked = sorted(self._sessions, key=lambda s: s["last_active"], reverse=True)
        else:
            ranked = self._hybrid_rank(query)

        # Skip re-render if results unchanged
        new_ids = [s["session_id"] for s in ranked]
        if new_ids == self._row_keys:
            return

        table = self.query_one("#session-table", DataTable)
        table.clear()
        self._row_keys = new_ids

        for s in ranked:
            when = _relative_time(s["last_active"])
            cwd = Path(s.get("cwd", "") or "").name or ""
            agents = ", ".join(s.get("agents", []))
            messages = s.get("first_messages", [])
            msg = (messages[0][:200] + "…") if messages and len(messages[0]) > 200 else (messages[0] if messages else "")
            table.add_row(when, cwd, agents, msg, height=None)

    def _hybrid_rank(self, query: str) -> list[dict]:
        """Rank using RRF fusion of semantic + BM25, with graceful fallback."""
        n = len(self._sessions)
        has_semantic = self._engine is not None and self._indexing_complete
        has_bm25 = self._ensure_bm25()

        semantic_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        if has_semantic:
            semantic_ranks = self._get_semantic_ranks(query)

        if has_bm25:
            bm25_ranks = self._get_bm25_ranks(query)

        if semantic_ranks or bm25_ranks:
            scores = _rrf_fuse(semantic_ranks, bm25_ranks, n)
            session_map = {s["session_id"]: s for s in self._sessions}
            ranked_ids = sorted(scores, key=lambda sid: scores[sid], reverse=True)
            matched = [session_map[sid] for sid in ranked_ids if sid in session_map]
            # Append non-matching sessions sorted by recency
            matched_set = set(scores)
            rest = sorted(
                [s for s in self._sessions if s["session_id"] not in matched_set],
                key=lambda s: s["last_active"],
                reverse=True,
            )
            return matched + rest
        return self._substring_filter(query, self._sessions)

    def _ensure_bm25(self) -> bool:
        """Lazily build the BM25 index. Returns True if available."""
        if self._bm25 is not None:
            return True
        if not self._sessions:
            return False
        try:
            _patch_tqdm()
            import bm25s
            import Stemmer  # type: ignore[import-not-found]
        except ImportError:
            return False

        stemmer = Stemmer.Stemmer("english")
        corpus_texts = [_build_bm25_text(s) for s in self._sessions]
        self._bm25_session_ids = [s["session_id"] for s in self._sessions]

        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self._bm25 = retriever
        return True

    def _get_bm25_ranks(self, query: str) -> dict[str, int]:
        """Query BM25 index and return session_id → rank mapping."""
        import bm25s
        import Stemmer  # type: ignore[import-not-found]

        stemmer = Stemmer.Stemmer("english")
        retriever = self._bm25

        query_text = " ".join(_tokenize_for_bm25(query))
        query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=stemmer)

        k = min(len(self._bm25_session_ids), 100)
        if k == 0:
            return {}

        results, scores = retriever.retrieve(query_tokens, k=k)  # type: ignore[union-attr]

        ranks: dict[str, int] = {}
        for rank_idx in range(results.shape[1]):
            doc_idx = results[0, rank_idx]
            score = scores[0, rank_idx]
            if score > 0:
                ranks[self._bm25_session_ids[doc_idx]] = rank_idx
        return ranks

    def _get_semantic_ranks(self, query: str) -> dict[str, int]:
        """Get semantic similarity ranks for all sessions."""
        if self._embeddings is None:
            self._embeddings = self._load_embeddings()
        session_ids, matrix = self._embeddings
        if matrix is None or len(session_ids) == 0:
            return {}

        query_emb = self._engine.embed(query)  # type: ignore[union-attr]
        scores = self._engine.similarity(query_emb, matrix)  # type: ignore[union-attr]

        scored = sorted(zip(session_ids, scores, strict=True), key=lambda x: x[1], reverse=True)

        threshold = 0.1
        ranks: dict[str, int] = {}
        for rank, (sid, score) in enumerate(scored):
            if score >= threshold:
                ranks[sid] = rank
        return ranks

    def _load_embeddings(self) -> tuple[list[str], object | None]:
        """Load embeddings from DB, deserialize, build matrix."""
        import numpy as np

        from synth_acp.db import load_all_embeddings_sync

        with contextlib.closing(sqlite3.connect(str(self._db_path))) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            rows = load_all_embeddings_sync(conn)

        if not rows:
            return ([], None)

        session_ids = [r[0] for r in rows]
        vectors = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        matrix = np.stack(vectors)
        return (session_ids, matrix)

    def _substring_filter(self, query: str, sessions: list[dict]) -> list[dict]:
        """Case-insensitive substring match against metadata text."""
        q = query.lower()
        results = []
        for s in sessions:
            text = " ".join([
                s.get("session_id", ""),
                " ".join(s.get("agents", [])),
                " ".join(s.get("tasks", [])),
                s.get("cwd", "") or "",
                " ".join(s.get("first_messages", [])),
            ]).lower()
            if q in text:
                results.append(s)
        return results

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Dismiss with the selected session_id."""
        row_idx = event.cursor_row
        if 0 <= row_idx < len(self._row_keys):
            self.dismiss(self._row_keys[row_idx])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
