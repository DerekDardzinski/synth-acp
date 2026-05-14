"""SessionPickerScreen — modal for selecting a session to restore."""

from __future__ import annotations

import contextlib
import sqlite3
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
            table.add_column("Message", width=self._msg_col_width, key="msg")
            yield table

    def on_mount(self) -> None:
        """Populate the initial table."""
        self._do_search("")

    def on_resize(self, event: Resize) -> None:
        """Recalculate Message column width when container is sized."""
        container = self.query_one("#picker-container")
        # Available width minus padding (2*2), border (2), other columns, separators
        available = container.size.width - 6 - _COL_TIME_W - _COL_DIR_W - 2
        new_width = max(20, available)
        if new_width != self._msg_col_width:
            self._msg_col_width = new_width
            table = self.query_one("#session-table", DataTable)
            table.columns["msg"].width = new_width
            # Re-populate to recalculate row heights with new width
            self._row_keys.clear()
            self._do_search(self._last_query)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Debounce search input (50ms timer)."""
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(0.05, lambda: self._do_search(event.value))

    def _do_search(self, query: str) -> None:
        """Rank sessions by semantic similarity or substring match."""
        self._last_query = query

        if not query:
            ranked = sorted(self._sessions, key=lambda s: s["last_active"], reverse=True)
        elif self._engine and self._indexing_complete:
            ranked = self._semantic_rank(query)
        else:
            ranked = self._substring_filter(query, self._sessions)

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
            messages = s.get("first_messages", [])
            msg = (messages[0][:200] + "…") if messages and len(messages[0]) > 200 else (messages[0] if messages else "")
            table.add_row(when, cwd, msg, height=None)

    def _semantic_rank(self, query: str) -> list[dict]:
        """Rank sessions by cosine similarity to query embedding."""
        if self._embeddings is None:
            self._embeddings = self._load_embeddings()
        session_ids, matrix = self._embeddings
        if matrix is None or len(session_ids) == 0:
            return self._substring_filter(query, self._sessions)

        query_emb = self._engine.embed(query)  # type: ignore[union-attr]
        scores = self._engine.similarity(query_emb, matrix)  # type: ignore[union-attr]

        score_map = dict(zip(session_ids, scores, strict=True))

        # Split into relevant (above threshold) and rest
        threshold = 0.25
        relevant = []
        rest = []
        for s in self._sessions:
            score = score_map.get(s["session_id"], -2.0)
            if score >= threshold:
                relevant.append((score, s))
            else:
                rest.append(s)

        relevant.sort(key=lambda x: x[0], reverse=True)
        rest.sort(key=lambda s: s["last_active"], reverse=True)
        return [s for _, s in relevant] + rest

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
