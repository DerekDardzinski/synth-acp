"""Tests for synth_acp.db embedding helpers and session retention."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from synth_acp.db import (
    BUSY_TIMEOUT_MS,
    RETENTION_DAYS,
    SCHEMA,
    AgentRenameError,
    ExpiryReport,
    configure_connection,
    ensure_schema_sync,
    expire_old_sessions_sync,
    get_unembedded_agents_sync,
    load_all_embeddings_sync,
    rename_agent_sync,
    session_last_activity_sync,
    store_embedding_sync,
)

DAY_MS = 86400 * 1000

# All retention tests pin one clock so that seeding and expiry cannot straddle a
# second boundary — the exactly-60-days boundary case is otherwise flaky.
_FIXED_NOW = 1_800_000_000


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> int:
    """Pin synth_acp.db's clock to _FIXED_NOW and return it."""
    import synth_acp.db as db_module

    monkeypatch.setattr(db_module.time, "time", lambda: float(_FIXED_NOW))
    return _FIXED_NOW


def _ms_ago(days: float) -> int:
    """Stored-millisecond timestamp *days* before the pinned clock."""
    return int(_FIXED_NOW * 1000 - days * DAY_MS)


def _seed_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    age_days: float,
    status: str = "inactive",
    ui_events: int = 2,
) -> None:
    """Insert one agent plus *ui_events* journal rows, all *age_days* old."""
    ts = _ms_ago(age_days)
    conn.execute(
        "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
        (f"{session_id}-a1", session_id, status, ts),
    )
    for seq in range(ui_events):
        conn.execute(
            "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, f"{session_id}-a1", seq, "chunk", "{}", ts),
        )
    conn.commit()


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count for every table retention touches."""
    tables = ("ui_events", "messages", "agent_commands", "session_embeddings", "agents")
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema_sync(conn)
    return conn


class TestSessionEmbeddingsSchema:
    def test_ensure_schema_creates_session_embeddings_table(self) -> None:
        conn = _make_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_embeddings'"
        ).fetchall()
        assert tables == [("session_embeddings",)]

    def test_migration_drops_old_embeddings_table(self) -> None:
        """Old single-PK schema is detected and recreated with agent_id."""
        conn = sqlite3.connect(":memory:")
        # Create old schema without agent_id
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                registered INTEGER NOT NULL,
                parent TEXT,
                task TEXT,
                acp_session_id TEXT,
                harness TEXT,
                agent_mode TEXT,
                cwd TEXT,
                PRIMARY KEY (agent_id, session_id)
            );
            CREATE TABLE IF NOT EXISTS session_embeddings (
                session_id TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO session_embeddings (session_id, text_hash, embedding, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("sess-1", "hash-1", b"\x00" * 1536, 1000),
        )
        conn.commit()
        # Run migration
        ensure_schema_sync(conn)
        # Verify new schema has agent_id column
        cur = conn.execute("PRAGMA table_info(session_embeddings)")
        col_names = {row[1] for row in cur.fetchall()}
        assert "agent_id" in col_names
        # Old data is gone (table was dropped and recreated)
        rows = conn.execute("SELECT * FROM session_embeddings").fetchall()
        assert rows == []

    def test_migration_preserves_new_schema(self) -> None:
        """Migration is idempotent on already-migrated DBs."""
        conn = _make_conn()
        # Insert a row with new schema
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-1", b"\x00" * 1536)
        # Re-run ensure_schema_sync (simulates restart)
        ensure_schema_sync(conn)
        # Row should survive
        result = load_all_embeddings_sync(conn)
        assert len(result) == 1
        assert result[0] == ("sess-1", "agent-1", b"\x00" * 1536)


class TestEmbeddingCRUD:
    def test_store_and_load_with_agent_id(self) -> None:
        """Roundtrip with agent_id returns correct tuple."""
        conn = _make_conn()
        blob = b"\x00" * 1536
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-abc", blob)
        result = load_all_embeddings_sync(conn)
        assert result == [("sess-1", "agent-1", blob)]

    def test_store_embedding_upserts_on_same_key(self) -> None:
        """Second store with same (session_id, agent_id) replaces, not duplicates."""
        conn = _make_conn()
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-1", b"\x01" * 1536)
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-2", b"\x02" * 1536)
        result = load_all_embeddings_sync(conn)
        assert len(result) == 1
        assert result[0] == ("sess-1", "agent-1", b"\x02" * 1536)

    def test_get_unembedded_agents_returns_missing_pairs(self) -> None:
        """Only (session_id, agent_id) pairs without embeddings are returned."""
        conn = _make_conn()
        # Insert two agents in same session
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a1", "sess-1", "restorable", 1000),
        )
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a2", "sess-1", "restorable", 2000),
        )
        conn.commit()
        # Embed only a1
        store_embedding_sync(conn, "sess-1", "a1", "hash-1", b"\x00" * 1536)
        # Only a2 should be returned
        result = get_unembedded_agents_sync(conn)
        assert result == [("sess-1", "a2")]

    def test_load_all_embeddings_ordered_by_session_id(self) -> None:
        """Results are ordered by session_id for reduceat grouping."""
        conn = _make_conn()
        # Insert in reverse order
        store_embedding_sync(conn, "sess-b", "a1", "h1", b"\x01" * 1536)
        store_embedding_sync(conn, "sess-a", "a1", "h2", b"\x02" * 1536)
        result = load_all_embeddings_sync(conn)
        assert result[0][0] == "sess-a"
        assert result[1][0] == "sess-b"


class _RecordingConnection(sqlite3.Connection):
    """Connection that records every statement it executes."""

    statements: list[str]

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        self.statements.append(sql)
        return super().execute(sql, *args, **kwargs)


def _recording_conn() -> _RecordingConnection:
    conn = sqlite3.connect(":memory:", factory=_RecordingConnection)
    conn.statements = []
    ensure_schema_sync(conn)
    conn.statements.clear()
    return conn


class _FailAfterNDeletes(sqlite3.Connection):
    """Connection that raises on the Nth DELETE, to test transaction atomicity."""

    fail_on_delete: int
    deletes: int

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if sql.lstrip().upper().startswith("DELETE"):
            self.deletes += 1
            if self.deletes == self.fail_on_delete:
                raise sqlite3.OperationalError("injected failure")
        return super().execute(sql, *args, **kwargs)


class _FailOnCommit(sqlite3.Connection):
    """Connection whose commit raises once armed, to test rollback on commit failure."""

    fail_commit: bool = False

    def commit(self) -> None:
        if self.fail_commit:
            raise sqlite3.OperationalError("injected commit failure")
        super().commit()


class TestSessionLastActivity:
    """Tests for session_last_activity_sync."""

    def test_activity_is_max_over_sources_in_epoch_seconds(self, frozen_now: int) -> None:
        """The newest of ui_events/messages/agent_commands/agents wins, converted ms -> s."""
        conn = _make_conn()
        _seed_session(conn, "sess-1", age_days=61)
        newest_ms = _ms_ago(3)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("sess-1", "a1", "a2", "hi", newest_ms),
        )
        conn.commit()

        activity = session_last_activity_sync(conn)

        assert activity == {"sess-1": newest_ms // 1000}

    def test_embeddings_are_not_an_activity_source(self, frozen_now: int) -> None:
        """A fresh derived embedding must not raise a session's last activity."""
        conn = _make_conn()
        _seed_session(conn, "sess-1", age_days=61)
        store_embedding_sync(conn, "sess-1", "sess-1-a1", "h", b"\x00" * 1536)

        assert session_last_activity_sync(conn) == {"sess-1": _ms_ago(61) // 1000}


class TestExpireOldSessions:
    """Tests for expire_old_sessions_sync."""

    def test_defaults_are_sixty_days_and_dry_run(self, frozen_now: int) -> None:
        """Called with no keyword arguments at all, expiry previews and deletes nothing."""
        assert RETENTION_DAYS == 60

        conn = _make_conn()
        _seed_session(conn, "old", age_days=61)
        before = _row_counts(conn)

        report = expire_old_sessions_sync(conn)

        assert report.dry_run is True
        assert report.sessions_expired == 1
        assert report.ui_events_deleted == 2
        assert _row_counts(conn) == before

    def test_terminated_inactive_agent_session_is_expired(self, frozen_now: int) -> None:
        """REGRESSION: terminate() sets status 'inactive'; those sessions never expired."""
        conn = _make_conn()
        _seed_session(conn, "dead", age_days=61, status="inactive")

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == 1
        assert report.agents_deleted == 1
        assert report.ui_events_deleted == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM ui_events WHERE session_id = 'dead'"
        ).fetchone()[0] == 0

    def test_recent_message_keeps_session_alive(self, frozen_now: int) -> None:
        """A 1-day-old pending message must protect a session with 61-day-old ui_events."""
        conn = _make_conn()
        _seed_session(conn, "sess-1", age_days=61)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("sess-1", "a1", "a2", "still pending", _ms_ago(1)),
        )
        conn.commit()

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == 0
        assert conn.execute(
            "SELECT body FROM messages WHERE session_id = 'sess-1'"
        ).fetchall() == [("still pending",)]
        assert conn.execute(
            "SELECT COUNT(*) FROM ui_events WHERE session_id = 'sess-1'"
        ).fetchone()[0] == 2

    def test_recent_agent_command_keeps_session_alive(self, frozen_now: int) -> None:
        """agent_commands is the second non-obvious activity source."""
        conn = _make_conn()
        _seed_session(conn, "sess-1", age_days=61)
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("sess-1", "a1", "launch", "{}", _ms_ago(1)),
        )
        conn.commit()

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM agent_commands WHERE session_id = 'sess-1'"
        ).fetchone()[0] == 1

    def test_multi_agent_session_kept_alive_by_newest_agent(self, frozen_now: int) -> None:
        """Expiry is per session, not per agent: one fresh agent protects the whole session."""
        conn = _make_conn()
        _seed_session(conn, "sess-1", age_days=90)
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("fresh", "sess-1", "inactive", _ms_ago(2)),
        )
        conn.commit()

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM agents WHERE session_id = 'sess-1'"
        ).fetchone()[0] == 2

    def test_embeddings_do_not_resurrect_session(self, frozen_now: int) -> None:
        """A fresh embedding is derived data and must not save a dead session."""
        conn = _make_conn()
        _seed_session(conn, "dead", age_days=61)
        store_embedding_sync(conn, "dead", "dead-a1", "h", b"\x00" * 1536)
        conn.execute(
            "UPDATE session_embeddings SET created_at = ? WHERE session_id = 'dead'",
            (_ms_ago(1),),
        )
        conn.commit()

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == 1
        assert report.embeddings_deleted == 1
        assert _row_counts(conn) == {
            "ui_events": 0,
            "messages": 0,
            "agent_commands": 0,
            "session_embeddings": 0,
            "agents": 0,
        }

    @pytest.mark.parametrize(
        ("age_seconds", "expected_expired"),
        [(60 * 86400, 0), (60 * 86400 + 1, 1)],
    )
    def test_boundary_is_strictly_greater_than(
        self, frozen_now: int, age_seconds: int, expected_expired: int
    ) -> None:
        """Exactly 60 days is NOT expired; 60 days plus one second IS."""
        conn = _make_conn()
        ts = (_FIXED_NOW - age_seconds) * 1000
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a1", "sess-1", "inactive", ts),
        )
        conn.commit()

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == expected_expired

    def test_cutoff_epoch_is_epoch_seconds(self, frozen_now: int) -> None:
        """cutoff_epoch is epoch SECONDS, not the milliseconds the columns store."""
        conn = _make_conn()

        report = expire_old_sessions_sync(conn)

        assert report.cutoff_epoch == _FIXED_NOW - 60 * 86400

    def test_dry_run_reports_counts_and_deletes_nothing(self, frozen_now: int) -> None:
        """Non-zero counts, zero mutations, across every table."""
        conn = _make_conn()
        _seed_session(conn, "old", age_days=61)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old", "a1", "a2", "b", _ms_ago(61)),
        )
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old", "a1", "launch", "{}", _ms_ago(61)),
        )
        store_embedding_sync(conn, "old", "old-a1", "h", b"\x00" * 1536)
        before = _row_counts(conn)

        report = expire_old_sessions_sync(conn, dry_run=True)

        assert (
            report.ui_events_deleted,
            report.messages_deleted,
            report.agent_commands_deleted,
            report.embeddings_deleted,
            report.agents_deleted,
        ) == (2, 1, 1, 1, 1)
        assert _row_counts(conn) == before

    def test_execute_report_counts_equal_rows_deleted(self, frozen_now: int) -> None:
        """Every report field must equal the rows its own table actually lost."""
        conn = _make_conn()
        _seed_session(conn, "old", age_days=61, ui_events=5)
        _seed_session(conn, "fresh", age_days=1, ui_events=3)
        for i in range(2):
            conn.execute(
                "INSERT INTO messages (session_id, from_agent, to_agent, body, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("old", "a1", "a2", f"b{i}", _ms_ago(61)),
            )
        for i in range(3):
            conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("old", "a1", "launch", f"{i}", _ms_ago(61)),
            )
        # One row per table for the live session, to prove the counters are
        # scoped to the expired session rather than counting everything.
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fresh", "a1", "a2", "keep", _ms_ago(1)),
        )
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fresh", "a1", "launch", "{}", _ms_ago(1)),
        )
        conn.commit()
        store_embedding_sync(conn, "old", "old-a1", "h", b"\x00" * 1536)
        store_embedding_sync(conn, "fresh", "fresh-a1", "h", b"\x00" * 1536)
        before = _row_counts(conn)

        report = expire_old_sessions_sync(conn, dry_run=False)
        after = _row_counts(conn)

        deltas = {table: before[table] - after[table] for table in before}
        assert deltas == {
            "ui_events": 5,
            "messages": 2,
            "agent_commands": 3,
            "session_embeddings": 1,
            "agents": 1,
        }
        assert (
            report.ui_events_deleted,
            report.messages_deleted,
            report.agent_commands_deleted,
            report.embeddings_deleted,
            report.agents_deleted,
        ) == (5, 2, 3, 1, 1)
        assert after == {
            "ui_events": 3,
            "messages": 1,
            "agent_commands": 1,
            "session_embeddings": 1,
            "agents": 1,
        }

    def test_no_bytes_reclaimed_and_dry_run_issues_no_delete_or_ddl(
        self, frozen_now: int
    ) -> None:
        """DELETE does not shrink the file, and a preview must not write at all."""
        assert "bytes_reclaimed" not in ExpiryReport.model_fields

        conn = _recording_conn()
        _seed_session(conn, "old", age_days=61)
        conn.statements.clear()

        expire_old_sessions_sync(conn, dry_run=True)

        forbidden = ("DELETE", "DROP", "CREATE", "ALTER", "INSERT", "UPDATE", "VACUUM")
        offenders = [
            sql
            for sql in conn.statements
            if sql.lstrip().upper().startswith(forbidden)
        ]
        assert offenders == []

    def test_preview_leaves_legacy_embeddings_schema_untouched(
        self, frozen_now: int
    ) -> None:
        """A preview must not run the migration that DROPs legacy session_embeddings."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        conn.executescript("""
            CREATE TABLE session_embeddings (
                session_id  TEXT PRIMARY KEY,
                text_hash   TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  INTEGER NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO session_embeddings (session_id, text_hash, embedding, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("legacy", "h", b"\x00" * 1536, _ms_ago(400)),
        )
        _seed_session(conn, "old", age_days=61)

        report = expire_old_sessions_sync(conn)

        assert report.sessions_expired == 1
        assert conn.execute("SELECT COUNT(*) FROM session_embeddings").fetchone()[0] == 1
        cols = {row[1] for row in conn.execute("PRAGMA table_info(session_embeddings)")}
        assert cols == {"session_id", "text_hash", "embedding", "created_at"}

    def test_failure_mid_session_leaves_session_intact(self, frozen_now: int) -> None:
        """A crash partway through one session must not half-delete it."""
        conn = sqlite3.connect(":memory:", factory=_FailAfterNDeletes)
        conn.fail_on_delete = 3
        conn.deletes = 0
        ensure_schema_sync(conn)
        _seed_session(conn, "old", age_days=61)
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old", "a1", "a2", "b", _ms_ago(61)),
        )
        conn.execute(
            "INSERT INTO agent_commands (session_id, from_agent, command, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old", "a1", "launch", "{}", _ms_ago(61)),
        )
        conn.commit()
        before = _row_counts(conn)

        with pytest.raises(sqlite3.OperationalError, match="injected failure"):
            expire_old_sessions_sync(conn, dry_run=False)

        assert _row_counts(conn) == before

    def test_commit_failure_rolls_back_and_leaves_session_intact(
        self, frozen_now: int
    ) -> None:
        """A failing commit must roll back, not leave the transaction open.

        Without the rollback the rows are already gone on this connection and
        the transaction is still open, so a caller that swallows the error can
        commit the deletion later by accident.
        """
        conn = sqlite3.connect(":memory:", factory=_FailOnCommit)
        ensure_schema_sync(conn)
        _seed_session(conn, "old", age_days=61)
        before = _row_counts(conn)
        conn.fail_commit = True

        with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
            expire_old_sessions_sync(conn, dry_run=False)

        assert conn.in_transaction is False
        conn.fail_commit = False
        assert _row_counts(conn) == before

    def test_session_refreshed_after_discovery_is_not_deleted(
        self, frozen_now: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session that came alive between selection and deletion must survive.

        Simulates the concurrent writer that commits inside the
        discovery-to-delete window by reporting stale activity for a session
        whose rows are actually fresh.
        """
        import synth_acp.db as db_module

        conn = _make_conn()
        _seed_session(conn, "sess-1", age_days=1)
        before = _row_counts(conn)
        monkeypatch.setattr(
            db_module, "session_last_activity_sync", lambda _conn: {"sess-1": _FIXED_NOW - 61 * 86400}
        )

        report = expire_old_sessions_sync(conn, dry_run=False)

        assert report.sessions_expired == 0
        assert report.ui_events_deleted == 0
        assert _row_counts(conn) == before


class TestConnectionConfiguration:
    """The busy_timeout invariant, and the structural guard that keeps it true.

    WAL allows one writer, so a slow write transaction blocks every other writer.
    With Python's 5s default those writers raise OperationalError, uncaught, and one
    of them runs inside an agent's synth-mcp subprocess where it surfaces to the
    agent as a broken tool call rather than as a retry.
    """

    def test_configure_connection_sets_wal_and_busy_timeout(self, tmp_path: Any) -> None:
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        try:
            configure_connection(conn)
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
        finally:
            conn.close()

    def test_busy_timeout_outlasts_the_sqlite3_default(self) -> None:
        """The margin over sqlite3's 5000ms default is the point of the constant, so
        assert it is meaningfully larger rather than merely non-zero."""
        assert BUSY_TIMEOUT_MS >= 30_000

    @staticmethod
    def _opens_connection(scope: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "connect"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "sqlite3"
            for n in ast.walk(scope)
        )

    @staticmethod
    def _configures_connection(scope: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "configure_connection"
            for n in ast.walk(scope)
        )

    def test_every_connect_site_configures_its_connection(self) -> None:
        """Guard: a new sqlite3.connect that skips configure_connection fails here.

        Without it the invariant decays silently -- an unconfigured connection behaves
        identically until the day another writer holds the lock past the default 5s.
        """
        src = Path(__file__).resolve().parents[1] / "src" / "synth_acp"
        offenders: list[str] = []
        for path in sorted(src.rglob("*.py")):
            for scope in ast.walk(ast.parse(path.read_text())):
                if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if self._opens_connection(scope) and not self._configures_connection(scope):
                    offenders.append(f"{path.relative_to(src)}:{scope.lineno} {scope.name}")

        assert offenders == [], (
            "these functions open a sqlite3 connection without calling "
            f"configure_connection: {offenders}"
        )

    def test_guard_detects_an_unconfigured_connect(self) -> None:
        """A guard that cannot fail is not a guard.  Feed it a function that opens a
        connection without configuring it and confirm both halves fire."""
        scope = ast.parse(
            "def leaky(p):\n"
            "    conn = sqlite3.connect(p)\n"
            "    return conn.execute('SELECT 1')\n"
        ).body[0]
        assert self._opens_connection(scope)
        assert not self._configures_connection(scope)


OLD = "worker"
NEW = "worker.h0000dead"
SID = "sess-abc"
OTHER_SID = "sess-other"


def _seed_handoff(conn: sqlite3.Connection, *, session_id: str = SID) -> None:
    """Seed one agent with a child and one row in every agent-referencing column.

    Every status bucket that the classification rules distinguish is represented, so a
    single rename call can be checked against all three rules at once.
    """
    conn.execute(
        "INSERT INTO agents "
        "(agent_id, session_id, status, registered, parent, task, harness, agent_mode, cwd, "
        "acp_session_id) "
        "VALUES (?, ?, 'active', 100, 'boss', 'do the thing', 'kiro', 'plan', '/tmp/wd', 'acp-1')",
        (OLD, session_id),
    )
    conn.execute(
        "INSERT INTO agents (agent_id, session_id, status, registered, parent) "
        "VALUES ('kid', ?, 'active', 101, ?)",
        (session_id, OLD),
    )
    for status in ("pending", "delivered", "expired"):
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 100)",
            (session_id, "kid", OLD, f"to-{status}", status),
        )
        conn.execute(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 100)",
            (session_id, OLD, "kid", f"from-{status}", status),
        )
    for status in ("pending", "processing", "processed", "rejected"):
        conn.execute(
            "INSERT INTO agent_commands "
            "(session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, ?, 'terminate', '{}', ?, 100)",
            (session_id, OLD, status),
        )
    conn.execute(
        "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
        'VALUES (?, ?, 0, \'MessageChunkReceived\', \'{"agent_id": "worker"}\', 100)',
        (session_id, OLD),
    )
    conn.execute(
        "INSERT INTO session_embeddings "
        "(session_id, agent_id, text_hash, embedding, created_at) VALUES (?, ?, 'h', X'00', 100)",
        (session_id, OLD),
    )
    conn.execute(
        "INSERT INTO rules (agent_id, tool_kind, session_id, decision) "
        "VALUES (?, 'shell', ?, 'reject_always')",
        (OLD, session_id),
    )
    conn.commit()


def _rules_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rules ("
        "agent_id TEXT, tool_kind TEXT, session_id TEXT, decision TEXT, "
        "PRIMARY KEY (agent_id, tool_kind, session_id))"
    )


def _msg_recipients(conn: sqlite3.Connection, status: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT to_agent FROM messages WHERE status = ? AND body LIKE 'to-%' "
            "ORDER BY to_agent",
            (status,),
        ).fetchall()
    ]


def _cmd_authors(conn: sqlite3.Connection, status: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT from_agent FROM agent_commands WHERE status = ?", (status,)
        ).fetchall()
    ]


class TestRenameAgent:
    """The per-column classification, which is the whole correctness risk of a handoff.

    An agent id appears in three semantically different roles and each gets a different
    rule: AUTHORED columns move with the predecessor, POINTER columns must not move so
    they keep resolving to whoever now occupies the id, and RECIPIENT/IN-FLIGHT rows stay
    at the original id so the successor receives them.  A wrong bucket fails SILENTLY --
    no exception, no log line -- which is why these assertions are exhaustive rather than
    representative.
    """

    @staticmethod
    def _conn() -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        ensure_schema_sync(conn)
        _rules_table(conn)
        return conn

    def test_rename_moves_authored_and_leaves_pointers(self) -> None:
        conn = self._conn()
        _seed_handoff(conn)

        result = rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
        )

        # The predecessor is retired under its new id and keeps its ACP session, which
        # is exactly what resurrect() requires.
        assert conn.execute(
            "SELECT status, acp_session_id, parent, task FROM agents "
            "WHERE agent_id = ? AND session_id = ?",
            (NEW, SID),
        ).fetchone() == ("inactive", "acp-1", "boss", "do the thing")

        # The successor occupies the original id with the predecessor's fields copied
        # and no ACP session of its own yet.
        assert conn.execute(
            "SELECT status, acp_session_id, registered, parent, task, harness, agent_mode, cwd "
            "FROM agents WHERE agent_id = ? AND session_id = ?",
            (OLD, SID),
        ).fetchone() == ("active", None, 999, "boss", "do the thing", "kiro", "plan", "/tmp/wd")

        # POINTER: the child still points at the original id, which now denotes the
        # successor.  This is the feature, and it is what a future maintainer will
        # mistake for a bug.
        assert conn.execute(
            "SELECT parent FROM agents WHERE agent_id = 'kid'"
        ).fetchone() == (OLD,)

        # RECIPIENT: a pending inbound message stays for the successor; historical ones
        # move with the predecessor.
        assert _msg_recipients(conn, "pending") == [OLD]
        assert _msg_recipients(conn, "delivered") == [NEW]
        assert _msg_recipients(conn, "expired") == [NEW]

        # IN-FLIGHT: pending and processing commands stay so parent-derived
        # authorization still passes; settled ones move.
        assert _cmd_authors(conn, "pending") == [OLD]
        assert _cmd_authors(conn, "processing") == [OLD]
        assert _cmd_authors(conn, "processed") == [NEW]
        assert _cmd_authors(conn, "rejected") == [NEW]

        # AUTHORED: outbound messages move for every status.
        assert [
            r[0]
            for r in conn.execute(
                "SELECT from_agent FROM messages WHERE body LIKE 'from-%'"
            ).fetchall()
        ] == [NEW, NEW, NEW]

        assert conn.execute("SELECT agent_id FROM ui_events").fetchone() == (NEW,)
        assert conn.execute("SELECT agent_id FROM session_embeddings").fetchone() == (NEW,)

        # OPERATIONAL: rules are inherited by the successor, so they do not move.
        assert conn.execute("SELECT agent_id, decision FROM rules").fetchone() == (
            OLD,
            "reject_always",
        )

        # The payload keeps the stale id on purpose: replay routes by the column.
        assert conn.execute("SELECT payload FROM ui_events").fetchone()[0] == (
            '{"agent_id": "worker"}'
        )

        assert result.retired_acp_session_id == "acp-1"
        assert result.rows_moved == {
            "agents.agent_id": 1,
            "messages.to_agent": 2,
            "agent_commands.from_agent": 2,
            "session_embeddings.agent_id": 1,
            "messages.from_agent": 3,
            "ui_events.agent_id": 1,
        }
        conn.close()

    def test_rename_is_scoped_to_one_session(self) -> None:
        """Every statement is session-scoped, or a rename corrupts every other session
        sharing this database -- silently, since nothing reads across sessions at once."""
        conn = self._conn()
        _seed_handoff(conn)
        _seed_handoff(conn, session_id=OTHER_SID)

        rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
        )

        assert conn.execute(
            "SELECT status FROM agents WHERE agent_id = ? AND session_id = ?",
            (OLD, OTHER_SID),
        ).fetchone() == ("active",)
        assert conn.execute(
            "SELECT COUNT(*) FROM ui_events WHERE agent_id = ? AND session_id = ?",
            (OLD, OTHER_SID),
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE from_agent = ? AND session_id = ?",
            (OLD, OTHER_SID),
        ).fetchone() == (3,)
        conn.close()

    def test_rename_raises_when_predecessor_absent(self) -> None:
        conn = self._conn()
        with pytest.raises(AgentRenameError, match="no such agent"):
            rename_agent_sync(
                conn, old_agent_id="ghost", new_agent_id=NEW, session_id=SID, now_ms=999
            )
        assert conn.execute("SELECT COUNT(*) FROM agents").fetchone() == (0,)
        conn.close()

    def test_rename_raises_when_target_taken(self) -> None:
        """Reusing a live id would merge two agents' histories under one name."""
        conn = self._conn()
        _seed_handoff(conn)
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) "
            "VALUES (?, ?, 'active', 100)",
            (NEW, SID),
        )
        conn.commit()

        with pytest.raises(AgentRenameError, match="already taken"):
            rename_agent_sync(
                conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
            )

        assert conn.execute(
            "SELECT status FROM agents WHERE agent_id = ? AND session_id = ?", (OLD, SID)
        ).fetchone() == ("active",)
        assert conn.execute(
            "SELECT status FROM agents WHERE agent_id = ? AND session_id = ?", (NEW, SID)
        ).fetchone() == ("active",)
        conn.close()

    def test_rename_rolls_back_on_failure(self) -> None:
        """A partial rename leaves a half-renamed keyspace that nothing detects, so the
        transaction must be all-or-nothing.  Fail the LAST statement, which is the one
        with the most already applied behind it."""

        class _FailOnUIEvents(sqlite3.Connection):
            """Simulates the driver raising mid-transaction, the one real boundary here."""

            def execute(self, sql: str, *args: Any) -> Any:  # type: ignore[override]
                if sql.startswith("UPDATE ui_events"):
                    raise sqlite3.OperationalError("disk I/O error")
                return super().execute(sql, *args)

        conn = sqlite3.connect(":memory:", factory=_FailOnUIEvents)
        ensure_schema_sync(conn)
        _rules_table(conn)
        _seed_handoff(conn)

        with pytest.raises(sqlite3.OperationalError):
            rename_agent_sync(
                conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
            )

        assert conn.execute(
            "SELECT status FROM agents WHERE agent_id = ? AND session_id = ?", (OLD, SID)
        ).fetchone() == ("active",)
        assert conn.execute(
            "SELECT COUNT(*) FROM agents WHERE agent_id = ?", (NEW,)
        ).fetchone() == (0,)
        assert _msg_recipients(conn, "delivered") == [OLD]
        assert _cmd_authors(conn, "processed") == [OLD]
        assert conn.execute("SELECT agent_id FROM session_embeddings").fetchone() == (OLD,)
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE from_agent = ?", (NEW,)
        ).fetchone() == (0,)
        conn.close()

    def test_rename_uses_no_or_replace(self) -> None:
        """On a table whose primary key contains agent_id, OR REPLACE silently DELETES
        the conflicting row instead of raising.  On rules that would drop a
        reject_always decision, turning a blocked tool into a prompting one."""
        src = Path(__file__).resolve().parents[1] / "src" / "synth_acp" / "db.py"
        fn = next(
            n
            for n in ast.walk(ast.parse(src.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "rename_agent_sync"
        )
        literals = [
            n.value.lower() for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        assert [s for s in literals if "or replace" in s] == []

    def test_agent_id_column_inventory_is_exhaustive(self, tmp_path: Any) -> None:
        """Reflection guard: a NEW agent-referencing column must fail this test.

        For the eight columns that exist today the classification is a design-time
        decision, fully covered by the assertions above.  For a column added later it is
        a runtime risk that fails silently: a pointer column wrongly moved severs a child
        from its parent so LOCAL-mode delivery quietly stops, and a pending row wrongly
        moved strands a message forever.  Neither raises and neither logs, so the only
        available defense is forcing the conversation.

        This test imports PermissionEngine on purpose.  The eighth column lives in the
        rules table, which is created in PermissionEngine.__init__ rather than in db.py,
        so a db.py-only guard could not see it.
        """
        from synth_acp.broker.permissions import PermissionEngine

        db_path = tmp_path / "synth.db"
        conn = sqlite3.connect(str(db_path))
        ensure_schema_sync(conn)
        conn.close()
        PermissionEngine(db_path=db_path, session_id="s1")

        conn = sqlite3.connect(str(db_path))
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            found = {
                (table, col[1])
                for table in tables
                for col in conn.execute(f"PRAGMA table_info({table})").fetchall()
                if col[1] in {"agent_id", "parent", "from_agent", "to_agent"}
            }
        finally:
            conn.close()

        assert found == {
            ("agents", "agent_id"),
            ("agents", "parent"),
            ("messages", "from_agent"),
            ("messages", "to_agent"),
            ("agent_commands", "from_agent"),
            ("ui_events", "agent_id"),
            ("session_embeddings", "agent_id"),
            ("rules", "agent_id"),
        }

    def test_rename_of_25k_ui_events_completes_under_one_second(self, tmp_path: Any) -> None:
        """The transaction is a whole-session write stall, so it must stay sub-second.

        Measurement conditions: local temp-file database in WAL mode, no competing
        writer, 25,000 ui_events rows plus 200 messages and 200 agent_commands rows for
        the renamed agent.  The design's reference figure is 433ms for the whole
        transaction on a real 978MB database whose worst agent had 23,822 ui_events rows
        and 9.5MB of payload.  That measurement documents the design but cannot protect
        it: an implementation that adds per-row Python work, a second commit, or a loop
        over rows regresses this silently, since every correctness assertion still
        passes.
        """
        import time

        db_path = tmp_path / "perf.db"
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        ensure_schema_sync(conn)
        _rules_table(conn)
        _seed_handoff(conn)
        payload = '{"agent_id": "worker", "chunk": "' + "x" * 300 + '"}'
        conn.executemany(
            "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, created_at) "
            "VALUES (?, ?, ?, 'MessageChunkReceived', ?, 100)",
            [(SID, OLD, seq, payload) for seq in range(1, 25_001)],
        )
        conn.executemany(
            "INSERT INTO messages (session_id, from_agent, to_agent, body, status, created_at) "
            "VALUES (?, ?, 'kid', 'b', 'delivered', 100)",
            [(SID, OLD)] * 200,
        )
        conn.executemany(
            "INSERT INTO agent_commands "
            "(session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, ?, 'terminate', '{}', 'processed', 100)",
            [(SID, OLD)] * 200,
        )
        conn.commit()

        started = time.perf_counter()
        result = rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
        )
        elapsed = time.perf_counter() - started

        assert result.rows_moved["ui_events.agent_id"] == 25_001
        assert elapsed < 1.0, f"rename took {elapsed:.3f}s for 25k ui_events rows"
        conn.close()


class TestHandoffLineageColumns:
    """retired_from/retired_at, which are what authorizes a successor to wake a past self.

    The failures here are silent.  A retired_from left NULL produces "Not authorized" on
    a resurrection that should be allowed, which reads exactly like the bug this replaced.
    A retired_from copied onto the SUCCESSOR makes a live agent describe itself as retired.
    """

    @staticmethod
    def _conn() -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        ensure_schema_sync(conn)
        _rules_table(conn)
        return conn

    def test_rename_records_lineage_on_predecessor_only(self) -> None:
        conn = self._conn()
        _seed_handoff(conn)

        rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
        )

        # retired_from names the BARE id, so whoever currently holds it is authorized —
        # not one specific generation.
        assert conn.execute(
            "SELECT retired_from, retired_at FROM agents WHERE agent_id = ? AND session_id = ?",
            (NEW, SID),
        ).fetchone() == (OLD, 999)
        # The successor is live and must not inherit either value.
        assert conn.execute(
            "SELECT retired_from, retired_at FROM agents WHERE agent_id = ? AND session_id = ?",
            (OLD, SID),
        ).fetchone() == (None, None)

    def test_rename_leaves_child_lineage_untouched(self) -> None:
        """A handoff retires one agent.  Nothing else in the session becomes retired."""
        conn = self._conn()
        _seed_handoff(conn)

        rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=999
        )

        assert conn.execute(
            "SELECT retired_from, retired_at FROM agents WHERE agent_id = 'kid' AND session_id = ?",
            (SID,),
        ).fetchone() == (None, None)

    def test_second_handoff_points_both_predecessors_at_the_same_id(self) -> None:
        """Two generations back is reachable, and retirement order is recoverable.

        Both predecessors carry the same retired_from, so the ONLY thing separating a
        direct predecessor from an earlier one is retired_at.  If that stops being
        written, an agent asking for its direct predecessor wakes an arbitrary one.
        """
        conn = self._conn()
        _seed_handoff(conn)

        rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=NEW, session_id=SID, now_ms=1000
        )
        second = f"{OLD}.h1111beef"
        rename_agent_sync(
            conn, old_agent_id=OLD, new_agent_id=second, session_id=SID, now_ms=2000
        )

        assert conn.execute(
            "SELECT agent_id, retired_at FROM agents "
            "WHERE session_id = ? AND retired_from = ? ORDER BY retired_at DESC",
            (SID, OLD),
        ).fetchall() == [(second, 2000), (NEW, 1000)]

    def test_migration_adds_columns_to_a_preexisting_database(self) -> None:
        """An existing database gains the columns without losing rows.

        Covers the primary-key rebuild path too, because that path recreates `agents`
        from a frozen 10-column DDL and the two new columns have to be added after it.
        """
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE agents (
                agent_id    TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                registered  INTEGER NOT NULL,
                parent      TEXT,
                task        TEXT,
                acp_session_id TEXT,
                harness     TEXT,
                agent_mode  TEXT,
                cwd         TEXT
            );
            INSERT INTO agents
                (agent_id, session_id, status, registered, parent, task,
                 acp_session_id, harness, agent_mode, cwd)
            VALUES ('legacy', 'sess-old', 'active', 5, 'boss', 'old work',
                    'acp-9', 'kiro', 'plan', '/tmp/legacy');
        """)

        ensure_schema_sync(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(agents)").fetchall()}
        assert "retired_from" in cols
        assert "retired_at" in cols
        # The pre-existing row survives the rebuild with its columns still aligned, and
        # reads as "never retired" rather than needing a backfill.
        assert conn.execute(
            "SELECT session_id, parent, task, acp_session_id, cwd, retired_from, retired_at "
            "FROM agents WHERE agent_id = 'legacy'"
        ).fetchone() == ("sess-old", "boss", "old work", "acp-9", "/tmp/legacy", None, None)
        conn.close()

    def test_migration_copies_by_name_not_by_position(self) -> None:
        """The primary-key rebuild must not depend on the legacy column ORDER.

        A positional copy (``SELECT *``) into the rebuild's fixed DDL silently shifts
        every value one column left or right when the orders differ: no exception, no log
        line, just a session_id sitting in `status` and a task in `cwd`.  The columns are
        named on both sides of the copy precisely so the order cannot matter, and this
        pins that.
        """
        conn = sqlite3.connect(":memory:")
        # Same ten columns as the frozen rebuild DDL, deliberately in a different order.
        conn.executescript("""
            CREATE TABLE agents (
                agent_id    TEXT PRIMARY KEY,
                cwd         TEXT,
                agent_mode  TEXT,
                harness     TEXT,
                acp_session_id TEXT,
                task        TEXT,
                parent      TEXT,
                registered  INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                session_id  TEXT NOT NULL
            );
            INSERT INTO agents
                (agent_id, cwd, agent_mode, harness, acp_session_id, task, parent,
                 registered, status, session_id)
            VALUES ('legacy', '/tmp/legacy', 'plan', 'kiro', 'acp-9', 'old work',
                    'boss', 5, 'inactive', 'sess-old');
        """)

        ensure_schema_sync(conn)

        assert conn.execute(
            "SELECT session_id, status, registered, parent, task, acp_session_id, "
            "harness, agent_mode, cwd FROM agents WHERE agent_id = 'legacy'"
        ).fetchone() == (
            "sess-old", "inactive", 5, "boss", "old work", "acp-9",
            "kiro", "plan", "/tmp/legacy",
        )
        conn.close()
