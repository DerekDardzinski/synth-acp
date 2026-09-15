"""Shared SQLite schema and helpers."""

from __future__ import annotations

import sqlite3
import time

from pydantic import BaseModel, ConfigDict

SCHEMA = """\
CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL,
    parent      TEXT,
    task        TEXT,
    acp_session_id TEXT,
    harness     TEXT,
    agent_mode  TEXT,
    cwd         TEXT,
    retired_from TEXT,
    retired_at  INTEGER,
    PRIMARY KEY (agent_id, session_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  INTEGER NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'chat',
    reply_to    INTEGER REFERENCES messages(id),
    delivered_at INTEGER
);
CREATE TABLE IF NOT EXISTS agent_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    command     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    created_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ui_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ui_events_replay
    ON ui_events (session_id, agent_id, seq);
CREATE INDEX IF NOT EXISTS idx_ui_events_user_prompts
    ON ui_events (session_id, event_type, seq);
"""


SESSION_EMBEDDINGS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS session_embeddings (
    session_id  TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (session_id, agent_id)
);
"""


BUSY_TIMEOUT_MS = 30_000
"""Milliseconds a blocked writer waits for the WAL write lock before giving up.

WAL allows exactly one writer at a time.  Python's ``sqlite3`` default is 5000 ms,
which is too short for a multi-table write transaction: every other writer in the
session raises ``sqlite3.OperationalError: database is locked``, and no write path
in this codebase catches it.  One of those paths runs inside each agent's
``synth-mcp`` subprocess, where the failure surfaces to the agent as a broken tool
call rather than as a retry.

30 seconds turns a slow writer into an invisible stall instead of an error.
"""


def configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Apply the PRAGMAs every synth SQLite connection requires.

    Call this immediately after opening a connection and before any read or write.
    Every ``sqlite3.connect`` site in ``synth_acp`` routes through here so the
    settings have a single point of truth; a guard test in ``tests/test_db.py``
    enforces that.

    Args:
        conn: A freshly opened connection.

    Returns:
        The same connection, so this can wrap a ``connect`` call inline.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def ensure_schema_sync(conn) -> None:
    """Execute schema DDL on a synchronous sqlite3 connection."""
    conn.executescript(SCHEMA)
    conn.executescript(SESSION_EMBEDDINGS_SCHEMA)
    _migrate_schema_sync(conn)


def _migrate_schema_sync(conn) -> None:
    """Apply one-time migrations for existing databases.

    Detects the old agent_id-only primary key and recreates the table with
    the correct composite (agent_id, session_id) key, preserving all rows.
    Then adds any columns introduced after that rebuild.
    """
    cur = conn.execute("PRAGMA table_info(agents)")
    cols = {row[1]: row[5] for row in cur.fetchall()}  # name -> pk position
    pk_cols = [name for name, pk in cols.items() if pk > 0]
    if pk_cols == ["agent_id"]:
        # The column list is spelled out in BOTH halves of the copy rather than using
        # SELECT *.  This DDL is frozen at the 10 columns an agent_id-only database can
        # have, so a column added to SCHEMA later is added by the ADD COLUMN pass below
        # instead; SELECT * here would silently shift values into the wrong columns the
        # first time those two counts disagreed.
        conn.executescript("""
            ALTER TABLE agents RENAME TO agents_old;
            CREATE TABLE agents (
                agent_id    TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                registered  INTEGER NOT NULL,
                parent      TEXT,
                task        TEXT,
                acp_session_id TEXT,
                harness     TEXT,
                agent_mode  TEXT,
                cwd         TEXT,
                PRIMARY KEY (agent_id, session_id)
            );
            INSERT OR IGNORE INTO agents
                (agent_id, session_id, status, registered, parent, task,
                 acp_session_id, harness, agent_mode, cwd)
            SELECT agent_id, session_id, status, registered, parent, task,
                   acp_session_id, harness, agent_mode, cwd
            FROM agents_old;
            DROP TABLE agents_old;
        """)
        conn.commit()

    # Handoff lineage columns.  Nullable, so existing rows need no backfill: a NULL
    # retired_from means "not a handoff predecessor", which is true of every row that
    # predates this migration.
    cur = conn.execute("PRAGMA table_info(agents)")
    agent_cols = {row[1] for row in cur.fetchall()}
    for name, decl in (("retired_from", "TEXT"), ("retired_at", "INTEGER")):
        if name not in agent_cols:
            conn.execute(f"ALTER TABLE agents ADD COLUMN {name} {decl}")
    conn.commit()

    # Migrate session_embeddings: detect old single-PK schema (no agent_id column)
    cur = conn.execute("PRAGMA table_info(session_embeddings)")
    emb_cols = {row[1] for row in cur.fetchall()}
    if emb_cols and "agent_id" not in emb_cols:
        conn.execute("DROP TABLE session_embeddings")
        conn.executescript(SESSION_EMBEDDINGS_SCHEMA)
        conn.commit()


# ------------------------------------------------------------------
# Session retention
# ------------------------------------------------------------------

RETENTION_DAYS: int = 60

# Activity sources, maxed per session to derive a session's last update.
# session_embeddings is deliberately absent: it is derived data, and an indexing
# pass must not resurrect a dead session.
_ACTIVITY_SOURCES: tuple[tuple[str, str], ...] = (
    ("ui_events", "created_at"),
    ("messages", "created_at"),
    ("agent_commands", "created_at"),
    ("agents", "registered"),
)

# Tables cleared for an expired session, paired with their ExpiryReport field.
# Children before agents.
_EXPIRE_TABLES: tuple[tuple[str, str], ...] = (
    ("ui_events", "ui_events_deleted"),
    ("messages", "messages_deleted"),
    ("agent_commands", "agent_commands_deleted"),
    ("session_embeddings", "embeddings_deleted"),
    ("agents", "agents_deleted"),
)


class ExpiryReport(BaseModel):
    """Counts of rows expired, or that would be expired under dry_run."""

    model_config = ConfigDict(frozen=True)

    sessions_expired: int
    ui_events_deleted: int
    messages_deleted: int
    agent_commands_deleted: int
    agents_deleted: int
    embeddings_deleted: int
    dry_run: bool
    cutoff_epoch: int


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    """Return the names of tables present in this database.

    Expiry must work on older databases that predate a table without creating
    it.  Calling ensure_schema_sync instead would be destructive: its
    _migrate_schema_sync step drops session_embeddings when it finds the legacy
    schema, which a dry-run preview must never do.
    """
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _activity_union(tables: set[str]) -> str:
    """Return a UNION ALL of (session_id, timestamp) over the existing activity sources."""
    return " UNION ALL ".join(
        f"SELECT session_id, {column} AS ts FROM {table}"
        for table, column in _ACTIVITY_SOURCES
        if table in tables
    )


def session_last_activity_sync(conn: sqlite3.Connection) -> dict[str, int]:
    """Return session_id -> last activity epoch seconds.

    Activity is max(ui_events.created_at, messages.created_at,
    agent_commands.created_at, agents.registered) per session.
    session_embeddings is excluded because it is derived.

    Every timestamp column in this schema is stored in MILLISECONDS, so the
    stored values are divided by 1000 to yield the epoch seconds this function
    and ExpiryReport.cutoff_epoch are defined in.
    """
    union = _activity_union(_existing_tables(conn))
    if not union:
        return {}
    rows = conn.execute(
        f"SELECT session_id, MAX(ts) FROM ({union}) GROUP BY session_id"
    ).fetchall()
    return {session_id: ms // 1000 for session_id, ms in rows if ms is not None}


def _session_activity_sync(conn: sqlite3.Connection, session_id: str) -> int | None:
    """Return last activity epoch seconds for one session, or None if it has none."""
    union = _activity_union(_existing_tables(conn))
    if not union:
        return None
    row = conn.execute(
        f"SELECT MAX(ts) FROM ({union}) WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0]) // 1000


def _delete_session_sync(
    conn: sqlite3.Connection, session_id: str, tables: set[str], cutoff_epoch: int
) -> dict[str, int] | None:
    """Delete one expired session's rows in a single transaction.

    Returns per-report-field deleted row counts, or None if the session was
    refreshed after it was selected and must be kept.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Candidates are selected outside any write transaction, so a concurrent
        # writer -- an MCP subprocess inserting a message, or the ui_events
        # journal; both share this database over WAL -- can make a session active
        # again in between.  Re-verify inside the write transaction rather than
        # trusting the earlier read: deleting a session that came back to life is
        # unrecoverable.
        activity = _session_activity_sync(conn, session_id)
        if activity is None or activity >= cutoff_epoch:
            conn.rollback()
            return None

        deleted: dict[str, int] = {}
        for table, field in _EXPIRE_TABLES:
            if table not in tables:
                continue
            cur = conn.execute(
                f"DELETE FROM {table} WHERE session_id = ?",
                (session_id,),
            )
            deleted[field] = cur.rowcount
        # Inside the try: a failing commit must roll back too, or the transaction
        # stays open with the rows already gone on this connection.
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return deleted


def expire_old_sessions_sync(
    conn: sqlite3.Connection,
    *,
    days: int = RETENTION_DAYS,
    dry_run: bool = True,
) -> ExpiryReport:
    """Delete sessions whose last activity is strictly older than *days*.

    Boundary: expired when (now - last_activity) > days * 86400.  Exactly
    *days* old is NOT expired.

    Deletes per expired session: ui_events, messages, agent_commands,
    session_embeddings, agents.  Removes rows for ALL agent statuses including
    'inactive', which is the status terminate() assigns and the reason nothing
    has ever expired.

    All deletions for a session happen in ONE transaction.

    dry_run defaults to True: computes the report and deletes nothing.  A dry
    run issues no DELETE and no DDL of any kind.

    Does NOT run VACUUM, so the report exposes row counts only and never claims
    bytes reclaimed.
    """
    cutoff_epoch = int(time.time()) - days * 86400
    tables = _existing_tables(conn)
    expired = sorted(
        session_id
        for session_id, last in session_last_activity_sync(conn).items()
        if last < cutoff_epoch
    )

    counts = {field: 0 for _, field in _EXPIRE_TABLES}
    kept = 0

    if dry_run:
        for table, field in _EXPIRE_TABLES:
            if table not in tables:
                continue
            per_session = dict(
                conn.execute(
                    f"SELECT session_id, COUNT(*) FROM {table} GROUP BY session_id"
                ).fetchall()
            )
            counts[field] = sum(per_session.get(session_id, 0) for session_id in expired)
    else:
        for session_id in expired:
            deleted = _delete_session_sync(conn, session_id, tables, cutoff_epoch)
            if deleted is None:
                kept += 1
                continue
            for field, rows in deleted.items():
                counts[field] += rows

    return ExpiryReport(
        sessions_expired=len(expired) - kept,
        dry_run=dry_run,
        cutoff_epoch=cutoff_epoch,
        **counts,
    )


# ------------------------------------------------------------------
# Agent rename (handoff)
# ------------------------------------------------------------------


class AgentRenameResult(BaseModel):
    """Outcome of an atomic rename, plus the predecessor fields copied to the successor.

    Returned so the caller can construct the successor's session without a second read,
    which would race the transaction that just committed.
    """

    model_config = ConfigDict(frozen=True)

    old_agent_id: str
    new_agent_id: str
    parent: str | None
    """The predecessor's parent, copied verbatim onto the successor row."""
    task: str
    harness: str
    agent_mode: str | None
    cwd: str
    retired_acp_session_id: str | None
    """The PREDECESSOR's ACP session id.  Stays with the predecessor so it remains
    resurrectable.  The successor row is inserted with acp_session_id NULL and has it
    filled in by the normal session-created callback."""
    rows_moved: dict[str, int]
    """Affected row counts keyed "<table>.<column>".  For assertions and logging only."""


class AgentRenameError(Exception):
    """Raised when an atomic agent rename cannot be performed.

    Raised when the predecessor row does not exist in this session, or the target id is
    already taken in this session.  The transaction is rolled back before this is
    raised, so the database is unchanged.
    """


def rename_agent_sync(
    conn: sqlite3.Connection,
    *,
    old_agent_id: str,
    new_agent_id: str,
    session_id: str,
    now_ms: int,
) -> AgentRenameResult:
    """Rename an agent and insert a successor under the freed id, atomically.

    Runs entirely inside one BEGIN IMMEDIATE transaction so no reader or writer can
    observe a half-renamed keyspace.  On any failure the transaction is rolled back and
    the database is left unchanged.

    The predecessor ends at status 'inactive', which is exactly what
    AgentLifecycle.resurrect requires, so it stays resurrectable.  The successor is
    inserted 'active' with the predecessor's parent, task, harness, agent_mode and cwd
    copied, and with acp_session_id NULL.

    Does NOT modify agents.parent on any other row: children continue to point at
    old_agent_id, which after this call denotes the successor.  That is the purpose of
    the operation, not an oversight.

    Does NOT modify the rules table: permission decisions are operational state and are
    inherited by the successor.

    Args:
        conn: Open connection.  This function owns the transaction on it; the caller
            must not already be inside one.
        old_agent_id: The agent being retired.  Must exist in this session.
        new_agent_id: The id it is renamed to.  Must not exist in this session.
        session_id: Scopes every statement.
        now_ms: Epoch milliseconds recorded as the successor's `registered`.

    Returns:
        The predecessor fields the caller needs to build the successor's session.

    Raises:
        AgentRenameError: predecessor missing, or new_agent_id already taken.
        sqlite3.OperationalError: write lock not acquired within busy_timeout.
    """
    tables = _existing_tables(conn)
    rows_moved: dict[str, int] = {}

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT parent, task, harness, agent_mode, cwd, acp_session_id "
            "FROM agents WHERE agent_id = ? AND session_id = ?",
            (old_agent_id, session_id),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise AgentRenameError(
                f"Cannot rename '{old_agent_id}': no such agent in session '{session_id}'"
            )
        parent, task, harness, agent_mode, cwd, retired_acp_session_id = row

        taken = conn.execute(
            "SELECT 1 FROM agents WHERE agent_id = ? AND session_id = ?",
            (new_agent_id, session_id),
        ).fetchone()
        if taken is not None:
            conn.rollback()
            raise AgentRenameError(
                f"Cannot rename '{old_agent_id}' to '{new_agent_id}': id already taken"
            )

        # Frees old_agent_id.  Touches neither acp_session_id nor parent, so the
        # predecessor keeps its ACP session, keeps naming the agent that actually
        # launched it, and stays resurrectable.
        #
        # retired_from is set to old_agent_id — the id the SUCCESSOR is about to take —
        # so "who may resurrect this predecessor" is stored data rather than something
        # recovered by parsing the '.h<hex>' suffix off an id.  Because it names the bare
        # id rather than one generation, whoever currently holds that id may resurrect
        # every earlier generation in the lineage.
        cur = conn.execute(
            "UPDATE agents SET agent_id = ?, status = 'inactive', "
            "retired_from = ?, retired_at = ? "
            "WHERE agent_id = ? AND session_id = ?",
            (new_agent_id, old_agent_id, now_ms, old_agent_id, session_id),
        )
        rows_moved["agents.agent_id"] = cur.rowcount

        # MUST be in this transaction.  Inserted after commit, the original id would be
        # briefly vacant and a surviving synth-mcp subprocess could create a ghost row
        # via _ensure_registered's INSERT OR IGNORE.  With the id occupied that stray
        # write is a harmless no-op.
        #
        # NO statement in this transaction touches agents.parent on any OTHER row.
        # Children keep pointing at old_agent_id, which now denotes the successor.
        # That omission IS the feature: it is what keeps agents.parent and the
        # registry's parent pointers agreeing, so no child ever learns anything
        # changed.  Do not "fix" it.
        # retired_from and retired_at are deliberately NOT in this column list.  They
        # describe a row's OWN retirement, so a freshly started successor has none yet;
        # copying the predecessor's values would make the live agent claim to be retired.
        conn.execute(
            "INSERT INTO agents "
            "(agent_id, session_id, status, registered, parent, task, harness, agent_mode, cwd) "
            "VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?)",
            (old_agent_id, session_id, now_ms, parent, task, harness, agent_mode, cwd),
        )

        # Recipient column: a row still 'pending' is picked up by
        # message_bus._poll_messages, which does not filter on recipient status, and is
        # delivered to the SUCCESSOR.  Moving it would strand it on a retired agent.
        cur = conn.execute(
            "UPDATE messages SET to_agent = ? "
            "WHERE to_agent = ? AND session_id = ? AND status IN ('delivered', 'expired')",
            (new_agent_id, old_agent_id, session_id),
        )
        rows_moved["messages.to_agent"] = cur.rowcount

        # In-flight column: handle_terminate_command and handle_resurrect_command
        # authorize by comparing from_agent against the target's parent, and children's
        # parent still reads the ORIGINAL id, so renaming an actionable row makes that
        # comparison fail with "Not authorized".  'processing' is actionable too,
        # because message_bus reverts it to 'pending' on startup.
        cur = conn.execute(
            "UPDATE agent_commands SET from_agent = ? "
            "WHERE from_agent = ? AND session_id = ? AND status IN ('processed', 'rejected')",
            (new_agent_id, old_agent_id, session_id),
        )
        rows_moved["agent_commands.from_agent"] = cur.rowcount

        if "session_embeddings" in tables:
            cur = conn.execute(
                "UPDATE session_embeddings SET agent_id = ? "
                "WHERE agent_id = ? AND session_id = ?",
                (new_agent_id, old_agent_id, session_id),
            )
            rows_moved["session_embeddings.agent_id"] = cur.rowcount
        else:
            rows_moved["session_embeddings.agent_id"] = 0

        cur = conn.execute(
            "UPDATE messages SET from_agent = ? WHERE from_agent = ? AND session_id = ?",
            (new_agent_id, old_agent_id, session_id),
        )
        rows_moved["messages.from_agent"] = cur.rowcount

        # The expensive one, and the last: 23,822 rows / 9.5MB measured at 76ms on the
        # worst real agent.
        #
        # ui_events.payload is NOT rewritten even though every row embeds agent_id as
        # JSON.  Journal replay routes by this COLUMN: load_journal selects only
        # event_type and payload, and the feed to replay into is resolved from the
        # agent_id ARGUMENT and passed in explicitly.  The single reachable read of a
        # payload-derived id is the UI's _coalesce_events, which compares adjacent
        # events to each other, so the whole consequence is one extra message-boundary
        # split where a buffer joins journal events to live ones.
        cur = conn.execute(
            "UPDATE ui_events SET agent_id = ? WHERE agent_id = ? AND session_id = ?",
            (new_agent_id, old_agent_id, session_id),
        )
        rows_moved["ui_events.agent_id"] = cur.rowcount

        # The rules table is deliberately absent from this transaction.  A rule means
        # "this role always allows this tool" -- operational state, not history -- so
        # leaving it lets the successor inherit the user's approvals with zero code.
        #
        # Inside the try: a failing commit must roll back too, or the transaction stays
        # open with the writes already applied on this connection.
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    return AgentRenameResult(
        old_agent_id=old_agent_id,
        new_agent_id=new_agent_id,
        parent=parent,
        task=task or "",
        harness=harness or "",
        agent_mode=agent_mode,
        cwd=cwd or "",
        retired_acp_session_id=retired_acp_session_id,
        rows_moved=rows_moved,
    )


# ------------------------------------------------------------------
# Embedding helpers
# ------------------------------------------------------------------


def store_embedding_sync(
    conn: sqlite3.Connection, session_id: str, agent_id: str, text_hash: str, embedding_blob: bytes
) -> None:
    """Upsert a per-agent embedding. embedding_blob is 1536 bytes (384 x float32)."""
    conn.execute(
        "INSERT OR REPLACE INTO session_embeddings (session_id, agent_id, text_hash, embedding, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, agent_id, text_hash, embedding_blob, int(time.time() * 1000)),
    )
    conn.commit()


def load_all_embeddings_sync(conn: sqlite3.Connection) -> list[tuple[str, str, bytes]]:
    """Return all (session_id, agent_id, embedding_blob) tuples ordered by session_id."""
    return conn.execute(
        "SELECT session_id, agent_id, embedding FROM session_embeddings ORDER BY session_id"
    ).fetchall()


def get_unembedded_agents_sync(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (session_id, agent_id) pairs that exist in agents but not in session_embeddings."""
    return conn.execute(
        "SELECT a.session_id, a.agent_id FROM agents a "
        "WHERE a.status IN ('restorable', 'active') "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM session_embeddings e "
        "  WHERE e.session_id = a.session_id AND e.agent_id = a.agent_id"
        ")"
    ).fetchall()




