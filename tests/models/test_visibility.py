"""Tests for visibility and resurrection authorization.

Both functions here gate silent behavior.  ``get_visible_agents`` is what
``send_message`` checks, so a missing entry is not an error — it is a message that can
never be addressed.  ``get_resurrectable_agents`` must agree exactly with
``AgentLifecycle.handle_resurrect_command``: listing a row the caller cannot revive
advertises a dead end, and omitting one hides a capability the agent has.
"""

from __future__ import annotations

import sqlite3

from synth_acp.db import ensure_schema_sync
from synth_acp.models.visibility import get_resurrectable_agents, get_visible_agents

SID = "sess-vis"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema_sync(conn)
    return conn


def _agent(
    conn: sqlite3.Connection,
    agent_id: str,
    *,
    status: str = "active",
    parent: str | None = None,
    task: str = "",
    retired_from: str | None = None,
    retired_at: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO agents "
        "(agent_id, session_id, status, registered, parent, task, retired_from, retired_at) "
        "VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
        (agent_id, SID, status, parent, task, retired_from, retired_at),
    )


class TestLineageVisibility:
    """A resurrected predecessor is a child of nobody and a sibling of nobody.

    Without the lineage branch it is unreachable under LOCAL: the successor that just
    resurrected it cannot message it and it cannot reply.  Nothing raises — the send
    simply returns "Agent not visible".
    """

    def test_successor_and_resurrected_predecessor_see_each_other(self) -> None:
        conn = _conn()
        _agent(conn, "root")
        _agent(
            conn, "root.hdead0001", status="inactive",
            retired_from="root", retired_at=100,
        )

        assert "root.hdead0001" not in get_visible_agents(conn, "root", SID, "LOCAL")

        conn.execute(
            "UPDATE agents SET status = 'active' WHERE agent_id = 'root.hdead0001'"
        )

        # Symmetric: both parties resolve to the same lineage anchor, so enabling one
        # direction enables the other.  send_message gates both sends on this function.
        assert get_visible_agents(conn, "root", SID, "LOCAL") == ["root.hdead0001"]
        assert get_visible_agents(conn, "root.hdead0001", SID, "LOCAL") == ["root"]

    def test_two_resurrected_predecessors_are_not_peers(self) -> None:
        """The branch joins a predecessor to its successor and stops there.

        Every extra visibility edge is one more agent a `send_message(to_agent="*")`
        reaches, so peer edges nothing asked for are not free.
        """
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "root.hdead0001", retired_from="root", retired_at=100)
        _agent(conn, "root.hdead0002", retired_from="root", retired_at=200)

        assert get_visible_agents(conn, "root.hdead0001", SID, "LOCAL") == ["root"]
        # The successor still sees both of its own live predecessors.
        assert sorted(get_visible_agents(conn, "root", SID, "LOCAL")) == [
            "root.hdead0001",
            "root.hdead0002",
        ]

    def test_retired_predecessor_stays_invisible_in_mesh(self) -> None:
        """MESH already returns every active agent, so the branch must not widen it."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "other")
        _agent(conn, "root.hdead0001", status="inactive", retired_from="root", retired_at=100)

        assert sorted(get_visible_agents(conn, "root", SID, "MESH")) == ["other"]

    def test_lineage_branch_does_not_reveal_unrelated_agents(self) -> None:
        """An agent with no lineage sees exactly what it saw before."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "kid", parent="root")
        _agent(conn, "stranger")
        _agent(conn, "stranger.hdead0003", retired_from="stranger", retired_at=100)

        assert get_visible_agents(conn, "root", SID, "LOCAL") == ["kid"]

    def test_resurrected_predecessor_still_sees_its_original_parent(self) -> None:
        """The rename never rewrites parent, so a non-root predecessor keeps its parent."""
        conn = _conn()
        _agent(conn, "boss")
        _agent(conn, "worker", parent="boss")
        _agent(conn, "worker.hdead0004", parent="boss", retired_from="worker", retired_at=100)

        visible = get_visible_agents(conn, "worker.hdead0004", SID, "LOCAL")
        assert sorted(visible) == ["boss", "worker"]


class TestResurrectableAgents:
    def test_lists_a_root_predecessor_and_inactive_children(self) -> None:
        """The root predecessor case is the entire bug: its parent is NULL."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "root.hdead0001", status="inactive", retired_from="root", retired_at=300)
        _agent(conn, "kid", status="inactive", parent="root", task="phase 1")

        assert get_resurrectable_agents(conn, "root", SID) == [
            ("root.hdead0001", None, "", "root", 300),
            ("kid", "root", "phase 1", None, None),
        ]

    def test_excludes_statuses_resurrect_would_refuse(self) -> None:
        """`resurrect` accepts only 'inactive'.  'restorable' is a real status in this
        schema, and advertising one produces a command marked 'processed' while nothing
        starts — a success the caller cannot distinguish from a real one."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "kid", status="restorable", parent="root")

        assert get_resurrectable_agents(conn, "root", SID) == []

    def test_excludes_active_and_unrelated_inactive_agents(self) -> None:
        """Anything the caller cannot actually revive must not be advertised."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "live-kid", parent="root")
        _agent(conn, "stranger", status="inactive")
        _agent(
            conn, "stranger.hdead0005", status="inactive",
            retired_from="stranger", retired_at=100,
        )

        assert get_resurrectable_agents(conn, "root", SID) == []

    def test_orders_newest_retirement_first_with_untimed_rows_last(self) -> None:
        """generations_back is assigned from this order, so the order is the contract."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "root.hdead0001", status="inactive", retired_from="root", retired_at=100)
        _agent(conn, "root.hdead0002", status="inactive", retired_from="root", retired_at=200)
        _agent(conn, "kid", status="inactive", parent="root")

        assert [r[0] for r in get_resurrectable_agents(conn, "root", SID)] == [
            "root.hdead0002",
            "root.hdead0001",
            "kid",
        ]

    def test_lists_a_launched_childs_predecessor(self) -> None:
        """A parent can already revive its terminated children; a child's retired self
        is such a row, and it is reported against the CHILD's lineage rather than the
        caller's."""
        conn = _conn()
        _agent(conn, "root")
        _agent(conn, "worker", parent="root")
        _agent(
            conn, "worker.hdead0006", status="inactive", parent="root",
            retired_from="worker", retired_at=400,
        )

        assert get_resurrectable_agents(conn, "root", SID) == [
            ("worker.hdead0006", "root", "", "worker", 400),
        ]
