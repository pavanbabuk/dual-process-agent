"""SessionRouter bounds and isolation tests.

Defect under test: `_sessions` grew without bound and `destroy()` was never
called by anything, so a gateway seeing many chat_ids accumulated one dispatcher
(and one SQLite handle plus session dir) per id forever.

The acceptance signal is real isolation on disk: two chat_ids must resolve to
two different memory databases, and eviction must not delete a chat's history.
"""

import os

import pytest

from dual_agent.gateway.session_router import DEFAULT_MAX_SESSIONS, SessionRouter


class FakeDispatcher:
    """Records which session_id it was built for; owns a real per-chat dir."""

    def __init__(self, session_id, base_dir):
        self.session_id = session_id
        self.dir = os.path.join(base_dir, str(session_id))
        os.makedirs(self.dir, exist_ok=True)
        self.memory_db = os.path.join(self.dir, "memory.db")


@pytest.fixture
def router_factory(tmp_path):
    def make(max_sessions=DEFAULT_MAX_SESSIONS):
        base = tmp_path / "sessions"
        return SessionRouter(
            dispatcher_factory=lambda session_id: FakeDispatcher(session_id, str(base)),
            base_data_dir=str(base),
            max_sessions=max_sessions,
        )

    return make


def test_default_cap_is_documented_and_bounded():
    """The default must exist, be finite, and be a positive integer."""
    assert isinstance(DEFAULT_MAX_SESSIONS, int)
    assert DEFAULT_MAX_SESSIONS > 0
    assert DEFAULT_MAX_SESSIONS <= 4096


def test_session_cap_holds_under_many_chat_ids(router_factory):
    """Creating far more chats than the cap never exceeds the cap."""
    router = router_factory(max_sessions=8)

    for i in range(100):
        router.get_or_create(f"chat-{i}")

    assert router.active_sessions == 8, (
        f"cap breached: {router.active_sessions} resident sessions"
    )
    assert router.evictions == 92
    stats = router.stats()
    assert stats["active_sessions"] == 8
    assert stats["max_sessions"] == 8


def test_lru_keeps_the_most_recently_used_session(router_factory):
    """Recency, not creation order, decides eviction."""
    router = router_factory(max_sessions=3)

    a = router.get_or_create("a")
    router.get_or_create("b")
    router.get_or_create("c")

    # Renew "a" so it is the warmest; "b" becomes least-recently-used.
    assert router.get_or_create("a") is a
    router.get_or_create("d")

    resident = router.stats()["chat_ids"]
    assert "a" in resident, "renewed session was evicted ahead of a colder one"
    assert "b" not in resident, "LRU order ignored; 'b' should have gone first"


def test_cap_none_disables_bounding(router_factory):
    router = router_factory(max_sessions=None)
    for i in range(50):
        router.get_or_create(f"chat-{i}")
    assert router.active_sessions == 50
    assert router.evictions == 0


def test_invalid_cap_rejected(router_factory, tmp_path):
    with pytest.raises(ValueError):
        SessionRouter(
            dispatcher_factory=lambda session_id: FakeDispatcher(session_id, str(tmp_path)),
            base_data_dir=str(tmp_path),
            max_sessions=0,
        )


def test_destroy_removes_a_session(router_factory):
    router = router_factory(max_sessions=10)
    router.get_or_create("gone")
    assert router.active_sessions == 1
    router.destroy("gone")
    assert router.active_sessions == 0
    # Destroying an unknown id is a no-op, not an error.
    router.destroy("never-existed")


def test_isolation_two_chats_do_not_share_memory(router_factory):
    """Each chat gets its own dispatcher object and its own memory database."""
    router = router_factory(max_sessions=10)

    d1 = router.get_or_create("chat-one")
    d2 = router.get_or_create("chat-two")

    assert d1 is not d2
    assert d1.session_id == "chat-one"
    assert d2.session_id == "chat-two"
    assert d1.memory_db != d2.memory_db
    assert os.path.dirname(d1.memory_db) != os.path.dirname(d2.memory_db)

    # A chat writes real rows into its own DB; the other must not see them.
    import sqlite3

    con = sqlite3.connect(d1.memory_db)
    con.execute("CREATE TABLE marker (v TEXT)")
    con.execute("INSERT INTO marker VALUES ('only-in-one')")
    con.commit()
    con.close()

    con2 = sqlite3.connect(d2.memory_db)
    tables = {r[0] for r in con2.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    con2.close()
    assert "marker" not in tables, "chat-two observed chat-one's database"


def test_eviction_preserves_on_disk_history(router_factory):
    """Eviction frees memory but must not delete a chat's memory directory."""
    router = router_factory(max_sessions=1)

    first = router.get_or_create("keeper")
    marker = os.path.join(first.dir, "memory.db")
    with open(marker, "w") as fh:
        fh.write("history")

    router.get_or_create("intruder")  # evicts "keeper"

    assert router.active_sessions == 1
    assert os.path.exists(marker), "eviction destroyed a chat's on-disk memory"
    # A returning chat re-opens its own directory rather than a fresh one.
    again = router.get_or_create("keeper")
    assert again.dir == first.dir
