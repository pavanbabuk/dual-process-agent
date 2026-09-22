"""Tests for MemoryEngine retention (prune), WAL checkpointing, and FTS hygiene.

The orphan-index test below is the regression test for a bug found by running
the product: sessions_fts had an AFTER INSERT trigger but no AFTER DELETE
trigger, so pruning sessions left the FTS index pointing at rows that were gone.
Every test uses a tmp_path-based db_path — the real ~/.dual_agent/memory.db is
never touched.
"""

import os

import pytest

from dual_agent.memory import MemoryEngine


@pytest.fixture
def mem(tmp_path):
    return MemoryEngine(db_path=str(tmp_path / "mem.db"))


def _save(mem, goal, outcome="done", completed=True):
    return mem.save_session(
        goal=goal,
        outcome=outcome,
        is_completed=completed,
        total_steps=2,
        system_one_steps=1,
        system_two_steps=1,
        total_latency_ms=100.0,
        tokens_used=50,
        token_savings_pct=None,  # preserve the Optional/NULL contract
    )


# ---------------------------------------------------------------------------
# prune_sessions
# ---------------------------------------------------------------------------


def test_prune_dry_run_deletes_nothing_but_reports_matches(mem):
    for i in range(5):
        _save(mem, f"dry run task {i}")

    before = mem.get_aggregate_stats()["total_sessions"]
    result = mem.prune_sessions(keep_last=2, dry_run=True)

    assert result["matched"] == 3
    assert result["deleted"] == 0
    assert result["retained"] == before
    assert mem.get_aggregate_stats()["total_sessions"] == before


def test_prune_older_than_days_dry_run(mem):
    for i in range(3):
        _save(mem, f"old task {i}")

    # Everything was just written, so nothing is older than 30 days.
    result = mem.prune_sessions(older_than_days=30, dry_run=True)
    assert result["matched"] == 0
    assert result["deleted"] == 0
    assert result["retained"] == 3


def test_prune_live_deletes_rows_and_retains_requested_number(mem):
    for i in range(6):
        _save(mem, f"live task {i}")

    result = mem.prune_sessions(keep_last=2, dry_run=False)

    assert result["matched"] == 4
    assert result["deleted"] == 4
    assert result["retained"] == 2
    assert mem.get_aggregate_stats()["total_sessions"] == 2

    remaining = mem.get_recent_sessions(limit=10)
    assert [r["goal"] for r in remaining] == ["live task 5", "live task 4"]


def test_prune_keep_last_zero_deletes_everything(mem):
    for i in range(3):
        _save(mem, f"all task {i}")

    result = mem.prune_sessions(keep_last=0, dry_run=False)
    assert result["matched"] == 3
    assert result["deleted"] == 3
    assert result["retained"] == 0


def test_prune_requires_retention_criterion(mem):
    _save(mem, "keep me")
    with pytest.raises(ValueError):
        mem.prune_sessions()
    # Nothing was deleted by the rejected call.
    assert mem.get_aggregate_stats()["total_sessions"] == 1


def test_prune_rejects_negative_keep_last(mem):
    with pytest.raises(ValueError):
        mem.prune_sessions(keep_last=-1)


def test_prune_rejects_negative_older_than_days(mem):
    with pytest.raises(ValueError):
        mem.prune_sessions(older_than_days=-5)


def test_prune_matched_equals_deleted_on_live_run(mem):
    for i in range(4):
        _save(mem, f"parity task {i}")

    report = mem.prune_sessions(keep_last=1, dry_run=True)
    live = mem.prune_sessions(keep_last=1, dry_run=False)
    assert report["matched"] == live["matched"]
    assert live["deleted"] == live["matched"]


# ---------------------------------------------------------------------------
# checkpoint_wal
# ---------------------------------------------------------------------------


def test_checkpoint_wal_returns_counters_and_db_stays_usable(mem, tmp_path):
    for i in range(20):
        _save(mem, f"wal task {i} with some padding text to grow the log")

    result = mem.checkpoint_wal()
    assert isinstance(result, dict)
    assert set(["busy", "log", "checkpointed"]).issubset(result.keys())
    assert result["busy"] == 0

    # DB must still be in WAL mode and fully usable afterwards.
    recent = mem.get_recent_sessions(limit=10)
    assert len(recent) == 10
    assert mem.get_aggregate_stats()["total_sessions"] == 20

    fts = mem.full_text_search("padding")
    assert len(fts) >= 1

    _save(mem, "post checkpoint write")
    assert mem.get_aggregate_stats()["total_sessions"] == 21


def test_journal_mode_stays_wal_after_checkpoint(mem):
    _save(mem, "journal mode check")
    mem.checkpoint_wal()
    mode = mem._get_connection().execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# ---------------------------------------------------------------------------
# FTS orphan regression
# ---------------------------------------------------------------------------


def test_prune_leaves_no_fts_orphans(mem):
    """The regression test: a pruned goal must not be findable via FTS.

    Against the old code (no AFTER DELETE trigger, no rebuild) the FTS shadow
    index kept its entry, so this failed with a hit for a goal whose only
    session had been deleted.
    """
    _save(mem, "unique zebra goal that will be pruned", "zebra outcome", True)
    _save(mem, "keeper session about walrus maintenance", "walrus outcome", True)

    assert len(mem.full_text_search("zebra")) >= 1

    result = mem.prune_sessions(keep_last=1, dry_run=False)
    assert result["deleted"] == 1

    # The zebra session is gone...
    assert mem.get_recent_sessions(limit=10) == [
        s for s in mem.get_recent_sessions(limit=10) if "zebra" not in s["goal"]
    ]
    # ...and, critically, the index must not still return it.
    assert mem.full_text_search("zebra") == []
    # The retained session is still searchable.
    assert len(mem.full_text_search("walrus")) == 1


def test_fts_search_never_returns_deleted_rows(mem):
    """Every hit from full_text_search must join back to a live session row."""
    for i in range(5):
        _save(mem, f"distinctive marmot task {i}", f"marmot outcome {i}", True)

    mem.prune_sessions(keep_last=2, dry_run=False)
    hits = mem.full_text_search("marmot", limit=10)
    assert len(hits) == 2
    live_ids = {r["id"] for r in mem.get_recent_sessions(limit=10)}
    for h in hits:
        assert h["id"] in live_ids


def test_delete_trigger_is_created_idempotently(mem):
    """sessions_ad must exist (and re-init must not fail while it exists)."""
    triggers = {r[0] for r in mem._get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()}
    assert "sessions_ad" in triggers
    assert "sessions_ai" in triggers

    # Re-running init on the same DB must be a no-op, not an error.
    again = MemoryEngine(db_path=mem.db_path)
    triggers2 = {r[0] for r in again._get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()}
    assert {"sessions_ai", "sessions_ad"}.issubset(triggers2)


def test_rebuild_fts_index_repairs_pre_existing_orphans(mem):
    """Simulate a DB written by the old code: orphan the index, then repair it.

    Note the orphan is *invisible* through full_text_search even when present,
    because that query JOINs back to `sessions` and the deleted row is gone.
    The damage is real regardless: the shadow index still holds the terms (and
    would resurface them under a reused rowid). So this test inspects the FTS
    table directly, then proves the rebuild clears it.
    """
    _save(mem, "orphaned ocelot goal", "ocelot outcome", True)
    _save(mem, "healthy ocelot keeper", "keeper outcome", True)

    # Old-style deletion: remove the row without the trigger firing, exactly
    # what the pre-fix code left behind.
    with mem._get_connection() as conn:
        conn.execute("DROP TRIGGER sessions_ad")
        conn.execute("DELETE FROM sessions WHERE goal LIKE 'orphaned%'")
        conn.commit()

    # Re-init must restore the trigger for existing databases.
    mem._init_database()
    triggers = {r[0] for r in mem._get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()}
    assert "sessions_ad" in triggers

    # The orphan is still in the shadow index until rebuilt.
    with mem._get_connection() as conn:
        orphan_count = conn.execute(
            "SELECT COUNT(*) FROM sessions_fts WHERE sessions_fts MATCH 'orphaned'"
        ).fetchone()[0]
    assert orphan_count >= 1

    indexed = mem.rebuild_fts_index()
    assert indexed == 1

    # After the rebuild the orphan terms are gone and the keeper survives.
    with mem._get_connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sessions_fts WHERE sessions_fts MATCH 'orphaned'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sessions_fts WHERE sessions_fts MATCH 'keeper'"
        ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Existing public API must keep working
# ---------------------------------------------------------------------------


def test_public_api_survives_prune_and_checkpoint(mem):
    _save(mem, "api task")
    mem.save_learned_skill(
        name="inspect repo",
        intent_keywords=["inspect", "repo"],
        tool_sequence=["list_directory"],
    )
    mem.save_project_context("/tmp/ws", {"language": "python"}, {"style": "terse"})

    mem.checkpoint_wal()
    mem.prune_sessions(keep_last=10, dry_run=False)

    stats = mem.get_aggregate_stats()
    assert stats["total_sessions"] == 1

    skills = mem.get_all_skills()
    assert len(skills) == 1
    assert mem.find_matching_skill("inspect repo now") is not None
    assert mem.get_project_context("/tmp/ws")["tech_stack"]["language"] == "python"

    mem.update_user_profile("prefers pytest")
    assert "prefers pytest" in mem.get_user_profile()

    assert isinstance(mem.build_recall_context("api task"), str)
    hits = mem.full_text_search("api")
    assert any(h["goal"] == "api task" for h in hits)


def test_save_session_token_savings_none_is_preserved(mem):
    mem.save_session(
        goal="null savings task",
        outcome="ok",
        is_completed=True,
        total_steps=1,
        system_one_steps=1,
        system_two_steps=0,
        total_latency_ms=1.0,
        tokens_used=1,
        token_savings_pct=None,
    )
    row = mem.get_recent_sessions(limit=1)[0]
    assert row["token_savings_pct"] is None
    # Not measured yet => None, not 0.0
    assert mem.get_aggregate_stats()["avg_token_savings_pct"] is None
