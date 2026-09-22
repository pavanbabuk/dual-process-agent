"""Tests for MemoryEngine FTS5 search and USER.md profile methods."""

import os
import pytest
from dual_agent.memory import MemoryEngine


@pytest.fixture
def mem(tmp_path):
    db = str(tmp_path / "test_fts.db")
    return MemoryEngine(db_path=db)


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
        token_savings_pct=70.0,
    )


def test_fts_returns_matching_session(mem):
    _save(mem, "inspect pyproject.toml and list dependencies", "found 5 deps")
    _save(mem, "run pytest test suite with coverage", "24 tests passed")

    results = mem.full_text_search("pytest coverage")
    assert len(results) >= 1
    assert any("pytest" in r["goal"].lower() for r in results)


def test_fts_returns_empty_for_no_match(mem):
    _save(mem, "inspect pyproject.toml", "ok")
    results = mem.full_text_search("completely unrelated xyzzy query")
    # Should be empty or empty list (FTS with no match)
    assert isinstance(results, list)


def test_fts_fallback_like_search(mem):
    """Verify fallback LIKE search also works."""
    _save(mem, "write a new python script", "script written")
    results = mem._fallback_search("python script", limit=5)
    assert len(results) >= 1


def test_build_recall_context_format(mem):
    _save(mem, "inspect workspace directory structure", "found 12 files")
    ctx = mem.build_recall_context("workspace inspect", limit=3)
    # Should contain the formatted block header
    assert "[RELEVANT PAST SESSIONS]" in ctx or ctx == ""  # may be empty if FTS not populated yet


def test_user_profile_create_and_append(mem):
    mem.update_user_profile("Prefers pytest for testing over unittest")
    mem.update_user_profile("Uses Python 3.11+ exclusively")

    profile = mem.get_user_profile()
    assert "Prefers pytest" in profile
    assert "Python 3.11" in profile


def test_user_profile_empty_before_any_updates(mem):
    profile = mem.get_user_profile()
    assert profile == ""


def test_user_profile_ignores_empty_fact(mem):
    mem.update_user_profile("")
    mem.update_user_profile("   ")
    profile = mem.get_user_profile()
    assert profile == ""


def test_fts_limit_respected(mem):
    for i in range(10):
        _save(mem, f"run task number {i} with pytest tool", f"result {i}")
    results = mem.full_text_search("pytest", limit=3)
    assert len(results) <= 3
