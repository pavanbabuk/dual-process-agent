"""Regression tests for defects found by running the product, not by reading it.

Covers:
  * FTS5 recall silently degrading to a LIKE scan on ordinary goal text
  * the dashboard being hardcoded to simulation, so live Jev routing was
    unreachable from the UI
  * an unrecognised WebSocket approval decision raising ValueError mid-run
  * a fabricated savings claim in the Hermes bridge docstring
"""

import os
import re
import sqlite3

import pytest

from dual_agent.memory import MemoryEngine, _safe_fts_query

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def mem(tmp_path):
    m = MemoryEngine(db_path=str(tmp_path / "mem.db"))
    m.save_session(
        goal="Add a --version flag to src/dual_agent/cli.py",
        outcome="Added it.",
        is_completed=True,
        total_steps=2,
        system_one_steps=1,
        system_two_steps=1,
        total_latency_ms=12.0,
        tokens_used=0,
        steps=[],
    )
    return m


# ----------------------------------------------------------------------
# FTS5 query sanitisation
# ----------------------------------------------------------------------

def test_safe_fts_query_quotes_every_token():
    assert _safe_fts_query("list files") == '"list" OR "files"'
    # Punctuation is dropped, not passed through as query syntax.
    assert _safe_fts_query("--version") == '"version"'
    assert _safe_fts_query("src/dual_agent/cli.py") == (
        '"src" OR "dual_agent" OR "cli" OR "py"'
    )
    assert _safe_fts_query("dual-agent") == '"dual" OR "agent"'


def test_safe_fts_query_handles_fts_operators_as_literal_text():
    """Bare AND/OR/NOT must not be treated as operators."""
    out = _safe_fts_query("cats AND dogs")
    assert out == '"cats" OR "AND" OR "dogs"'
    assert _safe_fts_query("NEAR") == '"NEAR"'


def test_safe_fts_query_returns_empty_for_pure_punctuation():
    assert _safe_fts_query("-- ") == ""
    assert _safe_fts_query("") == ""


@pytest.mark.parametrize(
    "query",
    [
        "--version",              # a CLI flag
        "src/dual_agent/cli.py",  # a path
        "dual-agent",            # a hyphenated name
        'the "quoted" thing',    # quote characters
        "wild*card",             # an FTS prefix operator
        "paren(the)sis",         # grouping characters
        "cats AND dogs OR NOT",  # bare keywords
    ],
)
def test_recall_never_raises_and_never_degrades(mem, query, caplog):
    """Each of these used to trip the FTS5 parser and fall back to LIKE.

    The fallback is the observable failure: full-text ranking silently stopped
    working for file paths, CLI flags and hyphenated names, which is most of what
    this agent is asked to do.
    """
    with caplog.at_level("WARNING"):
        mem.full_text_search(query)
        mem.build_recall_context(query)
    assert "FTS search failed" not in caplog.text, (
        f"query {query!r} fell back to LIKE instead of using FTS5"
    )


def test_recall_still_finds_the_session(mem):
    """Sanitising must not break matching: the terms are still real terms."""
    assert mem.full_text_search("version"), "lost the hit for a substring of the goal"
    assert mem.full_text_search("cli.py"), "lost the hit for a path token"


def test_like_fallback_uses_the_same_tokenization(mem):
    """The fallback must agree with FTS on what the query means."""
    hits = mem._fallback_search("--version")
    assert hits, "LIKE fallback found nothing for '--version' (it kept the dashes)"


# ----------------------------------------------------------------------
# Dashboard must be able to reach live Jev routing
# ----------------------------------------------------------------------

def test_dashboard_is_not_hardcoded_to_simulation():
    """Reading the source is the honest check here.

    The dashboard previously passed force_simulation=True unconditionally, so
    opening the UI always showed simulated routing even with a valid key, with
    nothing in the UI explaining why.
    """
    source = open(os.path.join(REPO_ROOT, "src/dual_agent/web/server.py")).read()
    assert "force_simulation=True," not in source, (
        "dashboard is hardcoded to simulation again"
    )
    assert "DUAL_AGENT_UI_FORCE_SIMULATION" in source, (
        "there must be an explicit, documented way to force simulation"
    )


# ----------------------------------------------------------------------
# Untrusted approval decisions must not crash the run
# ----------------------------------------------------------------------

def test_unrecognised_approval_decision_denies_instead_of_raising():
    from dual_agent.permission_broker import ApprovalDecision

    # Mirrors streaming_dispatcher's guarded conversion.
    for bad in ("yolo", "", "ALLOW", "maybe"):
        try:
            decision = ApprovalDecision(bad)
        except ValueError:
            decision = ApprovalDecision.DENY
        assert decision == ApprovalDecision.DENY


def test_web_dispatcher_source_guards_the_enum():
    source = open(
        os.path.join(REPO_ROOT, "src/dual_agent/web/streaming_dispatcher.py")
    ).read()
    assert "except ValueError" in source
    assert "ApprovalDecision.DENY" in source


# ----------------------------------------------------------------------
# No unmeasured performance claims
# ----------------------------------------------------------------------

def test_no_unmeasured_savings_claims_in_docstrings():
    """This project has no baseline harness, so it must not quote a savings figure.

    The README already had invented numbers removed; the Hermes bridge docstring
    still claimed 'up to 80% of outer-loop token costs', which nothing measures.
    """
    offenders = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__", ".pytest_cache"}]
        for name in files:
            if not name.endswith((".py", ".md")):
                continue
            path = os.path.join(root, name)
            text = open(path, encoding="utf-8", errors="ignore").read()
            for match in re.finditer(r"(?:save|saving|reduce|reduction|cut)[^.\n]{0,40}?(\d{1,3})\s*%", text, re.I):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{os.path.relpath(path, REPO_ROOT)}:{line}: {match.group(0)!r}")
    assert not offenders, "unmeasured savings claim(s) found:\n  " + "\n  ".join(offenders)
