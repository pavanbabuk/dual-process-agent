"""Tests for CronScheduler (NL cron parsing + job CRUD)."""

import pytest
from dual_agent.memory import MemoryEngine
from dual_agent.scheduler import CronScheduler, parse_nl_to_cron


# ── NL Parser tests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("description,expected", [
    ("every minute", "* * * * *"),
    ("every 5 minutes", "*/5 * * * *"),
    ("every 15 minutes", "*/15 * * * *"),
    ("every hour", "0 * * * *"),
    ("every night", "0 2 * * *"),
    ("nightly", "0 2 * * *"),
    ("every morning", "0 9 * * *"),
    ("every week", "0 9 * * 1"),
    ("weekly", "0 9 * * 1"),
    ("every day at 9am", "0 9 * * *"),
    ("daily at 9am", "0 9 * * *"),
    ("every day at 6pm", "0 18 * * *"),
    ("every day at 3:30", "30 3 * * *"),
    ("every Monday at 8am", "0 8 * * 1"),
    ("every Friday at 5pm", "0 17 * * 5"),
])
def test_parse_nl_to_cron(description, expected):
    result = parse_nl_to_cron(description)
    assert result == expected, f"'{description}' → '{result}' (expected '{expected}')"


def test_parse_nl_unknown_returns_none():
    assert parse_nl_to_cron("when the stars align") is None
    assert parse_nl_to_cron("") is None


# ── Scheduler CRUD tests ─────────────────────────────────────────────────────

@pytest.fixture
def scheduler(tmp_path):
    mem = MemoryEngine(db_path=str(tmp_path / "sched.db"))
    return CronScheduler(memory_engine=mem)


def test_add_job_with_nl_description(scheduler):
    job_id = scheduler.add_job(
        description="every day at 9am",
        goal="Send a daily git status report",
    )
    assert isinstance(job_id, int)
    assert job_id > 0


def test_add_job_with_explicit_cron(scheduler):
    job_id = scheduler.add_job(
        description="every day backup",
        goal="Run backup script",
        cron_expr="0 2 * * *",
    )
    jobs = scheduler.list_jobs()
    assert any(j["id"] == job_id and j["cron_expr"] == "0 2 * * *" for j in jobs)


def test_add_job_invalid_nl_raises(scheduler):
    with pytest.raises(ValueError, match="cron"):
        scheduler.add_job(description="banana time", goal="do something")


def test_list_jobs_returns_all(scheduler):
    scheduler.add_job("every minute", "check status")
    scheduler.add_job("every hour", "hourly report")
    jobs = scheduler.list_jobs()
    assert len(jobs) == 2


def test_remove_job(scheduler):
    job_id = scheduler.add_job("every minute", "ping")
    removed = scheduler.remove_job(job_id)
    assert removed is True
    assert scheduler.list_jobs() == []


def test_remove_nonexistent_job_returns_false(scheduler):
    assert scheduler.remove_job(9999) is False


def test_enable_disable_job(scheduler):
    job_id = scheduler.add_job("every hour", "report")
    scheduler.enable_job(job_id, enabled=False)
    jobs = scheduler.list_jobs()
    job = next(j for j in jobs if j["id"] == job_id)
    assert not job["enabled"]

    scheduler.enable_job(job_id, enabled=True)
    jobs = scheduler.list_jobs()
    job = next(j for j in jobs if j["id"] == job_id)
    assert job["enabled"]
