"""Scheduler wiring tests.

Defect under test: the dashboard constructed a CronScheduler whose jobs are
listed by /api/schedules, but nothing dispatched them for the dashboard's own
lifetime — a UI-created job never ran, and the API still reported running=True.

These tests assert dispatch actually happened, by observing a real dispatcher's
side effect (and the job's persisted run_count), not that a mock was called.
"""

import asyncio
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from starlette.testclient import TestClient


class RecordingDispatcher:
    """Minimal stand-in for DualProcessDispatcher that records real invocations.

    It is the *transport* into the agent that is stubbed (no network, no LLM),
    while the acceptance signal below — the dispatch actually happening end to
    end through the app's scheduler loop — is real.
    """

    def __init__(self, sink):
        self._sink = sink

    def run(self, goal, max_steps=10):
        self._sink.append(goal)

        class _Result:
            final_output = "recorded"

        return _Result()


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Dashboard app with a recording dispatcher installed in its scheduler."""
    monkeypatch.setenv("DUAL_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DUAL_AGENT_AUTO_ALLOW_PERMISSIONS", "true")

    fired = []
    import dual_agent.scheduler as sch
    from dual_agent.web.server import create_app

    original_init = sch.CronScheduler.__init__

    def recording_init(self, memory_engine, dispatcher_factory=None):
        # Force the factory so a dispatched job reaches RecordingDispatcher
        # instead of the real agent (which would need network + approvals).
        original_init(
            self,
            memory_engine,
            dispatcher_factory=lambda: RecordingDispatcher(fired),
        )

    monkeypatch.setattr(sch.CronScheduler, "__init__", recording_init)

    app = create_app(host="localhost", port=7860, open_browser=False)
    with TestClient(app) as client:
        yield client, fired


def test_scheduled_job_actually_dispatches_while_dashboard_runs(app_client):
    """A due job fires: real dispatch observed within one tick interval."""
    client, fired = app_client

    r = client.post("/api/schedule", json={
        "description": "every minute",
        "goal": "PROBE-DISPATCH",
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # The app's loop ticks every tick_seconds; allow one interval plus slack.
    tick = client.get("/api/schedules").json()["tick_seconds"]
    deadline = time.time() + tick + 15
    while time.time() < deadline and not fired:
        time.sleep(0.5)

    assert fired == ["PROBE-DISPATCH"], (
        "dashboard never dispatched the due job — /api/schedules would be "
        "advertising a job that cannot run"
    )

    # The run must be persisted, not merely observed in memory.
    jobs = client.get("/api/schedules").json()["jobs"]
    job = next(j for j in jobs if j["goal"] == "PROBE-DISPATCH")
    assert job["run_count"] >= 1
    assert job["last_run_at"] is not None
    assert job["last_status"] == "ok"


def test_scheduler_loop_starts_and_stops_with_lifespan():
    """scheduler_running reflects the live loop, not a hardcoded True."""
    import tempfile
    import dual_agent.scheduler as sch
    from dual_agent.web.server import create_app

    old_data = os.environ.get("DUAL_AGENT_DATA_DIR")
    os.environ["DUAL_AGENT_DATA_DIR"] = tempfile.mkdtemp(prefix="sched_life_")

    fired = []
    original_init = sch.CronScheduler.__init__

    def recording_init(self, memory_engine, dispatcher_factory=None):
        original_init(self, memory_engine,
                      dispatcher_factory=lambda: RecordingDispatcher(fired))

    sch.CronScheduler.__init__ = recording_init
    try:
        app = create_app(open_browser=False)
        with TestClient(app) as client:
            body = client.get("/api/schedules").json()
            assert body["scheduler_running"] is True
            assert body["tick_seconds"] > 0
            assert body["last_tick_error"] is None
    finally:
        sch.CronScheduler.__init__ = original_init
        if old_data is None:
            os.environ.pop("DUAL_AGENT_DATA_DIR", None)
        else:
            os.environ["DUAL_AGENT_DATA_DIR"] = old_data


def test_api_schedules_states_execution_and_delivery_limits(app_client):
    """Contract: jobs list must not imply chat notification or approval support."""
    client, _ = app_client
    body = client.get("/api/schedules").json()

    assert "jobs" in body
    # Delivery is explicitly stdout, and named as such in prose too.
    assert body["delivery"] == "stdout"
    note = body["note"]
    assert "NOT sent" in note and "chat" in note.lower()
    # The dashboard cannot answer approval prompts; that must be stated.
    assert "approval" in note.lower()
    assert "gateway" in note.lower()


def test_tick_continues_after_a_failing_job(tmp_path, monkeypatch):
    """One raising job must not abort the remaining jobs in the same pass."""
    monkeypatch.setenv("DUAL_AGENT_DATA_DIR", str(tmp_path))
    from dual_agent.memory import MemoryEngine
    from dual_agent.scheduler import CronScheduler

    memory = MemoryEngine()
    calls = []

    class Flaky:
        def run(self, goal, max_steps=10):
            calls.append(goal)
            if goal == "BOOM":
                raise RuntimeError("job blew up")

            class _R:
                final_output = "ok"

            return _R()

    sched = CronScheduler(memory_engine=memory, dispatcher_factory=lambda: Flaky())
    sched.add_job(description="every minute", goal="BOOM")
    sched.add_job(description="every minute", goal="FINE")

    sched._tick()  # must not raise

    assert calls == ["BOOM", "FINE"], "a failing job starved the next job"
    jobs = {j["goal"]: j for j in sched.list_jobs()}
    assert jobs["BOOM"]["last_status"].startswith("error:")
    assert jobs["FINE"]["last_status"] == "ok"


def test_scheduler_dedupes_within_the_same_minute(tmp_path, monkeypatch):
    """A due job runs once per minute window, not on every tick."""
    monkeypatch.setenv("DUAL_AGENT_DATA_DIR", str(tmp_path))
    from dual_agent.memory import MemoryEngine
    from dual_agent.scheduler import CronScheduler

    memory = MemoryEngine()
    fired = []
    sched = CronScheduler(
        memory_engine=memory,
        dispatcher_factory=lambda: RecordingDispatcher(fired),
    )
    sched.add_job(description="every minute", goal="ONCE")

    sched._tick()
    sched._tick()
    sched._tick()

    assert fired == ["ONCE"], f"job re-ran inside one minute window: {fired}"
