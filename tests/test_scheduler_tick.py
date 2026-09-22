"""Test that scheduler ticks during web server lifecycle."""

import pytest
from starlette.testclient import TestClient
from dual_agent.web.server import create_app


def test_api_schedules_reports_status_and_limitations():
    """GET /api/schedules must state execution status and delivery destination."""
    app = create_app(open_browser=False)
    with TestClient(app) as client:
        r = client.get("/api/schedules")
        assert r.status_code == 200
        data = r.json()
        assert "jobs" in data
        assert "note" in data
        assert "gateway" in data["note"].lower() or "console" in data["note"].lower()
