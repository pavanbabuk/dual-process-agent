"""Tests for the Web API (FastAPI REST endpoints)."""

import pytest
import os

# Skip all if fastapi not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a TestClient with an isolated tmp data directory."""
    data_dir = tmp_path_factory.mktemp("dual_agent_web_test")
    os.environ["DUAL_AGENT_DATA_DIR"] = str(data_dir)
    os.environ["DUAL_AGENT_AUTO_ALLOW_PERMISSIONS"] = "true"

    from dual_agent.web.server import create_app
    app = create_app(host="localhost", port=7860, open_browser=False)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_dashboard_html_serves(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Dual-Process Agent" in r.text
    assert "<html" in r.text.lower()


def test_api_tools_returns_list(client):
    r = client.get("/api/tools")
    assert r.status_code == 200
    data = r.json()
    assert "tools" in data
    assert isinstance(data["tools"], list)
    assert len(data["tools"]) >= 4
    names = [t["name"] for t in data["tools"]]
    assert "run_shell_command" in names
    assert "list_directory" in names


def test_api_tools_risk_fields(client):
    r = client.get("/api/tools")
    tools = {t["name"]: t for t in r.json()["tools"]}
    shell = tools["run_shell_command"]
    assert shell["requires_approval"] is True
    assert shell["risk_level"] == "high"
    write = tools["write_file"]
    assert write["requires_approval"] is True
    assert write["risk_level"] == "medium"
    read = tools["list_directory"]
    assert read["requires_approval"] is False


def test_api_skills_empty_initially(client):
    r = client.get("/api/skills")
    assert r.status_code == 200
    assert "skills" in r.json()
    assert isinstance(r.json()["skills"], list)


def test_api_memory_returns_stats(client):
    r = client.get("/api/memory")
    assert r.status_code == 200
    data = r.json()
    assert "stats" in data
    stats = data["stats"]
    assert "total_sessions" in stats
    assert "total_s1_steps" in stats


def test_api_schedules_empty(client):
    r = client.get("/api/schedules")
    assert r.status_code == 200
    assert "jobs" in r.json()


def test_api_add_schedule_valid(client):
    r = client.post("/api/schedule", json={
        "description": "every day at 9am",
        "goal": "Send daily git status"
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "job_id" in body

    # Now it should appear in list
    r2 = client.get("/api/schedules")
    jobs = r2.json()["jobs"]
    assert any(j["description"] == "every day at 9am" for j in jobs)


def test_api_add_schedule_invalid_nl(client):
    r = client.post("/api/schedule", json={
        "description": "whenever I feel like it",
        "goal": "Do something"
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "error" in body


def test_api_profile_empty(client):
    r = client.get("/api/profile")
    assert r.status_code == 200
    assert "profile" in r.json()


def test_websocket_ping_pong(client):
    with client.websocket_connect("/ws") as ws:
        init = ws.receive_json()  # consume init event
        assert init["type"] == "init"
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        assert msg["type"] == "pong"


def test_websocket_init_event(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "provider" in msg
        assert "tool_count" in msg
        assert msg["tool_count"] >= 4


def test_websocket_run_goal(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # consume init
        ws.send_json({"type": "run", "goal": "List the files in the current directory"})

        events = []
        # receive_json() blocks until data arrives — no timeout kwarg in TestClient
        for _ in range(30):
            try:
                msg = ws.receive_json()
                events.append(msg)
                if msg["type"] in ("done", "error"):
                    break
            except Exception:
                break  # WebSocket closed or real error

        types = [e["type"] for e in events]
        assert "started" in types
        assert "done" in types or "error" in types


def test_websocket_empty_goal_returns_error(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # consume init
        ws.send_json({"type": "run", "goal": ""})
        msg = ws.receive_json()
        assert msg["type"] == "error"
