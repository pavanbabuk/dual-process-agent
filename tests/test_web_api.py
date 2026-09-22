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


def test_api_schedules_reports_live_scheduler_state(client):
    """The contract must expose live scheduler state, not a hardcoded running=True.

    Previously /api/schedules returned `running: True` unconditionally while the
    dashboard's tick loop never dispatched, so the UI advertised jobs that could
    not run.
    """
    r = client.get("/api/schedules")
    assert r.status_code == 200
    body = r.json()

    assert body["scheduler_running"] is True  # this client runs the lifespan
    assert isinstance(body["tick_seconds"], int) and body["tick_seconds"] > 0
    assert "last_tick_error" in body

    # Delivery is stdout, and the note must say chat delivery does not happen.
    assert body["delivery"] == "stdout"
    note = body["note"]
    assert "NOT sent" in note and "chat" in note.lower()
    assert "gateway" in note.lower()


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


def test_api_config_get_and_post(client):
    # GET config
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg_data = r.json()
    assert "system_two_provider" in cfg_data
    assert "system_one_confidence_threshold" in cfg_data

    # POST config updates
    r2 = client.post("/api/config", json={
        "system_two_provider": "mock",
        "system_one_confidence_threshold": 0.90,
        "grok_api_key": "xai-test-key-12345",
    })
    assert r2.status_code == 200
    res = r2.json()
    assert res["ok"] is True
    assert res["provider"] == "MOCK"
    assert res["confidence_threshold"] == 0.90

    # Verify GET returns updated values
    r3 = client.get("/api/config")
    cfg_updated = r3.json()
    assert cfg_updated["system_two_provider"] == "mock"
    assert cfg_updated["system_one_confidence_threshold"] == 0.90
    assert cfg_updated["grok_api_key"] == "xai-test-key-12345"


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


def test_api_screen_preview(client):
    r = client.get("/api/screen/preview")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "data_url" in data
    assert data["data_url"].startswith("data:image/png;base64,")
    assert data["logical_width"] > 0
    assert data["logical_height"] > 0
    assert data["scale_x"] >= 1.0


def test_api_screen_preview_with_grid(client):
    r = client.get("/api/screen/preview?grid=true")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "data_url" in data
    assert data["data_url"].startswith("data:image/png;base64,")


def test_api_screen_click_bounds(client):
    r = client.post("/api/screen/click", json={"x": 99999, "y": 99999})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert "out of bounds" in data["error"]


def test_api_screen_type(client):
    # Empty payload returns error
    r = client.post("/api/screen/type", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is False

    # Invalid key returns error
    r_inv = client.post("/api/screen/type", json={"key": "nonexistent_special_key"})
    assert r_inv.status_code == 200
    assert r_inv.json()["ok"] is False
    assert "Unrecognized key" in r_inv.json()["error"]

    # Valid key with mocked actuation
    from unittest.mock import patch
    with patch("dual_agent.screen.send_key_press", return_value={"key": "return"}):
        r2 = client.post("/api/screen/type", json={"key": "return"})
        assert r2.status_code == 200
        assert r2.json()["ok"] is True


def test_api_config_vision_fields(client):
    r = client.post("/api/config", json={
        "vision_provider": "openai",
        "vision_model": "gpt-4o",
        "vision_base_url": "https://api.openai.com/v1",
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    cfg = r2.json()
    assert cfg["vision_provider"] == "openai"
    assert cfg["vision_model"] == "gpt-4o"
    assert cfg["vision_base_url"] == "https://api.openai.com/v1"
    assert "screen_control_enabled" in cfg

