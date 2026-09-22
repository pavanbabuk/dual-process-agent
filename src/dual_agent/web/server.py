"""FastAPI web server for Dual-Process Agent dashboard.

Endpoints:
  GET  /                  → Serve dashboard HTML
  GET  /api/skills        → List .SKILL.md files
  GET  /api/memory        → Aggregate stats + recent sessions
  GET  /api/schedules     → List cron jobs
  GET  /api/tools         → List registered MCP tools
  GET  /api/profile       → USER.md profile content
  POST /api/schedule      → Add a scheduled job
  WebSocket /ws           → Real-time bidirectional agent stream

Usage:
  dual-agent --ui           # starts on http://localhost:7860
  dual-agent --ui --port 8080
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def create_app(
    host: str = "localhost",
    port: int = 7860,
    open_browser: bool = True,
) -> "FastAPI":
    """Factory that creates and configures the FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required for the web UI.\n"
            "Install with:  pip install 'dual-agent[ui]'"
        )

    from dual_agent.config import load_config
    from dual_agent.memory import MemoryEngine
    from dual_agent.mcp_host import MCPHost
    from dual_agent.mcp_manager import MCPManager
    from dual_agent.typesafe_client import JevSystemOneClient
    from dual_agent.system_two import get_system_two_provider
    from dual_agent.dispatcher import DualProcessDispatcher
    from dual_agent.skills_manager import SkillsManager
    from dual_agent.scheduler import CronScheduler
    from dual_agent.web.streaming_dispatcher import StreamingDispatcher, resolve_permission

    # ── Bootstrap shared state ──────────────────────────────────────────────
    cfg = load_config()
    memory = MemoryEngine()
    mcp = MCPHost()
    MCPManager().attach_to_host(mcp)
    skills = SkillsManager()
    scheduler = CronScheduler(memory_engine=memory)

    s1 = JevSystemOneClient(
        api_key=cfg.typesafe_api_key,
        base_url=cfg.typesafe_base_url,
        force_simulation=True,
    )
    s2 = get_system_two_provider(cfg.system_two_provider)
    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=mcp,
        memory_engine=memory,
        skills_manager=skills,
        confidence_threshold=cfg.system_one_confidence_threshold,
    )

    # ── FastAPI app ─────────────────────────────────────────────────────────

    # Use lifespan instead of deprecated on_event
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if open_browser:
            async def _open():
                await asyncio.sleep(0.9)
                import webbrowser
                webbrowser.open(f"http://{host}:{port}")
            asyncio.create_task(_open())
        logger.info(f"[Dashboard] Running at http://{host}:{port}")
        yield
        # Shutdown (nothing to clean up)

    app = FastAPI(title="Dual-Process Agent", version="2.0.0", lifespan=lifespan)

    WEB_DIR = Path(__file__).parent
    TEMPLATES_DIR = WEB_DIR / "templates"
    STATIC_DIR = WEB_DIR / "static"
    STATIC_DIR.mkdir(exist_ok=True)

    # Cache the dashboard HTML at startup (avoid Jinja2 Python 3.14 cache bug)
    _dashboard_html_path = TEMPLATES_DIR / "dashboard.html"
    _dashboard_html_cache: dict = {}

    def _render_dashboard() -> str:
        key = str(_dashboard_html_path)
        if key not in _dashboard_html_cache or not _dashboard_html_cache[key]:
            raw = _dashboard_html_path.read_text(encoding="utf-8")
            _dashboard_html_cache[key] = raw
        html = _dashboard_html_cache[key]
        # Simple token substitution (no Jinja2 needed)
        return (
            html
            .replace("{{ provider }}", cfg.system_two_provider.upper())
            .replace("{{ tool_count }}", str(len(mcp.list_tools())))
            .replace("{{ version }}", "2.0.0")
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Routes ──────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(content=_render_dashboard())

    @app.get("/api/skills")
    async def api_skills():
        skill_list = skills.load_all_skills()
        return {
            "skills": [
                {
                    "name": s.name,
                    "keywords": s.intent_keywords,
                    "tool_sequence": s.tool_sequence,
                    "success_count": s.success_count,
                    "last_used": s.last_used,
                }
                for s in skill_list
            ]
        }

    @app.get("/api/memory")
    async def api_memory():
        stats = memory.get_aggregate_stats()
        recent = memory.full_text_search("", limit=0)  # just stats
        learned = memory.get_all_skills()
        return {
            "stats": stats,
            "learned_skills": [
                {"name": s.name, "success_count": s.success_count, "tool_sequence": s.tool_sequence}
                for s in learned[:10]
            ],
        }

    @app.get("/api/tools")
    async def api_tools():
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "requires_approval": t.requires_approval,
                    "risk_level": t.risk_level,
                }
                for t in mcp.list_tools()
            ]
        }

    @app.get("/api/schedules")
    async def api_schedules():
        return {"jobs": scheduler.list_jobs()}

    @app.post("/api/schedule")
    async def api_add_schedule(body: dict):
        description = body.get("description", "")
        goal = body.get("goal", description)
        try:
            job_id = scheduler.add_job(description=description, goal=goal)
            return {"ok": True, "job_id": job_id}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/profile")
    async def api_profile():
        return {"profile": memory.get_user_profile()}

    # ── WebSocket ────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        logger.info("[WS] Client connected")

        async def send_event(event: dict):
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                pass

        stream = StreamingDispatcher(dispatcher=dispatcher, send_event=send_event)

        # Send initial state
        await send_event({
            "type": "init",
            "provider": cfg.system_two_provider.upper(),
            "tool_count": len(mcp.list_tools()),
            "skills_count": len(skills.load_all_skills()),
        })

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "run":
                    goal = msg.get("goal", "").strip()
                    if not goal:
                        await send_event({"type": "error", "message": "Goal cannot be empty."})
                        continue
                    max_steps = int(msg.get("max_steps", 15))
                    asyncio.create_task(stream.run_streaming(goal=goal, max_steps=max_steps))

                elif msg_type == "permission_response":
                    request_id = msg.get("request_id", "")
                    decision = msg.get("decision", "deny")
                    edited_args = msg.get("edited_args")
                    await resolve_permission(request_id, decision, edited_args)

                elif msg_type == "ping":
                    await send_event({"type": "pong"})

        except WebSocketDisconnect:
            logger.info("[WS] Client disconnected")
        except Exception as e:
            logger.error(f"[WS] Error: {e}")

    return app


def run_server(host: str = "localhost", port: int = 7860, open_browser: bool = True):
    """Entry point called from CLI --ui flag.

    Binds to loopback by default. This dashboard exposes an unauthenticated
    WebSocket that can execute tools (including shell commands) with your
    privileges — binding it to a non-loopback address without the deliberate
    opt-in below would publish that endpoint to the network.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required.\n"
            "Install with:  pip install 'dual-agent[ui]'"
        )
    import uvicorn

    if host not in ("localhost", "127.0.0.1", "::1"):
        if os.getenv("DUAL_AGENT_UI_ALLOW_PUBLIC_BIND", "false").lower() != "true":
            raise SystemExit(
                f"Refusing to bind the agent dashboard to {host}: it has no "
                "authentication and can execute tools as you.\n"
                "Use --host=127.0.0.1, or set DUAL_AGENT_UI_ALLOW_PUBLIC_BIND=true "
                "if you have put your own auth/reverse proxy in front of it."
            )
        logger.warning(
            f"[Dashboard] Binding to {host} with NO authentication — any host that "
            "can reach this port can run tools as you."
        )

    # Print banner before starting uvicorn (uses stderr so it doesn't interfere)
    print(f"\n  ⚡ Dual-Process Agent Dashboard v2.0", flush=True)
    print(f"  → http://{host}:{port}", flush=True)
    print(f"  Press Ctrl+C to stop\n", flush=True)

    if open_browser:
        import webbrowser, threading
        def _open():
            import time; time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    app = create_app(host=host, port=port, open_browser=False)
    uvicorn.run(app, host=host, port=port, log_level="info")
