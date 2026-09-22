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
        # Previously hardcoded to True, which meant the dashboard could never
        # exercise real Jev routing: opening the UI always showed simulated
        # decisions even with a valid key configured, with nothing in the UI to
        # explain why. Now it goes live when a key exists, and simulation remains
        # available and explicit via DUAL_AGENT_UI_FORCE_SIMULATION=true.
        force_simulation=(
            os.getenv("DUAL_AGENT_UI_FORCE_SIMULATION", "false").lower() == "true"
            or not cfg.typesafe_api_key
        ),
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
    scheduler = CronScheduler(memory_engine=memory, dispatcher_factory=lambda: dispatcher)

    # ── FastAPI app ─────────────────────────────────────────────────────────

    # Tick the scheduler in this process for the dashboard's lifetime.
    #
    # The loop previously called scheduler._tick() on a fixed 20s wait loop.
    # That did dispatch, but it meant /api/schedules advertised jobs while the
    # dashboard was the only process running them — and a scheduled job that
    # needs approval blocked on stdin ("EOF when reading a line") because there
    # is no terminal here. So the dashboard now runs ticks (a job created in the
    # UI fires without a separate daemon) and names its own limitations in the
    # API, rather than listing jobs that look scheduled but cannot complete.
    SCHED_TICK_SECONDS = 20
    scheduler_state = {"task": None, "running": False, "last_error": None}

    async def _scheduler_loop() -> None:
        while True:
            try:
                # _tick() is blocking (SQLite + agent subprocesses); run it off
                # the event loop or every HTTP/WebSocket request stalls behind it.
                await asyncio.get_running_loop().run_in_executor(None, scheduler._tick)
                scheduler_state["last_error"] = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                scheduler_state["last_error"] = str(e)
                logger.warning(f"[Scheduler] Tick failed: {e}")
            await asyncio.sleep(SCHED_TICK_SECONDS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if open_browser:
            async def _open():
                await asyncio.sleep(0.9)
                import webbrowser
                webbrowser.open(f"http://{host}:{port}")
            asyncio.create_task(_open())

        scheduler_state["task"] = asyncio.create_task(_scheduler_loop())
        scheduler_state["running"] = True
        job_count = len(scheduler.list_jobs())
        logger.info(
            f"[Dashboard] Running at http://{host}:{port} "
            f"(scheduler ticking every {SCHED_TICK_SECONDS}s, {job_count} job(s))"
        )
        yield

        # Shutdown: stop the loop and let it unwind before the DB closes.
        scheduler_state["running"] = False
        task = scheduler_state["task"]
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            scheduler_state["task"] = None

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
        """List cron jobs plus the honest execution/delivery contract.

        `scheduler_running` is read from the live loop, not hardcoded: a previous
        version always returned running=True. `delivery` exists because job
        output goes to this process's stdout and is never pushed to a chat —
        the UI must not imply a notification that never arrives.
        """
        return {
            "jobs": scheduler.list_jobs(),
            "scheduler_running": bool(scheduler_state["running"]),
            "tick_seconds": SCHED_TICK_SECONDS,
            "last_tick_error": scheduler_state["last_error"],
            "delivery": "stdout",
            "note": (
                "Jobs run in whichever long-lived process is ticking the "
                "scheduler (this dashboard or `dual-agent gateway`). Output goes "
                "to that process's console and execution memory — it is NOT sent "
                "to any chat. Interactive tools that need approval are denied "
                "here because there is no terminal to answer the prompt; run "
                "`dual-agent gateway` with DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true "
                "for unattended runs."
            ),
        }

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

    @app.get("/api/screen/preview")
    async def api_screen_preview(grid: bool = False):
        try:
            from dual_agent.screen import capture_screenshot, render_grid_overlay
            from dual_agent.system_two import _encode_image_to_data_url

            meta = capture_screenshot()
            img_path = meta["path"]
            if grid:
                img_path = render_grid_overlay(img_path)

            data_url = _encode_image_to_data_url(img_path)
            return {
                "ok": True,
                "data_url": data_url,
                "logical_width": meta["logical_width"],
                "logical_height": meta["logical_height"],
                "pixel_width": meta["pixel_width"],
                "pixel_height": meta["pixel_height"],
                "scale_x": meta["scale_x"],
                "scale_y": meta["scale_y"],
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/screen/click")
    async def api_screen_click(body: dict):
        try:
            from dual_agent.screen import click_mouse
            x = float(body.get("x", 0))
            y = float(body.get("y", 0))
            btn = str(body.get("button", "left"))
            ctype = str(body.get("click_type", "single"))
            res = click_mouse(x=x, y=y, button=btn, click_type=ctype)
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/screen/type")
    async def api_screen_type(body: dict):
        try:
            from dual_agent.screen import send_key_press
            text = str(body.get("text", ""))
            key = body.get("key")
            modifiers = body.get("modifiers")
            if key:
                res = send_key_press(key=str(key), modifiers=modifiers)
            elif text:
                for char in text:
                    send_key_press(key=char)
                res = {"typed": text}
            else:
                return {"ok": False, "error": "No key or text provided"}
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/config")
    async def api_get_config():
        return {
            "typesafe_api_key": cfg.typesafe_api_key or "",
            "typesafe_base_url": cfg.typesafe_base_url,
            "system_two_provider": cfg.system_two_provider,
            "system_one_confidence_threshold": cfg.system_one_confidence_threshold,
            "deepseek_api_key": cfg.deepseek_api_key or "",
            "deepseek_model": cfg.deepseek_model or "deepseek-chat",
            "grok_api_key": cfg.grok_api_key or "",
            "grok_model": cfg.grok_model,
            "openai_api_key": cfg.openai_api_key or "",
            "anthropic_api_key": cfg.anthropic_api_key or "",
            "custom_llm_base_url": cfg.custom_llm_base_url or cfg.hermes_base_url or "http://localhost:11434/v1",
            "custom_llm_model": cfg.custom_llm_model or cfg.hermes_model or "llama3.1",
            "custom_llm_api_key": cfg.custom_llm_api_key or cfg.hermes_api_key or "",
            "vision_provider": cfg.vision_provider or "",
            "vision_model": cfg.vision_model or "",
            "vision_base_url": cfg.vision_base_url or "",
            "vision_api_key": cfg.vision_api_key or "",
            "screen_control_enabled": os.getenv("DUAL_AGENT_SCREEN_CONTROL", "1") != "0",
            "is_simulation": (
                os.getenv("DUAL_AGENT_UI_FORCE_SIMULATION", "false").lower() == "true"
                or not cfg.typesafe_api_key
            ),
        }

    @app.post("/api/config")
    async def api_post_config(body: dict):
        from dual_agent.config import save_config

        # Update config fields if provided
        if "typesafe_api_key" in body:
            cfg.typesafe_api_key = body["typesafe_api_key"].strip() or None
            if cfg.typesafe_api_key:
                os.environ["TYPESAFE_API_KEY"] = cfg.typesafe_api_key
            elif "TYPESAFE_API_KEY" in os.environ:
                del os.environ["TYPESAFE_API_KEY"]

        if "typesafe_base_url" in body and body["typesafe_base_url"].strip():
            cfg.typesafe_base_url = body["typesafe_base_url"].strip()

        if "system_two_provider" in body and body["system_two_provider"].strip():
            cfg.system_two_provider = body["system_two_provider"].strip()
            os.environ["SYSTEM_TWO_PROVIDER"] = cfg.system_two_provider

        if "system_one_confidence_threshold" in body:
            try:
                cfg.system_one_confidence_threshold = float(body["system_one_confidence_threshold"])
            except (ValueError, TypeError):
                pass

        if "deepseek_api_key" in body:
            cfg.deepseek_api_key = body["deepseek_api_key"].strip() or None
            if cfg.deepseek_api_key:
                os.environ["DEEPSEEK_API_KEY"] = cfg.deepseek_api_key
            elif "DEEPSEEK_API_KEY" in os.environ:
                del os.environ["DEEPSEEK_API_KEY"]

        if "deepseek_model" in body and body["deepseek_model"].strip():
            cfg.deepseek_model = body["deepseek_model"].strip()
            os.environ["DEEPSEEK_MODEL"] = cfg.deepseek_model

        if "grok_api_key" in body:
            cfg.grok_api_key = body["grok_api_key"].strip() or None
            if cfg.grok_api_key:
                os.environ["GROK_API_KEY"] = cfg.grok_api_key
            elif "GROK_API_KEY" in os.environ:
                del os.environ["GROK_API_KEY"]

        if "grok_model" in body and body["grok_model"].strip():
            cfg.grok_model = body["grok_model"].strip()

        if "openai_api_key" in body:
            cfg.openai_api_key = body["openai_api_key"].strip() or None
            if cfg.openai_api_key:
                os.environ["OPENAI_API_KEY"] = cfg.openai_api_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

        if "anthropic_api_key" in body:
            cfg.anthropic_api_key = body["anthropic_api_key"].strip() or None
            if cfg.anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = cfg.anthropic_api_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

        custom_url = body.get("custom_llm_base_url") or body.get("hermes_base_url")
        if custom_url and custom_url.strip():
            cfg.custom_llm_base_url = custom_url.strip()
            os.environ["CUSTOM_LLM_BASE_URL"] = cfg.custom_llm_base_url

        custom_model = body.get("custom_llm_model") or body.get("hermes_model")
        if custom_model and custom_model.strip():
            cfg.custom_llm_model = custom_model.strip()
            os.environ["CUSTOM_LLM_MODEL"] = cfg.custom_llm_model

        custom_key = body.get("custom_llm_api_key") or body.get("hermes_api_key")
        if custom_key is not None:
            cfg.custom_llm_api_key = custom_key.strip() or None
            if cfg.custom_llm_api_key:
                os.environ["CUSTOM_LLM_API_KEY"] = cfg.custom_llm_api_key
            elif "CUSTOM_LLM_API_KEY" in os.environ:
                del os.environ["CUSTOM_LLM_API_KEY"]

        if "vision_provider" in body:
            cfg.vision_provider = body["vision_provider"].strip() or None
            if cfg.vision_provider:
                os.environ["VISION_PROVIDER"] = cfg.vision_provider
            elif "VISION_PROVIDER" in os.environ:
                del os.environ["VISION_PROVIDER"]

        if "vision_model" in body:
            cfg.vision_model = body["vision_model"].strip() or None
            if cfg.vision_model:
                os.environ["VISION_MODEL"] = cfg.vision_model
            elif "VISION_MODEL" in os.environ:
                del os.environ["VISION_MODEL"]

        if "vision_base_url" in body:
            cfg.vision_base_url = body["vision_base_url"].strip() or None
            if cfg.vision_base_url:
                os.environ["VISION_BASE_URL"] = cfg.vision_base_url
            elif "VISION_BASE_URL" in os.environ:
                del os.environ["VISION_BASE_URL"]

        if "vision_api_key" in body:
            cfg.vision_api_key = body["vision_api_key"].strip() or None
            if cfg.vision_api_key:
                os.environ["VISION_API_KEY"] = cfg.vision_api_key
            elif "VISION_API_KEY" in os.environ:
                del os.environ["VISION_API_KEY"]

        # Persist updated configuration
        save_config(cfg)

        # Hot-reload live System 1 and System 2 services
        new_s1 = JevSystemOneClient(
            api_key=cfg.typesafe_api_key,
            base_url=cfg.typesafe_base_url,
            force_simulation=(
                os.getenv("DUAL_AGENT_UI_FORCE_SIMULATION", "false").lower() == "true"
                or not cfg.typesafe_api_key
            ),
        )
        new_s2 = get_system_two_provider(cfg.system_two_provider)

        dispatcher.s1 = new_s1
        dispatcher.s2 = new_s2
        dispatcher.confidence_threshold = cfg.system_one_confidence_threshold

        logger.info(f"[Config] Updated config: S1 (key_set={bool(cfg.typesafe_api_key)}), S2 ({cfg.system_two_provider})")

        return {
            "ok": True,
            "provider": cfg.system_two_provider.upper(),
            "confidence_threshold": cfg.system_one_confidence_threshold,
            "is_simulation": not bool(cfg.typesafe_api_key),
        }

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
