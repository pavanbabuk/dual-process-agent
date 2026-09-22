"""Native Desktop Application launcher for Dual-Process Agent using WebKit / pywebview."""

from __future__ import annotations
import os
import sys
import time
import socket
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def find_free_port(start_port: int = 7860) -> int:
    """Find an available loopback port starting from start_port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def run_desktop_app(
    port: Optional[int] = None,
    debug: bool = False,
    title: str = "Dual-Process Agent",
) -> None:
    """Start the local runtime engine and open a native desktop application window."""
    try:
        import webview
    except ImportError:
        print(
            "Desktop application window requires pywebview.\n"
            "Install with: pip install 'dual-agent[app]'\n"
            "Alternatively, run the browser interface with: dual-agent ui\n"
        )
        sys.exit(1)

    import uvicorn
    from dual_agent.web.server import create_app

    target_port = port or find_free_port(7860)
    app = create_app(host="127.0.0.1", port=target_port, open_browser=False)

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=target_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(server_config)

    # Start uvicorn server in background thread
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to bind
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", target_port)) == 0:
                break
        time.sleep(0.05)

    app_url = f"http://127.0.0.1:{target_port}"
    logger.info(f"Launching native desktop window pointing to {app_url}")

    # Create native desktop window
    window = webview.create_window(
        title=title,
        url=app_url,
        width=1280,
        height=850,
        min_size=(960, 640),
        text_select=True,
    )

    try:
        webview.start(debug=debug)
    finally:
        server.should_exit = True
