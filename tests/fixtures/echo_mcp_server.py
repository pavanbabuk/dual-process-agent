"""A REAL, minimal MCP stdio server used as a test fixture.

This is a genuine MCP server: it speaks newline-delimited JSON-RPC 2.0 on
stdin/stdout, performs the initialize handshake, answers `tools/list`, and
computes the result of `tools/call` itself. It is not a mock and it is not a
stand-in — the client under test connects to this process exactly as it would
connect to a published server.

Written against the protocol spec rather than any SDK so it cannot drift when
the `mcp` package changes.

Usage (the stdio client spawns it this way):

    python tests/fixtures/echo_mcp_server.py

Failure modes are selected by argv so a single file covers startup failure,
tool failure, and mid-session exit:

    --fail-startup     exit(3) before answering initialize
    --fail-tool        return isError=True from tools/call
    --exit-after-init  answer initialize, then exit(0)
    --sleep <seconds>  delay each response (used to prove timeouts fire)
"""

from __future__ import annotations
import json
import sys
import time

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "add_numbers",
        "description": "Add two integers and return the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First addend"},
                "b": {"type": "integer", "description": "Second addend"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "reverse_text",
        "description": "Reverse the characters of the supplied text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to reverse"},
            },
            "required": ["text"],
        },
    },
]


def _args() -> dict:
    return {"a": 0, "b": 0}


def _call(name: str, arguments: dict) -> dict:
    """Compute the real result. Nothing here consults the client."""
    if name == "add_numbers":
        return {"content": [{"type": "text", "text": str(int(arguments["a"]) + int(arguments["b"]))}]}
    if name == "reverse_text":
        return {"content": [{"type": "text", "text": str(arguments["text"])[::-1]}]}
    return {
        "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
        "isError": True,
    }


def main() -> int:
    argv = sys.argv[1:]
    fail_startup = "--fail-startup" in argv
    fail_tool = "--fail-tool" in argv
    exit_after_init = "--exit-after-init" in argv
    sleep_for = 0.0
    if "--sleep" in argv:
        sleep_for = float(argv[argv.index("--sleep") + 1])

    if fail_startup:
        # Real failure, before the protocol starts. The client must report this
        # rather than claiming the server was attached.
        sys.stderr.write("echo_mcp_server: deliberate startup failure (--fail-startup)\n")
        sys.stderr.flush()
        return 3

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        if sleep_for:
            time.sleep(sleep_for)

        method = request.get("method")
        request_id = request.get("id")

        # Notifications carry no id and expect no reply.
        if request_id is None:
            continue

        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "echo-mcp-server", "version": "1.0.0"},
            }
            if exit_after_init:
                _send({"jsonrpc": "2.0", "id": request_id, "result": result})
                return 0
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if fail_tool:
                result = {
                    "content": [{"type": "text", "text": f"tool '{name}' refused on purpose"}],
                    "isError": True,
                }
            else:
                result = _call(str(name), arguments)
        else:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )
            continue

        _send({"jsonrpc": "2.0", "id": request_id, "result": result})

    return 0


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
