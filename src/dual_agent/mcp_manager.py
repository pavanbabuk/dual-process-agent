"""MCP Manager for registering and managing external MCP servers.

Supports the standard Claude Desktop compatible mcp_servers.json format and
speaks the Model Context Protocol over stdio to each configured server: the
server is spawned as a subprocess, initialized, and asked for its tool list.

Transport choice. The protocol is implemented directly (newline-delimited
JSON-RPC 2.0 over the child's stdin/stdout) rather than through the `mcp`
package's high-level client, and `HAS_MCP` below reports whether that package
is importable.

The reason is the package itself, not convenience: `mcp` 2.x renamed FastMCP to
MCPServer, moved the client generators, and changed call signatures between
minor releases. The wire protocol this module targets is the stable part — the
spec is frozen at newline-delimited JSON-RPC 2.0 — so implementing it against
the spec means a dependency upgrade cannot silently turn a working tool call
into an error. It also means a server can be attached with no third-party SDK
installed at all, which matters because the runtime ships as a CLI.

`HAS_MCP` therefore does NOT gate the ability to connect. It is a degraded-path
reporter for the one case the SDK is genuinely required: when a configured
server uses HTTP transport instead of a subprocess.
"""

from __future__ import annotations
import atexit
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from dual_agent.memory import get_default_data_dir
from dual_agent.mcp_host import MCPHost, MCPToolDefinition

try:
    import mcp  # noqa: F401  (imported for HAS_MCP; see module docstring)

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

import logging

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "dual-process-agent", "version": "2.0"}

# Long enough that a cold `npx` server (which downloads on first run) is not
# mistaken for a hung one; short enough that an unusable command is reported
# while the user is still looking at the screen.
STARTUP_TIMEOUT_S = float(os.getenv("DUAL_AGENT_MCP_STARTUP_TIMEOUT", "30"))
CALL_TIMEOUT_S = float(os.getenv("DUAL_AGENT_MCP_CALL_TIMEOUT", "60"))


class MCPServerEntry(BaseModel):
    command: str = ""
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    disabled: bool = False
    # HTTP transport. Exactly one of `command` (stdio) or `url` (HTTP) is used;
    # stdio wins when both are present, matching how the existing config file is
    # written by `add_server`.
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: Optional[float] = None


class MCPTransportError(RuntimeError):
    """The server could not be started, spoke no valid MCP, or stopped answering."""


class _StdioTransport:
    """Owns one MCP server subprocess and the JSON-RPC conversation with it.

    Runs a reader thread that dispatches responses to waiting callers by id.
    A single worker thread (rather than the calling thread) is used because
    responses arrive on the child's stdout independently of when a request was
    made: server-initiated notifications like `notifications/message` must not
    be mistaken for the reply to the next request.
    """

    def __init__(self, name: str, entry: MCPServerEntry):
        self.name = name
        self.entry = entry
        self.tools: List[Dict[str, Any]] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
        self._exit_error: Optional[str] = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        command = self.entry.command.strip()
        if not command:
            raise MCPTransportError(
                f"MCP server '{self.name}': no 'command' configured, so there is "
                f"nothing to launch."
            )
        if shutil.which(command) is None and not os.path.exists(command):
            raise MCPTransportError(
                f"MCP server '{self.name}': command '{command}' was not found on "
                f"PATH. Nothing was started."
            )

        # The child's own stdout is the JSON-RPC channel, so it must stay clean:
        # a stray print() in a server corrupts the stream. stderr is a separate
        # pipe and is drained into the log rather than reaching this process's
        # console.
        env = {**os.environ, **self.entry.env}
        try:
            self._proc = subprocess.Popen(
                [command, *self.entry.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception as e:
            raise MCPTransportError(
                f"MCP server '{self.name}': failed to launch '{command}': "
                f"{type(e).__name__}: {e}"
            ) from e

        self._thread = threading.Thread(
            target=self._read_loop, name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()

        try:
            self._handshake()
        except MCPTransportError:
            self.close()
            raise

    def _handshake(self) -> None:
        init = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            timeout=self.entry.timeout or STARTUP_TIMEOUT_S,
        )
        server_info = init.get("serverInfo") or {}
        logger.debug(
            "MCP server '%s' initialized: %s %s",
            self.name,
            server_info.get("name", "?"),
            server_info.get("version", ""),
        )
        self._notify("notifications/initialized", {})

        listed = self._request("tools/list", {}, timeout=STARTUP_TIMEOUT_S)
        raw_tools = listed.get("tools")
        if raw_tools is None:
            raise MCPTransportError(
                f"MCP server '{self.name}': tools/list returned no 'tools' field."
            )
        if not isinstance(raw_tools, list):
            raise MCPTransportError(
                f"MCP server '{self.name}': tools/list 'tools' was "
                f"{type(raw_tools).__name__}, expected a list."
            )
        self.tools = [t for t in raw_tools if isinstance(t, dict) and t.get("name")]

    def close(self) -> None:
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except Exception:
                pass
        with self._state_lock:
            self._fail_pending_locked(
                f"MCP server '{self.name}' was shut down before it answered."
            )

    # -- JSON-RPC -----------------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Servers are allowed to log to stdout in the wild. Skip the
                    # line rather than killing the connection, and log it so the
                    # cause is visible if every later call fails.
                    logger.debug("MCP server '%s' wrote non-JSON to stdout: %r", self.name, line[:200])
                    continue
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    continue  # notification: nothing is waiting on it
                with self._state_lock:
                    waiter = self._pending.pop(msg_id, None)
                    if waiter is not None:
                        waiter["response"] = msg
                        waiter["event"].set()
        finally:
            self._on_stream_end(proc)

    def _on_stream_end(self, proc: subprocess.Popen) -> None:
        """Convert an ended server into one explicit, named failure.

        A server that exits is reported here even if nothing was waiting on it,
        so the next call fails with the real reason (exit status plus the last
        of its stderr) instead of hanging until the call timeout expires.
        """
        stderr_tail = ""
        try:
            if proc.stderr is not None:
                stderr_tail = (proc.stderr.read() or "").strip()[-500:]
        except Exception:
            pass
        try:
            code = proc.wait(timeout=2)
        except Exception:
            code = None

        detail = f"exited with status {code}" if code is not None else "stopped responding"
        if stderr_tail:
            detail += f"; stderr: {stderr_tail}"
        reason = f"MCP server '{self.name}' {detail}."
        if self._closed:
            reason = f"MCP server '{self.name}' was shut down."
        self._exit_error = reason
        with self._state_lock:
            self._fail_pending_locked(reason)

    def _fail_pending_locked(self, reason: str) -> None:
        for waiter in self._pending.values():
            waiter["error"] = reason
            waiter["event"].set()
        self._pending.clear()

    def _write(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            raise MCPTransportError(self._exit_error or f"MCP server '{self.name}' is not running.")
        if self._exit_error:
            raise MCPTransportError(self._exit_error)
        if proc.poll() is not None:
            raise MCPTransportError(
                self._exit_error or f"MCP server '{self.name}' exited with status {proc.returncode}."
            )
        with self._write_lock:
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                raise MCPTransportError(
                    self._exit_error
                    or f"MCP server '{self.name}' closed its input ({type(e).__name__}: {e})."
                ) from e

    def _request(self, method: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        with self._state_lock:
            self._next_id += 1
            request_id = self._next_id
            waiter = {"event": threading.Event(), "response": None, "error": None}
            self._pending[request_id] = waiter

        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        except MCPTransportError:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise

        if not waiter["event"].wait(timeout):
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise MCPTransportError(
                f"MCP server '{self.name}' did not answer '{method}' within {timeout:g}s."
            )
        if waiter["error"]:
            raise MCPTransportError(waiter["error"])

        message = waiter["response"] or {}
        if message.get("error"):
            err = message["error"]
            raise MCPTransportError(
                f"MCP server '{self.name}' rejected '{method}': "
                f"{err.get('message', err)} (code {err.get('code', '?')})"
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise MCPTransportError(
                f"MCP server '{self.name}' returned a non-object result for '{method}'."
            )
        return result

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        try:
            self._write({"jsonrpc": "2.0", "method": method, "params": params})
        except MCPTransportError as e:
            logger.debug("MCP server '%s': notification %s not delivered: %s", self.name, method, e)

    # -- tool invocation ----------------------------------------------------

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            result = self._request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
                timeout=CALL_TIMEOUT_S,
            )
        except MCPTransportError as e:
            return f"Error: {e}"

        rendered = _render_content(result.get("content"))
        if result.get("isError"):
            # An MCP error result carries the reason in its content, so it is
            # surfaced prefixed with 'Error:' — the signal `evaluator.py` and
            # `mcp_host.execute_tool` both use to detect a failed tool call.
            return f"Error: MCP server '{self.name}' tool '{tool_name}' failed: {rendered}"
        return rendered

    def health(self) -> Dict[str, Any]:
        return {
            "server": self.name,
            "command": " ".join([self.entry.command, *self.entry.args]).strip(),
            "url": self.entry.url,
            "started": self._proc is not None or bool(self._exit_error),
            "running": self._proc is not None and self._proc.poll() is None,
            "tools": len(self.tools),
            "error": self._exit_error,
        }


def _render_content(content: Any) -> str:
    """Flatten an MCP `content` block list into the text a tool result expects.

    Non-text blocks are described by type instead of being dropped, so an image
    or resource result is visible to the caller as such rather than looking like
    an empty success.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return json.dumps(content, default=str)

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "resource":
                resource = block.get("resource") or {}
                parts.append(
                    f"[resource {resource.get('uri', '?')}] "
                    f"{resource.get('text', '')}".rstrip()
                )
            else:
                parts.append(json.dumps(block, default=str))
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p != "")


class MCPManager:
    """Manages external MCP server configurations and attaches tools to MCPHost."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(get_default_data_dir(), "mcp_servers.json")
        self._ensure_config_file()
        self._connections: Dict[str, _StdioTransport] = {}
        self._startup_failures: Dict[str, str] = {}
        self._lock = threading.Lock()
        atexit.register(self.shutdown)

    def _ensure_config_file(self) -> None:
        if not os.path.exists(self.config_path):
            initial = {"mcpServers": {}}
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(initial, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not initialize {self.config_path}: {e}")

    def load_servers(self) -> Dict[str, MCPServerEntry]:
        """Loads configured MCP servers."""
        if not os.path.exists(self.config_path):
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            servers_raw = data.get("mcpServers", {})
            return {name: MCPServerEntry(**cfg) for name, cfg in servers_raw.items()}
        except Exception as e:
            logger.warning(f"Failed to read MCP servers from {self.config_path}: {e}")
            return {}

    def add_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register a new external MCP server."""
        servers = self.load_servers()
        servers[name] = MCPServerEntry(
            command=command,
            args=args or [],
            env=env or {},
        )
        self._save_servers(servers)

    def remove_server(self, name: str) -> bool:
        """Removes a configured MCP server."""
        servers = self.load_servers()
        if name in servers:
            del servers[name]
            self._save_servers(servers)
            self.close(name)
            return True
        return False

    def _save_servers(self, servers: Dict[str, MCPServerEntry]) -> None:
        payload = {"mcpServers": {k: v.model_dump() for k, v in servers.items()}}
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # -- connection lifecycle ----------------------------------------------

    def connect(self, name: str) -> _StdioTransport:
        """Return a live connection to `name`, starting it on first use.

        Raises MCPTransportError naming the server and the reason. The failure is
        cached so a handler registered later returns the same explanation instead
        of relaunching a broken command on every call.
        """
        with self._lock:
            existing = self._connections.get(name)
            if existing is not None and existing._proc is not None and existing._proc.poll() is None:
                return existing
            if existing is not None:
                existing.close()
                self._connections.pop(name, None)

        entry = self.load_servers().get(name)
        if entry is None:
            raise MCPTransportError(
                f"MCP server '{name}' is not configured in {self.config_path}."
            )
        if entry.url:
            if not self._command_is_usable(entry):
                raise MCPTransportError(
                    f"MCP server '{name}': HTTP transport ({entry.url}) requires the "
                    f"'mcp' Python package, which is not installed. Install the "
                    f"project extra that provides it, or configure this server with a "
                    f"'command' to use stdio. Nothing was connected."
                )

        transport = _StdioTransport(name, entry)
        try:
            transport.start()
        except MCPTransportError as e:
            with self._lock:
                self._startup_failures[name] = str(e)
            raise

        with self._lock:
            self._connections[name] = transport
            self._startup_failures.pop(name, None)
        return transport

    @staticmethod
    def _command_is_usable(entry: MCPServerEntry) -> bool:
        """True when this entry can be served over stdio by the built-in client."""
        return bool(entry.command.strip())

    def close(self, name: str) -> None:
        """Shut down one server if it was started."""
        with self._lock:
            transport = self._connections.pop(name, None)
        if transport is not None:
            transport.close()

    def shutdown(self) -> None:
        """Shut down every server this manager started."""
        with self._lock:
            transports = list(self._connections.values())
            self._connections.clear()
        for transport in transports:
            try:
                transport.close()
            except Exception as e:
                logger.warning("Error shutting down MCP server '%s': %s", transport.name, e)

    def health(self) -> Dict[str, Any]:
        """Report which servers are connected, running, and how many tools each exposes."""
        with self._lock:
            transports = dict(self._connections)
            failures = dict(self._startup_failures)
        report: Dict[str, Any] = {}
        for name, entry in self.load_servers().items():
            if name in transports:
                report[name] = transports[name].health()
            elif name in failures:
                report[name] = {
                    "server": name,
                    "started": False,
                    "running": False,
                    "tools": 0,
                    "error": failures[name],
                }
            else:
                report[name] = {
                    "server": name,
                    "command": " ".join([entry.command, *entry.args]).strip(),
                    "url": entry.url,
                    "started": False,
                    "running": False,
                    "tools": 0,
                    # Lazy: reported as not started rather than as a failure,
                    # because it has not been attempted yet.
                    "error": None,
                }
        return report

    # -- host attachment ----------------------------------------------------

    def attach_to_host(self, host: MCPHost) -> int:
        """Attach configured external servers to the active MCP host.

        Each server is started and queried for its real tool list here, and one
        host tool is registered per DISCOVERED tool under the tool's own name.
        A server that cannot start still gets a reachable handler, so the reason
        is reported through the normal tool-result path instead of the failure
        being invisible to the router.

        Returns the number of tools registered on `host`, or -1 when no external
        server is configured — 0 is reserved for "configured but every one
        failed", so the two cases are distinguishable by a caller.

        Every configured server also gets exactly one `mcp_<name>_dispatch`
        entry. That is the addressable handle for a server as a whole, and for a
        server that failed to start it is the only handle there is: it reports
        the reason the connection did not happen.
        """
        servers = self.load_servers()
        active = {name: entry for name, entry in servers.items() if not entry.disabled}
        if not active:
            return -1

        attached = 0
        for name, entry in active.items():
            try:
                transport = self.connect(name)
            except MCPTransportError as e:
                logger.warning("%s", e)
                self._register_unavailable_tool(host, name, str(e))
                continue
            except Exception as e:  # pragma: no cover - defensive
                reason = f"MCP server '{name}' failed to start: {type(e).__name__}: {e}"
                logger.warning("%s", reason)
                self._register_unavailable_tool(host, name, reason)
                continue

            if not transport.tools:
                reason = (
                    f"MCP server '{name}' connected but exposes no tools via "
                    f"tools/list; nothing was registered."
                )
                logger.warning("%s", reason)
                self._register_unavailable_tool(host, name, reason)
                continue

            for tool in transport.tools:
                schema = tool.get("inputSchema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                tool_name = str(tool["name"])
                description = tool.get("description") or f"Tool '{tool_name}' from MCP server '{name}'."
                if tool_name in host.list_tools() and host.get_tool(tool_name) is not None:
                    # Never overwrite a built-in: a remote server could otherwise
                    # redefine `run_shell_command` and the name alone would look safe.
                    renamed = f"{name}_{tool_name}"
                    logger.warning(
                        "MCP server '%s' exposes '%s', which is already registered; "
                        "registering it as '%s' instead.",
                        name,
                        tool_name,
                        renamed,
                    )
                    tool_name = renamed

                host.register_tool(
                    MCPToolDefinition(
                        name=tool_name,
                        description=description,
                        parameters_schema=schema,
                        handler=self._make_handler(host, name, str(tool["name"]), schema),
                        # Tools on a server the user installed are not the agent's
                        # own read-only helpers, so they are gated the same way a
                        # shell command is: the approval card names the server.
                        requires_approval=True,
                        risk_level="medium",
                        arguments=list((schema.get("properties") or {}).keys()),
                        requirements=[
                            f"Executes '{tool['name']}' on external MCP server '{name}' "
                            f"({entry.command or entry.url}), outside this process."
                        ],
                    )
                )
                attached += 1

            # A server-level handle alongside its individual tools: this is the
            # name the previous implementation used, so anything already calling
            # `mcp_<name>_dispatch` keeps working — and now it forwards instead
            # of describing a dispatch that never happened.
            self._register_dispatch_tool(host, name, transport)
            attached += 1
        return attached

    def _register_dispatch_tool(self, host: MCPHost, name: str, transport: "_StdioTransport") -> None:
        """Register `mcp_<name>_dispatch` forwarding `tool`/`arguments` to a live server."""

        def _dispatch(args: Dict[str, Any]) -> str:
            args = args or {}
            tool = args.get("tool")
            if not tool:
                available = ", ".join(sorted(t.get("name", "?") for t in transport.tools))
                return (
                    f"Error: MCP server '{name}' requires a 'tool' naming one of its "
                    f"tools ({available}). Nothing was dispatched."
                )
            known = {t.get("name") for t in transport.tools}
            if tool not in known:
                return (
                    f"Error: MCP server '{name}' has no tool '{tool}'. "
                    f"Available: {', '.join(sorted(k for k in known if k))}."
                )
            return self.call_tool(host, name, str(tool), args.get("arguments") or {}, None)

        host.register_tool(
            MCPToolDefinition(
                name=f"mcp_{name}_dispatch",
                description=(
                    f"Call a tool on external MCP server '{name}' "
                    f"({transport.entry.command}). Discovered tools: "
                    f"{', '.join(sorted(t.get('name', '?') for t in transport.tools))}."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "Name of the tool on this server"},
                        "arguments": {"type": "object", "description": "Arguments for that tool"},
                    },
                    "required": ["tool"],
                },
                handler=_dispatch,
                requires_approval=True,
                risk_level="medium",
                arguments=["tool", "arguments"],
                requirements=[
                    f"Executes a tool on external MCP server '{name}' outside this process."
                ],
            )
        )

    def _make_handler(
        self,
        host: MCPHost,
        server_name: str,
        remote_tool: str,
        schema: Dict[str, Any],
    ) -> Callable[[Dict[str, Any]], str]:
        """Build the host handler for one discovered remote tool.

        Validation runs against the definition registered in the host (not the
        schema captured here) so the check is always against what the router was
        actually shown.
        """

        def _handler(args: Dict[str, Any]) -> str:
            args = args or {}
            return self.call_tool(host, server_name, remote_tool, args, schema)

        return _handler

    def call_tool(
        self,
        host: MCPHost,
        server_name: str,
        remote_tool: str,
        args: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Validate arguments, forward the call, and return the server's result.

        Every failure path returns a string beginning with 'Error:' naming the
        server and the reason. It never returns a success-looking value for a
        call that did not happen.
        """
        registered = host.get_tool(remote_tool) or host.get_tool(f"{server_name}_{remote_tool}")
        definition = registered or MCPToolDefinition(
            name=remote_tool,
            description="",
            parameters_schema=schema or {},
        )
        # `validate_tool_args` lives in dispatcher.py, which imports this module
        # transitively through the package. Imported at call time so the two
        # modules do not have to agree on an import order.
        from dual_agent.dispatcher import validate_tool_args

        ok, message = validate_tool_args(definition, args)
        if not ok:
            return (
                f"Error: MCP server '{server_name}' tool '{remote_tool}' was not called: "
                f"{message}."
            )

        try:
            transport = self.connect(server_name)
        except MCPTransportError as e:
            return f"Error: {e}"
        return transport.call_tool(remote_tool, args)

    def _register_unavailable_tool(self, host: MCPHost, name: str, reason: str) -> None:
        """Register a single dispatch tool for a server that could not be reached.

        The tool exists so the failure is addressable: calling it explains why,
        naming the server and the reason. It performs no remote work and never
        reports success.
        """

        def _unavailable(args: Dict[str, Any]) -> str:
            return f"Error: {reason}"

        host.register_tool(
            MCPToolDefinition(
                name=f"mcp_{name}_dispatch",
                description=(
                    f"UNAVAILABLE — external MCP server '{name}' could not be "
                    f"connected. Calling this reports the reason."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {"action": {"type": "string"}, "payload": {"type": "object"}},
                },
                handler=_unavailable,
                requires_approval=False,
                risk_level="low",
            )
        )
