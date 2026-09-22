"""MCP Manager for registering and managing external MCP servers.

Supports standard Claude Desktop compatible mcp_servers.json format.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from dual_agent.memory import get_default_data_dir
from dual_agent.mcp_host import MCPHost, MCPToolDefinition

logger = logging.getLogger(__name__)


class MCPServerEntry(BaseModel):
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    disabled: bool = False


class MCPManager:
    """Manages external MCP server configurations and attaches tools to MCPHost."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(get_default_data_dir(), "mcp_servers.json")
        self._ensure_config_file()

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
            return True
        return False

    def _save_servers(self, servers: Dict[str, MCPServerEntry]) -> None:
        payload = {"mcpServers": {k: v.model_dump() for k, v in servers.items()}}
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def attach_to_host(self, host: MCPHost) -> int:
        """Attach configured external servers to the active MCP host."""
        servers = self.load_servers()
        attached_count = 0
        for name, srv in servers.items():
            # External MCP servers are placeholders until a real stdio client is implemented.
            # Fail loudly on execution rather than pretending dispatch succeeded.
            def _unconnected_handler(args, n=name, cmd=srv.command):
                raise RuntimeError(
                    f"External MCP server '{n}' ({cmd}) is configured but not connected: "
                    f"subprocess/JSON-RPC client is not implemented."
                )

            tool_name = f"mcp_{name}_dispatch"
            host.register_tool(
                MCPToolDefinition(
                    name=tool_name,
                    description=f"External MCP Server '{name}' ({srv.command}) [NOT CONNECTED]",
                    parameters_schema={
                        "type": "object",
                        "properties": {"action": {"type": "string"}, "payload": {"type": "object"}},
                    },
                    handler=_unconnected_handler,
                )
            )
            attached_count += 1
        return attached_count
