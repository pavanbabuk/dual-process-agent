"""Model Context Protocol (MCP) Host and Tool Registry."""

from __future__ import annotations
import os
import subprocess
import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPToolDefinition(BaseModel):
    """Metadata and execution handler for an MCP tool."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    handler: Optional[Callable[[Dict[str, Any]], Any]] = None
    is_safe: bool = True


class ToolExecutionResult(BaseModel):
    """Result of an MCP tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class MCPHost:
    """Manages MCP tool registration, schema conversion for Jev, and execution."""

    def __init__(self):
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._register_default_tools()

    def register_tool(self, tool: MCPToolDefinition) -> None:
        """Register an MCP tool into the host."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered MCP tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[MCPToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[MCPToolDefinition]:
        return list(self._tools.values())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Dictionary of tool names to their descriptions formatted for Jev Choice."""
        return {name: tool.description for name, tool in self._tools.items()}

    def get_formatted_tool_list_for_system_two(self) -> str:
        """Detailed formatted tool specifications for System 2 prompts."""
        lines = []
        for name, tool in self._tools.items():
            params_str = json.dumps(tool.parameters_schema.get("properties", {}), indent=2)
            lines.append(f"### Tool: `{name}`\n{tool.description}\nParameters:\n{params_str}\n")
        return "\n".join(lines)

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a registered tool and record timing."""
        tool = self.get_tool(name)
        if not tool:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' is not registered in MCP host.",
            )

        start = time.perf_counter()
        try:
            if tool.handler:
                result = tool.handler(arguments)
            else:
                result = f"Executed {name} with args {arguments}"

            elapsed_ms = (time.perf_counter() - start) * 1000
            return ToolExecutionResult(
                tool_name=name,
                success=True,
                output=result,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )

    def _register_default_tools(self) -> None:
        """Register built-in local workspace tools (FileSystem, Shell, System)."""
        
        # 1. list_directory
        def _list_dir(args: Dict[str, Any]) -> str:
            path = args.get("path", ".")
            try:
                entries = os.listdir(path)
                return json.dumps({"directory": path, "entries": entries[:50], "total": len(entries)})
            except Exception as e:
                return f"Error listing directory {path}: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="list_directory",
                description="List contents of a local directory or inspect workspace files.",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Target folder path"}},
                    "required": ["path"],
                },
                handler=_list_dir,
            )
        )

        # 2. read_file
        def _read_file(args: Dict[str, Any]) -> str:
            path = args.get("path", "")
            if not os.path.exists(path):
                return f"Error: File '{path}' does not exist."
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read(10000)
                return content
            except Exception as e:
                return f"Error reading '{path}': {e}"

        self.register_tool(
            MCPToolDefinition(
                name="read_file",
                description="Read contents of a text file from workspace.",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Path to file"}},
                    "required": ["path"],
                },
                handler=_read_file,
            )
        )

        # 3. write_file
        def _write_file(args: Dict[str, Any]) -> str:
            path = args.get("path", "")
            content = args.get("content", "")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Successfully wrote {len(content)} characters to {path}"
            except Exception as e:
                return f"Error writing to '{path}': {e}"

        self.register_tool(
            MCPToolDefinition(
                name="write_file",
                description="Create or overwrite a file in the workspace.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "Content string to write"},
                    },
                    "required": ["path", "content"],
                },
                handler=_write_file,
            )
        )

        # 4. run_shell_command
        def _run_cmd(args: Dict[str, Any]) -> str:
            command = args.get("command", "")
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                output = proc.stdout if proc.returncode == 0 else proc.stderr
                return f"ExitCode: {proc.returncode}\nOutput:\n{output.strip()[:2000]}"
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 15 seconds."
            except Exception as e:
                return f"Error executing command: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="run_shell_command",
                description="Execute a bash/shell command in the local environment and return stdout/stderr.",
                parameters_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                    "required": ["command"],
                },
                handler=_run_cmd,
            )
        )
