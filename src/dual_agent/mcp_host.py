"""Model Context Protocol (MCP) Host and Tool Registry."""

from __future__ import annotations
import ast
import os
import shlex
import subprocess
import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def validate_python_syntax(path: str, content: str) -> Optional[str]:
    """Return an error message if `content` is not valid Python, else None.

    Validating BEFORE writing is what makes a write non-destructive. This was
    added after watching the agent rewrite its own cli.py: it reproduced the file
    from memory rather than editing it, silently dropped the module docstring's
    triple quotes and an unrelated import, produced a file that could not be
    parsed at all — and still reported "Task completed". A write that corrupts
    the target and reports success is worse than a write that refuses, so
    invalid Python is rejected at the boundary instead of being persisted.
    """
    if not path.endswith(".py"):
        return None
    try:
        ast.parse(content)
    except SyntaxError as e:
        return (
            f"refusing to write invalid Python to {path}: "
            f"line {e.lineno}: {e.msg}. The file was NOT modified."
        )
    return None


class MCPToolDefinition(BaseModel):
    """Metadata and execution handler for an MCP tool."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    handler: Optional[Callable[[Dict[str, Any]], Any]] = None
    is_safe: bool = True
    # Permission broker fields (OpenMausBot-style approval gating)
    requires_approval: bool = False
    risk_level: str = "low"  # "low", "medium", "high"
    # Declared interface, used by the fast-path argument validator and surfaced on
    # the approval card. `requirements` states why the tool may ask for input,
    # so an approval is granted against a stated purpose rather than vibes.
    arguments: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    requires_user_input: bool = False
    accepts_user_input: bool = False


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
            syntax_error = validate_python_syntax(path, content)
            if syntax_error:
                return f"Error: {syntax_error}"
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
                description=(
                    "Create or overwrite a file in the workspace. Rewrites the ENTIRE file, "
                    "so prefer patch_file when changing an existing file: a full rewrite "
                    "requires reproducing every unrelated line from memory, and anything "
                    "misremembered is silently lost. Invalid Python is rejected."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "Content string to write"},
                    },
                    "required": ["path", "content"],
                },
                handler=_write_file,
                requires_approval=True,
                risk_level="medium",
            )
        )

        # 5. patch_file — surgical edit for an EXISTING file.
        #
        # write_file forces the model to reproduce the whole file from memory,
        # and anything it misremembers is silently destroyed. That is exactly how
        # the agent corrupted its own cli.py. A targeted replacement only needs
        # the lines it is actually changing, so unrelated content cannot be lost.
        def _patch_file(args: Dict[str, Any]) -> str:
            path = args.get("path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")
            replace_all = bool(args.get("replace_all", False))

            if not old_string:
                return "Error: 'old_string' must not be empty."
            if old_string == new_string:
                return "Error: 'old_string' and 'new_string' are identical; nothing to do."
            if not os.path.isfile(path):
                return f"Error: File '{path}' does not exist. Use write_file to create it."

            try:
                with open(path, "r", encoding="utf-8") as f:
                    original = f.read()
            except Exception as e:
                return f"Error reading '{path}': {e}"

            occurrences = original.count(old_string)
            # Ambiguity is refused rather than guessed: replacing the wrong one of
            # several identical snippets silently corrupts behaviour while looking
            # like a successful edit.
            if occurrences == 0:
                return (
                    f"Error: 'old_string' was not found in {path}. Read the file and "
                    "copy the text to replace exactly, including indentation."
                )
            if occurrences > 1 and not replace_all:
                return (
                    f"Error: 'old_string' appears {occurrences} times in {path}. "
                    "Include more surrounding context to make it unique, or pass "
                    "replace_all=true if every occurrence should change."
                )

            updated = (
                original.replace(old_string, new_string)
                if replace_all
                else original.replace(old_string, new_string, 1)
            )

            syntax_error = validate_python_syntax(path, updated)
            if syntax_error:
                return f"Error: {syntax_error}"

            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(updated)
            except Exception as e:
                return f"Error writing '{path}': {e}"

            return (
                f"Patched {path}: replaced {occurrences if replace_all else 1} "
                f"occurrence(s), {len(original)} -> {len(updated)} chars"
            )

        self.register_tool(
            MCPToolDefinition(
                name="patch_file",
                description=(
                    "Edit an existing file by replacing an exact string. Use this instead of "
                    "write_file to change one part of a file. 'old_string' must match the "
                    "file exactly (including indentation) and must be unique unless "
                    "replace_all is true. Invalid Python results are rejected and the file "
                    "is left unchanged."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "old_string": {"type": "string", "description": "Exact text to replace"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Replace every occurrence"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
                handler=_patch_file,
                requires_approval=True,
                risk_level="medium",
            )
        )

        # 6. run_shell_command
        def _run_cmd(args: Dict[str, Any]) -> str:
            command = args.get("command", "")
            if not command:
                return "Error: No command provided — refusing to execute an empty command."
            # SECURITY: no shell. `shell=True` on a command assembled from model
            # output plus a string-interpolated argument turns any quoting mistake
            # into command injection (e.g. path='"; rm -rf ~ #'). argv form also
            # means the permission card shows the real executable, so approving
            # "ls -la" cannot actually run something else.
            try:
                argv = shlex.split(command)
            except ValueError as e:
                return f"Error: could not parse command ({e}). Use simple argv-style commands."
            if not argv:
                return "Error: No command provided — refusing to execute an empty command."

            try:
                proc = subprocess.run(
                    argv,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                output = proc.stdout if proc.returncode == 0 else proc.stderr
                return f"ExitCode: {proc.returncode}\nOutput:\n{output.strip()[:2000]}"
            except FileNotFoundError as e:
                return f"Error: command not found: {e}"
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 15 seconds."
            except Exception as e:
                return f"Error executing command: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="run_shell_command",
                description=(
                    "Execute a shell command in the local environment and return stdout/stderr. "
                    "Arguments are passed as argv (no shell), so pipes, redirects, globbing and "
                    "$$VAR expansion are not available — call the target binary directly."
                ),
                # The card shows argv; requirements state why arguments were accepted.
                requires_user_input=True,
                accepts_user_input=True,
                parameters_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                    "required": ["command"],
                },
                handler=_run_cmd,
                requires_approval=True,
                risk_level="high",
                arguments=["command"],
                requirements=[
                    "run_shell_command executes with the agent's own privileges; "
                    "only commands needed for this task may be requested.",
                    "The agent must not request commands that delete, overwrite, or exfiltrate data.",
                ],
            )
        )
