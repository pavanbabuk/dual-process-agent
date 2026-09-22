"""Model Context Protocol (MCP) Host and Tool Registry."""

from __future__ import annotations
import ast
import os
import shlex
import shutil
import subprocess
import sys
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
                line_start = args.get("line_start")
                line_end = args.get("line_end")
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                total_lines = len(lines)

                if line_start is not None or line_end is not None:
                    start_num = max(1, int(line_start or 1))
                    end_num = min(total_lines, int(line_end or total_lines))
                    selected = lines[start_num - 1 : end_num]
                    numbered = [f"{start_num + i}: {l}" for i, l in enumerate(selected)]
                    return "".join(numbered)

                content = "".join(lines)
                limit = 10000
                total_chars = len(content)
                if total_chars > limit:
                    truncated_count = total_chars - limit
                    return (
                        content[:limit]
                        + f"\n...[truncated {truncated_count} characters of {total_chars}. "
                        f"Use read_file with line_start and line_end to view remaining lines.]"
                    )
                return content
            except Exception as e:
                return f"Error reading '{path}': {e}"

        self.register_tool(
            MCPToolDefinition(
                name="read_file",
                description=(
                    "Read contents of a text file from workspace. Supports optional line_start and "
                    "line_end (1-indexed) to inspect specific sections or large files."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "line_start": {"type": "integer", "description": "Optional 1-indexed starting line number"},
                        "line_end": {"type": "integer", "description": "Optional 1-indexed ending line number"},
                    },
                    "required": ["path"],
                },
                handler=_read_file,
            )
        )

        # 3. search_file
        def _search_file(args: Dict[str, Any]) -> str:
            path = args.get("path", "")
            query = args.get("query", "")
            if not query:
                return "Error: 'query' must not be empty."
            if not os.path.isfile(path):
                return f"Error: File '{path}' does not exist."
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                matches = []
                for i, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        matches.append(f"{i}: {line}")
                if not matches:
                    return f"No matches found for '{query}' in {path} ({len(lines)} lines searched)."
                return f"Found {len(matches)} match(es) for '{query}' in {path}:\n" + "".join(matches[:40])
            except Exception as e:
                return f"Error searching in '{path}': {e}"

        self.register_tool(
            MCPToolDefinition(
                name="search_file",
                description="Search for a text pattern or symbol in a file. Returns matching line numbers and contents.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to search"},
                        "query": {"type": "string", "description": "Text substring or symbol to search for"},
                    },
                    "required": ["path", "query"],
                },
                handler=_search_file,
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

            if argv[0] == "python" and not shutil.which("python"):
                argv[0] = sys.executable

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
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Full shell command line including all arguments (e.g. 'grep -n pattern file.py' or 'python -m py_compile file.py')",
                        }
                    },
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

        # 7. Screen perception, diffing, overlay, and actuation tools
        from dual_agent.screen import (
            capture_screenshot,
            compute_screen_diff,
            render_grid_overlay,
            click_mouse,
            move_mouse,
            send_key_press,
        )

        def _screenshot(args: Dict[str, Any]) -> str:
            out_path = args.get("output_path")
            try:
                meta = capture_screenshot(output_path=out_path)
                return json.dumps(meta)
            except Exception as e:
                return f"Error taking screenshot: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="screenshot",
                description="Capture a screenshot of the main screen with logical dimensions and Retina scale factor.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "Optional file path to save screenshot PNG"},
                    },
                },
                handler=_screenshot,
                risk_level="low",
                requires_approval=False,
            )
        )

        def _screen_diff(args: Dict[str, Any]) -> str:
            p1 = args.get("image_path_1", "")
            p2 = args.get("image_path_2", "")
            th = float(args.get("threshold", 0.002))
            try:
                res = compute_screen_diff(p1, p2, threshold=th)
                return json.dumps(res)
            except Exception as e:
                return f"Error computing screen diff: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="screen_diff",
                description="Compare two screenshot images and report whether the screen changed and the bounding box of differences.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "image_path_1": {"type": "string", "description": "Path to baseline screenshot PNG"},
                        "image_path_2": {"type": "string", "description": "Path to comparison screenshot PNG"},
                        "threshold": {"type": "number", "description": "Fraction difference threshold (default 0.002)"},
                    },
                    "required": ["image_path_1", "image_path_2"],
                },
                handler=_screen_diff,
                risk_level="low",
                requires_approval=False,
            )
        )

        def _grid_overlay(args: Dict[str, Any]) -> str:
            img = args.get("image_path", "")
            out = args.get("output_path")
            step = int(args.get("grid_step", 150))
            try:
                res_path = render_grid_overlay(img, output_path=out, grid_step=step)
                return json.dumps({"overlay_path": res_path})
            except Exception as e:
                return f"Error rendering grid overlay: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="grid_overlay",
                description="Annotate a screenshot with a labeled coordinate grid overlay to guide coordinate-based visual actions.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to screenshot image"},
                        "output_path": {"type": "string", "description": "Optional output path for annotated PNG"},
                        "grid_step": {"type": "integer", "description": "Pixel spacing between grid lines (default 150)"},
                    },
                    "required": ["image_path"],
                },
                handler=_grid_overlay,
                risk_level="low",
                requires_approval=False,
            )
        )

        def _mouse_click(args: Dict[str, Any]) -> str:
            try:
                x = float(args.get("x", 0))
                y = float(args.get("y", 0))
                btn = str(args.get("button", "left"))
                ctype = str(args.get("click_type", "single"))
                res = click_mouse(x=x, y=y, button=btn, click_type=ctype)
                return json.dumps(res)
            except Exception as e:
                return f"Error actuating mouse click: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="mouse_click",
                description="Click the mouse at logical screen coordinates (x, y). High risk — interacts directly with the live desktop.",
                requires_approval=True,
                risk_level="high",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate in logical screen points"},
                        "y": {"type": "number", "description": "Y coordinate in logical screen points"},
                        "button": {"type": "string", "enum": ["left", "right"], "description": "Mouse button (default 'left')"},
                        "click_type": {"type": "string", "enum": ["single", "double"], "description": "Click type (default 'single')"},
                    },
                    "required": ["x", "y"],
                },
                handler=_mouse_click,
                arguments=["x", "y", "button", "click_type"],
                requirements=["Actuates physical mouse clicks on the user's active desktop session."],
            )
        )

        def _mouse_move(args: Dict[str, Any]) -> str:
            try:
                x = float(args.get("x", 0))
                y = float(args.get("y", 0))
                res = move_mouse(x=x, y=y)
                return json.dumps(res)
            except Exception as e:
                return f"Error moving mouse: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="mouse_move",
                description="Move mouse cursor to logical screen coordinates (x, y). High risk — moves cursor on live desktop.",
                requires_approval=True,
                risk_level="high",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate in logical screen points"},
                        "y": {"type": "number", "description": "Y coordinate in logical screen points"},
                    },
                    "required": ["x", "y"],
                },
                handler=_mouse_move,
                arguments=["x", "y"],
                requirements=["Moves physical mouse cursor on the user's active desktop session."],
            )
        )

        def _key_press(args: Dict[str, Any]) -> str:
            try:
                key = str(args.get("key", ""))
                modifiers = args.get("modifiers")
                res = send_key_press(key=key, modifiers=modifiers)
                return json.dumps(res)
            except Exception as e:
                return f"Error sending key press: {e}"

        self.register_tool(
            MCPToolDefinition(
                name="key_press",
                description="Send a keystroke (character or named key like 'return', 'tab', 'escape') with optional modifiers.",
                requires_approval=True,
                risk_level="high",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Named key ('return', 'tab', 'escape', 'up', etc.) or single character"},
                        "modifiers": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["shift", "control", "option", "command"]},
                            "description": "Optional list of keyboard modifiers",
                        },
                    },
                    "required": ["key"],
                },
                handler=_key_press,
                arguments=["key", "modifiers"],
                requirements=["Sends physical keyboard strokes to the user's active desktop window."],
            )
        )
