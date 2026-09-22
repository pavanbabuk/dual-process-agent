"""Tests for safe file editing — the fix for a real self-corruption incident.

Background: the agent was asked to add a `--version` flag to its own
src/dual_agent/cli.py. It used write_file, which rewrites the whole file, and
reproduced the file from memory rather than editing it. It dropped the module
docstring's triple quotes and an unrelated import, producing a file that could
not be parsed — and still reported "Task completed".

Two defences came out of that:

1. `validate_python_syntax` rejects a write that would leave invalid Python, so
   a botched rewrite cannot be persisted at all.
2. A `patch_file` tool edits by exact string replacement, so unrelated content
   is never regenerated and therefore never lost.
"""

import ast
import os

import pytest

from dual_agent.mcp_host import MCPHost, validate_python_syntax

GOOD = '''"""A module docstring that must survive an edit."""

import sys


def main():
    print("hello")
'''

BROKEN = "A module docstring with no quotes\n\nimport sys\n"


# ----------------------------------------------------------------------
# validate_python_syntax
# ----------------------------------------------------------------------

def test_valid_python_passes():
    assert validate_python_syntax("x.py", GOOD) is None


def test_invalid_python_is_rejected():
    msg = validate_python_syntax("x.py", BROKEN)
    assert msg is not None
    assert "invalid Python" in msg
    assert "NOT modified" in msg


def test_non_python_files_are_not_syntax_checked():
    """A .md or .txt file must be written even if it is not valid Python."""
    assert validate_python_syntax("README.md", "not python at all !!! {{") is None
    assert validate_python_syntax("notes.txt", "anything") is None


# ----------------------------------------------------------------------
# write_file must refuse to corrupt an existing file
# ----------------------------------------------------------------------

def test_write_file_refuses_broken_python(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text(GOOD)
    before = target.read_text()

    result = MCPHost().execute_tool("write_file", {"path": str(target), "content": BROKEN})

    assert result.success is True  # the tool ran; it declined the write
    assert "invalid Python" in str(result.output)
    assert target.read_text() == before, "the original file must be left untouched"


def test_write_file_still_allows_valid_python(tmp_path):
    target = tmp_path / "new.py"
    result = MCPHost().execute_tool("write_file", {"path": str(target), "content": GOOD})
    assert "Successfully wrote" in str(result.output)
    ast.parse(target.read_text())


def test_write_file_allows_non_python_content(tmp_path):
    target = tmp_path / "data.json"
    result = MCPHost().execute_tool("write_file", {"path": str(target), "content": '{"a": 1}'})
    assert "Successfully wrote" in str(result.output)


# ----------------------------------------------------------------------
# patch_file makes surgical edits safely
# ----------------------------------------------------------------------

def test_patch_file_preserves_unrelated_content(tmp_path):
    """The exact failure from the incident: docstring and import must survive."""
    target = tmp_path / "cli.py"
    target.write_text(GOOD)

    result = MCPHost().execute_tool("patch_file", {
        "path": str(target),
        "old_string": '    print("hello")',
        "new_string": '    print("hello")\n    print("world")',
    })

    body = target.read_text()
    assert "Patched" in str(result.output)
    assert '"""A module docstring that must survive an edit."""' in body, "docstring lost"
    assert "import sys" in body, "unrelated import lost"
    assert 'print("world")' in body
    ast.parse(body)


def test_patch_file_refuses_edit_that_breaks_syntax(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text(GOOD)
    before = target.read_text()

    result = MCPHost().execute_tool("patch_file", {
        "path": str(target),
        "old_string": '"""A module docstring that must survive an edit."""',
        "new_string": "A module docstring with no quotes",
    })

    assert "invalid Python" in str(result.output)
    assert target.read_text() == before


def test_patch_file_refuses_missing_old_string(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text(GOOD)
    result = MCPHost().execute_tool("patch_file", {
        "path": str(target), "old_string": "this text is absent", "new_string": "x",
    })
    assert "not found" in str(result.output)


def test_patch_file_refuses_ambiguous_match(tmp_path):
    """Two identical snippets must not be guessed at."""
    target = tmp_path / "mod.py"
    target.write_text("x = 1\ny = 1\n")
    result = MCPHost().execute_tool("patch_file", {
        "path": str(target), "old_string": "= 1", "new_string": "= 2",
    })
    assert "appears 2 times" in str(result.output)
    assert target.read_text() == "x = 1\ny = 1\n"


def test_patch_file_replace_all_is_explicit(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("x = 1\ny = 1\n")
    result = MCPHost().execute_tool("patch_file", {
        "path": str(target), "old_string": "= 1", "new_string": "= 2", "replace_all": True,
    })
    assert "2 occurrence" in str(result.output)
    assert target.read_text() == "x = 2\ny = 2\n"


def test_patch_file_requires_existing_file(tmp_path):
    result = MCPHost().execute_tool("patch_file", {
        "path": str(tmp_path / "nope.py"), "old_string": "a", "new_string": "b",
    })
    assert "does not exist" in str(result.output)


def test_patch_file_rejects_empty_and_noop(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text(GOOD)
    host = MCPHost()
    assert "must not be empty" in str(host.execute_tool(
        "patch_file", {"path": str(target), "old_string": "", "new_string": "x"}).output)
    assert "identical" in str(host.execute_tool(
        "patch_file", {"path": str(target), "old_string": "import sys", "new_string": "import sys"}).output)


def test_patch_file_is_registered_and_requires_approval():
    tool = MCPHost().get_tool("patch_file")
    assert tool is not None
    assert tool.requires_approval is True
    assert "old_string" in tool.parameters_schema["required"]
