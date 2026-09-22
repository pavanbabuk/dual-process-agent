"""Tests for TeamManifestManager (export/import round-trip)."""

import os
import pytest
from dual_agent.team_manifest import TeamManifestManager, TeamManifest


@pytest.fixture
def manager():
    return TeamManifestManager()


def test_export_generates_valid_markdown(manager, tmp_path):
    out = str(tmp_path / "manifest.md")
    content = manager.export_manifest(output_path=out)
    assert os.path.exists(out)
    assert "---" in content
    assert "name:" in content
    assert "provider:" in content
    assert "# Dual-Process Agent" in content


def test_export_returns_string_without_path(manager):
    content = manager.export_manifest()
    assert isinstance(content, str)
    assert len(content) > 100


def test_import_round_trip(manager, tmp_path):
    """Export then import should give back a valid TeamManifest."""
    out = str(tmp_path / "manifest.md")
    manager.export_manifest(output_path=out)

    m = manager._parse_manifest(open(out).read())
    assert isinstance(m, TeamManifest)
    assert m.name != ""
    assert m.version == "1.0"
    assert isinstance(m.preferences, dict)


def test_import_preferences_parsed(manager, tmp_path):
    content = """\
---
name: Test Agent
description: A test manifest
version: "1.0"
provider: grok
preferences:
  confidence_threshold: 0.90
---

# Test Agent
"""
    m = manager._parse_manifest(content)
    assert m.provider == "grok"
    assert m.preferences.get("confidence_threshold") == pytest.approx(0.90)


def test_import_mcp_servers_parsed(manager):
    content = """\
---
name: Dev Agent
description: Dev setup
version: "1.0"
provider: anthropic
mcp_servers:
  - name: github
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
---

# Dev Agent
"""
    m = manager._parse_manifest(content)
    assert len(m.mcp_servers) == 1
    assert m.mcp_servers[0]["name"] == "github"
    assert m.mcp_servers[0]["command"] == "npx"


def test_import_skills_list_parsed(manager):
    content = """\
---
name: Python Agent
description: Python dev
version: "1.0"
provider: mock
skills:
  - inspect-pyproject
  - write-pytest-test
---

# Python Agent
"""
    m = manager._parse_manifest(content)
    assert "inspect-pyproject" in m.skills
    assert "write-pytest-test" in m.skills


def test_import_invalid_format_raises(manager):
    with pytest.raises(ValueError, match="frontmatter"):
        manager._parse_manifest("This is just plain markdown without frontmatter")


def test_export_saves_to_file(manager, tmp_path):
    out = str(tmp_path / "my_team.md")
    manager.export_manifest(output_path=out)
    assert os.path.exists(out)
    size = os.path.getsize(out)
    assert size > 200
