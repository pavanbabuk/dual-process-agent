"""Team Manifest — OpenMausBot-style portable agent configuration.

Export the complete agent configuration (provider, MCP servers, active skills,
preferences) as a human-readable Markdown file with YAML frontmatter.

Import a manifest from disk or GitHub URL to instantly configure the agent.

Format (compatible with OpenMausBot's BotMRR standard):
---
name: My Agent Team
description: A dual-process agent setup for Python development
version: "1.0"
provider: grok
mcp_servers:
  - name: github
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
skills:
  - inspect-pyproject
  - write-pytest-test
preferences:
  confidence_threshold: 0.85
---

# My Agent Team

...playbook prose...
"""

from __future__ import annotations
import os
import re
import json
import logging
import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TeamManifest:
    name: str
    description: str
    version: str
    provider: str
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    playbook: str = ""


class TeamManifestManager:
    """Export and import agent team configurations as portable Markdown manifests."""

    def __init__(self, memory_engine=None, config=None, mcp_manager=None, skills_manager=None):
        self.memory = memory_engine
        self.config = config
        self.mcp = mcp_manager
        self.skills = skills_manager

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_manifest(self, output_path: Optional[str] = None) -> str:
        """Generate a team manifest and optionally save it to disk.

        Returns the manifest as a string.
        """
        name = "Dual-Process Agent"
        description = "Auto-exported Dual-Process Agent configuration."
        provider = "mock"
        confidence_threshold = 0.85
        mcp_servers_list: List[Dict[str, Any]] = []
        skills_list: List[str] = []

        # Pull from config if available
        if self.config:
            provider = getattr(self.config, "system_two_provider", "mock")
            confidence_threshold = getattr(self.config, "system_one_confidence_threshold", 0.85)

        # Pull MCP servers
        if self.mcp:
            servers = self.mcp.load_servers()
            for sname, srv in servers.items():
                mcp_servers_list.append({
                    "name": sname,
                    "command": srv.command,
                    "args": srv.args,
                })

        # Pull skills
        if self.skills:
            for skill in self.skills.load_all_skills():
                skills_list.append(skill.name.lower().replace(" ", "-"))

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build YAML frontmatter
        mcp_yaml = ""
        if mcp_servers_list:
            mcp_yaml = "mcp_servers:\n"
            for srv in mcp_servers_list:
                args_str = json.dumps(srv.get("args", []))
                mcp_yaml += f'  - name: {srv["name"]}\n'
                mcp_yaml += f'    command: {srv["command"]}\n'
                mcp_yaml += f'    args: {args_str}\n'

        skills_yaml = ""
        if skills_list:
            skills_yaml = "skills:\n" + "\n".join(f"  - {s}" for s in skills_list) + "\n"

        frontmatter = f"""---
name: {name}
description: {description}
version: "1.0"
exported_at: {now}
provider: {provider}
{mcp_yaml}{skills_yaml}preferences:
  confidence_threshold: {confidence_threshold}
---"""

        playbook = f"""
# {name}

{description}

## How to Use

This manifest was exported from a **Dual-Process Agent** instance.
Import it with:

```bash
dual-agent /import <path-to-this-file>
```

## Provider

- **System 2 Provider:** `{provider}`
- **Confidence Threshold:** `{confidence_threshold}`

## MCP Servers

{"No external MCP servers configured." if not mcp_servers_list else chr(10).join(f"- `{s['name']}`: `{s['command']} {' '.join(s.get('args', []))}`" for s in mcp_servers_list)}

## Skills

{"No custom skills synthesized yet." if not skills_list else chr(10).join(f"- {s}" for s in skills_list)}

## Notes

- Install this manifest to instantly configure a fresh agent with the same tools and skills.
- Credentials are **not** included — set them in `.env` or via `dual-agent config`.
- Skills are listed by name; copy the `.SKILL.md` files from `~/.dual_agent/skills/` separately.
"""

        manifest_content = frontmatter + playbook

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(manifest_content)
            logger.info(f"[TeamManifest] Exported to {output_path}")

        return manifest_content

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_manifest(self, source: str) -> TeamManifest:
        """Import a manifest from a file path or GitHub raw URL.

        Applies MCP server configurations and preferences to the running agent.
        Returns the parsed TeamManifest.
        """
        content = self._fetch_content(source)
        manifest = self._parse_manifest(content)

        # Apply MCP servers
        if self.mcp and manifest.mcp_servers:
            for srv in manifest.mcp_servers:
                try:
                    self.mcp.add_server(
                        name=srv.get("name", "imported"),
                        command=srv.get("command", ""),
                        args=srv.get("args", []),
                    )
                    logger.info(f"[TeamManifest] Registered MCP server '{srv['name']}'")
                except Exception as e:
                    logger.warning(f"[TeamManifest] Could not register MCP server: {e}")

        # Apply preferences to config (if mutable)
        if self.config and manifest.preferences:
            ct = manifest.preferences.get("confidence_threshold")
            if ct is not None:
                try:
                    self.config.system_one_confidence_threshold = float(ct)
                    logger.info(f"[TeamManifest] Set confidence_threshold={ct}")
                except Exception:
                    pass

        logger.info(f"[TeamManifest] Imported '{manifest.name}' from {source}")
        return manifest

    def _fetch_content(self, source: str) -> str:
        """Fetch manifest content from a file path or URL."""
        if source.startswith("http://") or source.startswith("https://"):
            import httpx
            response = httpx.get(source, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
            return response.text
        else:
            with open(source, "r", encoding="utf-8") as f:
                return f.read()

    def _parse_manifest(self, content: str) -> TeamManifest:
        """Parse a Markdown manifest into a TeamManifest dataclass."""
        fm_match = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
        if not fm_match:
            raise ValueError("Invalid manifest format: missing YAML frontmatter (--- ... ---)")

        fm_text, playbook = fm_match.groups()

        def _get(key: str, default="") -> str:
            m = re.search(rf"^{key}:\s*(.+)$", fm_text, re.MULTILINE)
            return m.group(1).strip().strip('"') if m else default

        def _get_list_block(key: str) -> List[Dict[str, Any]]:
            """Parse a YAML block list of objects under a key."""
            # Match the block starting at the key
            m = re.search(rf"^{key}:\n((?:[ \t]+.+\n?)+)", fm_text, re.MULTILINE)
            if not m:
                return []
            block = m.group(1)
            items: List[Dict[str, Any]] = []
            current: Optional[Dict[str, Any]] = None
            for line in block.splitlines():
                # New list item: "  - name: value" or "  - name: value"
                new_item_m = re.match(r"\s+-\s+(\w+):\s*(.*)", line)
                if new_item_m:
                    if current is not None:
                        items.append(current)
                    k, v = new_item_m.group(1), new_item_m.group(2).strip()
                    if v.startswith("["):
                        try:
                            v = json.loads(v)
                        except Exception:
                            pass
                    current = {k: v}
                    continue
                # Continuation key-value: "    key: value"
                if current is not None:
                    kv_m = re.match(r"\s+(\w+):\s*(.*)", line)
                    if kv_m:
                        k, v = kv_m.group(1), kv_m.group(2).strip()
                        if v.startswith("["):
                            try:
                                v = json.loads(v)
                            except Exception:
                                pass
                        current[k] = v
            if current is not None:
                items.append(current)
            return items


        def _get_simple_list(key: str) -> List[str]:
            m = re.search(rf"^{key}:\n((?:\s+-.+\n?)+)", fm_text, re.MULTILINE)
            if not m:
                return []
            return [re.sub(r"^\s+-\s*", "", line) for line in m.group(1).splitlines() if line.strip()]

        preferences: Dict[str, Any] = {}
        pref_m = re.search(r"^preferences:\n((?:\s+.+\n?)+)", fm_text, re.MULTILINE)
        if pref_m:
            for pline in pref_m.group(1).splitlines():
                pkv = re.match(r"\s+(\w+):\s*(.+)", pline)
                if pkv:
                    try:
                        preferences[pkv.group(1)] = float(pkv.group(2))
                    except ValueError:
                        preferences[pkv.group(1)] = pkv.group(2).strip()

        return TeamManifest(
            name=_get("name", "Unknown Agent"),
            description=_get("description"),
            version=_get("version", "1.0"),
            provider=_get("provider", "mock"),
            mcp_servers=_get_list_block("mcp_servers"),
            skills=_get_simple_list("skills"),
            preferences=preferences,
            playbook=playbook.strip(),
        )
