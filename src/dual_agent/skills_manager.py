"""Skills Manager — Hermes-style autonomous .SKILL.md synthesis and retrieval.

Skills are procedural memory files written in human-readable Markdown with YAML
frontmatter. Compatible with the agentskills.io open standard.

Directory: ~/.dual_agent/skills/*.SKILL.md
"""

from __future__ import annotations
import os
import re
import logging
import datetime
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SKILLS_DIR_ENV = "DUAL_AGENT_SKILLS_DIR"


def get_skills_dir() -> str:
    """Returns the skills directory, creating it if needed."""
    from dual_agent.memory import get_default_data_dir
    path = os.getenv(SKILLS_DIR_ENV, os.path.join(get_default_data_dir(), "skills"))
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


@dataclass
class Skill:
    name: str
    description: str
    intent_keywords: List[str]
    tool_sequence: List[str]
    success_count: int
    last_used: str
    file_path: str
    body: str = ""


class SkillsManager:
    """Manages .SKILL.md files for the Dual-Process Agent."""

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = skills_dir or get_skills_dir()

    # ------------------------------------------------------------------
    # Synthesis: write a new .SKILL.md after a successful task
    # ------------------------------------------------------------------

    def synthesize_skill(
        self,
        goal: str,
        tool_sequence: List[str],
        outcome: str,
        success_count: int = 1,
    ) -> Optional[str]:
        """Write (or update) a .SKILL.md file encoding a discovered procedure.

        Returns the file path written, or None if skill was skipped.
        """
        if not tool_sequence:
            return None

        # Build a clean file-safe name from the goal
        slug = self._slugify(goal)
        file_path = os.path.join(self.skills_dir, f"{slug}.SKILL.md")

        # If the file already exists, just bump the success_count
        if os.path.exists(file_path):
            existing = self._parse_file(file_path)
            if existing:
                new_count = existing.success_count + 1
                self._write_skill_file(
                    file_path=file_path,
                    name=existing.name,
                    description=existing.description,
                    intent_keywords=existing.intent_keywords,
                    tool_sequence=existing.tool_sequence,
                    success_count=new_count,
                )
                logger.info(f"[SkillsManager] Updated skill '{existing.name}' (uses={new_count})")
                return file_path

        # New skill
        keywords = self._extract_keywords(goal)
        description = f"Procedure synthesized from goal: {goal}"
        self._write_skill_file(
            file_path=file_path,
            name=slug.replace("-", " ").title(),
            description=description,
            intent_keywords=keywords,
            tool_sequence=tool_sequence,
            success_count=success_count,
        )
        logger.info(f"[SkillsManager] Created new skill '{slug}'")
        return file_path

    def _write_skill_file(
        self,
        file_path: str,
        name: str,
        description: str,
        intent_keywords: List[str],
        tool_sequence: List[str],
        success_count: int,
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        kw_yaml = ", ".join(intent_keywords)
        seq_yaml = "\n".join(f"  - {t}" for t in tool_sequence)

        content = f"""---
name: {name}
description: {description}
intent_keywords: [{kw_yaml}]
success_count: {success_count}
last_used: {now}
---

# {name}

{description}

## Tool Sequence

```
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(tool_sequence))}
```

## When to Use

Invoke this skill when the user goal contains keywords like: **{", ".join(intent_keywords)}**.

## Notes

Auto-synthesized by Dual-Process Agent on {now}. Edit this file to refine the procedure.
"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # Retrieval: find relevant skills for a given goal
    # ------------------------------------------------------------------

    def find_relevant_skills(self, goal: str, top_k: int = 3) -> List[Skill]:
        """Return the top-k most relevant skills for the given goal."""
        skills = self.load_all_skills()
        if not skills:
            return []

        goal_words = set(goal.lower().split())
        scored: List[tuple[int, Skill]] = []
        for skill in skills:
            kw_set = set(k.lower() for k in skill.intent_keywords)
            overlap = len(kw_set & goal_words)
            if overlap > 0:
                scored.append((overlap * skill.success_count, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def build_skill_context(self, goal: str) -> str:
        """Build a formatted context block for System 2 prompt injection."""
        relevant = self.find_relevant_skills(goal)
        if not relevant:
            return ""

        lines = ["[RELEVANT SKILLS FROM MEMORY]"]
        for skill in relevant:
            lines.append(f"\n### {skill.name} (used {skill.success_count}x)")
            lines.append(f"Keywords: {', '.join(skill.intent_keywords)}")
            lines.append(f"Tool Sequence: {' → '.join(skill.tool_sequence)}")
        lines.append("\n[END SKILLS CONTEXT]\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Listing & Parsing
    # ------------------------------------------------------------------

    def load_all_skills(self) -> List[Skill]:
        """Load and parse all .SKILL.md files from the skills directory."""
        skills = []
        try:
            for fname in sorted(os.listdir(self.skills_dir)):
                if not fname.endswith(".SKILL.md"):
                    continue
                fpath = os.path.join(self.skills_dir, fname)
                skill = self._parse_file(fpath)
                if skill:
                    skills.append(skill)
        except FileNotFoundError:
            pass
        return sorted(skills, key=lambda s: s.success_count, reverse=True)

    def _parse_file(self, file_path: str) -> Optional[Skill]:
        """Parse a .SKILL.md file and return a Skill object."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()

            # Extract YAML frontmatter between --- delimiters
            fm_match = re.match(r"^---\n(.*?)\n---\n?(.*)", raw, re.DOTALL)
            if not fm_match:
                return None

            fm_text, body = fm_match.groups()

            def _get(key: str, default="") -> str:
                m = re.search(rf"^{key}:\s*(.+)$", fm_text, re.MULTILINE)
                return m.group(1).strip() if m else default

            def _get_list(key: str) -> List[str]:
                m = re.search(rf"^{key}:\s*\[(.+)\]", fm_text, re.MULTILINE)
                if m:
                    return [x.strip() for x in m.group(1).split(",") if x.strip()]
                # YAML block list
                lines = re.findall(rf"^\s+-\s+(.+)$", fm_text, re.MULTILINE)
                return lines

            # Parse tool sequence from the numbered list in the body
            seq_matches = re.findall(r"^\d+\.\s+(.+)$", body, re.MULTILINE)

            return Skill(
                name=_get("name", os.path.basename(file_path)),
                description=_get("description"),
                intent_keywords=_get_list("intent_keywords"),
                tool_sequence=seq_matches if seq_matches else _get_list("tool_sequence"),
                success_count=int(_get("success_count", "1")),
                last_used=_get("last_used", ""),
                file_path=file_path,
                body=body.strip(),
            )
        except Exception as e:
            logger.warning(f"[SkillsManager] Could not parse {file_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slugify(self, text: str, max_len: int = 40) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        text = re.sub(r"\s+", "-", text.strip())
        text = re.sub(r"-+", "-", text)
        return text[:max_len].rstrip("-")

    def _extract_keywords(self, goal: str, max_kw: int = 6) -> List[str]:
        stopwords = {
            "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
            "is", "are", "was", "be", "by", "with", "that", "this", "it", "all",
            "find", "get", "show", "list", "from", "into", "then", "can", "my",
        }
        words = re.findall(r"[a-z]+", goal.lower())
        return [w for w in words if w not in stopwords and len(w) > 3][:max_kw]
