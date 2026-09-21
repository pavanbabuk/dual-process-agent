"""Persistent SQLite Memory Layer for Dual-Process Agent.

Stores execution sessions, project profiles, and learned skills across agent runs.
"""

from __future__ import annotations
import os
import json
import sqlite3
import datetime
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def get_default_data_dir() -> str:
    """Returns ~/.dual_agent or DUAL_AGENT_HOME environment directory."""
    path = os.getenv("DUAL_AGENT_HOME", os.path.expanduser("~/.dual_agent"))
    os.makedirs(path, exist_ok=True)
    return path


class LearnedSkill(BaseModel):
    name: str
    intent_keywords: List[str]
    tool_sequence: List[str]
    success_count: int = 1


class MemoryEngine:
    """Manages persistent SQLite memory for the agent."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = get_default_data_dir()
            self.db_path = os.path.join(data_dir, "memory.db")
        else:
            self.db_path = db_path
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Create tables if they do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    goal TEXT NOT NULL,
                    outcome TEXT,
                    is_completed BOOLEAN,
                    total_steps INTEGER,
                    system_one_steps INTEGER,
                    system_two_steps INTEGER,
                    total_latency_ms REAL,
                    tokens_used INTEGER,
                    token_savings_pct REAL,
                    steps_json TEXT
                )
            """)

            # Learned skills table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    intent_keywords TEXT,
                    tool_sequence_json TEXT,
                    success_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Project context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_context (
                    workspace_path TEXT PRIMARY KEY,
                    tech_stack_json TEXT,
                    user_preferences_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_session(
        self,
        goal: str,
        outcome: Optional[str],
        is_completed: bool,
        total_steps: int,
        system_one_steps: int,
        system_two_steps: int,
        total_latency_ms: float,
        tokens_used: int,
        token_savings_pct: float,
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Persist a completed task session into SQLite."""
        steps_str = json.dumps(steps or [])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (
                    goal, outcome, is_completed, total_steps, system_one_steps,
                    system_two_steps, total_latency_ms, tokens_used, token_savings_pct, steps_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal, outcome, is_completed, total_steps, system_one_steps,
                    system_two_steps, total_latency_ms, tokens_used, token_savings_pct, steps_str
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent task sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def save_learned_skill(self, name: str, intent_keywords: List[str], tool_sequence: List[str]) -> None:
        """Store or increment a learned routine."""
        keywords_str = " ".join([k.lower().strip() for k in intent_keywords])
        seq_str = json.dumps(tool_sequence)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO learned_skills (name, intent_keywords, tool_sequence_json, success_count, last_used_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(name) DO UPDATE SET
                    success_count = success_count + 1,
                    last_used_at = ?
                """,
                (name, keywords_str, seq_str, now, now),
            )
            conn.commit()

    def find_matching_skill(self, goal: str) -> Optional[LearnedSkill]:
        """Find a cached skill that matches words in the goal."""
        words = set(goal.lower().split())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learned_skills ORDER BY success_count DESC")
            for row in cursor.fetchall():
                kw_set = set(row["intent_keywords"].split())
                if kw_set and kw_set.issubset(words):
                    return LearnedSkill(
                        name=row["name"],
                        intent_keywords=list(kw_set),
                        tool_sequence=json.loads(row["tool_sequence_json"]),
                        success_count=row["success_count"],
                    )
        return None

    def get_all_skills(self) -> List[LearnedSkill]:
        """Return all learned skills."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learned_skills ORDER BY success_count DESC")
            return [
                LearnedSkill(
                    name=r["name"],
                    intent_keywords=r["intent_keywords"].split(),
                    tool_sequence=json.loads(r["tool_sequence_json"]),
                    success_count=r["success_count"],
                )
                for r in cursor.fetchall()
            ]

    def save_project_context(
        self,
        workspace_path: str,
        tech_stack: Dict[str, Any],
        user_preferences: Dict[str, Any],
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO project_context (workspace_path, tech_stack_json, user_preferences_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(workspace_path) DO UPDATE SET
                    tech_stack_json = ?,
                    user_preferences_json = ?,
                    updated_at = ?
                """,
                (
                    workspace_path, json.dumps(tech_stack), json.dumps(user_preferences), now,
                    json.dumps(tech_stack), json.dumps(user_preferences), now
                ),
            )
            conn.commit()

    def get_project_context(self, workspace_path: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM project_context WHERE workspace_path = ?", (workspace_path,))
            row = cursor.fetchone()
            if row:
                return {
                    "workspace_path": row["workspace_path"],
                    "tech_stack": json.loads(row["tech_stack_json"]),
                    "user_preferences": json.loads(row["user_preferences_json"]),
                    "updated_at": row["updated_at"],
                }
        return None

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Calculates cumulative statistics across all sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(total_steps) as total_steps,
                    SUM(system_one_steps) as total_s1_steps,
                    SUM(system_two_steps) as total_s2_steps,
                    SUM(tokens_used) as total_tokens,
                    AVG(token_savings_pct) as avg_token_savings_pct
                FROM sessions
            """)
            row = cursor.fetchone()
            if row and row["total_sessions"]:
                return {
                    "total_sessions": row["total_sessions"] or 0,
                    "total_steps": row["total_steps"] or 0,
                    "total_s1_steps": row["total_s1_steps"] or 0,
                    "total_s2_steps": row["total_s2_steps"] or 0,
                    "total_tokens": row["total_tokens"] or 0,
                    "avg_token_savings_pct": round(row["avg_token_savings_pct"] or 0.0, 1),
                }
            return {
                "total_sessions": 0,
                "total_steps": 0,
                "total_s1_steps": 0,
                "total_s2_steps": 0,
                "total_tokens": 0,
                "avg_token_savings_pct": 0.0,
            }
