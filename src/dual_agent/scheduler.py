"""Cron Scheduler — Hermes-style natural-language automations.

Stores scheduled jobs in SQLite and dispatches them in a background asyncio loop
when the daemon is running (`dual-agent --gateway` or `dual-agent --daemon`).

Usage (from shell):
    /schedule "Run git status every day at 9am"

The scheduler uses a very simple natural-language → cron expression parser for
common patterns. For advanced expressions, users can pass raw cron strings.
"""

from __future__ import annotations
import asyncio
import logging
import re
import json
import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Natural-language cron parser (covers the 90% case without heavy deps)
# ---------------------------------------------------------------------------

NL_PATTERNS: List[tuple[re.Pattern, str]] = [
    # "every minute"
    (re.compile(r"every\s+minute", re.IGNORECASE), "* * * * *"),
    # "every 5 minutes"
    (re.compile(r"every\s+(\d+)\s+minutes?", re.IGNORECASE), r"*/\1 * * * *"),
    # "every hour"
    (re.compile(r"every\s+hour", re.IGNORECASE), "0 * * * *"),
    # "every day at 9am" / "daily at 09:00"
    (re.compile(r"(?:every\s+day|daily)\s+at\s+(\d{1,2})(?::(\d{2}))?(?:\s*am)?", re.IGNORECASE), None),
    (re.compile(r"(?:every\s+day|daily)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*pm", re.IGNORECASE), None),
    # "every Monday at 9am"
    (re.compile(r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?(?:\s*am)?", re.IGNORECASE), None),
    # "every night" / "nightly"
    (re.compile(r"every\s+night|nightly", re.IGNORECASE), "0 2 * * *"),
    # "every week" / "weekly"
    (re.compile(r"every\s+week(?:ly)?", re.IGNORECASE), "0 9 * * 1"),
    # "every morning"
    (re.compile(r"every\s+morning", re.IGNORECASE), "0 9 * * *"),
]

DAY_MAP = {
    "monday": "1", "tuesday": "2", "wednesday": "3", "thursday": "4",
    "friday": "5", "saturday": "6", "sunday": "0",
}


def parse_nl_to_cron(description: str) -> Optional[str]:
    """Attempt to convert a natural language schedule description to a cron expression."""
    d = description.lower()

    # "every N minutes"
    m = re.search(r"every\s+(\d+)\s+minutes?", d, re.IGNORECASE)
    if m:
        return f"*/{m.group(1)} * * * *"

    # "every minute"
    if re.search(r"every\s+minute", d, re.IGNORECASE):
        return "* * * * *"

    # "every hour" / "hourly"
    if re.search(r"every\s+hour(?:ly)?|^hourly$", d, re.IGNORECASE):
        return "0 * * * *"

    # "every night" / "nightly"
    if re.search(r"every\s+night|nightly", d, re.IGNORECASE):
        return "0 2 * * *"

    # "every morning"
    if re.search(r"every\s+morning", d, re.IGNORECASE):
        return "0 9 * * *"

    # "every week" / "weekly"
    if re.search(r"every\s+week(?:ly)?|^weekly$", d, re.IGNORECASE):
        return "0 9 * * 1"

    # "every day at H[:M] [am|pm]"
    m = re.search(r"(?:every\s+day|daily)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", d, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if m.group(3) and m.group(3).lower() == "pm" and hour < 12:
            hour += 12
        return f"{minute} {hour} * * *"

    # "every Monday at H[:M] [am|pm]"
    m = re.search(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        d, re.IGNORECASE,
    )
    if m:
        dow = DAY_MAP[m.group(1).lower()]
        hour = int(m.group(2))
        minute = int(m.group(3) or 0)
        if m.group(4) and m.group(4).lower() == "pm" and hour < 12:
            hour += 12
        return f"{minute} {hour} * * {dow}"

    return None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CronScheduler:
    """Stores and dispatches scheduled agent jobs via SQLite + asyncio."""

    def __init__(self, memory_engine, dispatcher_factory: Optional[Callable] = None):
        """
        Args:
            memory_engine:      MemoryEngine instance (for job persistence).
            dispatcher_factory: Callable that returns a DualProcessDispatcher.
                                Used to create a fresh dispatcher per job run.
        """
        self.memory = memory_engine
        self.dispatcher_factory = dispatcher_factory
        self._init_table()

    def _init_table(self) -> None:
        with self.memory._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    cron_expr TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    last_run_at TEXT,
                    run_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_job(self, description: str, goal: str, cron_expr: Optional[str] = None) -> int:
        """Add a new scheduled job. Parses cron from description if cron_expr is None."""
        expr = cron_expr or parse_nl_to_cron(description)
        if not expr:
            raise ValueError(
                f"Could not parse a cron expression from: '{description}'\n"
                "Try: 'every day at 9am', 'every Monday at 6pm', 'every 30 minutes'"
            )
        with self.memory._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO scheduled_jobs (description, cron_expr, goal) VALUES (?, ?, ?)",
                (description, expr, goal),
            )
            conn.commit()
            job_id = cursor.lastrowid
        logger.info(f"[Scheduler] Added job #{job_id}: '{description}' ({expr})")
        return job_id

    def remove_job(self, job_id: int) -> bool:
        with self.memory._get_connection() as conn:
            cur = conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self.memory._get_connection() as conn:
            conn.row_factory = __import__("sqlite3").Row
            cur = conn.execute("SELECT * FROM scheduled_jobs ORDER BY id")
            return [dict(r) for r in cur.fetchall()]

    def enable_job(self, job_id: int, enabled: bool = True) -> None:
        with self.memory._get_connection() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = ? WHERE id = ?",
                (int(enabled), job_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Async tick loop
    # ------------------------------------------------------------------

    async def run_forever(self, interval_seconds: int = 60) -> None:
        """Main scheduler loop — call inside asyncio event loop from the daemon."""
        logger.info("[Scheduler] Daemon loop started (interval=60s)")
        while True:
            try:
                self._tick()
            except Exception as e:
                logger.warning(f"[Scheduler] Tick error: {e}")
            await asyncio.sleep(interval_seconds)

    def _tick(self) -> None:
        """Check all enabled jobs and dispatch any that are due."""
        now = datetime.datetime.now(datetime.timezone.utc)
        jobs = self.list_jobs()
        for job in jobs:
            if not job["enabled"]:
                continue
            if self._is_due(job["cron_expr"], job["last_run_at"], now):
                self._run_job(job)

    def _is_due(self, cron_expr: str, last_run_at: Optional[str], now: datetime.datetime) -> bool:
        """Very lightweight cron-due check (minute-resolution)."""
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return False
            minute_e, hour_e, dom_e, month_e, dow_e = parts

            def matches(expr: str, value: int) -> bool:
                if expr == "*":
                    return True
                if expr.startswith("*/"):
                    step = int(expr[2:])
                    return value % step == 0
                return str(value) == expr

            if not (
                matches(minute_e, now.minute)
                and matches(hour_e, now.hour)
                and matches(dom_e, now.day)
                and matches(month_e, now.month)
                and matches(dow_e, now.weekday() + 1 if now.weekday() < 6 else 0)
            ):
                return False

            # Don't run more than once per minute window
            if last_run_at:
                last = datetime.datetime.fromisoformat(last_run_at.replace("Z", "+00:00"))
                if (now - last).total_seconds() < 55:
                    return False
            return True
        except Exception:
            return False

    def _run_job(self, job: Dict[str, Any]) -> None:
        """Dispatch a scheduled job to the agent."""
        from rich.console import Console
        console = Console()
        job_id = job["id"]
        goal = job["goal"]
        logger.info(f"[Scheduler] Running job #{job_id}: {goal[:60]}")
        console.print(f"\n[bold magenta]⏰ Scheduled Job #{job_id}:[/bold magenta] {goal}")

        try:
            if self.dispatcher_factory:
                dispatcher = self.dispatcher_factory()
                result = dispatcher.run(goal)
                console.print(f"[dim]Job #{job_id} result: {result.final_output or 'Done'}[/dim]")
            else:
                console.print(f"[yellow]Job #{job_id}: No dispatcher configured — skipped.[/yellow]")
        except Exception as e:
            logger.error(f"[Scheduler] Job #{job_id} failed: {e}")
        finally:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with self.memory._get_connection() as conn:
                conn.execute(
                    "UPDATE scheduled_jobs SET last_run_at = ?, run_count = run_count + 1 WHERE id = ?",
                    (now, job_id),
                )
                conn.commit()
