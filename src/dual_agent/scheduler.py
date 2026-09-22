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

MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DOW_NAMES = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "7": 0,
}


def _field_matches(
    expr: str,
    value: int,
    name_map: Optional[Dict[str, int]] = None,
    min_val: int = 0,
    max_val: int = 59,
) -> bool:
    """Evaluate a single cron field against an integer value."""
    expr = expr.strip().lower()
    if expr == "*":
        return True

    if "," in expr:
        return any(
            _field_matches(sub, value, name_map, min_val, max_val)
            for sub in expr.split(",")
        )

    if name_map:
        for name, num in name_map.items():
            expr = re.sub(rf"\b{name}\b", str(num), expr)

    step = 1
    if "/" in expr:
        parts = expr.split("/", 1)
        expr = parts[0]
        step = int(parts[1])

    if expr == "*":
        start, end = min_val, max_val
    elif "-" in expr:
        range_parts = expr.split("-", 1)
        start, end = int(range_parts[0]), int(range_parts[1])
    else:
        if step == 1:
            return value == int(expr)
        start, end = int(expr), max_val

    if start <= value <= end:
        return (value - start) % step == 0
    return False


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
        """Evaluate if cron expression is due at the given datetime."""
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return False
            minute_e, hour_e, dom_e, month_e, dow_e = parts

            # Weekday in standard cron: 0=Sun, 1=Mon, ..., 6=Sat
            dow_val = now.weekday() + 1 if now.weekday() < 6 else 0

            if not (
                _field_matches(minute_e, now.minute, min_val=0, max_val=59)
                and _field_matches(hour_e, now.hour, min_val=0, max_val=23)
                and _field_matches(dom_e, now.day, min_val=1, max_val=31)
                and _field_matches(month_e, now.month, name_map=MONTH_NAMES, min_val=1, max_val=12)
                and _field_matches(dow_e, dow_val, name_map=DOW_NAMES, min_val=0, max_val=7)
            ):
                return False

            # Don't run more than once per minute window
            if last_run_at:
                last = datetime.datetime.fromisoformat(last_run_at.replace("Z", "+00:00"))
                if (now - last).total_seconds() < 55:
                    return False
            return True
        except Exception as e:
            logger.debug(f"[Scheduler] _is_due error on '{cron_expr}': {e}")
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
