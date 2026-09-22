"""Session Router — maps incoming gateway messages to per-chat agent sessions.

Each chat_id gets an isolated MemoryEngine and DualProcessDispatcher so that
conversations on Telegram (or any other platform) don't bleed into each other.

Bounded lifetime: a bot that sees many distinct chat_ids (group joins, spam,
one-shot DMs) would otherwise accumulate one dispatcher — and one open SQLite
handle plus one session directory — per id, forever. `get_or_create` therefore
evicts the least-recently-used session once the cap is reached. Eviction only
touches this process's in-memory dispatcher map; the chat's on-disk memory
directory is left intact, so a returning chat re-opens its own history.
"""

from __future__ import annotations
import os
import logging
import threading
from collections import OrderedDict
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Documented default: 256 concurrently resident chat sessions. A dispatcher is
# a SQLite handle plus MCP state, so the ceiling's job is to bound descriptors
# and RAM, not to model "how many users exist" — well above any real chat volume.
DEFAULT_MAX_SESSIONS = 256


class SessionRouter:
    """Manages one dispatcher per chat_id with isolated memory, LRU-bounded."""

    def __init__(
        self,
        dispatcher_factory: Callable,
        base_data_dir: Optional[str] = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ):
        """
        Args:
            dispatcher_factory: Callable() → DualProcessDispatcher.
                                Called with session_id= for each new chat_id.
            base_data_dir:      Root dir; each chat gets a subdir.
            max_sessions:       Maximum resident sessions before LRU eviction.
                                Must be >= 1; None or 0 disables the cap.
        """
        if max_sessions is not None and max_sessions < 1:
            raise ValueError(
                f"max_sessions must be >= 1 or None, got {max_sessions!r}"
            )
        self.dispatcher_factory = dispatcher_factory
        self.base_data_dir = base_data_dir or os.path.expanduser("~/.dual_agent/sessions")
        os.makedirs(self.base_data_dir, mode=0o700, exist_ok=True)
        self.max_sessions = max_sessions
        # Ordered least-recently-used first. A plain dict insertion order is not
        # enough: renewing an existing chat must move it to the warm end, or LRU
        # would evict the busiest chat simply for being created first.
        self._sessions: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.RLock()
        self._evictions = 0

    def get_or_create(self, chat_id: str):
        """Return the dispatcher for this chat, creating one if needed."""
        with self._lock:
            if chat_id in self._sessions:
                self._sessions.move_to_end(chat_id)
                return self._sessions[chat_id]

            logger.info(f"[SessionRouter] Creating new session for chat_id={chat_id!r}")
            dispatcher = self.dispatcher_factory(session_id=chat_id)
            self._sessions[chat_id] = dispatcher
            self._sessions.move_to_end(chat_id)
            self._enforce_cap_locked()
            return self._sessions[chat_id]

    def _enforce_cap_locked(self) -> None:
        """Evict least-recently-used sessions until the cap is satisfied.

        Caller must hold self._lock.
        """
        if not self.max_sessions:
            return
        while len(self._sessions) > self.max_sessions:
            evicted_id, _ = self._sessions.popitem(last=False)
            self._evictions += 1
            # No removal of the session directory: history stays on disk and a
            # returning chat reloads it. Deleting it here would be data loss.
            logger.info(
                f"[SessionRouter] Evicted LRU session chat_id={evicted_id!r} "
                f"(cap={self.max_sessions})"
            )

    def destroy(self, chat_id: str) -> None:
        """Remove a session (e.g., when a user deletes their chat)."""
        with self._lock:
            self._sessions.pop(chat_id, None)

    @property
    def active_sessions(self) -> int:
        """Number of resident (in-memory) sessions."""
        with self._lock:
            return len(self._sessions)

    @property
    def evictions(self) -> int:
        """Total sessions evicted by the cap since construction."""
        with self._lock:
            return self._evictions

    def stats(self) -> dict:
        """Resident count and cap, for health endpoints and drift checks."""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "evictions": self._evictions,
                "chat_ids": list(self._sessions.keys()),
            }
