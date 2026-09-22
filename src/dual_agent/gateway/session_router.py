"""Session Router — maps incoming gateway messages to per-chat agent sessions.

Each chat_id gets an isolated MemoryEngine and DualProcessDispatcher so that
conversations on Telegram (or any other platform) don't bleed into each other.
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SessionRouter:
    """Manages one dispatcher per chat_id with isolated memory."""

    def __init__(self, dispatcher_factory, base_data_dir: Optional[str] = None):
        """
        Args:
            dispatcher_factory: Callable() → DualProcessDispatcher.
                                Called once per new chat_id.
            base_data_dir:      Root dir; each chat gets a subdir.
        """
        self.dispatcher_factory = dispatcher_factory
        self.base_data_dir = base_data_dir or os.path.expanduser("~/.dual_agent/sessions")
        os.makedirs(self.base_data_dir, mode=0o700, exist_ok=True)
        self._sessions: Dict[str, object] = {}  # chat_id → dispatcher

    def get_or_create(self, chat_id: str):
        """Return the dispatcher for this chat, creating one if needed."""
        if chat_id not in self._sessions:
            logger.info(f"[SessionRouter] Creating new session for chat_id={chat_id!r}")
            dispatcher = self.dispatcher_factory(session_id=chat_id)
            self._sessions[chat_id] = dispatcher
        return self._sessions[chat_id]

    def destroy(self, chat_id: str) -> None:
        """Remove a session (e.g., when a user deletes their chat)."""
        self._sessions.pop(chat_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
