"""Gateway base class for Dual-Process Agent — Hermes-style omni-channel messaging."""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class IncomingMessage:
    """Normalized inbound message from any platform."""
    def __init__(
        self,
        chat_id: str,
        user_id: str,
        text: str,
        platform: str,
        raw: Any = None,
    ):
        self.chat_id = chat_id
        self.user_id = user_id
        self.text = text
        self.platform = platform
        self.raw = raw

    def __repr__(self) -> str:
        return f"<IncomingMessage platform={self.platform} chat={self.chat_id} text={self.text[:40]!r}>"


MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, None]]


class GatewayAdapter(ABC):
    """Abstract base for all platform gateways (Telegram, Discord, Slack…)."""

    def __init__(self, on_message: MessageHandler):
        self.on_message = on_message

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a reply to the given chat."""

    @abstractmethod
    async def start(self) -> None:
        """Begin polling/webhook listening. Runs until cancelled."""

    async def send_chunked(self, chat_id: str, text: str, chunk_size: int = 4000) -> None:
        """Send long text in chunks (avoids platform limits)."""
        for i in range(0, len(text), chunk_size):
            await self.send_message(chat_id, text[i: i + chunk_size])
            if i + chunk_size < len(text):
                await asyncio.sleep(0.1)
