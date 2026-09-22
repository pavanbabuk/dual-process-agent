"""Telegram Gateway Adapter for Dual-Process Agent.

Bridges Telegram messages to the agent dispatcher using python-telegram-bot.

Requires: pip install "dual-agent[gateway]"
Env var:  TELEGRAM_BOT_TOKEN=<your-bot-token>

Usage:
    dual-agent --gateway
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import Iterable, Optional

from dual_agent.gateway.base import GatewayAdapter, IncomingMessage, MessageHandler

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"


class TelegramAdapter(GatewayAdapter):
    """Telegram gateway adapter using python-telegram-bot."""

    def __init__(
        self,
        on_message: MessageHandler,
        token: Optional[str] = None,
        allowed_user_ids: Optional[Iterable[str]] = None,
    ):
        super().__init__(on_message)
        self.token = token or os.getenv(TELEGRAM_TOKEN_ENV, "")
        if not self.token:
            raise ValueError(
                "Telegram bot token not found. Set TELEGRAM_BOT_TOKEN in .env "
                "or pass token= to TelegramAdapter()."
            )

        # Default-secure: with no allow-list nobody is authorized. This gateway
        # runs shell commands and writes files on this machine, so an open bot
        # is remote code execution for anyone who finds it.
        env_ids = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
        if allowed_user_ids is not None:
            self.allowed_user_ids = {str(i).strip() for i in allowed_user_ids if str(i).strip()}
        else:
            self.allowed_user_ids = {i.strip() for i in env_ids.split(",") if i.strip()}

        if not self.allowed_user_ids:
            logger.warning(
                "[Telegram] No TELEGRAM_ALLOWED_USER_IDS configured. Every incoming "
                "message will be rejected until an allowed user id is set."
            )
        self._app = None

    async def send_message(self, chat_id: str, text: str) -> None:
        if self._app:
            try:
                await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=text,
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning(f"[Telegram] send_message failed: {e}")

    async def start(self) -> None:
        try:
            from telegram.ext import Application, MessageHandler as TGHandler, filters
            from telegram import Update
            from telegram.ext import ContextTypes
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for the gateway.\n"
                "Install it with:  pip install 'dual-agent[gateway]'"
            )

        self._app = Application.builder().token(self.token).build()

        async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message or not update.message.text:
                return

            sender_id = str(update.effective_user.id) if update.effective_user else ""

            # --- AUTHORIZATION GATE ---
            # This gateway can write files and run shell commands on the host.
            # Anyone who can message the bot would otherwise inherit that
            # access, so an explicit allow-list is required before a single
            # command is accepted. Default-secure: empty list admits nobody.
            if sender_id not in self.allowed_user_ids:
                logger.warning(
                    f"[Telegram] Rejected message from unauthorized user_id={sender_id!r}. "
                    "Add it to TELEGRAM_ALLOWED_USER_IDS to grant access."
                )
                await self.send_message(
                    str(update.effective_chat.id),
                    "⛔ Not authorized. This bot is restricted to an explicit user allow-list.",
                )
                return

            msg = IncomingMessage(
                chat_id=str(update.effective_chat.id),
                user_id=sender_id,
                text=update.message.text,
                platform="telegram",
                raw=update,
            )
            logger.info(f"[Telegram] Received: {msg}")
            try:
                await self.on_message(msg)
            except Exception as e:
                logger.error(f"[Telegram] Handler error: {e}")
                await self.send_message(msg.chat_id, f"⚠️ Agent error: {e}")

        self._app.add_handler(TGHandler(filters.TEXT & ~filters.COMMAND, _handle))

        logger.info("[Telegram] Starting polling…")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        # Run until cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
