"""Gateway runner — the entry point that makes the messaging gateway reachable.

The adapter and session router existed but nothing ever constructed them, and
`dual-agent --gateway` had no CLI handler — so the documented command failed with
"unrecognized arguments: --gateway". That made the entire gateway package dead
code, and with it the scheduler: cron jobs are only dispatched by a long-lived
process, which did not exist either.

This module wires the pieces together:

    Telegram ──> TelegramAdapter ──> SessionRouter ──> per-chat Dispatcher
                    (allow-list)        (isolated memory)     (MCP tools)

    CronScheduler ──> the same dispatcher factory, ticking in the background

Security posture: the gateway accepts only Telegram user ids present in
TELEGRAM_ALLOWED_USER_IDS, and risky tools are denied rather than prompted for,
because there is no terminal behind a chat to answer an approval card.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from dual_agent.config import AgentConfig, load_config
from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.mcp_host import MCPHost
from dual_agent.mcp_manager import MCPManager
from dual_agent.memory import MemoryEngine, get_default_data_dir
from dual_agent.permission_broker import PermissionBroker
from dual_agent.scheduler import CronScheduler
from dual_agent.skills_manager import SkillsManager
from dual_agent.system_two import get_system_two_provider
from dual_agent.typesafe_client import JevSystemOneClient

from dual_agent.gateway.base import IncomingMessage
from dual_agent.gateway.session_router import SessionRouter
from dual_agent.gateway.telegram_adapter import TelegramAdapter

logger = logging.getLogger(__name__)


class GatewayRunner:
    """Owns the long-lived gateway process: adapter, sessions, and scheduler."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        max_steps: int = 10,
        session_root: Optional[str] = None,
    ):
        self.config = config or load_config()
        self.max_steps = max_steps
        self.session_root = session_root or os.path.join(
            get_default_data_dir(), "sessions"
        )
        os.makedirs(self.session_root, mode=0o700, exist_ok=True)

        # One shared tool host and one shared router client: a Jev client holds
        # fallback state, and we want a live-API failure to be global and honest
        # rather than per-chat and inconsistent.
        self.mcp_host = MCPHost()
        MCPManager().attach_to_host(self.mcp_host)
        self.s1 = JevSystemOneClient(
            api_key=self.config.typesafe_api_key,
            base_url=self.config.typesafe_base_url,
        )
        self.s2 = get_system_two_provider(self.config.system_two_provider)

        # The scheduler reads jobs from the main memory DB, which is the same one
        # the interactive shell writes /schedule entries into.
        self.memory = MemoryEngine()
        self.scheduler = CronScheduler(
            memory_engine=self.memory,
            dispatcher_factory=lambda session_id="scheduler": self.build_dispatcher(session_id),
        )

        self.router = SessionRouter(
            dispatcher_factory=self.build_dispatcher,
            base_data_dir=self.session_root,
        )
        self.adapter: Optional[TelegramAdapter] = None

    # ------------------------------------------------------------------
    # Dispatcher construction
    # ------------------------------------------------------------------

    def build_dispatcher(self, session_id: str = "shared") -> DualProcessDispatcher:
        """Build an isolated dispatcher for one chat (its own memory DB).

        `session_id` is passed by SessionRouter as a keyword, so it must be
        accepted by name.
        """
        session_dir = os.path.join(self.session_root, str(session_id))
        os.makedirs(session_dir, mode=0o700, exist_ok=True)
        memory = MemoryEngine(db_path=os.path.join(session_dir, "memory.db"))

        auto_allow = (
            os.getenv("DUAL_AGENT_AUTO_ALLOW_PERMISSIONS", "false").lower() == "true"
        )
        broker = PermissionBroker(
            memory_engine=memory,
            auto_allow=auto_allow,
            # No terminal behind a chat: deny rather than block on stdin.
            non_interactive=not auto_allow,
        )

        return DualProcessDispatcher(
            system_one_client=self.s1,
            system_two_provider=self.s2,
            mcp_host=self.mcp_host,
            memory_engine=memory,
            skills_manager=SkillsManager(),
            permission_broker=broker,
            confidence_threshold=self.config.system_one_confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def handle_message(self, msg: IncomingMessage) -> None:
        """Run one inbound chat message through that chat's agent."""
        dispatcher = self.router.get_or_create(msg.chat_id)
        logger.info(f"[Gateway] chat={msg.chat_id} goal={msg.text[:80]!r}")

        loop = asyncio.get_running_loop()
        try:
            # run() is blocking (SQLite + tool subprocesses); keep the event loop free.
            result: DispatchResult = await loop.run_in_executor(
                None, lambda: dispatcher.run(goal=msg.text, max_steps=self.max_steps)
            )
        except Exception as e:
            logger.exception(f"[Gateway] Agent run failed: {e}")
            await self.send(msg.chat_id, f"⚠️ Agent error: {e}")
            return

        await self.send(msg.chat_id, self.format_reply(result))

    @staticmethod
    def format_reply(result: DispatchResult) -> str:
        """Render a run result for a chat client.

        Reports measured values only, and says plainly when the router was
        simulated, so a chat user is never told they got a Jev-routed answer
        when they got a local stub.
        """
        lines = [result.final_output or "Task completed."]
        lines.append(
            f"— {result.total_steps} steps "
            f"(S1: {result.system_one_steps}, S2: {result.system_two_steps}) "
            f"in {result.total_latency_ms:.0f}ms"
        )
        if result.used_simulated_system_one:
            lines.append(
                f"⚠️ Router was SIMULATED ({result.system_one_fallback_reason}) — "
                "not the Jev model."
            )
        return "\n".join(lines)

    async def send(self, chat_id: str, text: str) -> None:
        if self.adapter is not None:
            await self.adapter.send_chunked(chat_id, text)
        else:
            logger.info(f"[Gateway] (no adapter) reply to {chat_id}: {text[:120]}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _preflight(self) -> None:
        """Fail fast, and loudly, before polling starts."""
        # A missing bot token is the most likely first-run mistake, and it is
        # fully predictable — it should never surface as a raw traceback.
        if not os.getenv("TELEGRAM_BOT_TOKEN", "").strip():
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set, so there is nothing to poll as.\n"
                "  1. Create a bot with @BotFather and copy its token.\n"
                "  2. Find your numeric Telegram user id (e.g. message @userinfobot).\n"
                "  3. Put both in an env file:\n"
                f"       {os.path.join(get_default_data_dir(), '.env')}\n"
                "         TELEGRAM_BOT_TOKEN=<token>\n"
                "         TELEGRAM_ALLOWED_USER_IDS=<your numeric id>\n"
                "  4. Re-run: dual-agent --gateway"
            )

        if self.s1.force_simulation:
            logger.warning(
                f"System 1 will run SIMULATED ({self.s1.simulation_reason}). "
                "Set TYPESAFE_API_KEY in "
                f"{os.path.join(get_default_data_dir(), '.env')} for live Jev routing."
            )
        if not os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip():
            logger.warning(
                "TELEGRAM_ALLOWED_USER_IDS is empty — the bot will reject every "
                "message. Set it to your Telegram user id before expecting replies."
            )
        if os.getenv("DUAL_AGENT_AUTO_ALLOW_PERMISSIONS", "false").lower() != "true":
            logger.warning(
                "Risky tools (write_file, run_shell_command) will be DENIED: there "
                "is no terminal behind a chat to approve them. Set "
                "DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true to allow them unattended."
            )

    async def run(self) -> None:
        """Start polling and the scheduler, and run until cancelled."""
        self._preflight()

        self.adapter = TelegramAdapter(on_message=self.handle_message)

        logger.info(
            f"[Gateway] Started. Authorized users: "
            f"{sorted(self.adapter.allowed_user_ids) or 'NONE (all messages rejected)'}"
        )
        logger.info(f"[Gateway] Sessions dir: {self.session_root}")
        logger.info(
            f"[Gateway] Scheduler jobs: {len(self.scheduler.list_jobs())} "
            "(dispatched every 60s while this process runs)"
        )

        # The scheduler must tick concurrently with polling; otherwise cron jobs
        # never fire and /schedule silently does nothing.
        scheduler_task = asyncio.create_task(self.scheduler.run_forever())
        try:
            await self.adapter.start()
        finally:
            scheduler_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass


def run_gateway(max_steps: int = 10, session_root: Optional[str] = None) -> None:
    """Blocking entry point used by `dual-agent --gateway`."""
    runner = GatewayRunner(max_steps=max_steps, session_root=session_root)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("[Gateway] Stopped by user.")
