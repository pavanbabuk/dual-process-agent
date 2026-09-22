"""Permission Broker — OpenMausBot-style interactive approval cards.

Intercepts tool calls flagged as `requires_approval=True` and presents the user
with a Rich-rendered inline card:

  ┌─ 🔐 Permission Required ─────────────────┐
  │ Tool:    run_shell_command                │
  │ Command: rm -rf build/                   │
  │ Risk:    HIGH                            │
  └──────────────────────────────────────────┘
  [A] Allow once  [S] Allow session  [D] Deny  [E] Edit args

Decisions are logged to SQLite for audit.
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

logger = logging.getLogger(__name__)
console = Console()


class ApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"
    EDITED = "edited"


class PermissionBroker:
    """Intercepts risky tool calls and requests user approval via Rich cards."""

    def __init__(
        self,
        memory_engine=None,
        auto_allow: bool = False,
        non_interactive: bool = False,
    ):
        """
        Args:
            memory_engine: Optional MemoryEngine for audit logging.
            auto_allow:    If True, all requests are auto-approved (useful for CI/testing).
            non_interactive: No terminal exists to answer a prompt — e.g. the
                           Telegram gateway, where the agent runs server-side.
                           An approval request is then DENIED instead of
                           blocking on stdin or raising EOFError.
        """
        self.memory = memory_engine
        self.auto_allow = auto_allow
        self.non_interactive = non_interactive
        # Tool names allowed for the entire session (ALLOW_SESSION decisions)
        self._session_allowed: Set[str] = set()
        # Tool names denied for the entire session
        self._session_denied: Set[str] = set()

    def request_approval(
        self,
        tool_name: str,
        args: Dict[str, Any],
        risk_level: str = "high",
    ) -> Tuple[ApprovalDecision, Dict[str, Any]]:
        """Show an approval card and return the user's decision + final args.

        Returns:
            (decision, args): args may be modified if user chose 'Edit'.
        """
        # --- Session-level overrides ---
        if tool_name in self._session_allowed:
            logger.debug(f"[PermissionBroker] Auto-allowed (session): {tool_name}")
            return ApprovalDecision.ALLOW_ONCE, args

        if tool_name in self._session_denied:
            logger.debug(f"[PermissionBroker] Auto-denied (session): {tool_name}")
            return ApprovalDecision.DENY, args

        # --- Auto-allow mode (CI / tests) ---
        if self.auto_allow:
            return ApprovalDecision.ALLOW_ONCE, args

        # --- No terminal available (gateway/web server) ---
        # Prompt.ask below would block on stdin or raise EOFError, so an
        # unanswerable approval is denied instead of hanging the daemon.
        if self.non_interactive:
            logger.warning(
                f"[PermissionBroker] DENIED '{tool_name}' ({risk_level} risk): no "
                "interactive terminal to approve it. Set "
                "DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true to allow risky tools "
                "unattended in gateway mode."
            )
            self._log_decision(tool_name, args, ApprovalDecision.DENY)
            return ApprovalDecision.DENY, args

        # --- Render the approval card ---
        risk_color = {
            "high": "bold red",
            "medium": "bold yellow",
            "low": "bold green",
        }.get(risk_level.lower(), "bold yellow")

        args_display = "\n".join(
            f"  [dim]{k}:[/dim] [white]{str(v)[:120]}[/white]" for k, v in args.items()
        )

        card_body = (
            f"[bold]Tool:[/bold]    [cyan]{tool_name}[/cyan]\n"
            f"[bold]Risk:[/bold]    [{risk_color}]{risk_level.upper()}[/{risk_color}]\n"
            f"[bold]Args:[/bold]\n{args_display or '  (none)'}"
        )

        console.print()
        console.print(
            Panel(
                card_body,
                title="[bold yellow]🔐 Permission Required[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                expand=False,
            )
        )
        console.print(
            "[dim]  [A] Allow once  "
            "[S] Allow for session  "
            "[D] Deny  "
            "[E] Edit args[/dim]"
        )

        while True:
            choice = Prompt.ask(
                "[bold yellow]Your choice[/bold yellow]",
                choices=["a", "s", "d", "e", "A", "S", "D", "E"],
                default="a",
            ).lower()

            if choice == "a":
                self._log_decision(tool_name, args, ApprovalDecision.ALLOW_ONCE)
                return ApprovalDecision.ALLOW_ONCE, args

            elif choice == "s":
                self._session_allowed.add(tool_name)
                self._log_decision(tool_name, args, ApprovalDecision.ALLOW_SESSION)
                console.print(f"[green]✓ '{tool_name}' will be auto-allowed for this session.[/green]")
                return ApprovalDecision.ALLOW_SESSION, args

            elif choice == "d":
                self._log_decision(tool_name, args, ApprovalDecision.DENY)
                console.print(f"[red]✗ Tool '{tool_name}' denied.[/red]")
                return ApprovalDecision.DENY, args

            elif choice == "e":
                # Let the user edit the raw args JSON
                import json
                console.print("[dim]Edit the arguments (JSON). Press Enter to confirm.[/dim]")
                try:
                    raw = console.input(f"[cyan]args>[/cyan] ")
                    edited_args = json.loads(raw) if raw.strip() else args
                    self._log_decision(tool_name, edited_args, ApprovalDecision.EDITED)
                    return ApprovalDecision.EDITED, edited_args
                except Exception as ex:
                    console.print(f"[red]Invalid JSON: {ex}. Keeping original args.[/red]")
                    return ApprovalDecision.ALLOW_ONCE, args

    def reset_session_memory(self) -> None:
        """Clear all session-level allow/deny decisions."""
        self._session_allowed.clear()
        self._session_denied.clear()

    def _log_decision(
        self,
        tool_name: str,
        args: Dict[str, Any],
        decision: ApprovalDecision,
    ) -> None:
        """Persist the approval decision to the audit log table."""
        if not self.memory:
            return
        try:
            import json
            import datetime
            with self.memory._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO approval_audit
                        (tool_name, args_json, decision, decided_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        tool_name,
                        json.dumps(args),
                        decision.value,
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"[PermissionBroker] Could not log decision: {e}")
