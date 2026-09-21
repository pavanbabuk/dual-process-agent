"""Interactive Terminal TUI Shell for Dual-Process Agent.

Provides an ongoing conversational REPL with slash commands and memory persistence.
"""

from __future__ import annotations
import sys
import os
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from dual_agent.config import load_config, run_configuration_wizard
from dual_agent.memory import MemoryEngine
from dual_agent.mcp_host import MCPHost
from dual_agent.mcp_manager import MCPManager
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.system_two import get_system_two_provider
from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult

console = Console()


class InteractiveShell:
    """The conversational TUI shell for Dual-Process Agent."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryEngine] = None,
        mcp_host: Optional[MCPHost] = None,
        force_simulation: bool = False,
    ):
        self.config = config or load_config()
        self.memory = memory or MemoryEngine()
        self.mcp_host = mcp_host or MCPHost()
        self.mcp_manager = MCPManager()
        self.mcp_manager.attach_to_host(self.mcp_host)

        self.s1 = JevSystemOneClient(
            api_key=self.config.typesafe_api_key,
            base_url=self.config.typesafe_base_url,
            force_simulation=force_simulation,
        )
        self.s2 = get_system_two_provider(self.config.system_two_provider)
        self.dispatcher = DualProcessDispatcher(
            system_one_client=self.s1,
            system_two_provider=self.s2,
            mcp_host=self.mcp_host,
            memory_engine=self.memory,
            confidence_threshold=self.config.system_one_confidence_threshold,
        )

    def print_welcome(self):
        banner_text = (
            "[bold cyan]Dual-Process Agent Shell (v1.0)[/bold cyan]\n"
            f"[dim]System 1: TypeSafe AI Jev | System 2: {self.config.system_two_provider.upper()} | "
            f"Active Tools: {len(self.mcp_host.list_tools())}[/dim]\n"
            "[dim]Type your instruction, or [yellow]/help[/yellow] for available commands.[/dim]"
        )
        console.print(Panel(banner_text, border_style="cyan", box=box.ROUNDED))

    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if shell should continue, False to exit."""
        cmd_lower = cmd.strip().lower()

        if cmd_lower in ("/exit", "/quit", "exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            return False

        elif cmd_lower in ("/help", "help"):
            self._show_help()

        elif cmd_lower in ("/tools", "tools"):
            self._show_tools()

        elif cmd_lower in ("/memory", "/stats", "memory"):
            self._show_memory()

        elif cmd_lower in ("/config", "config"):
            self.config = run_configuration_wizard()
            # Refresh providers
            self.s1 = JevSystemOneClient(
                api_key=self.config.typesafe_api_key,
                base_url=self.config.typesafe_base_url,
            )
            self.s2 = get_system_two_provider(self.config.system_two_provider)
            self.dispatcher.s1 = self.s1
            self.dispatcher.s2 = self.s2

        elif cmd_lower in ("/clear", "clear"):
            console.clear()
            self.print_welcome()

        else:
            # Execute agent task
            self._execute_goal(cmd)

        return True

    def _show_help(self):
        table = Table(title="Available Commands", box=box.SIMPLE_HEAVY)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        table.add_row("/tools", "List all registered MCP tools and their descriptions")
        table.add_row("/memory", "View learned skills and cumulative performance stats")
        table.add_row("/config", "Open interactive configuration wizard")
        table.add_row("/clear", "Clear screen and reset active view")
        table.add_row("/exit", "Quit the interactive shell")
        table.add_row("<text>", "Any natural language goal or programming instruction")
        console.print(table)

    def _show_tools(self):
        table = Table(title="Registered MCP Tools", box=box.SIMPLE_HEAVY)
        table.add_column("Tool Name", style="green", no_wrap=True)
        table.add_column("Description", style="white")

        for tool in self.mcp_host.list_tools():
            table.add_row(tool.name, tool.description)
        console.print(table)

    def _show_memory(self):
        stats = self.memory.get_aggregate_stats()
        skills = self.memory.get_all_skills()

        stat_table = Table(title="Memory & Savings Statistics", box=box.SIMPLE_HEAVY)
        stat_table.add_column("Metric", style="cyan")
        stat_table.add_column("Value", style="bold green")

        stat_table.add_row("Total Executed Sessions", str(stats["total_sessions"]))
        stat_table.add_row("Total Steps Completed", str(stats["total_steps"]))
        stat_table.add_row("System 1 Fast Steps", f"{stats['total_s1_steps']} (~15ms each)")
        stat_table.add_row("System 2 Slow Steps", str(stats['total_s2_steps']))
        stat_table.add_row("Average Token Savings", f"{stats['avg_token_savings_pct']}%")
        console.print(stat_table)

        if skills:
            skill_table = Table(title="Learned Routine Skills", box=box.SIMPLE_HEAVY)
            skill_table.add_column("Skill Name", style="yellow")
            skill_table.add_column("Tool Sequence", style="cyan")
            skill_table.add_column("Uses", style="green")

            for s in skills[:10]:
                skill_table.add_row(s.name, " -> ".join(s.tool_sequence), str(s.success_count))
            console.print(skill_table)

    def _execute_goal(self, goal: str):
        console.print(f"\n[bold yellow]Executing Task:[/bold yellow] {goal}\n")
        with console.status("[bold green]Dual-Process loop running (Jev System 1)..."):
            result: DispatchResult = self.dispatcher.run(goal)

        status_icon = "✅" if result.is_completed else "⚠️"
        console.print(f"{status_icon} [bold]Result:[/bold] {result.final_output or 'Completed.'}")
        console.print(
            f"[dim]Stats: {result.total_steps} steps (S1: {result.system_one_steps}, S2: {result.system_two_steps}) | "
            f"Latency: {result.total_latency_ms:.1f}ms ({result.speedup_ratio}x faster) | "
            f"Token Savings: {result.estimated_token_savings_pct}%[/dim]\n"
        )

    def start(self):
        self.print_welcome()
        while True:
            try:
                user_input = console.input("[bold cyan]dual-agent>[/bold cyan] ")
                if not user_input.strip():
                    continue
                should_continue = self.handle_command(user_input)
                if not should_continue:
                    break
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Exiting Dual-Process Agent.[/yellow]")
                break


def main():
    shell = InteractiveShell()
    shell.start()


if __name__ == "__main__":
    main()
