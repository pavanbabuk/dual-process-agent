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
from dual_agent.skills_manager import SkillsManager
from dual_agent.scheduler import CronScheduler
from dual_agent.team_manifest import TeamManifestManager

console = Console()


class InteractiveShell:
    """The conversational TUI shell for Dual-Process Agent."""

    def __init__(
        self,
        config=None,
        memory: Optional[MemoryEngine] = None,
        mcp_host: Optional[MCPHost] = None,
        force_simulation: bool = False,
    ):
        self.config = config or load_config()
        self.memory = memory or MemoryEngine()
        self.mcp_host = mcp_host or MCPHost()
        self.mcp_manager = MCPManager()
        self.mcp_manager.attach_to_host(self.mcp_host)
        self.skills = SkillsManager()
        self.scheduler = CronScheduler(memory_engine=self.memory)
        self.manifest_manager = TeamManifestManager(
            memory_engine=self.memory,
            config=self.config,
            mcp_manager=self.mcp_manager,
            skills_manager=self.skills,
        )

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
            skills_manager=self.skills,
            confidence_threshold=self.config.system_one_confidence_threshold,
        )

    def print_welcome(self):
        banner_text = (
            "[bold cyan]Dual-Process Agent Shell (v2.0)[/bold cyan]\n"
            f"[dim]System 1: TypeSafe AI Jev | System 2: {self.config.system_two_provider.upper()} | "
            f"Active Tools: {len(self.mcp_host.list_tools())}[/dim]\n"
            "[dim]Type your instruction, or [yellow]/help[/yellow] for available commands.[/dim]"
        )
        console.print(Panel(banner_text, border_style="cyan", box=box.ROUNDED))

    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if shell should continue, False to exit."""
        cmd_stripped = cmd.strip()
        cmd_lower = cmd_stripped.lower()

        if cmd_lower in ("/exit", "/quit", "exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            return False

        elif cmd_lower in ("/help", "help"):
            self._show_help()

        elif cmd_lower in ("/tools", "tools"):
            self._show_tools()

        elif cmd_lower in ("/memory", "/stats", "memory"):
            self._show_memory()

        elif cmd_lower in ("/skills", "skills"):
            self._show_skills()

        elif cmd_lower.startswith("/recall ") or cmd_lower.startswith("recall "):
            query = cmd_stripped.split(" ", 1)[1].strip()
            self._recall_sessions(query)

        elif cmd_lower in ("/whoami", "whoami"):
            self._show_user_profile()

        elif cmd_lower.startswith("/schedule ") or cmd_lower.startswith("schedule "):
            description = cmd_stripped.split(" ", 1)[1].strip()
            self._add_schedule(description)

        elif cmd_lower in ("/schedules", "schedules"):
            self._show_schedules()

        elif cmd_lower in ("/export", "export"):
            self._export_manifest()

        elif cmd_lower.startswith("/import ") or cmd_lower.startswith("import "):
            source = cmd_stripped.split(" ", 1)[1].strip()
            self._import_manifest(source)

        elif cmd_lower in ("/config", "config"):
            self.config = run_configuration_wizard()
            self.s1 = JevSystemOneClient(
                api_key=self.config.typesafe_api_key,
                base_url=self.config.typesafe_base_url,
            )
            self.s2 = get_system_two_provider(self.config.system_two_provider)
            self.dispatcher.s1 = self.s1
            self.dispatcher.s2 = self.s2

        elif cmd_lower in ("/update", "update"):
            from dual_agent.updater import perform_update
            perform_update()

        elif cmd_lower in ("/clear", "clear"):
            console.clear()
            self.print_welcome()

        else:
            # Execute agent task
            self._execute_goal(cmd_stripped)

        return True

    def _show_help(self):
        table = Table(title="Available Commands", box=box.SIMPLE_HEAVY)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        table.add_row("/tools", "List all registered MCP tools and their descriptions")
        table.add_row("/memory", "View learned skills and cumulative performance stats")
        table.add_row("/skills", "List all synthesized .SKILL.md procedural memory files")
        table.add_row("/recall <query>", "Full-text search across past sessions")
        table.add_row("/whoami", "Display your auto-maintained USER.md profile")
        table.add_row("/schedule <desc>", "Schedule a recurring task (e.g. 'every day at 9am')")
        table.add_row("/schedules", "List all scheduled jobs")
        table.add_row("/export", "Export agent config as a portable team manifest .md file")
        table.add_row("/import <path|url>", "Import a team manifest from disk or GitHub URL")
        table.add_row("/config", "Open interactive configuration wizard")
        table.add_row("/update", "Pull latest updates from GitHub and upgrade dependencies")
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
        stat_table.add_row("System 1 Fast Steps", str(stats['total_s1_steps']))
        stat_table.add_row("System 2 Slow Steps", str(stats['total_s2_steps']))
        avg_savings = stats.get("avg_token_savings_pct")
        stat_table.add_row(
            "Average Token Savings",
            "n/a (no baseline run)" if avg_savings is None else f"{avg_savings}%",
        )
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
            f"Latency: {result.total_latency_ms:.1f}ms"
            + (" [SIMULATED S1]" if result.used_simulated_system_one else "")
            + "[/dim]\n"
        )

    # ------------------------------------------------------------------
    # New command handlers (Hermes + OpenMausBot inspired)
    # ------------------------------------------------------------------

    def _show_skills(self):
        """List all .SKILL.md files from the skills directory."""
        skills = self.skills.load_all_skills()
        if not skills:
            console.print("[dim]No .SKILL.md files yet. Complete a task to auto-synthesize one.[/dim]")
            return
        t = Table(title=f"Synthesized Skills ({len(skills)} total)", box=box.SIMPLE_HEAVY)
        t.add_column("Skill Name", style="yellow")
        t.add_column("Keywords", style="cyan")
        t.add_column("Tool Sequence", style="white")
        t.add_column("Uses", style="green")
        t.add_column("Last Used", style="dim")
        for s in skills[:15]:
            t.add_row(
                s.name,
                ", ".join(s.intent_keywords[:4]),
                " → ".join(s.tool_sequence[:4]) + ("..." if len(s.tool_sequence) > 4 else ""),
                str(s.success_count),
                s.last_used[:10] if s.last_used else "—",
            )
        console.print(t)
        skills_dir = self.skills.skills_dir
        console.print(f"[dim]Files at: {skills_dir}[/dim]")

    def _recall_sessions(self, query: str):
        """Full-text search across past sessions."""
        if not query:
            console.print("[yellow]Usage: /recall <search terms>[/yellow]")
            return
        results = self.memory.full_text_search(query, limit=8)
        if not results:
            console.print(f"[dim]No past sessions matched '{query}'.[/dim]")
            return
        t = Table(title=f"Recall Results for '{query}'", box=box.SIMPLE_HEAVY)
        t.add_column("Date", style="dim", no_wrap=True)
        t.add_column("Goal", style="cyan")
        t.add_column("Outcome", style="white")
        t.add_column("✓", style="green")
        for r in results:
            t.add_row(
                str(r.get("created_at", ""))[:10],
                str(r.get("goal", ""))[:60],
                str(r.get("outcome", "") or "")[:60],
                "✅" if r.get("is_completed") else "⚠️",
            )
        console.print(t)

    def _show_user_profile(self):
        """Display the USER.md user profile."""
        profile = self.memory.get_user_profile()
        if not profile:
            console.print(
                "[dim]No USER.md profile yet. The agent will build one as you work.[/dim]"
            )
            return
        console.print(Panel(profile, title="[bold cyan]Your User Profile (USER.md)[/bold cyan]",
                            border_style="cyan", box=box.ROUNDED))

    def _add_schedule(self, description: str):
        """Parse a natural language description and add a scheduled job."""
        # The goal is the description itself (user can refine later)
        goal = description
        try:
            job_id = self.scheduler.add_job(description=description, goal=goal)
            console.print(
                f"[green]✓ Scheduled Job #{job_id} added![/green]\n"
                f"[dim]Description: {description}[/dim]\n"
                f"[dim]Start the gateway daemon with [cyan]dual-agent --gateway[/cyan] "
                f"to activate scheduled jobs.[/dim]"
            )
        except ValueError as e:
            console.print(f"[red]Could not parse schedule: {e}[/red]")

    def _show_schedules(self):
        """List all scheduled jobs."""
        jobs = self.scheduler.list_jobs()
        if not jobs:
            console.print("[dim]No scheduled jobs. Use /schedule <description> to add one.[/dim]")
            return
        t = Table(title="Scheduled Jobs", box=box.SIMPLE_HEAVY)
        t.add_column("ID", style="dim", no_wrap=True)
        t.add_column("Description", style="cyan")
        t.add_column("Cron", style="yellow")
        t.add_column("Runs", style="green")
        t.add_column("Last Run", style="dim")
        t.add_column("Enabled", style="white")
        for j in jobs:
            t.add_row(
                str(j["id"]),
                j["description"][:40],
                j["cron_expr"],
                str(j["run_count"]),
                str(j.get("last_run_at") or "Never")[:16],
                "✅" if j["enabled"] else "⏸",
            )
        console.print(t)

    def _export_manifest(self):
        """Export the current agent config as a team manifest .md file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        from dual_agent.memory import get_default_data_dir
        out_path = os.path.join(get_default_data_dir(), f"team_manifest_{timestamp}.md")
        content = self.manifest_manager.export_manifest(output_path=out_path)
        console.print(
            f"[green]✓ Team manifest exported![/green]\n"
            f"[dim]Saved to: {out_path}[/dim]\n"
            f"[dim]Share this file to replicate your agent setup on another machine.[/dim]"
        )

    def _import_manifest(self, source: str):
        """Import a team manifest from a file path or GitHub URL."""
        try:
            manifest = self.manifest_manager.import_manifest(source)
            console.print(
                f"[green]✓ Manifest '{manifest.name}' imported![/green]\n"
                f"[dim]Provider: {manifest.provider} | "
                f"MCP Servers: {len(manifest.mcp_servers)} | "
                f"Skills: {len(manifest.skills)}[/dim]"
            )
        except Exception as e:
            console.print(f"[red]Failed to import manifest: {e}[/red]")

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
