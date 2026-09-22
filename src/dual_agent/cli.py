"""Command-line interface, launcher, and benchmarking telemetry for Dual-Process Agent."""

from __future__ import annotations
import argparse
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.system_two import get_system_two_provider
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.config import load_config, run_configuration_wizard

console = Console()


def render_banner():
    banner_text = (
        "[bold cyan]Dual-Process Agent Runtime[/bold cyan]\n"
        "[dim]System 1: TypeSafe AI Jev (Sub-20ms Reflex) | System 2: LLMs (Hermes/Grok) | Interface: MCP[/dim]"
    )
    console.print(Panel(banner_text, box=box.ROUNDED, border_style="cyan"))


def run_agent_task(
    goal: str,
    provider: Optional[str] = None,
    threshold: Optional[float] = None,
    max_steps: int = 10,
    force_simulation: bool = False,
):
    render_banner()

    cfg = load_config()
    active_provider = provider or cfg.system_two_provider
    active_threshold = threshold if threshold is not None else cfg.system_one_confidence_threshold

    console.print(f"[bold yellow]Task Goal:[/bold yellow] {goal}")
    console.print(f"[dim]System 2 Fallback:[/dim] [green]{active_provider}[/green] | [dim]Confidence Threshold:[/dim] [green]{active_threshold}[/green]\n")

    s1_client = JevSystemOneClient(
        api_key=cfg.typesafe_api_key,
        base_url=cfg.typesafe_base_url,
        force_simulation=force_simulation,
    )
    s2_provider = get_system_two_provider(active_provider)
    dispatcher = DualProcessDispatcher(
        system_one_client=s1_client,
        system_two_provider=s2_provider,
        confidence_threshold=active_threshold,
    )

    with console.status("[bold green]Running Dual-Process Agent Loop..."):
        result: DispatchResult = dispatcher.run(goal=goal, max_steps=max_steps)

    # Telemetry table — MEASURED VALUES ONLY.
    # This table used to print a "Traditional LLM Baseline" column computed from
    # hardcoded constants (1200ms / 1500 tokens per step) and an "Improvement"
    # column derived from it. Those numbers were not measured, so they are gone.
    table = Table(title="Execution Telemetry (measured)", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row(
        "Total Steps",
        f"{result.total_steps} (S1: {result.system_one_steps}, S2: {result.system_two_steps})",
    )
    table.add_row("Total Latency", f"{result.total_latency_ms:.1f} ms")
    table.add_row("System 1 Latency", f"{result.system_one_latency_ms:.1f} ms")
    table.add_row("System 2 Latency", f"{result.system_two_latency_ms:.1f} ms")
    table.add_row("Tokens Consumed", f"{result.tokens_used:,} tokens")

    console.print(table)

    if result.used_simulated_system_one:
        console.print(
            Panel(
                "[bold yellow]⚠ System 1 is SIMULATED — these numbers do not measure Jev.[/bold yellow]\n"
                f"Reason: {result.system_one_fallback_reason}\n"
                f"Of the System 1 latency above, {result.simulated_latency_ms:.1f} ms is an emulated\n"
                "network roundtrip inserted by the local stub, not model time.\n"
                "Set TYPESAFE_API_KEY (and install typesafe-sdk) for real System 1 routing.",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    # A real savings/speedup figure requires running the same goal through a
    # plain single-model agent loop and comparing. Not implemented yet, so we
    # say so instead of printing an invented percentage.
    console.print(
        "[dim]Token-savings and speedup figures are omitted: no baseline run of this\n"
        "goal against a plain-LLM agent was performed. Do not cite one without it.[/dim]"
    )

    status_icon = "✅" if result.is_completed else "⚠️"
    console.print(f"\n{status_icon} [bold]Outcome:[/bold] {result.final_output or 'Task completed.'}\n")


def main():
    # If no arguments provided, launch interactive conversational shell
    if len(sys.argv) == 1:
        from dual_agent.shell import InteractiveShell
        shell = InteractiveShell()
        shell.start()
        return

    # If first argument is 'config', run configuration wizard
    if sys.argv[1].lower() in ("config", "--config"):
        run_configuration_wizard()
        return

    # If first argument is 'update', run self-updater
    if sys.argv[1].lower() in ("update", "--update"):
        from dual_agent.updater import perform_update
        perform_update()
        return

    # If first argument is '--ui', launch the web dashboard
    if sys.argv[1].lower() in ("ui", "--ui"):
        port = 7860
        no_browser = False
        for arg in sys.argv[2:]:
            if arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
            elif arg == "--no-browser":
                no_browser = True
            else:
                try:
                    port = int(arg)
                except ValueError:
                    pass
        from dual_agent.web.server import run_server
        run_server(host="localhost", port=port, open_browser=not no_browser)
        return

    # Messaging gateway (Telegram) + scheduler daemon.
    # These were documented in the README, install.sh, the shell help and the
    # scheduler docstring, but had no handler here — so `dual-agent --gateway`
    # died with "unrecognized arguments", leaving the gateway package and all
    # scheduled jobs unreachable.
    if sys.argv[1].lower() in ("--gateway", "gateway", "--daemon", "daemon"):
        max_steps = 10
        for arg in sys.argv[2:]:
            if arg.startswith("--max-steps="):
                try:
                    max_steps = int(arg.split("=", 1)[1])
                except ValueError:
                    pass
        from dual_agent.gateway.runner import run_gateway
        try:
            run_gateway(max_steps=max_steps)
        except ImportError as e:
            console.print(
                f"[bold red]Gateway dependencies missing.[/bold red]\n{e}\n"
                "[dim]Install with:  pip install 'dual-agent[gateway]'[/dim]"
            )
            raise SystemExit(1)
        except ValueError as e:
            # Configuration problems (missing token) are user errors, not crashes.
            console.print(f"[bold red]Gateway not started.[/bold red]\n{e}")
            raise SystemExit(2)
        return

    parser = argparse.ArgumentParser(description="Run Dual-Process Agent with Jev and MCP.")
    parser.add_argument(
        "positional_goal",
        nargs="?",
        default=None,
        help="Optional positional goal string.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal description for the agent.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["mock", "hermes", "grok", "anthropic", "openai"],
        help="System 2 model provider.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for System 1 fast path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum agent steps before forcing completion.",
    )

    args = parser.parse_args()
    goal = args.goal or args.positional_goal or "Inspect current project directory and summarize files."

    run_agent_task(
        goal=goal,
        provider=args.provider,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
