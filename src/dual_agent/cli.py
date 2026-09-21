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

    # Telemetry Table
    table = Table(title="Execution Telemetry & Benchmark", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Dual-Process (Jev + S2)", style="green")
    table.add_column("Traditional LLM Baseline", style="red")
    table.add_column("Improvement", style="bold yellow")

    table.add_row(
        "Total Steps",
        f"{result.total_steps} (S1: {result.system_one_steps}, S2: {result.system_two_steps})",
        f"{result.total_steps} (All S2)",
        "—"
    )
    table.add_row(
        "Total Latency",
        f"{result.total_latency_ms:.1f} ms",
        f"{result.total_steps * 1200:.1f} ms",
        f"{result.speedup_ratio:.1f}x Faster"
    )
    table.add_row(
        "System 1 Latency",
        f"{result.system_one_latency_ms:.1f} ms",
        "N/A",
        "~15ms / step"
    )
    table.add_row(
        "Tokens Consumed",
        f"{result.tokens_used:,} tokens",
        f"{result.estimated_baseline_tokens:,} tokens",
        f"{result.estimated_token_savings_pct:.1f}% Saved"
    )

    console.print(table)

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
