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
from dual_agent import __version__

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

    if result.system_two_is_mock:
        console.print(
            Panel(
                "[bold yellow]⚠ System 2 used the MOCK provider — nothing was generated.[/bold yellow]\n"
                f"Reason: {result.system_two_degraded_reason}\n"
                "The 'result' above is canned text from a stub, not model output.\n"
                "Configure a real model, e.g.:\n"
                "  SYSTEM_TWO_PROVIDER=hermes\n"
                "  HERMES_BASE_URL=http://localhost:20128/v1\n"
                "  HERMES_API_KEY=<key>\n"
                "  HERMES_MODEL=auto/cheap",
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


UNIFIED_HELP = """Dual-Process Agent — routes each step with Jev (System 1) and escalates to a
model (System 2) only when the fast path is not confident enough.

USAGE
  dual-agent                                    Interactive shell (no arguments)
  dual-agent --goal "..." [options]             Run one goal
  dual-agent "..." [options]                    Run one goal (positional form)
  dual-agent app                                Native desktop application (WebKit window)
  dual-agent ui [--port=7860] [--no-browser]    Web dashboard (browser)
  dual-agent gateway [--max-steps=N]            Telegram bot + cron scheduler daemon
  dual-agent config                             Configuration wizard
  dual-agent update                             Self-update from git

OPTIONS for a single goal
  --goal TEXT                Goal to achieve
  --provider NAME            System 2 provider: mock | hermes | grok | openai
  --threshold FLOAT          System 1 confidence needed to trust the fast path
  --max-steps INT            Step budget (default 10)
  --version                  Show the Dual-Process Agent version and exit

ENVIRONMENT
  DUAL_AGENT_HOME                 Data dir. Default ~/.dual_agent.
  TYPESAFE_API_KEY                Enables live Jev routing. Without it System 1
                                  is a local stub and every run says so.
  TYPESAFE_BASE_URL               Default https://api.typesafe.ai

  SYSTEM_TWO_PROVIDER             mock | hermes | grok | openai (default mock).
  HERMES_BASE_URL                 OpenAI-compatible System 2 endpoint.
  HERMES_API_KEY                  Bearer token for that endpoint.
  HERMES_MODEL                    Model name.
  OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL
  GROK_API_KEY / GROK_MODEL
  SYSTEM_TWO_TIMEOUT              Seconds (default 45).
  SYSTEM_TWO_MAX_TOKENS           Generation cap (default 4000).

  DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true   Skip approval prompts. Required for
                                           unattended use (gateway), where there
                                           is no terminal to answer them.
  DUAL_AGENT_SCREEN_CONTROL=0              Kill switch: disable all mouse and
                                           keyboard actuation.
  DUAL_AGENT_SCREEN_LOOP=1                 Enable screen perception loop.

SCREEN CONTROL REQUIRES A VISION MODEL (it is not free)
  Actuation (`mouse_click`, `key_press`, `screen_diff`) is local and costs
  nothing. Sight does not: it needs a multimodal model, and without one the
  screen loop is blind and the run aborts naming that precondition instead of
  falling back to the mock. The mock's canned text is never a screen reading.

    DUAL_AGENT_VISION_ENABLED=1          Required for the settings below to
                                        apply. Also vision_enabled: true in
                                        ~/.dual_agent/config.json.
    VISION_PROVIDER                      openai | grok | custom
    VISION_MODEL                         e.g. gpt-4o, grok-2-vision-1212,
                                        qwen2-vl for a local server
    VISION_BASE_URL                      OpenAI-compatible endpoint
    VISION_API_KEY                       Credential; not needed for localhost
    VISION_MAX_DIMENSION                 Long-edge bound for a sent screenshot
                                        (default 1400; 3420x2224 -> 1400x910)

  Measured on this machine, 2026-09-22 — not assumed:
    - No local VLM is installed. Ports 11434 (Ollama), 1234 (LM Studio) and
      8080 (vLLM) are all closed.
    - The configured deepseek-chat (served as deepseek-flash) IS vision-capable:
      it named Red/Blue/Green for solid frames and described an image with no
      textual question. The 'deepseek is text-only' note elsewhere in this repo
      is stale. It answered 'White' for a pure black frame, so its readings are
      evidence and not ground truth.
    - Direct api.deepseek.com vision calls returned 200 on 12/12 attempts. The
      429s seen here come from the local OmniRoute proxy on :20128
      (504 RATE_LIMIT_EXECUTION_TIMEOUT, then 429 model_cooldown). A 429 is
      always reported as a rate limit and never as a screen reading.
    - Grid overlays help a model land near a target; they do not eliminate
      coordinate misses. No accuracy figure is claimed anywhere.

  A run therefore needs a paid vision model or a locally installed VLM.

  VISION_PROVIDER / VISION_MODEL / VISION_BASE_URL are read when deliberately
  set; config.json remains the source of truth otherwise.
  DUAL_AGENT_UI_FORCE_SIMULATION=true      Keep the dashboard on the offline stub.
  DUAL_AGENT_UI_ALLOW_PUBLIC_BIND=true     Allow a non-loopback dashboard bind.
  TELEGRAM_BOT_TOKEN                       Required by `gateway`.
  TELEGRAM_ALLOWED_USER_IDS                Required by `gateway`. Comma-separated
                                           numeric ids; everyone else is refused.

QUICK START
  dual-agent config                 # set keys, saved to ~/.dual_agent/config.json
  dual-agent --goal "list the files in this directory"

The free path: point System 2 at a local OpenAI-compatible server (Ollama, vLLM,
LM Studio, OmniRoute) instead of a paid API:
  SYSTEM_TWO_PROVIDER=hermes HERMES_BASE_URL=http://localhost:20128/v1 \\
  HERMES_MODEL=auto/cheap dual-agent --goal "..."
"""


def main():
    # `main()` dispatches on the first argument before argparse runs, so the
    # argparse help could only ever show the single-goal flags. `dual-agent
    # --help` therefore never mentioned ui, gateway, config or update — half the
    # interface was invisible to the one command everyone types first.
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("-h", "--help", "help"):
        console.print(UNIFIED_HELP)
        return

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

    # If first argument is 'app' or 'desktop', launch the native desktop application
    if sys.argv[1].lower() in ("app", "--app", "desktop", "--desktop"):
        if any(h in sys.argv[2:] for h in ("--help", "-h", "help")):
            console.print("Usage: dual-agent app [--port=PORT]\n\nLaunch native standalone desktop application.")
            return
        port = None
        for arg in sys.argv[2:]:
            if arg.startswith("--port="):
                try:
                    port = int(arg.split("=", 1)[1])
                except ValueError:
                    pass
        from dual_agent.app import run_desktop_app
        run_desktop_app(port=port)
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

    parser = argparse.ArgumentParser(
        description="Run Dual-Process Agent with Jev and MCP.",
        epilog=(
            "Other entry points (handled before this parser, so they are not listed "
            "above): ui [--port=N] [--no-browser], gateway [--max-steps=N], config, "
            "update. Run 'dual-agent --help' for the full interface and environment "
            "variables."
        ),
    )
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
        choices=["mock", "hermes", "grok", "openai"],
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
    parser.add_argument(
        "--version",
        action="version",
        version=f"dual-agent {__version__}",
        help="Show the Dual-Process Agent version and exit.",
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
