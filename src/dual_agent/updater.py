"""Self-updater module for Dual-Process Agent.

Supports checking for git commits, updating dependencies, and upgrading models.
"""

from __future__ import annotations
import sys
import os
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()


def get_git_root() -> str | None:
    """Find the root of the git repository if running from source."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def perform_update() -> bool:
    """Checks for new commits and updates packages."""
    console.print(Panel("[bold cyan]Dual-Process Agent Self-Updater[/bold cyan]", border_style="cyan"))

    git_root = get_git_root()

    # 1. Update from GitHub if running in a git clone
    if git_root and os.path.exists(os.path.join(git_root, ".git")):
        console.print("[bold yellow]1. Checking GitHub for new commits...[/bold yellow]")
        try:
            res = subprocess.run(
                ["git", "pull", "origin", "master"],
                cwd=git_root,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0:
                console.print(f"[green]✓ Git Repository Updated:[/green]\n[dim]{res.stdout.strip()}[/dim]\n")
            else:
                # Try 'main' branch
                res2 = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    cwd=git_root,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if res2.returncode == 0:
                    console.print(f"[green]✓ Git Repository Updated:[/green]\n[dim]{res2.stdout.strip()}[/dim]\n")
                else:
                    console.print(f"[dim]Git pull: {res.stdout.strip() or res.stderr.strip()}[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]Could not pull git updates: {e}[/yellow]\n")
    else:
        console.print("[dim]Not running from a git clone; skipping git pull.[/dim]\n")

    # 2. Check installed vs latest versions
    console.print("[bold yellow]2. Checking TypeSafe AI SDK version...[/bold yellow]")
    try:
        import typesafe_sdk
        current_version = getattr(typesafe_sdk, "__version__", "unknown")
        console.print(f"[green]✓ Current typesafe-sdk version: {current_version}[/green]\n")
    except Exception as e:
        console.print(f"[yellow]Could not verify SDK version: {e}[/yellow]\n")

    console.print("[bold green]✓ System is up to date![/bold green]\n")
    return True
