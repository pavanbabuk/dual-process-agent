"""Configuration manager and setup wizard for Dual-Process Agent."""

from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from dual_agent.memory import get_default_data_dir

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration settings for Dual-Process Agent."""
    typesafe_api_key: Optional[str] = None
    typesafe_base_url: str = "https://api.typesafe.ai"
    system_two_provider: str = "mock"
    system_one_confidence_threshold: float = 0.85
    
    # Provider-specific keys & endpoints
    grok_api_key: Optional[str] = None
    grok_model: str = "grok-2-latest"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    hermes_base_url: str = "http://localhost:11434/v1"
    hermes_model: str = "nous-hermes-3-llama-3.1-8b"
    
    # User / workspace preferences
    auto_learn_skills: bool = True
    auto_scan_workspace: bool = True


def get_config_file_path() -> str:
    return os.path.join(get_default_data_dir(), "config.json")


def load_config() -> AgentConfig:
    """Loads configuration from ~/.dual_agent/config.json, with fallback to environment variables."""
    cfg_path = get_config_file_path()
    data: Dict[str, Any] = {}

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {cfg_path}: {e}")

    # Fallback to current environment variables
    env_typesafe = os.getenv("TYPESAFE_API_KEY")
    if env_typesafe and not data.get("typesafe_api_key"):
        data["typesafe_api_key"] = env_typesafe

    env_provider = os.getenv("SYSTEM_TWO_PROVIDER")
    if env_provider and not data.get("system_two_provider"):
        data["system_two_provider"] = env_provider

    env_grok = os.getenv("GROK_API_KEY")
    if env_grok and not data.get("grok_api_key"):
        data["grok_api_key"] = env_grok

    env_openai = os.getenv("OPENAI_API_KEY")
    if env_openai and not data.get("openai_api_key"):
        data["openai_api_key"] = env_openai

    env_anthropic = os.getenv("ANTHROPIC_API_KEY")
    if env_anthropic and not data.get("anthropic_api_key"):
        data["anthropic_api_key"] = env_anthropic

    return AgentConfig(**data)


def save_config(config: AgentConfig) -> str:
    """Save configuration to ~/.dual_agent/config.json."""
    cfg_path = get_config_file_path()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
    return cfg_path


def run_configuration_wizard() -> AgentConfig:
    """Interactive command-line wizard to configure API keys and preferences."""
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel

    console = Console()
    console.print(Panel("[bold cyan]Dual-Process Agent Configuration Wizard[/bold cyan]", border_style="cyan"))

    current = load_config()

    # 1. TypeSafe AI API Key
    ts_key = Prompt.ask(
        "[bold yellow]TypeSafe AI API Key[/bold yellow] (from https://console.typesafe.ai)",
        default=current.typesafe_api_key or "",
    )
    if ts_key:
        current.typesafe_api_key = ts_key.strip()
        os.environ["TYPESAFE_API_KEY"] = current.typesafe_api_key

    # 2. System 2 Provider
    provider = Prompt.ask(
        "[bold yellow]Select System 2 Reasoner Provider[/bold yellow]",
        choices=["mock", "hermes", "grok", "anthropic", "openai"],
        default=current.system_two_provider,
    )
    current.system_two_provider = provider

    # 3. Provider-specific keys
    if provider == "grok":
        key = Prompt.ask("xAI Grok API Key", default=current.grok_api_key or "")
        current.grok_api_key = key.strip()
    elif provider == "anthropic":
        key = Prompt.ask("Anthropic API Key", default=current.anthropic_api_key or "")
        current.anthropic_api_key = key.strip()
    elif provider == "openai":
        key = Prompt.ask("OpenAI API Key", default=current.openai_api_key or "")
        current.openai_api_key = key.strip()
    elif provider == "hermes":
        url = Prompt.ask("Hermes Ollama/vLLM Base URL", default=current.hermes_base_url)
        current.hermes_base_url = url.strip()

    # 4. Save
    saved_path = save_config(current)
    console.print(f"[bold green]Configuration saved successfully to:[/bold green] {saved_path}\n")
    return current
