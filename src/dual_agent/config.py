"""Configuration manager and setup wizard for Dual-Process Agent."""

from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, List, Optional
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
    # Needed for hosted OpenAI-compatible gateways (OmniRoute, OpenRouter, a
    # remote vLLM). Local Ollama/vLLM needs no key, so this stays optional.
    hermes_api_key: Optional[str] = None
    
    # User / workspace preferences
    auto_learn_skills: bool = True
    auto_scan_workspace: bool = True


def get_config_file_path() -> str:
    return os.path.join(get_default_data_dir(), "config.json")


def _candidate_env_files() -> List[str]:
    """Paths that may hold a `.env`, in increasing order of precedence."""
    return [
        os.path.join(os.getcwd(), ".env"),          # repo-local (development)
        os.path.join(get_default_data_dir(), ".env"),  # ~/.dual_agent/.env (installed)
    ]


def load_env_files() -> List[str]:
    """Load KEY=VALUE pairs from `.env` files into os.environ.

    The README and install.sh both instruct users to put credentials in a `.env`
    file, but nothing ever read one — no dotenv dependency, no parser. Following
    the documented setup silently left the agent in simulation mode with no
    error and no warning, so it looked like it worked while never contacting
    Jev at all.

    Real environment variables always win: an exported key is never clobbered by
    a stale file, so `TYPESAFE_API_KEY=... dual-agent` still overrides .env.
    """
    loaded: List[str] = []
    for path in _candidate_env_files():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    # Strip surrounding quotes and trailing inline comments.
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = value
            loaded.append(path)
        except Exception as e:
            logger.warning(f"Could not read env file {path}: {e}")
    if loaded:
        logger.debug(f"Loaded environment from: {', '.join(loaded)}")
    return loaded


def load_config() -> AgentConfig:
    """Loads configuration from ~/.dual_agent/config.json, with fallback to environment variables."""
    # Read .env first: it is the setup path the docs describe, and without this
    # the documented flow silently configures nothing.
    load_env_files()

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

    env_hermes = os.getenv("HERMES_API_KEY")
    if env_hermes and not data.get("hermes_api_key"):
        data["hermes_api_key"] = env_hermes
    env_hermes_url = os.getenv("HERMES_BASE_URL")
    if env_hermes_url:
        data["hermes_base_url"] = env_hermes_url
    env_hermes_model = os.getenv("HERMES_MODEL")
    if env_hermes_model:
        data["hermes_model"] = env_hermes_model

    config = AgentConfig(**data)

    # Export resolved provider settings back into the environment.
    #
    # The System 2 providers read os.environ directly (see system_two.py), so
    # without this the values a user saves via `dual-agent config` were parsed,
    # stored and then silently ignored — the provider kept using its defaults and
    # usually fell back to the mock. Real environment variables still take
    # precedence, so a shell export always overrides the config file.
    _export_if_absent("HERMES_BASE_URL", config.hermes_base_url)
    _export_if_absent("HERMES_MODEL", config.hermes_model)
    _export_if_absent("HERMES_API_KEY", config.hermes_api_key)

    return config


def _export_if_absent(name: str, value: Optional[str]) -> None:
    """Set an environment variable only if it is not already set and is truthy."""
    if value and not os.environ.get(name):
        os.environ[name] = value


def save_config(config: AgentConfig) -> str:
    """Save configuration to ~/.dual_agent/config.json with strict 0600 permissions."""
    cfg_path = get_config_file_path()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
    try:
        os.chmod(cfg_path, 0o600)
    except Exception:
        pass
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
        model = Prompt.ask("Model name", default=current.hermes_model)
        current.hermes_model = model.strip()
        # Optional: only hosted gateways need this. Blank is correct for a local
        # server, which is why it is not a required prompt.
        key = Prompt.ask(
            "API key (blank for a local server)",
            default=current.hermes_api_key or "",
            password=True,
        )
        current.hermes_api_key = key.strip()

    # 4. Save
    saved_path = save_config(current)
    console.print(f"[bold green]Configuration saved successfully to:[/bold green] {saved_path}\n")
    return current
