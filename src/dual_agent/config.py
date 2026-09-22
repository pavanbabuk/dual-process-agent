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

    # DeepSeek Provider
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"

    # Provider-specific keys & endpoints
    grok_api_key: Optional[str] = None
    grok_model: str = "grok-2-latest"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Custom / Local OpenAI-compatible LLM endpoint
    custom_llm_base_url: str = "http://localhost:11434/v1"
    custom_llm_model: str = "llama3.1"
    custom_llm_api_key: Optional[str] = None

    # Backward compatibility aliases for existing config files
    hermes_base_url: Optional[str] = None
    hermes_model: Optional[str] = None
    hermes_api_key: Optional[str] = None

    # User / workspace preferences
    auto_learn_skills: bool = True
    auto_scan_workspace: bool = True

    # Vision Provider (paid/local VLM for screen control)
    vision_provider: Optional[str] = None
    vision_model: Optional[str] = None
    vision_base_url: Optional[str] = None
    vision_api_key: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Sync backward-compatible hermes_* fields with custom_llm_* fields."""
        if self.hermes_base_url and not self.custom_llm_base_url:
            self.custom_llm_base_url = self.hermes_base_url
        if self.hermes_model and not self.custom_llm_model:
            self.custom_llm_model = self.hermes_model
        if self.hermes_api_key and not self.custom_llm_api_key:
            self.custom_llm_api_key = self.hermes_api_key


def get_config_file_path() -> str:
    return os.path.join(get_default_data_dir(), "config.json")


def _candidate_env_files() -> List[str]:
    """Paths that may hold a `.env`, in increasing order of precedence."""
    return [
        os.path.join(os.getcwd(), ".env"),          # repo-local (development)
        os.path.join(get_default_data_dir(), ".env"),  # ~/.dual_agent/.env (installed)
    ]


_LOADED_FROM_ENV_FILE: set[str] = set()


def load_env_files() -> List[str]:
    """Load KEY=VALUE pairs from `.env` files into os.environ.

    Returns the list of .env paths that were loaded. Keys populated from these
    files are recorded in _LOADED_FROM_ENV_FILE so load_config() knows they are
    defaults that must not override explicit config.json values.
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
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = value
                        _LOADED_FROM_ENV_FILE.add(key)
            loaded.append(path)
        except Exception as e:
            logger.warning(f"Could not read env file {path}: {e}")
    return loaded


def load_config() -> AgentConfig:
    """Loads configuration from ~/.dual_agent/config.json, with fallback to environment variables.

    Hierarchy:
      1. Explicit shell environment variables (os.environ, not from .env file)
      2. config.json (primary source of truth for saved settings)
      3. .env file defaults
      4. Hardcoded code defaults
    """
    load_env_files()

    cfg_path = get_config_file_path()
    data: Dict[str, Any] = {}

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {cfg_path}: {e}")

    # Synchronize legacy hermes_* keys into custom_llm_* if custom_llm_* is not explicitly set
    if "hermes_base_url" in data and "custom_llm_base_url" not in data:
        data["custom_llm_base_url"] = data["hermes_base_url"]
    if "hermes_model" in data and "custom_llm_model" not in data:
        data["custom_llm_model"] = data["hermes_model"]
    if "hermes_api_key" in data and "custom_llm_api_key" not in data:
        data["custom_llm_api_key"] = data["hermes_api_key"]

    def _resolve(key: str, env_var: str, default: Any = None) -> Any:
        # 1. Shell environment variable (present in os.environ and not from .env file)
        if env_var in os.environ and env_var not in _LOADED_FROM_ENV_FILE:
            return os.environ[env_var]
        # 2. config.json
        if key in data and data[key] is not None and data[key] != "":
            return data[key]
        # 3. .env file default (or fallback env)
        if env_var in os.environ:
            return os.environ[env_var]
        # 4. Code default
        return data.get(key, default)

    resolved: Dict[str, Any] = {
        "typesafe_api_key": _resolve("typesafe_api_key", "TYPESAFE_API_KEY", None),
        "typesafe_base_url": _resolve("typesafe_base_url", "TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
        "system_two_provider": _resolve("system_two_provider", "SYSTEM_TWO_PROVIDER", "mock"),
        "system_one_confidence_threshold": float(_resolve("system_one_confidence_threshold", "SYSTEM_ONE_CONFIDENCE_THRESHOLD", 0.85)),
        "deepseek_api_key": _resolve("deepseek_api_key", "DEEPSEEK_API_KEY", None),
        "deepseek_model": _resolve("deepseek_model", "DEEPSEEK_MODEL", "deepseek-chat"),
        "grok_api_key": _resolve("grok_api_key", "GROK_API_KEY", None),
        "grok_model": _resolve("grok_model", "GROK_MODEL", "grok-2-latest"),
        "openai_api_key": _resolve("openai_api_key", "OPENAI_API_KEY", None),
        "anthropic_api_key": _resolve("anthropic_api_key", "ANTHROPIC_API_KEY", None),
        "custom_llm_base_url": (
            _resolve("custom_llm_base_url", "CUSTOM_LLM_BASE_URL", None)
            or _resolve("hermes_base_url", "HERMES_BASE_URL", "http://localhost:11434/v1")
        ),
        "custom_llm_model": (
            _resolve("custom_llm_model", "CUSTOM_LLM_MODEL", None)
            or _resolve("hermes_model", "HERMES_MODEL", "llama3.1")
        ),
        "custom_llm_api_key": (
            _resolve("custom_llm_api_key", "CUSTOM_LLM_API_KEY", None)
            or _resolve("hermes_api_key", "HERMES_API_KEY", None)
        ),
        "vision_provider": _resolve("vision_provider", "VISION_PROVIDER", None),
        "vision_model": _resolve("vision_model", "VISION_MODEL", None),
        "vision_base_url": _resolve("vision_base_url", "VISION_BASE_URL", None),
        "vision_api_key": _resolve("vision_api_key", "VISION_API_KEY", None),
        "auto_learn_skills": bool(data.get("auto_learn_skills", True)),
        "auto_scan_workspace": bool(data.get("auto_scan_workspace", True)),
    }

    config = AgentConfig(**resolved)

    # Export resolved values back into env if not already set, so subprocesses / adapters see them
    _export_if_absent("TYPESAFE_API_KEY", config.typesafe_api_key)
    _export_if_absent("DEEPSEEK_API_KEY", config.deepseek_api_key)
    _export_if_absent("DEEPSEEK_MODEL", config.deepseek_model)
    _export_if_absent("CUSTOM_LLM_BASE_URL", config.custom_llm_base_url)
    _export_if_absent("CUSTOM_LLM_MODEL", config.custom_llm_model)
    _export_if_absent("CUSTOM_LLM_API_KEY", config.custom_llm_api_key)
    _export_if_absent("HERMES_BASE_URL", config.custom_llm_base_url)
    _export_if_absent("HERMES_MODEL", config.custom_llm_model)
    _export_if_absent("HERMES_API_KEY", config.custom_llm_api_key)

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
        choices=["mock", "deepseek", "grok", "openai", "custom"],
        default=current.system_two_provider,
    )
    current.system_two_provider = provider

    # 3. Provider-specific keys
    if provider == "deepseek":
        key = Prompt.ask("DeepSeek API Key (from https://platform.deepseek.com)", default=current.deepseek_api_key or "")
        current.deepseek_api_key = key.strip()
        model = Prompt.ask("DeepSeek Model (deepseek-chat or deepseek-reasoner)", default=current.deepseek_model)
        current.deepseek_model = model.strip()
    elif provider == "grok":
        key = Prompt.ask("xAI Grok API Key", default=current.grok_api_key or "")
        current.grok_api_key = key.strip()
    elif provider == "openai":
        key = Prompt.ask("OpenAI API Key", default=current.openai_api_key or "")
        current.openai_api_key = key.strip()
    elif provider in ("custom", "hermes"):
        url = Prompt.ask("Local LLM Base URL (Ollama / vLLM)", default=current.custom_llm_base_url)
        current.custom_llm_base_url = url.strip()
        model = Prompt.ask("Model name", default=current.custom_llm_model)
        current.custom_llm_model = model.strip()
        key = Prompt.ask(
            "API key (blank for local server)",
            default=current.custom_llm_api_key or "",
            password=True,
        )
        current.custom_llm_api_key = key.strip()

    # 4. Save
    saved_path = save_config(current)
    console.print(f"[bold green]Configuration saved successfully to:[/bold green] {saved_path}\n")
    return current
