"""Test that config.json values are honored and beat code defaults."""

import os
import json
import pytest
from dual_agent.config import load_config, AgentConfig
from dual_agent.system_two import get_system_two_provider, CustomLLMProvider


def test_config_json_beats_code_default(tmp_path, monkeypatch):
    """config.json settings for hermes / custom_llm must be honored, not ignored."""
    # Ensure env vars are clean
    for key in [
        "SYSTEM_TWO_PROVIDER",
        "HERMES_BASE_URL",
        "CUSTOM_LLM_BASE_URL",
        "HERMES_MODEL",
        "CUSTOM_LLM_MODEL",
        "HERMES_API_KEY",
        "CUSTOM_LLM_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    cfg_dir = tmp_path / ".dual_agent"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text(json.dumps({
        "system_two_provider": "hermes",
        "hermes_base_url": "http://localhost:20128/v1",
        "hermes_model": "auto/best-chat",
        "hermes_api_key": "configured-secret-key"
    }))
    monkeypatch.setenv("DUAL_AGENT_HOME", str(cfg_dir))

    cfg = load_config()
    assert cfg.custom_llm_base_url == "http://localhost:20128/v1"
    assert cfg.custom_llm_model == "auto/best-chat"
    assert cfg.custom_llm_api_key == "configured-secret-key"

    # The provider factory must use the configured endpoint
    provider = get_system_two_provider(config=cfg)
    assert isinstance(provider, CustomLLMProvider)
    assert provider.base_url == "http://localhost:20128/v1"
    assert provider.model == "auto/best-chat"
    assert provider.api_key == "configured-secret-key"


def test_unreachable_endpoint_reports_endpoint_in_degraded_reason():
    """Unreachable System 2 endpoint must report the attempted base_url and not appear to succeed."""
    provider = CustomLLMProvider(
        base_url="http://localhost:99999/v1",
        model="test-model",
        timeout=0.5,
    )
    res = provider.generate_step("Say hello")
    assert res.is_mock is True
    assert "http://localhost:99999/v1" in res.degraded_reason
    assert "failed" in res.degraded_reason.lower()


def test_dispatcher_fails_loudly_when_s2_unreachable():
    """Dispatcher must report failure and not claim completion when real S2 is unreachable."""
    from dual_agent.dispatcher import DualProcessDispatcher
    from dual_agent.typesafe_client import JevSystemOneClient
    from dual_agent.mcp_host import MCPHost
    from dual_agent.memory import MemoryEngine

    unreachable_s2 = CustomLLMProvider(
        base_url="http://127.0.0.1:1/v1",
        model="unreachable-model",
        timeout=0.2,
    )
    # Force slow path by using confidence_threshold=1.0
    dispatcher = DualProcessDispatcher(
        system_one_client=JevSystemOneClient(force_simulation=True),
        system_two_provider=unreachable_s2,
        mcp_host=MCPHost(),
        memory_engine=MemoryEngine(),
        confidence_threshold=1.0,
    )
    res = dispatcher.run("Complex creative synthesis task")
    assert res.is_completed is False
    assert res.system_two_is_mock is True
    assert "System 2 failure" in (res.final_output or "")
