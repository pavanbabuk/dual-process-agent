"""Tests for Phase 1 Fix 1: config.json is the source of truth for the provider."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from dual_agent.config import AgentConfig, save_config, load_config
from dual_agent.system_two import CustomLLMProvider, get_system_two_provider
from dual_agent.dispatcher import DualProcessDispatcher


def test_config_json_base_url_used_when_no_env_var(tmp_path, monkeypatch):
    """A temp config with a known base_url and no env var asserts the client uses that URL."""
    # Ensure no environment variables override
    for k in ("CUSTOM_LLM_BASE_URL", "HERMES_BASE_URL", "SYSTEM_TWO_PROVIDER"):
        monkeypatch.delenv(k, raising=False)

    cfg_dir = tmp_path / "agent_home"
    cfg_dir.mkdir()
    monkeypatch.setenv("DUAL_AGENT_HOME", str(cfg_dir))

    target_url = "https://custom-target-endpoint.example.com/v1"
    config = AgentConfig(
        system_two_provider="custom",
        custom_llm_base_url=target_url,
    )
    save_config(config)

    # 1. CustomLLMProvider without arguments must resolve to config.json target_url
    provider = CustomLLMProvider()
    assert provider.base_url == target_url

    # 2. get_system_two_provider() must resolve to CustomLLMProvider with target_url
    s2 = get_system_two_provider()
    assert isinstance(s2, CustomLLMProvider)
    assert s2.base_url == target_url

    # 3. DualProcessDispatcher with no s2 passed must use the configured base_url
    dispatcher = DualProcessDispatcher()
    assert isinstance(dispatcher.s2, CustomLLMProvider)
    assert dispatcher.s2.base_url == target_url


def test_env_var_overrides_config_json(tmp_path, monkeypatch):
    """When deliberately set in env, the env var overrides config.json."""
    cfg_dir = tmp_path / "agent_home"
    cfg_dir.mkdir()
    monkeypatch.setenv("DUAL_AGENT_HOME", str(cfg_dir))

    config = AgentConfig(
        system_two_provider="custom",
        custom_llm_base_url="https://stored-in-json.example.com/v1",
    )
    save_config(config)

    override_url = "https://override-from-env.example.com/v1"
    monkeypatch.setenv("CUSTOM_LLM_BASE_URL", override_url)

    provider = CustomLLMProvider()
    assert provider.base_url == override_url


def test_misconfigured_endpoint_reports_url_attempted():
    """A failed call reports the exact URL attempted in degraded_reason."""
    bad_url = "http://127.0.0.1:59999/v1"
    provider = CustomLLMProvider(base_url=bad_url, timeout=0.2)
    resp = provider.generate_step("hello")
    assert resp.is_mock is True
    assert resp.degraded_reason is not None
    assert bad_url in resp.degraded_reason
