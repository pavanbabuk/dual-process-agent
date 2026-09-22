"""Tests for DeepSeek provider, Custom LLM provider, and system_two factory."""

import pytest
import os
from dual_agent.system_two import (
    DeepSeekProvider,
    CustomLLMProvider,
    MockSystemTwoProvider,
    get_system_two_provider,
)
from dual_agent.config import AgentConfig, load_config


def test_deepseek_provider_init():
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-deepseek"
    os.environ["DEEPSEEK_MODEL"] = "deepseek-reasoner"
    provider = DeepSeekProvider()
    assert provider.api_key == "sk-test-deepseek"
    assert provider.model == "deepseek-reasoner"
    assert provider.base_url == "https://api.deepseek.com"


def test_deepseek_provider_missing_key_falls_back():
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]
    provider = DeepSeekProvider()
    res = provider.generate_step("Test prompt")
    assert res.is_mock is True
    assert "deepseek_api_key" in res.degraded_reason.lower()


def test_custom_llm_provider_init():
    os.environ["CUSTOM_LLM_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["CUSTOM_LLM_MODEL"] = "llama3.2"
    provider = CustomLLMProvider()
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model == "llama3.2"


def test_get_system_two_provider_factory():
    ds = get_system_two_provider("deepseek")
    assert isinstance(ds, DeepSeekProvider)

    ds_reasoner = get_system_two_provider("deepseek-reasoner")
    assert isinstance(ds_reasoner, DeepSeekProvider)

    custom = get_system_two_provider("custom")
    assert isinstance(custom, CustomLLMProvider)

    local = get_system_two_provider("local")
    assert isinstance(local, CustomLLMProvider)


def test_agent_config_deepseek_fields():
    cfg = AgentConfig(
        deepseek_api_key="sk-deepseek-123",
        deepseek_model="deepseek-reasoner",
        custom_llm_base_url="http://localhost:8080/v1",
    )
    assert cfg.deepseek_api_key == "sk-deepseek-123"
    assert cfg.deepseek_model == "deepseek-reasoner"
    assert cfg.custom_llm_base_url == "http://localhost:8080/v1"
