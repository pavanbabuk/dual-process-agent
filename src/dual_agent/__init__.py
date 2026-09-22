"""Dual-Process Agent Runtime combining TypeSafe AI Jev (System 1) with MCP and System 2 LLMs."""

from dual_agent.state import AgentState, StepRecord, StepType
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision
from dual_agent.mcp_host import MCPHost, MCPToolDefinition
from dual_agent.system_two import SystemTwoProvider, get_system_two_provider
from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.evaluator import JevEvaluator
from dual_agent.memory import MemoryEngine, LearnedSkill
from dual_agent.config import AgentConfig, load_config, save_config
from dual_agent.mcp_manager import MCPManager
from dual_agent.shell import InteractiveShell

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "AgentState",
    "StepRecord",
    "StepType",
    "JevSystemOneClient",
    "JevDecision",
    "MCPHost",
    "MCPToolDefinition",
    "SystemTwoProvider",
    "get_system_two_provider",
    "DualProcessDispatcher",
    "DispatchResult",
    "JevEvaluator",
    "MemoryEngine",
    "LearnedSkill",
    "AgentConfig",
    "load_config",
    "save_config",
    "MCPManager",
    "InteractiveShell",
]
