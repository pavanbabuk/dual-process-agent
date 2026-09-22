# dual_agent/gateway/__init__.py
"""Dual-Process Agent — Omni-Channel Gateway (Hermes-style)."""

from dual_agent.gateway.base import GatewayAdapter, IncomingMessage
from dual_agent.gateway.session_router import SessionRouter

__all__ = ["GatewayAdapter", "IncomingMessage", "SessionRouter"]
