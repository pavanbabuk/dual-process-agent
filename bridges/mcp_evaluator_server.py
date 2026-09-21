"""Standalone MCP Server exposing TypeSafe AI Jev primitives (Score and Noul).

Can be used by any standard MCP client (Claude Desktop, Cursor, Antigravity, etc.)
to gain ultra-fast (15ms) guardrail validation and rubric scoring.
"""

from __future__ import annotations
import sys
import json
import logging
from typing import Any, Dict

from dual_agent.typesafe_client import JevSystemOneClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_evaluator_server")


def handle_call_tool(name: str, arguments: Dict[str, Any], client: JevSystemOneClient) -> Dict[str, Any]:
    """Execute evaluation tool logic."""
    if name == "jev_evaluate_noul":
        state_text = arguments.get("state", "")
        criteria = arguments.get("criteria", "Is this state valid and error-free?")
        
        # Call Jev to evaluate predicate
        decision = client.evaluate_state_and_route(
            state_text=f"STATE: {state_text}\nCRITERIA: {criteria}",
            tool_options={"valid": "Meets criteria", "invalid": "Violates criteria"},
        )
        prob = decision.probabilities.get("valid", 0.9)
        return {
            "is_met": prob >= 0.5,
            "probability": prob,
            "latency_ms": decision.latency_ms,
        }

    elif name == "jev_score_rubric":
        content = arguments.get("content", "")
        rubric = arguments.get("rubric", "General accuracy and quality")
        score = client.evaluate_output_score(content, rubric)
        return {
            "score": score,
            "max_score": 5,
            "rubric": rubric,
        }

    else:
        raise ValueError(f"Unknown tool: {name}")


def main():
    """Simple JSON-RPC stdio runner for the evaluator server."""
    client = JevSystemOneClient()
    logger.info("Jev MCP Evaluator Server ready on stdio.")
    
    # Tool manifest
    tools_manifest = [
        {
            "name": "jev_evaluate_noul",
            "description": "Evaluate a semantic predicate/boolean criteria on a state in ~15ms using TypeSafe AI Jev.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "The text or state to evaluate"},
                    "criteria": {"type": "string", "description": "The question or assertion to verify"},
                },
                "required": ["state", "criteria"],
            },
        },
        {
            "name": "jev_score_rubric",
            "description": "Grade an output or code sample (1-5) against a rubric in ~15ms using TypeSafe AI Jev.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The text or code to score"},
                    "rubric": {"type": "string", "description": "The scoring criteria or rubric"},
                },
                "required": ["content"],
            },
        },
    ]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")

            if method == "tools/list":
                res = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools_manifest}}
            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                out = handle_call_tool(tool_name, args, client)
                res = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(out)}]},
                }
            elif method == "initialize":
                res = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "typesafe-jev-evaluator", "version": "1.0.0"},
                    },
                }
            else:
                res = {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_res = {"jsonrpc": "2.0", "id": req.get("id") if 'req' in locals() else None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
