"""End-to-end live smoke test verifying real provider interaction without silent fallbacks.

Asserts:
1. Agent terminates under step budget against a network endpoint.
2. A concrete artifact is produced and verified on disk.
3. System 2 does not silently degrade to mock: it explicitly reports whether a mock was used.
4. When a failure occurs, the degradation reason is loudly recorded.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.system_two import CustomLLMProvider
from dual_agent.mcp_host import MCPHost
from dual_agent.permission_broker import PermissionBroker


class StubOpenAIHandler(BaseHTTPRequestHandler):
    """Stub HTTP handler simulating an OpenAI-compatible /chat/completions server."""
    step_count = 0
    target_artifact_path = ""

    def do_POST(self):
        if "/chat/completions" in self.path:
            content_length = int(self.headers.get("Content-Length", 0))
            req_data = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length > 0 else {}
            user_msg = ""
            for msg in req_data.get("messages", []):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")

            if "Planning Engine" in user_msg or "plan_subgoals" in user_msg:
                # Planning step
                msg_content = json.dumps({
                    "thought": "Plan out the subgoals for this task",
                    "action": "plan_subgoals",
                    "args": {
                        "subgoals": [
                            "Create concrete artifact file on disk",
                            "Verify artifact and finish task",
                        ]
                    },
                })
            else:
                StubOpenAIHandler.step_count += 1
                if StubOpenAIHandler.step_count == 1:
                    # Step 1: Instruct agent to write the concrete artifact
                    msg_content = json.dumps({
                        "thought": "Create the required smoke test artifact file.",
                        "action": "write_file",
                        "args": {
                            "path": StubOpenAIHandler.target_artifact_path,
                            "content": "# Smoke Test Concrete Artifact\nSMOKE_TEST_PASSED = True\n",
                        },
                    })
                else:
                    # Step 2: Conclude task
                    msg_content = json.dumps({
                        "thought": "Artifact created and verified. Task completed.",
                        "action": "finish_task",
                        "args": {"result": "Smoke test artifact created successfully."},
                    })

            response_payload = {
                "id": "chatcmpl-stub-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "stub-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": msg_content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "total_tokens": 165,
                },
            }
            body = json.dumps(response_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence HTTP server log output during tests
        pass


@pytest.fixture
def stub_server(tmp_path):
    """Start local stub HTTP server on an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), StubOpenAIHandler)
    port = server.server_address[1]
    StubOpenAIHandler.step_count = 0
    StubOpenAIHandler.target_artifact_path = str(tmp_path / "smoke_artifact.py")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}/v1", StubOpenAIHandler.target_artifact_path

    server.shutdown()
    server.server_close()


def test_live_smoke_end_to_end_execution(stub_server):
    """Assert agent executes end-to-end against network stub without silent fallback."""
    base_url, artifact_path = stub_server

    provider = CustomLLMProvider(
        base_url=base_url,
        model="stub-model",
        api_key="test-key",
        timeout=5.0,
    )

    dispatcher = DualProcessDispatcher(
        system_two_provider=provider,
        mcp_host=MCPHost(),
        permission_broker=PermissionBroker(auto_allow=True),
        confidence_threshold=0.99,  # Force System 2 execution
    )

    max_steps = 6
    result = dispatcher.run(
        goal=f"Create a smoke test artifact at {artifact_path}",
        max_steps=max_steps,
    )

    # 1. Must terminate under budget
    assert result.total_steps <= max_steps
    assert result.is_completed is True

    # 2. Concrete artifact must be produced and verified on disk
    assert os.path.exists(artifact_path), "Smoke test artifact was not written"
    with open(artifact_path, "r") as f:
        content = f.read()
    assert "SMOKE_TEST_PASSED = True" in content

    # 3. Must not silently fall back to mock
    assert result.system_two_is_mock is False, "Agent silently fell back to mock provider"
    assert result.system_two_degraded_reason is None
    assert result.tokens_used > 0, "Real network inference must record consumed tokens"


def test_network_failure_is_reported_loudly_not_silent():
    """Assert that a network failure reports degradation loudly, not silently."""
    # Attempt connection to a closed port
    provider = CustomLLMProvider(
        base_url="http://127.0.0.1:1",
        model="unreachable-model",
        api_key="test-key",
        timeout=0.5,
    )

    dispatcher = DualProcessDispatcher(
        system_two_provider=provider,
        mcp_host=MCPHost(),
        permission_broker=PermissionBroker(auto_allow=True),
        confidence_threshold=0.99,
    )

    result = dispatcher.run(goal="Do work with unreachable server", max_steps=2)

    # Must flag the degradation and name the reason
    assert result.system_two_is_mock is True
    assert result.system_two_degraded_reason is not None
    assert "failed" in result.system_two_degraded_reason.lower() or "connect" in result.system_two_degraded_reason.lower()
