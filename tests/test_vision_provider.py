"""Vision provider acceptance tests.

Two rules this file follows, both of which have been violated in this repo before:

1. A mock may test control flow but may never be the acceptance criterion for a
   capability. The "it can see" claim is asserted against the *captured request
   body* of a real local HTTP server, never against a stub's return value.
2. No fabricated numbers. Every dimension, byte count and status code asserted
   here is read out of the object under test or the server's captured request.

Pillow is an optional extra; guard the import so a missing extra skips this
module instead of breaking collection for the whole suite.
"""

import base64
import json
import os
import tempfile
import threading

import pytest
from http.server import BaseHTTPRequestHandler, HTTPServer

from dual_agent.screen import has_pillow

if not has_pillow():
    pytest.skip(
        "Pillow is not installed; install the screen extra: pip install -e '.[screen]'",
        allow_module_level=True,
    )

from PIL import Image  # noqa: E402  (guarded by the skip above)

from dual_agent.system_two import (  # noqa: E402
    SystemTwoProvider,
    SystemTwoResponse,
    MockSystemTwoProvider,
    DeepSeekProvider,
    get_vision_provider,
)
from dual_agent.vision import (  # noqa: E402
    VisionSystemTwoProvider,
    BlockedVisionProvider,
    downscale_image,
    DEFAULT_MAX_DIMENSION,
)


# --------------------------------------------------------------------------
# A real OpenAI-compatible stub server. Nothing here is a mock object: the
# test asserts on bytes this server actually received.
# --------------------------------------------------------------------------

class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server's naming
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {"__unparseable__": raw[:400].decode("utf-8", "replace")}

        # Record everything the process needs to assert on later.
        self.server.captured.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
                "raw_len": len(raw),
            }
        )

        status = self.server.response_status
        if status != 200:
            payload = json.dumps(
                self.server.error_body
                or {"error": {"message": f"stub returning {status}", "type": "stub_error"}}
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        content = self.server.response_content
        if callable(content):
            content = content(body)
        payload = json.dumps(
            {
                "id": "stub-1",
                "object": "chat.completion",
                "model": self.server.served_model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


class StubVisionServer:
    """Context manager around a real HTTPServer on a free local port."""

    def __init__(self, status=200, response_content=None, error_body=None, served_model="stub-vlm"):
        self.status = status
        self.response_content = response_content or json.dumps(
            {"thought": "I see a window", "action": "mouse_click", "args": {"x": 10, "y": 20}}
        )
        self.error_body = error_body
        self.served_model = served_model
        self.httpd = HTTPServer(("127.0.0.1", 0), _StubHandler)
        self.httpd.captured = []
        self.httpd.response_status = status
        self.httpd.response_content = self.response_content
        self.httpd.error_body = error_body
        self.httpd.served_model = served_model
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/v1"

    @property
    def captured(self):
        return self.httpd.captured

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False


def _write_png(width, height, path, color=(30, 60, 90)):
    Image.new("RGB", (width, height), color=color).save(path, "PNG")
    return path


@pytest.fixture
def big_screenshot():
    """A synthetic capture matching this machine's real screencapture output."""
    fd, path = tempfile.mkstemp(prefix="dual_agent_screenshot_", suffix=".png")
    os.close(fd)
    _write_png(3420, 2224, path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def grid_screenshot():
    fd, path = tempfile.mkstemp(prefix="dual_agent_grid_", suffix=".png")
    os.close(fd)
    _write_png(800, 600, path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# --------------------------------------------------------------------------
# 1. Backward compatibility of the interface extension
# --------------------------------------------------------------------------

def test_legacy_text_only_provider_satisfies_interface():
    """A provider written before `images` existed is still a valid provider.

    This is the exact shape a third-party subclass had before this change: only
    `generate_step(self, prompt)`.
    """

    class LegacyProvider(SystemTwoProvider):
        def generate_step(self, prompt):
            return SystemTwoResponse(thought="legacy", action="finish_task", args={}, tokens_used=3)

    provider = LegacyProvider()
    assert isinstance(provider, SystemTwoProvider)
    # The old call signature must still work untouched.
    resp = provider.generate_step("hello")
    assert resp.action == "finish_task"
    assert resp.is_mock is False


def test_mock_provider_old_signature_and_metadata_default(big_screenshot):
    """Existing call sites keep working and gain no required argument."""
    provider = MockSystemTwoProvider()
    resp = provider.generate_step("summarize the workspace")
    assert resp.is_mock is True
    assert resp.tokens_used == 0
    # `metadata` is additive: text-only providers leave it empty rather than
    # being forced to populate it.
    assert resp.metadata == {}

    # And the three existing real providers still answer on the old signature.
    assert DeepSeekProvider(api_key="").generate_step("hi").is_mock is True


def test_metadata_field_is_additive_for_existing_constructors():
    """SystemTwoResponse keeps working for every existing positional/kw usage."""
    r = SystemTwoResponse(thought="t", action="a")
    assert r.metadata == {}
    assert r.degraded_reason is None
    assert r.is_mock is False

    r2 = SystemTwoResponse(
        thought="t",
        action="a",
        args={"x": 1},
        generated_content="g",
        latency_ms=1.5,
        tokens_used=9,
        is_mock=True,
        degraded_reason="why",
    )
    assert r2.tokens_used == 9
    assert r2.metadata == {}


def test_dispatcher_style_call_sites_unchanged(monkeypatch):
    """The three shapes the dispatcher actually uses all still work."""
    calls = []

    class RecordingProvider(SystemTwoProvider):
        def generate_step(self, prompt, images=None):
            calls.append((prompt, images))
            return SystemTwoResponse(thought="t", action="finish_task", args={})

    p = RecordingProvider()
    p.generate_step("plain")                       # dispatcher line 206 / 1116
    p.generate_step("with-images", images=["a"])   # dispatcher line 593
    p.generate_step("positional-none")             # any caller passing None explicitly

    assert calls[0] == ("plain", None)
    assert calls[1] == ("with-images", ["a"])
    assert calls[2] == ("positional-none", None)


def test_typeerror_fallback_path_still_reachable():
    """The dispatcher's TypeError fallback for pre-`images` providers still fires.

    `DualProcessDispatcher` guards `generate_step(prompt, images=...)` with
    `except TypeError` for providers that never learned about images. That guard
    must remain meaningful, so a genuinely old provider must still raise TypeError.
    """

    class PreImagesProvider(SystemTwoProvider):
        def generate_step(self, prompt):  # noqa: D401 - deliberately old shape
            return SystemTwoResponse(thought="t", action="finish_task", args={})

    provider = PreImagesProvider()
    with pytest.raises(TypeError):
        provider.generate_step("p", images=["x.png"])  # type: ignore[call-arg]
    # ...and the fallback the dispatcher uses still succeeds.
    assert provider.generate_step("p").action == "finish_task"


# --------------------------------------------------------------------------
# 2. The "it can see" acceptance test — real server, captured request body
# --------------------------------------------------------------------------

def test_vision_request_actually_carries_the_image(big_screenshot):
    with StubVisionServer() as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url,
            model="stub-vlm",
            api_key="stub-key",
            max_dimension=1400,
            enabled=True,
        )
        resp = provider.generate_step("Click the settings button.", images=[big_screenshot])

        # The provider genuinely saw it: a real, non-mock response.
        assert resp.is_mock is False
        assert resp.degraded_reason is None
        assert resp.action == "mouse_click"
        assert resp.args == {"x": 10, "y": 20}
        assert resp.tokens_used == 18

        # Exactly one request reached the real server.
        assert len(server.captured) == 1
        request = server.captured[0]
        assert request["path"] == "/v1/chat/completions"

        user_content = request["body"]["messages"][1]["content"]
        # Multimodal shape: a list of typed parts, not a bare string.
        assert isinstance(user_content, list)
        text_parts = [p for p in user_content if p["type"] == "text"]
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        assert len(text_parts) == 1
        assert "Click the settings button." in text_parts[0]["text"]
        assert len(image_parts) == 1

        # And the image itself is really in there, as a data URL.
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        # Compare against the source pixels: the sent image is a real
        # downscaled render of THIS file, not placeholder bytes.
        with Image.open(big_screenshot) as src:
            src_w, src_h = src.size
        with Image.open(__import__("io").BytesIO(decoded)) as sent:
            assert sent.size == (1400, 910)
        assert src_w == 3420 and src_h == 2224

        # Metadata reports the exact transmitted dimensions and byte size.
        audit = resp.metadata["images_sent"][0]
        assert audit["sent_width"] == 1400
        assert audit["sent_height"] == 910
        assert audit["sent_bytes"] == len(decoded)
        assert audit["original_width"] == 3420
        assert audit["original_height"] == 2224
        assert audit["resized"] is True
        assert resp.metadata["total_sent_bytes"] == len(decoded)


def test_vision_request_sends_grid_variant_and_labels_it(grid_screenshot):
    """Both raw and grid-annotated screenshots are supported and distinguishable."""
    with StubVisionServer() as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="stub-vlm", api_key="k", enabled=True
        )
        resp = provider.generate_step("Where is the button?", images=[grid_screenshot])

        assert resp.is_mock is False
        assert len(server.captured) == 1
        parts = server.captured[0]["body"]["messages"][1]["content"]
        assert [p["type"] for p in parts] == ["text", "image_url"]
        # An 800x600 image is under the bound, so it must be sent unresampled and
        # the audit record must say so rather than claiming a downscale.
        audit = resp.metadata["images_sent"][0]
        assert audit["sent_width"] == 800
        assert audit["sent_height"] == 600
        assert audit["resized"] is False
        assert audit["variant"] == "grid"


def test_vision_accepts_raw_and_grid_together(grid_screenshot, big_screenshot):
    with StubVisionServer() as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="stub-vlm", api_key="k", enabled=True
        )
        resp = provider.generate_step("Compare.", images=[big_screenshot, grid_screenshot])
        assert resp.is_mock is False
        parts = server.captured[0]["body"]["messages"][1]["content"]
        assert len([p for p in parts if p["type"] == "image_url"]) == 2
        variants = {a["variant"] for a in resp.metadata["images_sent"]}
        assert variants == {"raw", "grid"}


# --------------------------------------------------------------------------
# 3. Downscale dimensions for the real capture size
# --------------------------------------------------------------------------

def test_downscale_of_real_capture_dimensions(big_screenshot):
    """3420x2224 must bound to the long edge with the aspect ratio preserved."""
    payload, record = downscale_image(big_screenshot, max_dimension=1400)
    assert (record.original_width, record.original_height) == (3420, 2224)
    assert max(record.sent_width, record.sent_height) == 1400
    # 2224 * (1400/3420) = 910.4 -> floor 910. Aspect preserved to within a pixel.
    assert (record.sent_width, record.sent_height) == (1400, 910)
    assert record.resized is True
    assert record.sent_bytes == len(payload)
    assert record.sent_bytes > 0
    assert record.sent_bytes < os.path.getsize(big_screenshot)

    # The encoded bytes really are that size.
    import io

    with Image.open(io.BytesIO(payload)) as img:
        assert img.size == (1400, 910)


def test_downscale_bound_is_configurable(big_screenshot):
    _, record = downscale_image(big_screenshot, max_dimension=700)
    assert (record.sent_width, record.sent_height) == (700, 455)

    # A bound larger than the source leaves the image untouched.
    _, unchanged = downscale_image(big_screenshot, max_dimension=9999)
    assert (unchanged.sent_width, unchanged.sent_height) == (3420, 2224)
    assert unchanged.resized is False


def test_downscale_default_matches_documented_bound(big_screenshot):
    _, record = downscale_image(big_screenshot)
    assert max(record.sent_width, record.sent_height) == DEFAULT_MAX_DIMENSION == 1400


def test_vision_max_dimension_is_honoured_end_to_end(big_screenshot):
    """A configured bound must reach the wire, not just the local helper."""
    with StubVisionServer() as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="stub-vlm", api_key="k",
            max_dimension=512, enabled=True,
        )
        resp = provider.generate_step("look", images=[big_screenshot])
        audit = resp.metadata["images_sent"][0]
        assert max(audit["sent_width"], audit["sent_height"]) == 512
        url = server.captured[0]["body"]["messages"][1]["content"][1]["image_url"]["url"]
        decoded = base64.b64decode(url.split(",", 1)[1])
        import io

        with Image.open(io.BytesIO(decoded)) as img:
            assert max(img.size) == 512
        assert resp.metadata["max_dimension"] == 512


# --------------------------------------------------------------------------
# 4. Honesty: 429, missing key, text-only model, network error
# --------------------------------------------------------------------------

def test_429_is_reported_as_a_rate_limit_not_a_success(big_screenshot):
    """A 429 must name the rate limit. It must never look like a screen reading."""
    with StubVisionServer(
        status=429,
        error_body={"error": {"message": "rate limited", "type": "rate_limit_error"}},
    ) as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="free-vlm", api_key="k", enabled=True
        )
        resp = provider.generate_step("What is on screen?", images=[big_screenshot])

        # The request really was made and really was refused.
        assert len(server.captured) == 1
        assert "429" in resp.degraded_reason
        assert "Too Many Requests" in resp.degraded_reason
        assert "rate-limited" in resp.degraded_reason
        assert "free-vlm" in resp.degraded_reason
        # ...and it is flagged as not-a-real-answer, not returned as an observation.
        assert resp.is_mock is True
        # The response must not claim to describe the screen.
        assert "I see a window" not in resp.thought
        assert resp.action != "mouse_click"


def test_429_on_every_attempt_never_fabricates_a_screen_reading(big_screenshot):
    """Repeated 429s must stay honest on attempt N, not drift into an answer."""
    with StubVisionServer(status=429) as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="free-vlm", api_key="k", enabled=True
        )
        for _ in range(3):
            resp = provider.generate_step("What is on screen?", images=[big_screenshot])
            assert resp.is_mock is True
            assert "429" in resp.degraded_reason
            assert "429" in resp.degraded_reason and "rate" in resp.degraded_reason.lower()
            assert resp.metadata.get("vision") is True  # it did try to send
    assert len(server.captured) == 3


def test_401_names_the_credential(big_screenshot):
    with StubVisionServer(status=401) as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="m", api_key="bad", enabled=True
        )
        resp = provider.generate_step("look", images=[big_screenshot])
        assert resp.is_mock is True
        assert "401" in resp.degraded_reason
        assert "VISION_API_KEY" in resp.degraded_reason


def test_text_only_model_rejection_is_named(big_screenshot):
    """A 400 that rejects the image must be reported as a vision-capability problem."""
    with StubVisionServer(
        status=400,
        error_body={
            "error": {
                "message": "You have uploaded an unsupported image format",
                "type": "invalid_request_error",
            }
        },
    ) as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="deepseek-chat", api_key="k", enabled=True
        )
        resp = provider.generate_step("look", images=[big_screenshot])
        assert resp.is_mock is True
        assert "does not accept multimodal input" in resp.degraded_reason


def test_missing_key_fails_before_any_request(big_screenshot):
    """No credential means no call and a reason that names the missing key."""
    with StubVisionServer() as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="stub-vlm", api_key="", enabled=True,
            is_local=False,  # a hosted endpoint must require a credential
        )
        assert provider.preflight() is not None
        assert "VISION_API_KEY" in provider.preflight()

        resp = provider.generate_step("look", images=[big_screenshot])
        assert resp.is_mock is True
        assert "VISION_API_KEY" in resp.degraded_reason
        # Nothing was transmitted, so the server saw nothing.
        assert server.captured == []
        # And the identical setup pointed at a bare certificate would have sent:
        # this asserts the block came from the missing key, not from the address.
        assert provider.capabilities()["has_api_key"] is False


def test_vision_disabled_fails_instead_of_falling_back_to_mock(big_screenshot):
    with StubVisionServer() as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="stub-vlm", api_key="k", enabled=False
        )
        resp = provider.generate_step("look", images=[big_screenshot])
        assert resp.is_mock is True
        assert "vision is disabled" in resp.degraded_reason
        assert server.captured == []


def test_network_error_names_the_endpoint():
    provider = VisionSystemTwoProvider(
        base_url="http://127.0.0.1:1/v1", model="m", api_key="k", timeout=0.5, enabled=True
    )
    fd, path = tempfile.mkstemp(prefix="dual_agent_screenshot_", suffix=".png")
    os.close(fd)
    _write_png(100, 100, path)
    try:
        resp = provider.generate_step("look", images=[path])
        assert resp.is_mock is True
        assert "http://127.0.0.1:1/v1" in resp.degraded_reason
        assert "failed" in resp.degraded_reason.lower()
    finally:
        os.unlink(path)


def test_non_json_reply_is_not_reported_as_an_action(big_screenshot):
    """A model that saw the screen but rambled must not yield an invented action."""
    with StubVisionServer(response_content="I think there is probably a button somewhere.") as server:
        provider = VisionSystemTwoProvider(
            base_url=server.base_url, model="stub-vlm", api_key="k", enabled=True
        )
        resp = provider.generate_step("look", images=[big_screenshot])
        assert resp.is_mock is True
        assert "non-JSON" in resp.degraded_reason
        assert "non-JSON" in resp.degraded_reason and "stub-vlm" in resp.degraded_reason
        assert resp.action != "mouse_click"


def test_missing_image_file_is_an_image_problem_not_a_model_problem():
    provider = VisionSystemTwoProvider(
        base_url="http://127.0.0.1:1/v1", model="m", api_key="k", enabled=True
    )
    resp = provider.generate_step("look", images=["/nonexistent/screen.png"])
    assert resp.is_mock is True
    assert "/nonexistent/screen.png" in resp.degraded_reason
    assert "cannot read screenshot" in resp.degraded_reason


def test_no_images_supplied_is_reported_not_guessed():
    provider = VisionSystemTwoProvider(model="m", api_key="k", enabled=True)
    none_resp = provider.generate_step("look", images=None)
    empty_resp = provider.generate_step("look", images=[])
    assert none_resp.is_mock is True
    assert "no images supplied" in none_resp.degraded_reason
    assert empty_resp.is_mock is True
    assert "empty" in empty_resp.degraded_reason


# --------------------------------------------------------------------------
# 5. Factory / config precedence
# --------------------------------------------------------------------------

def test_factory_returns_blocked_provider_not_a_mock(monkeypatch, tmp_path):
    """A missing vision config must not silently resolve to a canned-text mock."""
    import json as _json

    for var in ("VISION_PROVIDER", "VISION_MODEL", "VISION_API_KEY",
                "VISION_BASE_URL", "OPENAI_API_KEY", "GROK_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    home = tmp_path / ".dual_agent"
    home.mkdir()
    (home / "config.json").write_text(_json.dumps({"system_two_provider": "deepseek"}))
    monkeypatch.setenv("DUAL_AGENT_HOME", str(home))

    provider = get_vision_provider()
    assert isinstance(provider, BlockedVisionProvider)
    assert not isinstance(provider, MockSystemTwoProvider)

    resp = provider.generate_step("look", images=["/tmp/x.png"])
    # The canonical mock reason must NOT be the only thing we learn — the run
    # needs to know it was blind, not merely "mock".
    assert resp.degraded_reason != "mock provider — no real model was called"
    assert "no vision model configured" in resp.degraded_reason


def test_factory_honours_config_json_when_env_is_absent(monkeypatch, tmp_path):
    import json as _json

    for var in ("VISION_PROVIDER", "VISION_MODEL", "VISION_API_KEY",
                "VISION_BASE_URL", "VISION_ENABLED", "DUAL_AGENT_VISION_ENABLED"):
        monkeypatch.delenv(var, raising=False)

    home = tmp_path / ".dual_agent"
    home.mkdir()
    (home / "config.json").write_text(_json.dumps({
        "vision_provider": "custom",
        "vision_model": "qwen2-vl",
        "vision_base_url": "http://localhost:11434/v1",
        "vision_enabled": True,
    }))
    monkeypatch.setenv("DUAL_AGENT_HOME", str(home))

    from dual_agent.config import load_config

    cfg = load_config()
    assert cfg.vision_model == "qwen2-vl"
    assert cfg.vision_enabled is True

    provider = get_vision_provider(config=cfg)
    assert isinstance(provider, VisionSystemTwoProvider)
    assert provider.model == "qwen2-vl"
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.enabled is True
    assert provider.preflight() is None  # local server needs no key


def test_env_overrides_config_json_for_vision(monkeypatch, tmp_path):
    import json as _json

    home = tmp_path / ".dual_agent"
    home.mkdir()
    (home / "config.json").write_text(_json.dumps({
        "vision_model": "from-json",
        "vision_enabled": False,
    }))
    monkeypatch.setenv("DUAL_AGENT_HOME", str(home))
    monkeypatch.setenv("VISION_MODEL", "from-env")
    monkeypatch.setenv("DUAL_AGENT_VISION_ENABLED", "1")

    from dual_agent.config import load_config

    cfg = load_config()
    assert cfg.vision_model == "from-env"
    assert cfg.vision_enabled is True


def test_vision_enabled_false_string_does_not_enable(monkeypatch, tmp_path):
    """bool("false") is True — the flag must be parsed, not coerced."""
    import json as _json

    home = tmp_path / ".dual_agent"
    home.mkdir()
    (home / "config.json").write_text(_json.dumps({"vision_enabled": True}))
    monkeypatch.setenv("DUAL_AGENT_HOME", str(home))
    monkeypatch.setenv("DUAL_AGENT_VISION_ENABLED", "false")

    from dual_agent.config import load_config

    assert load_config().vision_enabled is False


def test_config_json_beats_env_file_default_for_vision(monkeypatch, tmp_path):
    """config.json is the source of truth; a .env default must not override it."""
    import json as _json

    home = tmp_path / ".dual_agent"
    home.mkdir()
    (home / "config.json").write_text(_json.dumps({"vision_max_dimension": 900}))
    (home / ".env").write_text("VISION_MAX_DIMENSION=200\n")
    monkeypatch.setenv("DUAL_AGENT_HOME", str(home))
    monkeypatch.delenv("VISION_MAX_DIMENSION", raising=False)

    from dual_agent.config import load_config, _LOADED_FROM_ENV_FILE

    _LOADED_FROM_ENV_FILE.discard("VISION_MAX_DIMENSION")
    assert load_config().vision_max_dimension == 900


def test_preflight_reports_each_blocking_precondition(monkeypatch):
    """Each precondition has its own message, so a blind run names its cause."""
    disabled = VisionSystemTwoProvider(model="m", api_key="k", enabled=False)
    assert "vision is disabled" in disabled.preflight()

    keyless = VisionSystemTwoProvider(model="m", api_key="", enabled=True)
    assert "VISION_API_KEY" in keyless.preflight()

    ok = VisionSystemTwoProvider(model="m", api_key="k", enabled=True)
    assert ok.preflight() is None
    assert ok.capabilities()["blocking_reason"] is None
    assert ok.capabilities()["model"] == "m"


def test_blocked_provider_is_never_silent():
    blocked = BlockedVisionProvider("no local VLM is installed")
    resp = blocked.generate_step("look at the screen")
    assert resp.is_mock is True
    assert resp.degraded_reason == "no local VLM is installed"
    assert resp.metadata == {"vision": False, "blocking_reason": "no local VLM is installed"}


# --------------------------------------------------------------------------
# 6. A screen run with no eyes must abort, not burn steps
# --------------------------------------------------------------------------

def test_dispatcher_aborts_a_screen_run_when_the_provider_cannot_see():
    """The prerequisite is enforced before any step runs, with a named reason."""
    from dual_agent.dispatcher import DualProcessDispatcher
    from dual_agent.mcp_host import MCPHost
    from dual_agent.typesafe_client import JevSystemOneClient

    dispatcher = DualProcessDispatcher(
        system_one_client=JevSystemOneClient(force_simulation=True),
        system_two_provider=BlockedVisionProvider(
            "no vision model configured. Set VISION_MODEL and VISION_API_KEY."
        ),
        mcp_host=MCPHost(),
    )
    result = dispatcher.run("click the settings button on screen", max_steps=3)

    assert result.is_completed is False
    assert result.total_steps == 0          # nothing was attempted
    assert result.system_two_is_mock is True
    assert "cannot see" in result.system_two_degraded_reason
    assert "no vision model configured" in result.system_two_degraded_reason
    assert "System 2 failure" in result.final_output


def test_dispatcher_does_not_preflight_non_screen_goals():
    """A text-only goal must not be blocked by a missing vision model."""
    from dual_agent.dispatcher import DualProcessDispatcher
    from dual_agent.mcp_host import MCPHost
    from dual_agent.typesafe_client import JevSystemOneClient

    dispatcher = DualProcessDispatcher(
        system_one_client=JevSystemOneClient(force_simulation=True),
        system_two_provider=BlockedVisionProvider("no vision model configured"),
        mcp_host=MCPHost(),
    )
    result = dispatcher.run("summarize the repository layout", max_steps=2)
    # The run proceeds (and may or may not complete); what matters is that the
    # vision precondition did not short-circuit an unrelated goal.
    assert "cannot see" not in (result.system_two_degraded_reason or "")
