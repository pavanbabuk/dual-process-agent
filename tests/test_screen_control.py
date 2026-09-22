"""Tests for Screen Perception, Diffing, Grid Overlay, and Actuation.

Pillow and pyobjc-framework-Quartz are OPTIONAL extras (`pip install -e '.[screen]'`).
This module must import without them: an unguarded `from PIL import ...` breaks
pytest *collection* for the entire suite, so one missing optional dependency made
all 231 unrelated tests unrunnable. Guard the imports and skip instead.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from dual_agent.screen import has_pillow, has_quartz

if not has_pillow():
    pytest.skip(
        "Pillow is not installed; install the screen extra: pip install -e '.[screen]'",
        allow_module_level=True,
    )

from PIL import Image, ImageDraw  # noqa: E402  (guarded by the skip above)

from dual_agent.screen import (
    get_display_geometry,
    capture_screenshot,
    compute_screen_diff,
    render_grid_overlay,
    click_mouse,
    move_mouse,
    send_key_press,
    is_screen_control_enabled,
    check_accessibility_permission,
)
from dual_agent.mcp_host import MCPHost
from dual_agent.system_two import (
    DeepSeekProvider,
    OpenAICompatibleProvider,
    MockSystemTwoProvider,
    get_vision_provider,
)
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision

# The OpenAI-compatible stub server lives with the vision tests. Imported rather
# than duplicated so one real server definition backs both suites.
from tests.test_vision_provider import StubVisionServer


def test_screen_geometry_returns_valid_scaling():
    geom = get_display_geometry()
    assert geom["logical_width"] > 0
    assert geom["logical_height"] > 0
    assert geom["pixel_width"] > 0
    assert geom["pixel_height"] > 0
    assert geom["scale_x"] >= 1.0
    assert geom["scale_y"] >= 1.0


def test_screenshot_capture_and_metadata():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        res = capture_screenshot(output_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        assert res["path"] == path
        assert res["pixel_width"] > 0
        assert res["pixel_height"] > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_screen_diff_identical_and_modified():
    # Create two test images
    img1 = Image.new("RGB", (400, 300), color=(240, 240, 240))
    img2 = img1.copy()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        p1 = f1.name
        p2 = f2.name

    try:
        img1.save(p1)
        img2.save(p2)

        # Baseline: identical images
        diff_same = compute_screen_diff(p1, p2)
        assert diff_same["changed"] is False
        assert diff_same["diff_fraction"] == 0.0
        assert diff_same["bounding_box"] is None

        # Draw a small 20x20 button click change
        draw = ImageDraw.Draw(img2)
        draw.rectangle([100, 100, 120, 120], fill=(0, 120, 255))
        img2.save(p2)

        diff_mod = compute_screen_diff(p1, p2)
        assert diff_mod["changed"] is True
        assert diff_mod["diff_fraction"] > 0.0
        assert diff_mod["bounding_box"] == [100, 100, 121, 121]
    finally:
        for p in (p1, p2):
            if os.path.exists(p):
                os.unlink(p)


def test_grid_overlay_creates_annotated_file():
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        src_path = f.name
    img.save(src_path)

    try:
        overlay_path = render_grid_overlay(src_path, grid_step=50)
        assert os.path.isfile(overlay_path)
        assert os.path.getsize(overlay_path) > 0
        out_img = Image.open(overlay_path)
        assert out_img.size == (300, 200)
    finally:
        if os.path.exists(src_path):
            os.unlink(src_path)
        if os.path.exists(overlay_path):
            os.unlink(overlay_path)


def test_mouse_out_of_bounds_raises_error():
    geom = get_display_geometry()
    max_w = geom["logical_width"]
    max_h = geom["logical_height"]

    with pytest.raises(ValueError, match="out of bounds"):
        click_mouse(max_w + 500, max_h + 500)

    with pytest.raises(ValueError, match="out of bounds"):
        move_mouse(-10, 100)


def test_key_press_allow_list_validation():
    # Valid key with valid modifier
    # Will either raise PermissionError (if no AX permissions) or succeed
    try:
        send_key_press("a", modifiers=["shift"])
    except PermissionError:
        pass  # expected without accessibility permission

    # Invalid named key must raise ValueError
    with pytest.raises(ValueError, match="Unrecognized key"):
        send_key_press("nonexistent_special_key")

    # Invalid modifier must raise ValueError
    with pytest.raises(ValueError, match="Invalid modifier"):
        send_key_press("return", modifiers=["hyper"])


def test_screen_control_kill_switch(monkeypatch):
    monkeypatch.setenv("DUAL_AGENT_SCREEN_CONTROL", "0")
    assert not is_screen_control_enabled()

    with pytest.raises(PermissionError, match="kill switch"):
        click_mouse(100, 100)

    with pytest.raises(PermissionError, match="kill switch"):
        move_mouse(100, 100)

    with pytest.raises(PermissionError, match="kill switch"):
        send_key_press("return")


def test_mcp_host_screen_tools_registered():
    host = MCPHost()
    tool_names = [t.name for t in host.list_tools()]
    for expected in ("screenshot", "screen_diff", "grid_overlay", "mouse_click", "mouse_move", "key_press"):
        assert expected in tool_names, f"Expected {expected} registered in MCPHost"

    # Verify risk levels and approval requirements
    for act_tool in ("mouse_click", "mouse_move", "key_press"):
        defn = host.get_tool(act_tool)
        assert defn.requires_approval is True
        assert defn.risk_level == "high"

    for read_tool in ("screenshot", "screen_diff", "grid_overlay"):
        defn = host.get_tool(read_tool)
        assert defn.requires_approval is False
        assert defn.risk_level == "low"


def test_deepseek_sends_images_rather_than_assuming_text_only(tmp_path):
    """The old hard-coded text-only refusal is gone.

    It asserted `deepseek-chat` cannot see. Measured 2026-09-22, the configured
    endpoint IS vision-capable (image_url payloads returned HTTP 200, and solid
    red/green frames were named Red/Green), so the refusal discarded images the
    endpoint would have accepted and substituted canned text. This test pins the
    replacement behaviour: the image is transmitted, and the endpoint decides.
    """
    img_path = str(tmp_path / "shot.png")
    Image.new("RGB", (10, 10)).save(img_path)

    with StubVisionServer() as server:
        provider = DeepSeekProvider(api_key="test_key", base_url=server.base_url, model="deepseek-chat")
        resp = provider.generate_step("What is on this screen?", images=[img_path])

        # The image really went out, as a multimodal part.
        assert len(server.captured) == 1
        content = server.captured[0]["body"]["messages"][1]["content"]
        assert isinstance(content, list)
        assert [p["type"] for p in content] == ["text", "image_url"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

        # And the endpoint's answer is reported, not fabricated.
        assert resp.is_mock is False
        assert resp.action == "mouse_click"


def test_deepseek_reports_a_multimodal_rejection_when_the_endpoint_refuses(tmp_path):
    """If an endpoint genuinely cannot see, its refusal is what gets reported."""
    img_path = str(tmp_path / "shot.png")
    Image.new("RGB", (10, 10)).save(img_path)

    with StubVisionServer(
        status=400,
        error_body={"error": {"message": "You have uploaded an unsupported image format"}},
    ) as server:
        provider = DeepSeekProvider(api_key="test_key", base_url=server.base_url, model="deepseek-chat")
        resp = provider.generate_step("What is on this screen?", images=[img_path])
        assert resp.is_mock is True
        assert "400" in resp.degraded_reason
        # Not presented as an observation of the screen.
        assert "unsupported image" in resp.degraded_reason


def test_openai_compatible_multimodal_request():
    provider = OpenAICompatibleProvider(api_key="test_key", base_url="https://mock.api/v1")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    try:
        Image.new("RGB", (10, 10), color="blue").save(img_path)
        with patch("httpx.Client.post") as mock_post:
            mock_res = MagicMock()
            mock_res.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"thought": "Saw blue box", "action": "mouse_click", "args": {"x": 5, "y": 5}}'
                    }
                }],
                "usage": {"total_tokens": 120}
            }
            mock_res.raise_for_status = MagicMock()
            mock_post.return_value = mock_res

            resp = provider.generate_step("Inspect screen", images=[img_path])
            assert resp.action == "mouse_click"
            assert resp.args == {"x": 5, "y": 5}

            # Check that request payload had multimodal image_url
            sent_payload = mock_post.call_args[1]["json"]
            user_msg = sent_payload["messages"][1]["content"]
            assert isinstance(user_msg, list)
            assert any(item.get("type") == "image_url" for item in user_msg)
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


def test_dispatcher_screen_loop_visual_verification():
    """Verify that screen actuation triggers visual verification in dispatcher."""
    host = MCPHost()
    # Mock mouse_click to execute without needing macOS accessibility permission
    host._tools["mouse_click"].handler = lambda args: '{"success": true, "x": 100, "y": 100}'

    s1 = MagicMock()
    s1.evaluate_state_and_route.return_value = JevDecision(
        is_terminal=False,
        selected_tool="mouse_click",
        confidence=0.5,  # escalate to System 2
        latency_ms=5.0,
        needs_generation=True,
    )
    s1.force_simulation = True
    s1.simulation_reason = "unit test"

    from dual_agent.system_two import SystemTwoResponse
    s2 = MagicMock()
    s2.is_mock = False
    s2.generate_step.side_effect = [
        SystemTwoResponse(
            thought="Click button",
            action="mouse_click",
            args={"x": 100, "y": 100},
            generated_content="Clicking button",
            latency_ms=10.0,
            tokens_used=10,
        ),
        SystemTwoResponse(
            thought="Done",
            action="finish_task",
            args={"result": "Done"},
            generated_content="Done",
            latency_ms=5.0,
            tokens_used=5,
        ),
    ]

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=host,
    )

    result = dispatcher.run(goal="Click on the settings button on screen", max_steps=4, enable_screen_loop=True)
    assert result.total_steps >= 1
    # Check that s2 was called with images in the screen loop
    any_images = any(
        "images" in call[1] and call[1]["images"] is not None
        for call in s2.generate_step.call_args_list
    )
    assert any_images
