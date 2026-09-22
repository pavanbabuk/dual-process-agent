"""Screen perception, diffing, grid overlay, and actuation primitives.

Provides tools for the agent to inspect the visual desktop and actuate mouse
and keyboard events via macOS Quartz APIs, with safety bounds, kill-switch,
and retina scaling calibration.
"""

from __future__ import annotations
import os
import sys
import time
import tempfile
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Allow-list of named keys for key_press tool
NAMED_KEY_CODES: Dict[str, int] = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "backspace": 51,
    "delete": 51,
    "escape": 53,
    "esc": 53,
    "command": 55,
    "shift": 56,
    "capslock": 57,
    "option": 58,
    "control": 59,
    "right_shift": 60,
    "right_option": 61,
    "right_control": 62,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
}

VALID_MODIFIERS = {"shift", "control", "option", "command"}


def is_screen_control_enabled() -> bool:
    """Return False if DUAL_AGENT_SCREEN_CONTROL is explicitly '0' or 'false'."""
    val = os.environ.get("DUAL_AGENT_SCREEN_CONTROL", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def has_pillow() -> bool:
    """Check if Pillow is installed."""
    try:
        import PIL.Image  # noqa: F401
        return True
    except ImportError:
        return False


def has_quartz() -> bool:
    """Check if Quartz is available."""
    try:
        import Quartz  # noqa: F401
        return True
    except ImportError:
        return False


def check_accessibility_permission() -> bool:
    """Check whether the current process has macOS Accessibility permission."""
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        app_services = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        return bool(app_services.AXIsProcessTrusted())
    except Exception as e:
        logger.warning(f"Failed to check macOS AXIsProcessTrusted: {e}")
        return False


def get_display_geometry() -> Dict[str, float]:
    """Return main display logical bounds and retina scale factor.

    Returns:
        dict with keys: logical_width, logical_height, pixel_width,
        pixel_height, scale_x, scale_y
    """
    if not has_quartz():
        raise ImportError(
            "pyobjc-framework-Quartz is required for display geometry on macOS. "
            "Install with: pip install 'dual-agent[screen]'"
        )

    import Quartz
    main_id = Quartz.CGMainDisplayID()
    bounds = Quartz.CGDisplayBounds(main_id)
    logical_w = float(bounds.size.width)
    logical_h = float(bounds.size.height)

    # Inspect actual pixel resolution via CGDisplayCreateImage
    cg_img = Quartz.CGDisplayCreateImage(main_id)
    if cg_img is not None:
        pixel_w = float(Quartz.CGImageGetWidth(cg_img))
        pixel_h = float(Quartz.CGImageGetHeight(cg_img))
    else:
        pixel_w = logical_w
        pixel_h = logical_h

    scale_x = pixel_w / logical_w if logical_w > 0 else 1.0
    scale_y = pixel_h / logical_h if logical_h > 0 else 1.0

    return {
        "logical_width": logical_w,
        "logical_height": logical_h,
        "pixel_width": pixel_w,
        "pixel_height": pixel_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def capture_screenshot(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Capture the current screen and save to output_path.

    Returns metadata dictionary including dimensions, retina scaling, and path.
    """
    if not has_pillow():
        raise ImportError(
            "Pillow is required for screenshot capture. "
            "Install with: pip install 'dual-agent[screen]'"
        )

    from PIL import Image

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="dual_agent_screen_", suffix=".png")
        os.close(fd)

    geom = get_display_geometry()

    # Capture via Quartz CGDisplayCreateImage
    import Quartz
    main_id = Quartz.CGMainDisplayID()
    cg_img = Quartz.CGDisplayCreateImage(main_id)

    if cg_img is None:
        # Fallback to screencapture utility
        import subprocess
        res = subprocess.run(["screencapture", "-x", output_path], capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to capture screen: screencapture exited with code {res.returncode}. "
                "Screen Recording permission may be required in System Settings -> Privacy & Security -> Screen Recording."
            )
        img = Image.open(output_path)
    else:
        width = Quartz.CGImageGetWidth(cg_img)
        height = Quartz.CGImageGetHeight(cg_img)
        provider = Quartz.CGImageGetDataProvider(cg_img)
        data = Quartz.CGDataProviderCopyData(provider)
        bpr = Quartz.CGImageGetBytesPerRow(cg_img)
        img = Image.frombytes("RGBA", (width, height), bytes(data), "raw", "BGRA", bpr, 1)
        img.save(output_path, "PNG")

    return {
        "path": output_path,
        "logical_width": geom["logical_width"],
        "logical_height": geom["logical_height"],
        "pixel_width": img.size[0],
        "pixel_height": img.size[1],
        "scale_x": geom["scale_x"],
        "scale_y": geom["scale_y"],
    }


def compute_screen_diff(
    image_path_1: str,
    image_path_2: str,
    threshold: float = 0.002,
) -> Dict[str, Any]:
    """Compare two screenshot images and report whether the screen changed.

    Args:
        image_path_1: Path to baseline screenshot.
        image_path_2: Path to subsequent screenshot.
        threshold: Fraction of differing pixels required to flag a change (default 0.2%).

    Returns:
        Dict with 'changed' (bool), 'diff_fraction' (float), 'diff_percentage' (str),
        and 'bounding_box' ([min_x, min_y, max_x, max_y] or None).
    """
    if not has_pillow():
        raise ImportError(
            "Pillow is required for screen_diff. "
            "Install with: pip install 'dual-agent[screen]'"
        )

    from PIL import Image, ImageChops

    if not os.path.isfile(image_path_1):
        raise FileNotFoundError(f"Baseline image not found: {image_path_1}")
    if not os.path.isfile(image_path_2):
        raise FileNotFoundError(f"Comparison image not found: {image_path_2}")

    img1 = Image.open(image_path_1).convert("RGB")
    img2 = Image.open(image_path_2).convert("RGB")

    if img1.size != img2.size:
        # Resize comparison image to match baseline if necessary
        img2 = img2.resize(img1.size)

    diff = ImageChops.difference(img1, img2)
    bbox = diff.getbbox()

    if not bbox:
        return {
            "changed": False,
            "diff_fraction": 0.0,
            "diff_percentage": "0.00%",
            "bounding_box": None,
            "details": "Screenshots are identical.",
        }

    # Count pixels exceeding difference threshold
    # Fast path: greyscale difference histogram
    diff_grey = diff.convert("L")
    hist = diff_grey.histogram()
    # hist[0] is count of pixels with 0 difference; values > 10 indicate perceptible diff
    total_pixels = img1.size[0] * img1.size[1]
    diff_pixels = sum(hist[12:])  # filter out subtle sensor/clock noise
    diff_fraction = round(diff_pixels / total_pixels, 5)
    changed = diff_fraction >= threshold or (bbox and diff_pixels > 100)

    return {
        "changed": changed,
        "diff_fraction": diff_fraction,
        "diff_percentage": f"{diff_fraction * 100:.2f}%",
        "bounding_box": list(bbox),
        "details": f"Screen changed: {diff_fraction * 100:.2f}% pixels differ within region {bbox}."
        if changed
        else "No significant visual change detected above threshold.",
    }


def render_grid_overlay(
    image_path: str,
    output_path: Optional[str] = None,
    grid_step: int = 150,
) -> str:
    """Render a labeled coordinate grid overlay over a screenshot.

    Draws grid lines and (x, y) coordinates to help vision models pinpoint
    exact UI coordinates accurately.
    """
    if not has_pillow():
        raise ImportError(
            "Pillow is required for grid_overlay. "
            "Install with: pip install 'dual-agent[screen]'"
        )

    from PIL import Image, ImageDraw, ImageFont

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="dual_agent_grid_", suffix=".png")
        os.close(fd)

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    grid_color = (255, 0, 0, 90)
    label_color = (255, 255, 0, 220)
    bg_color = (0, 0, 0, 160)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw vertical and horizontal grid lines
    for x in range(0, w, grid_step):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, grid_step):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Draw coordinate labels at cell intersections
    for x in range(0, w, grid_step):
        for y in range(0, h, grid_step):
            text = f"{x},{y}"
            draw.rectangle([x + 2, y + 2, x + 55, y + 16], fill=bg_color)
            draw.text((x + 4, y + 3), text, fill=label_color, font=font)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(output_path, "PNG")
    return output_path


def click_mouse(
    x: float,
    y: float,
    button: str = "left",
    click_type: str = "single",
) -> Dict[str, Any]:
    """Actuate a mouse click at logical coordinates (x, y).

    Args:
        x: Horizontal coordinate in logical screen points.
        y: Vertical coordinate in logical screen points.
        button: "left" or "right".
        click_type: "single" or "double".
    """
    if not is_screen_control_enabled():
        raise PermissionError(
            "Screen actuation is disabled via DUAL_AGENT_SCREEN_CONTROL=0 kill switch."
        )

    geom = get_display_geometry()
    max_w = geom["logical_width"]
    max_h = geom["logical_height"]

    if x < 0 or x > max_w or y < 0 or y > max_h:
        raise ValueError(
            f"Coordinates ({x}, {y}) out of bounds. Display bounds are [0..{max_w}, 0..{max_h}]."
        )

    if not check_accessibility_permission():
        raise PermissionError(
            "macOS Accessibility permission required to actuate mouse events. "
            "Grant permission to the application in System Settings -> Privacy & Security -> Accessibility."
        )

    import Quartz

    btn = Quartz.kCGMouseButtonRight if button == "right" else Quartz.kCGMouseButtonLeft
    down_type = Quartz.kCGEventRightMouseDown if button == "right" else Quartz.kCGEventLeftMouseDown
    up_type = Quartz.kCGEventRightMouseUp if button == "right" else Quartz.kCGEventLeftMouseUp

    pt = Quartz.CGPoint(x, y)

    # Move cursor first
    move_ev = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, pt, btn)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_ev)
    time.sleep(0.02)

    click_count = 2 if click_type == "double" else 1

    for c in range(1, click_count + 1):
        down_ev = Quartz.CGEventCreateMouseEvent(None, down_type, pt, btn)
        Quartz.CGEventSetIntegerValueField(down_ev, Quartz.kCGMouseEventClickState, c)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_ev)
        time.sleep(0.05)

        up_ev = Quartz.CGEventCreateMouseEvent(None, up_type, pt, btn)
        Quartz.CGEventSetIntegerValueField(up_ev, Quartz.kCGMouseEventClickState, c)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_ev)
        time.sleep(0.05)

    return {
        "success": True,
        "x": x,
        "y": y,
        "button": button,
        "click_type": click_type,
    }


def move_mouse(x: float, y: float) -> Dict[str, Any]:
    """Move cursor to logical coordinates (x, y)."""
    if not is_screen_control_enabled():
        raise PermissionError(
            "Screen actuation is disabled via DUAL_AGENT_SCREEN_CONTROL=0 kill switch."
        )

    geom = get_display_geometry()
    max_w = geom["logical_width"]
    max_h = geom["logical_height"]

    if x < 0 or x > max_w or y < 0 or y > max_h:
        raise ValueError(
            f"Coordinates ({x}, {y}) out of bounds. Display bounds are [0..{max_w}, 0..{max_h}]."
        )

    if not check_accessibility_permission():
        raise PermissionError(
            "macOS Accessibility permission required to actuate mouse events. "
            "Grant permission to the application in System Settings -> Privacy & Security -> Accessibility."
        )

    import Quartz
    pt = Quartz.CGPoint(x, y)
    move_ev = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventMouseMoved, pt, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_ev)

    return {"success": True, "x": x, "y": y}


def send_key_press(key: str, modifiers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Actuate keyboard keystroke with optional modifier keys.

    Args:
        key: Named key (e.g. 'return', 'tab', 'escape', 'left') or single character ('a', '1', etc.)
        modifiers: List of modifiers: 'shift', 'control', 'option', 'command'
    """
    if not is_screen_control_enabled():
        raise PermissionError(
            "Screen actuation is disabled via DUAL_AGENT_SCREEN_CONTROL=0 kill switch."
        )

    # Validate key and modifiers
    key_lower = key.lower()
    if key_lower not in NAMED_KEY_CODES and len(key) != 1:
        raise ValueError(
            f"Unrecognized key '{key}'. Allowed named keys: {sorted(NAMED_KEY_CODES.keys())} "
            "or a single character."
        )

    if modifiers:
        for mod in modifiers:
            if mod.lower() not in VALID_MODIFIERS:
                raise ValueError(
                    f"Invalid modifier '{mod}'. Allowed modifiers: {sorted(VALID_MODIFIERS)}"
                )

    if not check_accessibility_permission():
        raise PermissionError(
            "macOS Accessibility permission required to actuate keystrokes. "
            "Grant permission to the application in System Settings -> Privacy & Security -> Accessibility."
        )

    # Bind Quartz in THIS scope. The module previously relied on the import inside
    # has_quartz(), which is a different function's local name — so every call that
    # passed a modifier raised NameError: name 'Quartz' is not defined, i.e. any
    # shifted keystroke crashed. Import here, after the permission check, so the
    # dependency error and the permission error stay distinguishable.
    try:
        import Quartz
    except ImportError as e:
        raise RuntimeError(
            "pyobjc-framework-Quartz is required for key actuation on macOS. "
            "Install with: pip install 'dual-agent[screen]'"
        ) from e

    flags = 0
    if modifiers:
        for mod in modifiers:
            m = mod.lower()
            if m == "shift":
                flags |= Quartz.kCGEventFlagMaskShift
            elif m == "control":
                flags |= Quartz.kCGEventFlagMaskControl
            elif m == "option":
                flags |= Quartz.kCGEventFlagMaskAlternate
            elif m == "command":
                flags |= Quartz.kCGEventFlagMaskCommand

    if key_lower in NAMED_KEY_CODES:
        keycode = NAMED_KEY_CODES[key_lower]
        down_ev = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
        up_ev = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
        if flags:
            Quartz.CGEventSetFlags(down_ev, flags)
            Quartz.CGEventSetFlags(up_ev, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_ev)
        time.sleep(0.04)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_ev)
    elif len(key) == 1:
        # Single Unicode character
        down_ev = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        up_ev = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
        if flags:
            Quartz.CGEventSetFlags(down_ev, flags)
            Quartz.CGEventSetFlags(up_ev, flags)
        Quartz.CGEventKeyboardSetUnicodeString(down_ev, len(key), key)
        Quartz.CGEventKeyboardSetUnicodeString(up_ev, len(key), key)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_ev)
        time.sleep(0.04)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_ev)
    else:
        raise ValueError(
            f"Unrecognized key '{key}'. Allowed named keys: {sorted(NAMED_KEY_CODES.keys())} "
            "or a single character."
        )

    return {
        "success": True,
        "key": key,
        "modifiers": modifiers or [],
    }
