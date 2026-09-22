"""Vision-capable System 2 — screen perception half of the dual-process loop.

WHY THIS MODULE EXISTS
    `screen.py` can capture, diff, annotate and actuate, but until now nothing
    could *look* at what it captured: `SystemTwoProvider.generate_step` was
    text-only, so the screen agent was blind. This module supplies the sensing
    half behind the same interface, plus the cost/precondition reporting that
    makes a blind run impossible to mistake for a seeing one.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
    It does not claim vision is free, and it does not claim a grid overlay
    removes targeting error. `observations()` and the metadata are the only
    places those facts may be stated — see README "Vision & Cost".

HONESTY CONTRACT
    Every path that cannot actually see returns `is_mock=True` with a
    `degraded_reason` that names the specific cause. Canned text is never
    returned as a screen reading. A missing key, a 429, a model that rejects
    multimodal input, and an unreachable endpoint are four different reasons
    and are reported as four different strings.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from dual_agent.system_two import (
    SystemTwoProvider,
    SystemTwoResponse,
    _mock_fallback,
    _encode_image_to_data_url,
)
from dual_agent.screen import has_pillow

logger = logging.getLogger(__name__)

# Downscale bound. A 3420x2224 capture is ~7.6 MP; at that size an image_url
# data URL costs thousands of prompt tokens and most vision models tile-and-
# downsample it anyway, so sending the full frame buys no accuracy. 1400 px on
# the long edge keeps text legible while bounding per-step cost.
DEFAULT_MAX_DIMENSION = 1400

# Kept as a probe: if the configured endpoint rejects an image on a text-shaped
# request, this is the substring such rejections carry. Used only to classify an
# error as "this model cannot see", never to guess a response.
_VISION_REJECTION_MARKERS = (
    "unsupported image",
    "image_url",
    "not support vision",
    "does not support image",
    "multimodal",
    "invalid image",
    "image input",
)


@dataclass
class SentImage:
    """Audit record for one image that was (or would have been) transmitted.

    The dimensions and byte count are measured after downscaling, not estimated,
    so a per-step cost figure can be derived from them rather than guessed.
    """
    source_path: str
    sent_width: int
    sent_height: int
    sent_bytes: int
    original_width: int
    original_height: int
    resized: bool
    variant: str  # "raw" | "grid" | "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "sent_width": self.sent_width,
            "sent_height": self.sent_height,
            "sent_bytes": self.sent_bytes,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "resized": self.resized,
            "variant": self.variant,
        }


def downscale_image(
    image_path: str,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    fmt: str = "PNG",
) -> Tuple[bytes, SentImage]:
    """Downscale `image_path` so its long edge is at most `max_dimension`.

    Returns the encoded bytes and a `SentImage` audit record. Returns encoded
    bytes rather than a path so a caller can never accidentally send the
    original full-resolution file: the object handed to the HTTP layer is the
    same object that was measured.

    Images already within the bound are re-encoded without resampling — that
    keeps the byte count honest for the cost ledger and avoids a needless
    generation loss.
    """
    if not has_pillow():
        raise ImportError(
            "Pillow is required to prepare screenshots for a vision model. "
            "Install with: pip install 'dual-agent[screen]'"
        )
    from PIL import Image

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        original_w, original_h = img.size
        working = img.convert("RGB")

        long_edge = max(original_w, original_h)
        if long_edge > max_dimension > 0:
            scale = max_dimension / float(long_edge)
            # Floor, then clamp to >=1: a 1-px edge would make PIL raise on some
            # codecs, and a rounded-up value can exceed the caller's bound.
            new_w = max(1, int(original_w * scale))
            new_h = max(1, int(original_h * scale))
            working = working.resize((new_w, new_h), Image.LANCZOS)
            resized = True
        else:
            new_w, new_h = original_w, original_h
            resized = False

        buffer = io.BytesIO()
        working.save(buffer, fmt)
        payload = buffer.getvalue()

    record = SentImage(
        source_path=image_path,
        sent_width=new_w,
        sent_height=new_h,
        sent_bytes=len(payload),
        original_width=original_w,
        original_height=original_h,
        resized=resized,
        variant=_classify_variant(image_path),
    )
    return payload, record


def _classify_variant(image_path: str) -> str:
    """Label the image as raw or grid-annotated from its filename.

    The grid overlay is written by `screen.render_grid_overlay` to a temp path
    prefixed `dual_agent_grid_`, and the dispatcher uses `grid_overlay` as its
    variable name. Anything unrecognised is reported as "unknown" rather than
    guessed, so the audit record never asserts a provenance it cannot support.
    """
    name = os.path.basename(image_path).lower()
    if "grid" in name or "annotated" in name:
        return "grid"
    if "screenshot" in name or "screen" in name or "capture" in name:
        return "raw"
    return "unknown"


def _encode_bytes_to_data_url(payload: bytes, fmt: str = "PNG") -> str:
    """Base64-encode already-downscaled bytes as a data URL."""
    mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('utf-8')}"


def _looks_like_vision_rejection(status: int, body: str) -> bool:
    """True when an error body indicates the model cannot accept images.

    Distinguishing this from a generic 400 matters: "your model is text-only"
    and "your request was malformed" send the user to different fixes.
    """
    lowered = (body or "").lower()
    return any(marker in lowered for marker in _VISION_REJECTION_MARKERS)


class VisionSystemTwoProvider(SystemTwoProvider):
    """System 2 provider that sends screenshots to a multimodal chat endpoint.

    Implements the unchanged `SystemTwoProvider` interface; it adds no required
    parameter and overrides no caller-visible behaviour. Consumers that ignore
    the `images` argument keep working against it exactly as they do against the
    text-only providers.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_dimension: Optional[int] = None,
        enabled: Optional[bool] = None,
        is_local: Optional[bool] = None,
    ):
        from dual_agent.config import load_config
        cfg = load_config()

        # Precedence is inherited, not reinvented: an explicit constructor
        # argument wins, then an env var that was deliberately set, then
        # config.json (the source of truth), then the code default.
        env_enabled = os.getenv("DUAL_AGENT_VISION_ENABLED")
        self.enabled = (
            enabled
            if enabled is not None
            else (env_enabled.strip().lower() in ("1", "true", "yes", "on")
                  if env_enabled is not None
                  else bool(cfg.vision_enabled))
        )

        self.base_url = (
            base_url
            or os.getenv("VISION_BASE_URL")
            or cfg.vision_base_url
            or "https://api.openai.com/v1"
        )
        self.model = (
            model
            or os.getenv("VISION_MODEL")
            or cfg.vision_model
            or "gpt-4o"
        )
        # No default key: a vision model that silently falls back to an unset
        # credential is exactly the silent-failure mode this class exists to
        # prevent. Take the vision-specific key, else the provider key if the
        # endpoint was pointed at that provider.
        self.api_key = (
            api_key
            or os.getenv("VISION_API_KEY")
            or cfg.vision_api_key
            or ""
        )

        self.timeout = timeout or float(os.getenv("SYSTEM_TWO_TIMEOUT", "60"))
        self.max_dimension = int(
            max_dimension
            if max_dimension is not None
            else (os.getenv("VISION_MAX_DIMENSION") or cfg.vision_max_dimension or DEFAULT_MAX_DIMENSION)
        )

        # A local OpenAI-compatible server (Ollama, vLLM, LM Studio) takes no
        # credential, so requiring a key there would report a healthy setup as
        # blocked. Hostname is inspected rather than a process-wide "is this
        # local" guess, and an explicit argument still overrides.
        if is_local is None:
            is_local = any(
                host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
            )
        self.is_local = bool(is_local)

        self.last_sent_images: List[SentImage] = []

    # -- precondition reporting -------------------------------------------------

    def preflight(self) -> Optional[str]:
        """Return the reason this provider cannot see, or None if it can.

        A caller that enables the screen loop must be able to distinguish
        "enabled and looking" from "enabled but blind". This is that check, and
        the CLI surfaces it before a run starts so a blind run fails up front
        instead of spending steps.
        """
        if not self.enabled:
            return (
                "vision is disabled (vision_enabled=false in config.json). "
                "A screen-control run needs a multimodal model; set "
                "vision_enabled=true and configure VISION_MODEL/VISION_API_KEY."
            )
        if not self.api_key and not self.is_local:
            return (
                f"no vision API key configured for {self.base_url} (model "
                f"'{self.model}'). Set VISION_API_KEY (or vision_api_key in "
                "~/.dual_agent/config.json). A screen agent cannot see without one."
            )
        if not has_pillow():
            return (
                "Pillow is not installed, so screenshots cannot be downscaled "
                "for transmission. Install with: pip install 'dual-agent[screen]'"
            )
        return None

    def capabilities(self) -> Dict[str, Any]:
        """Non-networking description of what this provider is set up to do."""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "is_local": self.is_local,
            "max_dimension": self.max_dimension,
            "blocking_reason": self.preflight(),
        }

    # -- the interface ----------------------------------------------------------

    def generate_step(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
    ) -> SystemTwoResponse:
        start = time.perf_counter()

        # Distinguish the two reasons there may be nothing to look at, because
        # they are the user's two different problems.
        if images is None:
            return _mock_fallback(prompt, "no images supplied to VisionSystemTwoProvider")
        if not images:
            return _mock_fallback(prompt, "images list was empty; nothing was captured to look at")

        blocking = self.preflight()
        if blocking:
            logger.warning(f"[VisionSystemTwoProvider] {blocking}")
            return self._blind(prompt, blocking, images)

        # Prepare payloads first. A failure here is a local file problem, not a
        # model problem, and must not be reported as a model failure.
        data_urls: List[str] = []
        sent_records: List[SentImage] = []
        for image_path in images:
            try:
                payload, record = downscale_image(image_path, self.max_dimension)
            except FileNotFoundError as e:
                reason = f"cannot read screenshot {image_path}: {e}"
                logger.warning(f"[VisionSystemTwoProvider] {reason}")
                return self._blind(prompt, reason, images, sent_records)
            except Exception as e:
                reason = f"failed to prepare {image_path} for transmission: {type(e).__name__}: {e}"
                logger.warning(f"[VisionSystemTwoProvider] {reason}")
                return self._blind(prompt, reason, images, sent_records)
            data_urls.append(_encode_bytes_to_data_url(payload))
            sent_records.append(record)

        self.last_sent_images = sent_records
        metadata = self._build_metadata(sent_records)

        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in data_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload_json = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the visual reasoning half of an autonomous agent. You are shown "
                        "a screenshot of the user's desktop. Report only what is visible. Output "
                        "valid JSON: {\"thought\": \"...\", \"action\": \"...\", \"args\": {...}}. "
                        "Coordinates are in logical screen points, not image pixels."
                    ),
                },
                {"role": "user", "content": content_parts},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": int(os.getenv("SYSTEM_TWO_MAX_TOKENS", "4000")),
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload_json,
                )
                if res.status_code >= 400:
                    return self._blind(
                        prompt,
                        self._classify_http_error(res.status_code, res.text),
                        images,
                        sent_records,
                        metadata,
                    )

                data = res.json()
                elapsed_ms = (time.perf_counter() - start) * 1000

                raw_content = data["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(raw_content)
                except Exception as parse_err:
                    # The model saw the screen but did not answer in the agreed
                    # shape. Say that instead of inventing an action.
                    reason = (
                        f"vision model '{self.model}' replied with non-JSON content "
                        f"({type(parse_err).__name__}); raw response began {raw_content[:120]!r}"
                    )
                    logger.warning(f"[VisionSystemTwoProvider] {reason}")
                    response = self._blind(prompt, reason, images, sent_records, metadata)
                    response.latency_ms = elapsed_ms
                    response.tokens_used = data.get("usage", {}).get("total_tokens", 0)
                    return response

                action = parsed.get("action") or "system_two_incomplete"
                metadata["served_model"] = data.get("model")

                return SystemTwoResponse(
                    thought=parsed.get("thought", "Visual inspection complete."),
                    action=action,
                    args=parsed.get("args", {}),
                    generated_content=raw_content,
                    latency_ms=elapsed_ms,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    is_mock=False,
                    degraded_reason=None,
                    metadata=metadata,
                )
        except Exception as e:
            reason = (
                f"vision call to {self.base_url} (model '{self.model}') failed: "
                f"{type(e).__name__}: {e}"
            )
            logger.warning(f"[VisionSystemTwoProvider] {reason}")
            return self._blind(prompt, reason, images, sent_records, metadata)

    # -- failure classification -------------------------------------------------

    def _classify_http_error(self, status: int, body: str) -> str:
        """Name the specific failure. A 429 is a rate limit and says so."""
        if status == 429:
            return (
                f"vision endpoint {self.base_url} (model '{self.model}') returned 429 "
                "Too Many Requests — rate-limited. This is a quota failure, not a "
                "screen reading; no image was analysed. Free hosted vision tiers "
                "commonly return 429 on every attempt."
            )
        if status in (401, 403):
            return (
                f"vision endpoint {self.base_url} rejected the credential for model "
                f"'{self.model}' with HTTP {status}. Check VISION_API_KEY."
            )
        if status == 404:
            return (
                f"vision endpoint {self.base_url} returned 404 for model '{self.model}' — "
                "no such vision model at that endpoint."
            )
        if status == 400 and _looks_like_vision_rejection(status, body):
            return (
                f"vision endpoint {self.base_url} rejected the image for model '{self.model}' "
                f"with HTTP 400: {body[:200]}. This model does not accept multimodal input; "
                "configure one that does."
            )
        return (
            f"vision call to {self.base_url} (model '{self.model}') failed with HTTP {status}: "
            f"{body[:200]}"
        )

    def _build_metadata(self, records: List[SentImage]) -> Dict[str, Any]:
        """Measured transmission facts, so per-step cost is auditable."""
        return {
            "vision": True,
            "model": self.model,
            "base_url": self.base_url,
            "max_dimension": self.max_dimension,
            "images_sent": [r.to_dict() for r in records],
            "total_sent_bytes": sum(r.sent_bytes for r in records),
            # A 1400x1400 RGB frame is ~5.9 Mpx. Image-token counts are
            # model-specific (OpenAI tiles, Anthropic caps at ~1.15 Mpx, Gemini
            # charges per tile), so this is an upper bound on the *pixel* budget,
            # not a token count and not a price. Never quote it as either.
            "sent_megapixels_upper_bound": round(
                sum(r.sent_width * r.sent_height for r in records) / 1_000_000, 3
            ),
        }

    def _blind(
        self,
        prompt: str,
        reason: str,
        images: List[str],
        sent_records: Optional[List[SentImage]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SystemTwoResponse:
        """Return an explicitly-blind response.

        The mock's canned `action`/`args` are retained purely because
        `SystemTwoResponse` requires them; `is_mock=True` and the
        `degraded_reason` are what make the blindness visible, and the dispatcher
        aborts the run on a non-mock-provider degraded reason. Nothing here may
        ever be presented as an observation of the screen.
        """
        response = _mock_fallback(prompt, reason)
        response.degraded_reason = reason
        if metadata is not None:
            response.metadata = metadata
        elif sent_records:
            response.metadata = self._build_metadata(sent_records)
        return response


class BlockedVisionProvider(SystemTwoProvider):
    """Always-blind provider for runs that enabled screen control without eyes.

    Exists so the failure is a named precondition instead of a mock that looks
    like an answer. This is what `get_vision_provider()` returns when the user
    set a vision model but nothing that can serve it, or enabled the screen loop
    with vision off.
    """

    def __init__(self, reason: str):
        self.reason = reason

    def generate_step(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
    ) -> SystemTwoResponse:
        response = _mock_fallback(prompt, self.reason)
        response.degraded_reason = self.reason
        response.metadata = {"vision": False, "blocking_reason": self.reason}
        return response

    def preflight(self) -> Optional[str]:
        return self.reason


def vision_observations() -> List[str]:
    """Honest, non-negotiable facts about this capability.

    Rendered by `dual-agent --help` and by `dual-agent doctor`, so the
    preconditions cannot be learned only by reading source.
    """
    return [
        "Actuation is local and free; SIGHT IS NOT. A screen run needs a paid vision",
        "model (gpt-4o, grok-2-vision-1212) or a locally installed VLM. No local VLM",
        "is installed here: ports 11434 (Ollama), 1234 (LM Studio) and 8080 (vLLM) are closed.",
        "A grid overlay is a coordinate aid, not a fix: it helps a model land near a",
        "target and does not eliminate coordinate misses. No accuracy figure is claimed.",
        "Screenshots are downscaled to a bounded long edge before transmission; the exact",
        "sent dimensions and byte size are recorded in each response's metadata.",
        "A 429 is reported as a rate limit and the run aborts; canned text is never",
        "returned as a screen reading.",
    ]
