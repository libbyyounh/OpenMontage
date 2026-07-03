"""Grsai image generation via the /v1/api/generate endpoint (国内节点).

Exposes 13 models through one tool:
  nano-banana series (11):
    nano-banana, nano-banana-fast, nano-banana-2, nano-banana-2-cl,
    nano-banana-2-2k-cl, nano-banana-2-4k-cl, nano-banana-pro,
    nano-banana-pro-vt, nano-banana-pro-cl, nano-banana-pro-vip,
    nano-banana-pro-4k-vip
  gpt-image-2 series (2):
    gpt-image-2, gpt-image-2-vip

nano-banana series accepts imageSize (1K/2K/4K) + aspectRatio (ratio).
gpt-image-2 series accepts aspectRatio only (ratio or pixel value like "1024x1024").
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)
from tools.video._shared import (
    grsai_generate,
    poll_grsai,
    probe_output,
)


_NANO_BANANA_MODELS = [
    "nano-banana",
    "nano-banana-fast",
    "nano-banana-2",
    "nano-banana-2-cl",
    "nano-banana-2-2k-cl",
    "nano-banana-2-4k-cl",
    "nano-banana-pro",
    "nano-banana-pro-vt",
    "nano-banana-pro-cl",
    "nano-banana-pro-vip",
    "nano-banana-pro-4k-vip",
]

_GPT_IMAGE_MODELS = [
    "gpt-image-2",
    "gpt-image-2-vip",
]

_ALL_MODELS = _NANO_BANANA_MODELS + _GPT_IMAGE_MODELS


def _file_to_data_uri(path_str: str) -> str:
    """Read a local file and return a base64 data URI."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path_str}")
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class GrsaiImage(BaseTool):
    name = "grsai_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "grsai"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:GRSAI_API_KEY"]
    install_instructions = (
        "Set GRSAI_API_KEY to your Grsai API key.\n"
        "  Get one from https://grsai.ai/zh/dashboard/api-keys"
    )
    agent_skills = ["grsai"]

    capabilities = ["generate_image", "text_to_image", "image_to_image"]
    supports = {
        "text_to_image": True,
        "image_to_image": True,        # via images field
        "multiple_reference_images": True,
        "aspect_ratio": True,
        "custom_size": True,           # nano-banana imageSize + gpt-image-2-vip pixel values
        "seed": False,
    }
    best_for = [
        "nano-banana-2 — Google Gemini Image family, strong prompt adherence",
        "nano-banana-pro / pro-vip — higher quality nano-banana variants",
        "nano-banana-pro-4k-vip — 4K photorealistic output",
        "gpt-image-2 / gpt-image-2-vip — GPT Image family with pixel-precise sizing",
        "reference-image conditioned generation (up to N images, base64 or URL)",
    ]
    not_good_for = ["offline generation", "seed-controlled reproducibility"]
    fallback_tools = ["flux_image", "google_imagen", "openai_image", "recraft_image", "runninghub_image"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "model"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "enum": _ALL_MODELS,
                "default": "nano-banana-2",
            },
            "aspect_ratio": {
                "type": "string",
                "default": "1:1",
                "description": (
                    "Aspect ratio (e.g. '1:1', '16:9', '9:16', '4:3', '3:4', '3:2', "
                    "'2:3', '5:4', '4:5', '21:9', 'auto') or pixel value "
                    "'WxH' for gpt-image-2-vip (e.g. '1024x1024', '2048x1152'). "
                    "nano-banana-2 series also accepts '1:4', '4:1', '1:8', '8:1'."
                ),
            },
            "image_size": {
                "type": "string",
                "enum": ["1K", "2K", "4K"],
                "default": "1K",
                "description": "Resolution tier for nano-banana series only. Ignored by gpt-image-2 models.",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Reference images (base64 data URI or HTTP URL).",
            },
            "image_path": {
                "type": "string",
                "description": "Local reference image path (auto-encoded to base64 data URI).",
            },
            "image_url": {
                "type": "string",
                "description": "Remote reference image URL.",
            },
            "reply_type": {
                "type": "string",
                "enum": ["json", "async"],
                "default": "json",
                "description": "json=synchronous (wait up to 300s), async=return task_id immediately and poll /v1/api/result.",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "aspect_ratio", "image_size", "reply_type"]
    side_effects = ["writes image file to output_path", "calls Grsai API"]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def _get_api_key(self) -> str | None:
        key = os.environ.get("GRSAI_API_KEY")
        if not key:
            return None
        return re.sub(r"^Bearer\s+", "", key, flags=re.IGNORECASE)

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Grsai pricing not exposed — return 0.0
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model", "nano-banana-2")
        if "4k" in model:
            return 180.0
        if "pro" in model or "vip" in model:
            return 90.0
        return 45.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="GRSAI_API_KEY not set. " + self.install_instructions,
            )

        start = time.time()
        model = inputs.get("model", "nano-banana-2")
        if model not in _ALL_MODELS:
            return ToolResult(
                success=False,
                error=f"Unknown model: {model}. Available: {', '.join(_ALL_MODELS)}",
            )
        prompt = inputs["prompt"]

        try:
            payload = self._build_payload(model, inputs)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to build payload: {e}")

        try:
            response = grsai_generate(payload, api_key)
        except Exception as e:
            return ToolResult(success=False, error=f"Grsai generate request failed: {e}")

        reply_type = inputs.get("reply_type", "json")
        try:
            if reply_type == "async":
                task_id = response.get("id")
                if not task_id:
                    return ToolResult(
                        success=False,
                        error=f"Grsai async submission missing task id: {response}",
                    )
                _url, image_bytes = poll_grsai(task_id, api_key)
            else:
                # json mode: response should already contain results[0].url
                status = response.get("status")
                if status != "succeeded":
                    err = response.get("error") or response.get("status") or "unknown"
                    return ToolResult(
                        success=False,
                        error=f"Grsai task did not succeed (status={status}): {err}",
                    )
                results = response.get("results") or []
                if not results or not results[0].get("url"):
                    return ToolResult(
                        success=False,
                        error=f"Grsai response missing result URL: {response}",
                    )
                import requests
                dl = requests.get(results[0]["url"], timeout=120)
                dl.raise_for_status()
                image_bytes = dl.content
        except Exception as e:
            return ToolResult(success=False, error=f"Grsai image generation failed: {e}")

        output_path = Path(inputs.get("output_path", f"grsai_{model.replace('/', '_')}.png"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "grsai",
                "model": model,
                "prompt": prompt,
                "aspect_ratio": inputs.get("aspect_ratio", "1:1"),
                "image_size": inputs.get("image_size") if model in _NANO_BANANA_MODELS else None,
                "output": str(output_path),
                "output_path": str(output_path),
                "format": output_path.suffix.lstrip(".") or "png",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )

    def _build_payload(self, model: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build the /v1/api/generate request payload.

        nano-banana series: includes imageSize + aspectRatio.
        gpt-image-2 series: includes aspectRatio only (no imageSize).
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": inputs["prompt"],
            "replyType": inputs.get("reply_type", "json"),
        }

        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        payload["aspectRatio"] = aspect_ratio

        if model in _NANO_BANANA_MODELS:
            payload["imageSize"] = inputs.get("image_size", "1K")

        images: list[str] = list(inputs.get("images") or [])
        if inputs.get("image_url"):
            images.append(inputs["image_url"])
        if inputs.get("image_path"):
            images.append(_file_to_data_uri(inputs["image_path"]))
        if images:
            payload["images"] = images

        return payload
