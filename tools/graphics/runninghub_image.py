"""RunningHub image generation via workflow-based async API.

Covers 5 workflows:
  - duanju          (2052744677727715329) 短剧专用图片模型, custom WxH
  - zimage_portrait (2003681895185563650) Z-image 超真实感定妆照
  - qwen_image_t2i  (1970396677775499266) Qwen-image 文生图
  - qwen_image_edit (2029488621429989377) Qwen Image Edit 2511 图生图
  - zimage_8k       (2058719340626796546) Z-Image 在线 8K 直出
"""

from __future__ import annotations

import os
import random
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
    poll_runninghub,
    probe_output,
    runninghub_resolve_media,
    runninghub_submit,
)


_QWEN_T2I_RATIO_MAP = {
    "3:2": "1", "2:3": "2", "16:9": "3", "9:16": "4",
    "4:3": "5", "3:4": "6", "1:1": "7",
}

_QWEN_LORA_NAME = "国潮面部插画qwen触发词gc.safetensors"


class RunninghubImage(BaseTool):
    name = "runninghub_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "runninghub"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:RUNNINGHUB_API_KEY"]
    install_instructions = (
        "Set RUNNINGHUB_API_KEY to your RunningHub API key.\n"
        "  Get one from https://www.runninghub.cn"
    )
    agent_skills = ["runninghub"]

    WORKFLOWS = {
        "duanju":          {"id": "2052744677727715329", "modes": ["text"]},
        "zimage_portrait": {"id": "2003681895185563650", "modes": ["text"]},
        "qwen_image_t2i":  {"id": "1970396677775499266", "modes": ["text"]},
        "qwen_image_edit": {"id": "2029488621429989377", "modes": ["singleImage"]},
        "zimage_8k":       {"id": "2058719340626796546", "modes": ["text"]},
    }

    capabilities = ["generate_image", "text_to_image", "image_to_image"]
    supports = {
        "text_to_image": True,
        "image_to_image": True,
        "negative_prompt": True,
        "seed": True,
        "custom_size": True,
        "aspect_ratio": True,
    }
    best_for = [
        "Chinese短剧 / AI portrait styling via Z-image workflows",
        "Qwen-image text-to-image with multi-aspect-ratio + negative prompts",
        "image editing via Qwen Image Edit (single reference image)",
        "8K photorealistic stills via Z-Image online",
    ]
    not_good_for = ["non-Chinese-prompt-optimized workflows", "offline generation"]
    fallback_tools = ["flux_image", "google_imagen", "openai_image", "recraft_image"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "model"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "enum": list(WORKFLOWS.keys()),
                "default": "zimage_portrait",
            },
            "negative_prompt": {"type": "string", "default": ""},
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4"],
                "default": "1:1",
            },
            "size": {
                "type": "string",
                "enum": ["1K", "2K", "4K"],
                "default": "1K",
                "description": "Size tier for duanju model (1K=96x, 2K=128x, 4K=256x base on aspect ratio)",
            },
            "seed": {"type": "integer"},
            "image_path": {"type": "string", "description": "Local reference image for qwen_image_edit"},
            "image_url": {"type": "string", "description": "Remote reference image URL for qwen_image_edit"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "aspect_ratio", "size", "seed"]
    side_effects = ["writes image file to output_path", "calls RunningHub API"]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def _get_api_key(self) -> str | None:
        key = os.environ.get("RUNNINGHUB_API_KEY")
        if not key:
            return None
        return re.sub(r"^Bearer\s+", "", key, flags=re.IGNORECASE)

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # RunningHub pricing not exposed in source — return 0.0
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="RUNNINGHUB_API_KEY not set. " + self.install_instructions,
            )

        start = time.time()
        model_alias = inputs.get("model", "zimage_portrait")
        if model_alias not in self.WORKFLOWS:
            return ToolResult(
                success=False,
                error=f"Unknown model: {model_alias}. Available: {', '.join(self.WORKFLOWS.keys())}",
            )
        workflow_id = self.WORKFLOWS[model_alias]["id"]
        prompt = inputs["prompt"]

        try:
            node_info_list = self._build_image_nodes(model_alias, inputs, api_key)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to build workflow nodes: {e}")

        try:
            task_id = runninghub_submit(workflow_id, node_info_list, api_key)
            _url, image_bytes = poll_runninghub(task_id, api_key)
        except Exception as e:
            return ToolResult(success=False, error=f"RunningHub image generation failed: {e}")

        output_path = Path(inputs.get("output_path", f"runninghub_{model_alias}.png"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "runninghub",
                "model": workflow_id,
                "model_alias": model_alias,
                "prompt": prompt,
                "output": str(output_path),
                "output_path": str(output_path),
                "format": output_path.suffix.lstrip(".") or "png",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=workflow_id,
        )

    def _build_image_nodes(
        self,
        model_alias: str,
        inputs: dict[str, Any],
        api_key: str,
    ) -> list[dict[str, Any]]:
        """Build the nodeInfoList for the given workflow. Mirrors TS imageRequest()."""
        prompt = inputs["prompt"]

        if model_alias == "duanju":
            width, height = self._compute_duanju_size(
                inputs.get("aspect_ratio", "1:1"),
                inputs.get("size", "1K"),
            )
            return [
                {"nodeId": "49", "fieldName": "text", "fieldValue": prompt, "description": "提示词"},
                {"nodeId": "60", "fieldName": "value", "fieldValue": str(width), "description": "宽"},
                {"nodeId": "61", "fieldName": "value", "fieldValue": str(height), "description": "高"},
            ]

        if model_alias == "qwen_image_t2i":
            ratio = inputs.get("aspect_ratio", "1:1")
            ratio_select = _QWEN_T2I_RATIO_MAP.get(ratio, "7")
            seed = inputs.get("seed") or random.randint(0, 10**15 - 1)
            negative = inputs.get("negative_prompt", "")
            # 25-field schema from TS lines 362-394. LoRA fields are pre-zeroed.
            return [
                {"nodeId": "932", "fieldName": "prompt", "fieldValue": prompt, "description": "正向提示词"},
                {"nodeId": "931", "fieldName": "text", "fieldValue": negative, "description": "反向提示词"},
                {"nodeId": "887", "fieldName": "select", "fieldValue": ratio_select, "description": "设置比例"},
                {"nodeId": "889", "fieldName": "batch_size", "fieldValue": "1", "description": "出图张数"},
                {"nodeId": "925", "fieldName": "lora_name", "fieldValue": _QWEN_LORA_NAME, "description": "lora1_name"},
                {"nodeId": "925", "fieldName": "strength_model", "fieldValue": "0", "description": "lora1_strength"},
                {"nodeId": "925", "fieldName": "strength_clip", "fieldValue": "0", "description": "lora1_strength_clip"},
                {"nodeId": "933", "fieldName": "text1", "fieldValue": "", "description": "text1-lora触发词"},
                {"nodeId": "926", "fieldName": "lora_name", "fieldValue": _QWEN_LORA_NAME, "description": "lora2_name"},
                {"nodeId": "926", "fieldName": "strength_model", "fieldValue": "0", "description": "lora2_strength"},
                {"nodeId": "926", "fieldName": "strength_clip", "fieldValue": "0", "description": "lora2_strength_clip"},
                {"nodeId": "933", "fieldName": "text2", "fieldValue": "", "description": "text2-lora触发词"},
                {"nodeId": "927", "fieldName": "lora_name", "fieldValue": _QWEN_LORA_NAME, "description": "lora3_name"},
                {"nodeId": "927", "fieldName": "strength_model", "fieldValue": "0", "description": "lora3_strength"},
                {"nodeId": "927", "fieldName": "strength_clip", "fieldValue": "0", "description": "lora3_strength_clip"},
                {"nodeId": "933", "fieldName": "text3", "fieldValue": "", "description": "text3-lora触发词"},
                {"nodeId": "928", "fieldName": "lora_name", "fieldValue": _QWEN_LORA_NAME, "description": "lora4_name"},
                {"nodeId": "928", "fieldName": "strength_model", "fieldValue": "0", "description": "lora4_strength"},
                {"nodeId": "928", "fieldName": "strength_clip", "fieldValue": "0", "description": "lora4_strength_clip"},
                {"nodeId": "933", "fieldName": "text4", "fieldValue": "", "description": "text4-lora触发词"},
                {"nodeId": "929", "fieldName": "lora_name", "fieldValue": _QWEN_LORA_NAME, "description": "lora5_name"},
                {"nodeId": "929", "fieldName": "strength_model", "fieldValue": "0", "description": "lora5_strength"},
                {"nodeId": "929", "fieldName": "strength_clip", "fieldValue": "0", "description": "lora5_strength_clip"},
                {"nodeId": "933", "fieldName": "text5", "fieldValue": "", "description": "text5-lora触发词"},
                {"nodeId": "860", "fieldName": "seed", "fieldValue": str(seed), "description": "种子"},
                {"nodeId": "860", "fieldName": "steps", "fieldValue": "4", "description": "步数"},
                {"nodeId": "860", "fieldName": "sampler_name", "fieldValue": "euler", "description": "采样器"},
                {"nodeId": "860", "fieldName": "scheduler", "fieldValue": "simple", "description": "调度器"},
                {"nodeId": "938", "fieldName": "unet_name", "fieldValue": "qwen_image_fp8_e4m3fn.safetensors", "description": "unet_name"},
            ]

        if model_alias == "qwen_image_edit":
            image_value = inputs.get("image_url") or inputs.get("image_path")
            if not image_value:
                raise ValueError("qwen_image_edit requires image_url or image_path")
            image_url = runninghub_resolve_media(image_value, api_key)
            return [
                {"nodeId": "41", "fieldName": "image", "fieldValue": image_url, "description": "image"},
                {"nodeId": "68", "fieldName": "prompt", "fieldValue": prompt, "description": "prompt"},
            ]

        if model_alias == "zimage_8k":
            return [
                {"nodeId": "163", "fieldName": "text", "fieldValue": prompt, "description": "text"},
            ]

        # zimage_portrait: value only
        return [
            {"nodeId": "59", "fieldName": "value", "fieldValue": prompt, "description": "提示词"},
        ]

    @staticmethod
    def _compute_duanju_size(aspect_ratio: str, size_tier: str) -> tuple[int, int]:
        """Compute width/height for duanju model from aspect_ratio + size tier.

        Mirrors TS parseAspectRatio + size calc (lines 322-347):
          factor = 96 (1K) | 128 (2K) | 256 (4K)
          width = ratio_w * factor, height = ratio_h * factor
          clamp to [512, 4096], round to multiple of 8
        """
        ratio_str = aspect_ratio.replace("/", ":")
        parts = ratio_str.split(":")
        if len(parts) != 2:
            parts = ["1", "1"]
        try:
            rw, rh = int(parts[0]), int(parts[1])
        except ValueError:
            rw, rh = 1, 1
        factor = {"1K": 96, "2K": 128, "4K": 256}.get(size_tier, 96)
        width = max(512, min(4096, round(rw * factor / 8) * 8))
        height = max(512, min(4096, round(rh * factor / 8) * 8))
        return width, height
