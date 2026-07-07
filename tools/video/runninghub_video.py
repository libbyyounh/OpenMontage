"""RunningHub video generation via workflow-based async API.

All 4 workflows are image-to-video (no text-to-video):
  - wan22             (1956699246381469698) WAN2.2 官方加速, 480P, 5s
  - ltx23             (2029759632314474498) LTX2.3 图生视频, 720P, 5s/10s
  - ltx23_long        (2055155307592077313) LTX2.3 多镜头分段, 720P, 10s
  - ltx23_four_frames (2054820963426021378) LTX2.3 四帧丝滑流转, 720P, 5s,
                                              uses /run/workflow/ endpoint
"""

from __future__ import annotations

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
    poll_runninghub,
    probe_output,
    runninghub_resolve_media,
    runninghub_submit,
)


class RunninghubVideo(BaseTool):
    name = "runninghub_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
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
        "wan22":             {"id": "1956699246381469698", "endpoint": "ai-app",   "durations": [5],     "resolutions": ["480P"]},
        "ltx23":             {"id": "2029759632314474498", "endpoint": "ai-app",   "durations": [5, 10], "resolutions": ["720P"]},
        "ltx23_long":        {"id": "2055155307592077313", "endpoint": "ai-app",   "durations": [10],    "resolutions": ["720P"]},
        "ltx23_four_frames": {"id": "2054820963426021378", "endpoint": "workflow", "durations": [5],     "resolutions": ["720P"]},
    }

    capabilities = ["image_to_video", "reference_to_video"]
    supports = {
        "image_to_video": True,
        "reference_to_video": True,
        "multiple_reference_images": True,  # four_frames accepts up to 4
        "aspect_ratio": True,
        "seed": False,                      # RunningHub workflows don't expose seed
    }
    best_for = [
        "WAN2.2 image-to-video at 480P (5s, official accelerated channel)",
        "LTX2.3 image-to-video at 720P (5s or 10s)",
        "LTX2.3 multi-shot long video with time-segmented prompt control (10s)",
        "LTX2.3 four-frame silky flow with up to 4 reference images (5s, 720P)",
    ]
    not_good_for = ["text-to-video (all 4 workflows require images)", "offline generation"]
    fallback_tools = ["seedance_video", "veo_video", "kling_video", "minimax_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "model"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "enum": list(WORKFLOWS.keys()),
                "default": "ltx23",
            },
            "image_url": {"type": "string", "description": "Primary reference image URL"},
            "image_path": {"type": "string", "description": "Primary reference image local path (auto-uploaded)"},
            "reference_image_urls": {
                "type": "array", "items": {"type": "string"},
                "description": "For ltx23_four_frames: up to 4 reference image URLs",
            },
            "reference_image_paths": {
                "type": "array", "items": {"type": "string"},
                "description": "For ltx23_four_frames: up to 4 local reference image paths (auto-uploaded)",
            },
            "duration": {"type": "integer", "enum": [5, 10], "default": 5},
            "resolution": {"type": "string", "enum": ["480P", "720P"], "default": "720P"},
            "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16"], "default": "16:9"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "duration", "resolution", "aspect_ratio"]
    side_effects = ["writes video file to output_path", "calls RunningHub API"]
    user_visible_verification = ["Watch generated clip for motion coherence and visual quality"]

    def _get_api_key(self) -> str | None:
        key = os.environ.get("RUNNINGHUB_API_KEY")
        if not key:
            return None
        return re.sub(r"^Bearer\s+", "", key, flags=re.IGNORECASE)

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # RunningHub pricing not exposed in source — return 0.0
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        model_alias = inputs.get("model", "ltx23")
        # 480P is roughly 2x faster than 720P
        if model_alias == "wan22":
            return 120.0
        return 300.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="RUNNINGHUB_API_KEY not set. " + self.install_instructions,
            )

        start = time.time()
        model_alias = inputs.get("model", "ltx23")
        if model_alias not in self.WORKFLOWS:
            return ToolResult(
                success=False,
                error=f"Unknown model: {model_alias}. Available: {', '.join(self.WORKFLOWS.keys())}",
            )
        workflow = self.WORKFLOWS[model_alias]
        workflow_id = workflow["id"]

        duration = inputs.get("duration", 5)
        if duration not in workflow["durations"]:
            return ToolResult(
                success=False,
                error=f"duration={duration} not supported by {model_alias}. Allowed: {workflow['durations']}",
            )
        resolution = inputs.get("resolution", "720P")
        if resolution not in workflow["resolutions"]:
            return ToolResult(
                success=False,
                error=f"resolution={resolution} not supported by {model_alias}. Allowed: {workflow['resolutions']}",
            )

        prompt = inputs.get("prompt", "")

        try:
            node_info_list, use_workflow_endpoint = self._build_video_nodes(
                model_alias, inputs, api_key, prompt, duration, resolution,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to build workflow nodes: {e}")

        # Retry submit/poll on transient errors (queue limit / 5xx).
        # RunningHub concurrency=1 — server-side queue can transiently look full
        # while a prior task drains, so real backoff between retries matters.
        TRANSIENT_TASK_MSG_MARKERS = (
            "queue limit reached",
            "503",
            "502",
            "504",
            "Service Unavailable",
        )
        SUBMIT_RETRY_BACKOFF = (5, 10, 20, 40, 60)
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                task_id = runninghub_submit(
                    workflow_id, node_info_list, api_key,
                    use_workflow_endpoint=use_workflow_endpoint,
                )
                _url, video_bytes = poll_runninghub(task_id, api_key)
                break
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if any(m in err_str for m in TRANSIENT_TASK_MSG_MARKERS) and attempt < 4:
                    time.sleep(SUBMIT_RETRY_BACKOFF[attempt])
                    continue
                return ToolResult(
                    success=False,
                    error=f"RunningHub video generation failed: {e}",
                )
        else:
            return ToolResult(
                success=False,
                error=f"RunningHub video generation failed after 5 attempts: {last_exc}",
            )

        output_path = Path(inputs.get("output_path", f"runninghub_{model_alias}.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "runninghub",
                "model": workflow_id,
                "model_alias": model_alias,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": inputs.get("aspect_ratio", "16:9"),
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=workflow_id,
        )

    def _build_video_nodes(
        self,
        model_alias: str,
        inputs: dict[str, Any],
        api_key: str,
        prompt: str,
        duration: int,
        resolution: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Build nodeInfoList for the given video workflow.

        Returns (node_info_list, use_workflow_endpoint). Mirrors TS videoRequest().
        """
        aspect = inputs.get("aspect_ratio", "16:9")

        if model_alias == "wan22":
            # 1956699246381469698: fixed 848x480
            image_value = inputs.get("image_url") or inputs.get("image_path")
            if not image_value:
                raise ValueError("wan22 requires image_url or image_path")
            image_url = runninghub_resolve_media(image_value, api_key)
            nodes = [
                {"nodeId": "790", "fieldName": "image", "fieldValue": image_url, "description": "输入图片"},
                {"nodeId": "809", "fieldName": "value", "fieldValue": prompt, "description": "输入提示词"},
                {"nodeId": "789", "fieldName": "value", "fieldValue": str(duration), "description": "时长"},
                {"nodeId": "791", "fieldName": "max_width", "fieldValue": "848", "description": "输入宽"},
                {"nodeId": "791", "fieldName": "max_height", "fieldValue": "480", "description": "输入高"},
            ]
            return nodes, False

        if model_alias == "ltx23_long":
            # 2055155307592077313: image(584) + prompt(620) + prompt分段(621)
            image_value = inputs.get("image_url") or inputs.get("image_path")
            if not image_value:
                raise ValueError("ltx23_long requires image_url or image_path")
            image_url = runninghub_resolve_media(image_value, api_key)
            width, height = self._compute_ltx_size(aspect, resolution)
            nodes = [
                {"nodeId": "584", "fieldName": "image", "fieldValue": image_url, "description": "image"},
                {"nodeId": "620", "fieldName": "prompt", "fieldValue": prompt, "description": "prompt主提示词"},
                {"nodeId": "621", "fieldName": "prompt", "fieldValue": prompt, "description": "prompt分阶段提示词"},
            ]
            # ltx23_long does not send width/height — workflow controls resolution.
            # (TS doesn't either; only the standard ltx23 path sends them.)
            return nodes, False

        if model_alias == "ltx23_four_frames":
            # 2054820963426021378: 4 reference images (1361-1364) + prompt(1473)
            # Uses /run/workflow/ endpoint with addMetadata=true
            ref_urls: list[str] = list(inputs.get("reference_image_urls") or [])
            for local_path in inputs.get("reference_image_paths") or []:
                ref_urls.append(runninghub_resolve_media(local_path, api_key))
            # Also accept single image_url/image_path as the first reference
            single = inputs.get("image_url") or inputs.get("image_path")
            if single and not ref_urls:
                ref_urls.append(runninghub_resolve_media(single, api_key))
            if not ref_urls:
                raise ValueError("ltx23_four_frames requires at least one reference image (reference_image_urls or reference_image_paths or image_url/image_path)")
            # Pad to 4 by repeating the last image (TS line 473)
            while len(ref_urls) < 4:
                ref_urls.append(ref_urls[-1])
            ref_urls = ref_urls[:4]
            nodes = [
                {"nodeId": "1361", "fieldName": "image", "fieldValue": ref_urls[0], "description": "参考图1"},
                {"nodeId": "1362", "fieldName": "image", "fieldValue": ref_urls[1], "description": "参考图2"},
                {"nodeId": "1363", "fieldName": "image", "fieldValue": ref_urls[2], "description": "参考图3"},
                {"nodeId": "1364", "fieldName": "image", "fieldValue": ref_urls[3], "description": "参考图4"},
                {"nodeId": "1473", "fieldName": "text", "fieldValue": prompt, "description": "自定义剧情提示词"},
            ]
            return nodes, True  # use_workflow_endpoint=True

        # ltx23 (2029759632314474498): image(98) + video_length(185) + width(222) + height(223) + prompt(224)
        image_value = inputs.get("image_url") or inputs.get("image_path")
        if not image_value:
            raise ValueError("ltx23 requires image_url or image_path")
        image_url = runninghub_resolve_media(image_value, api_key)
        width, height = self._compute_ltx_size(aspect, resolution)
        # LTX uses 24fps; video_length = duration * 24
        video_length = round(duration * 24)
        nodes = [
            {"nodeId": "98", "fieldName": "image", "fieldValue": image_url, "description": "上传图片"},
            {"nodeId": "185", "fieldName": "value", "fieldValue": str(video_length), "description": "视频长度"},
            {"nodeId": "222", "fieldName": "value", "fieldValue": str(width), "description": "视频宽度"},
            {"nodeId": "223", "fieldName": "value", "fieldValue": str(height), "description": "视频高度"},
            {"nodeId": "224", "fieldName": "value", "fieldValue": prompt, "description": "提示词"},
        ]
        return nodes, False

    @staticmethod
    def _compute_ltx_size(aspect_ratio: str, resolution: str) -> tuple[int, int]:
        """Compute width/height for LTX2.3 workflows.

        Mirrors TS lines 439-445:
          baseW = 1920 (1080) | 1280 (720) | 854 (else)
          baseH = 1080 (1080) | 720 (720)  | 480 (else)
          For 9:16, swap width/height.
        """
        if "1080" in resolution:
            base_w, base_h = 1920, 1080
        elif "720" in resolution:
            base_w, base_h = 1280, 720
        else:
            base_w, base_h = 854, 480
        if aspect_ratio == "9:16":
            return base_h, base_w
        return base_w, base_h
