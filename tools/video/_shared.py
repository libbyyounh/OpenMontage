"""Shared helpers for provider-specific video generation tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import ToolResult, ToolStatus


HEYGEN_PROVIDERS = {
    "veo_3_1": {"name": "Google VEO 3.1", "quality": "highest", "speed": "slow"},
    "veo_3_1_fast": {"name": "Google VEO 3.1 Fast", "quality": "high", "speed": "medium"},
    "veo3": {"name": "Google VEO 3", "quality": "high", "speed": "slow"},
    "veo3_fast": {"name": "Google VEO 3 Fast", "quality": "high", "speed": "medium"},
    "veo2": {"name": "Google VEO 2", "quality": "medium", "speed": "medium"},
    "kling_pro": {"name": "Kling Pro", "quality": "high", "speed": "medium"},
    "kling_v2": {"name": "Kling v2", "quality": "medium", "speed": "fast"},
    "sora_v2": {"name": "Sora v2", "quality": "high", "speed": "slow"},
    "sora_v2_pro": {"name": "Sora v2 Pro", "quality": "highest", "speed": "slow"},
    "runway_gen4": {"name": "Runway Gen-4", "quality": "high", "speed": "medium"},
    # NOTE: HeyGen's `seedance_lite` / `seedance_pro` provider strings map to
    # Seedance 1.x. Seedance 2.0 on HeyGen is exposed through Video Agent and
    # Avatar Shots endpoints, NOT via the workflow provider parameter. For 2.0
    # access today, use `seedance_video` (fal.ai) or `seedance_replicate`.
    "seedance_lite": {"name": "Seedance Lite (1.x)", "quality": "medium", "speed": "fast"},
    "seedance_pro": {"name": "Seedance Pro (1.x)", "quality": "high", "speed": "medium"},
    "ltx_distilled": {"name": "LTX Distilled", "quality": "low", "speed": "fastest"},
}

WAN_VARIANTS = {
    "wan2.1-1.3b": {
        "name": "Wan 2.1 (1.3B)",
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "hf_i2v_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "pipeline_class": "WanPipeline",
        "vram_mb": 8000,
        "quality": "high",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 832,
        "default_height": 480,
        "default_num_frames": 81,
        "fps": 16,
    },
    "wan2.1-14b": {
        "name": "Wan 2.1 (14B)",
        "hf_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "hf_i2v_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "pipeline_class": "WanPipeline",
        "vram_mb": 24000,
        "quality": "highest",
        "speed": "slow",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 1280,
        "default_height": 720,
        "default_num_frames": 81,
        "fps": 16,
    },
}

HUNYUAN_VARIANTS = {
    "hunyuan-1.5": {
        "name": "HunyuanVideo 1.5",
        "hf_id": "tencent/HunyuanVideo-1.5",
        "pipeline_class": "HunyuanVideoPipeline",
        "vram_mb": 14000,
        "quality": "high",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 848,
        "default_height": 480,
        "default_num_frames": 121,
        "fps": 24,
    },
}

LTX_LOCAL_VARIANTS = {
    "ltx2-local": {
        "name": "LTX-2 (Local)",
        "hf_id": "Lightricks/LTX-2",
        "pipeline_class": "LTXPipeline",
        "vram_mb": 12000,
        "quality": "high",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "LTX-2-Community",
        "default_width": 768,
        "default_height": 512,
        "default_num_frames": 121,
        "fps": 30,
    },
}

COGVIDEO_VARIANTS = {
    "cogvideo-5b": {
        "name": "CogVideoX 1.5 (5B)",
        "hf_id": "THUDM/CogVideoX-5b",
        "pipeline_class": "CogVideoXPipeline",
        "vram_mb": 12000,
        "quality": "medium",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 720,
        "default_height": 480,
        "default_num_frames": 49,
        "fps": 8,
    },
    "cogvideo-2b": {
        "name": "CogVideoX (2B)",
        "hf_id": "THUDM/CogVideoX-2b",
        "pipeline_class": "CogVideoXPipeline",
        "vram_mb": 6000,
        "quality": "medium",
        "speed": "fast",
        "t2v": True,
        "i2v": False,
        "license": "Apache-2.0",
        "default_width": 720,
        "default_height": 480,
        "default_num_frames": 49,
        "fps": 8,
    },
}

LTX2_FRAME_COUNTS = {
    "1s": 25,
    "2s": 49,
    "3s": 73,
    "4s": 97,
    "5s": 121,
    "6.7s": 161,
    "8s": 193,
}


def get_torch_device() -> str:
    """Return best available torch device: cuda > mps (Apple Silicon Metal) > cpu.

    Priority order:
      1. cuda  — NVIDIA GPU (fastest for most diffusion models)
      2. mps   — Apple Silicon Metal (M1/M2/M3/M4/M5, macOS >= 12.3)
      3. cpu   — fallback, always available but slow

    MPS detection is guarded for torch builds that lack ``torch.backends.mps``
    (e.g. older pip wheels or Linux builds).  We check both build-time support
    (``is_built()``) and runtime availability (``is_available()``).
    """
    try:
        import torch as _torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if _torch.cuda.is_available():
        return "cuda"
    # Guard: torch.backends.mps may not exist on older/non-macOS builds
    try:
        mps_backend = getattr(_torch, "backends", None)
        mps_backend = getattr(mps_backend, "mps", None) if mps_backend else None
        if mps_backend is not None:
            # Check build-time support first, then runtime availability
            is_built = getattr(mps_backend, "is_built", lambda: True)()
            is_available = getattr(mps_backend, "is_available", lambda: False)()
            if is_built and is_available:
                return "mps"
    except Exception:
        pass
    return "cpu"


def local_generation_enabled() -> bool:
    return os.environ.get("VIDEO_GEN_LOCAL_ENABLED", "").lower() in {"true", "1", "yes"}


def local_generation_status() -> ToolStatus:
    if not local_generation_enabled():
        return ToolStatus.UNAVAILABLE
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return ToolStatus.UNAVAILABLE
    return ToolStatus.AVAILABLE


def local_install_instructions() -> str:
    return (
        "Enable local video generation and install the diffusers stack:\n"
        "  export VIDEO_GEN_LOCAL_ENABLED=true\n"
        "  uv pip install diffusers transformers accelerate torch pillow requests\n"
        "\n"
        "GPU support — pick what matches your hardware:\n"
        "  NVIDIA CUDA    — works out of the box with the above\n"
        "  Apple Silicon (MPS, macOS >= 12.3) — works out of the box; no extra build\n"
        "  CPU fallback   — slow but functional on any machine\n"
        "\n"
        "VRAM profile: see the selected tool's resource_profile for minimum VRAM."
    )


def estimate_quality_cost(quality: str) -> float:
    if quality == "highest":
        return 0.50
    if quality == "high":
        return 0.35
    if quality == "low":
        return 0.15
    return 0.20


def estimate_speed_runtime(speed: str) -> float:
    return {"fastest": 30.0, "fast": 60.0, "medium": 120.0, "slow": 300.0}.get(speed, 120.0)


def estimate_local_runtime(speed: str) -> float:
    return {"fast": 120.0, "medium": 240.0, "slow": 600.0}.get(speed, 240.0)


def load_diffusers_pipeline(pipeline_class: str, model_id: str, enable_offload: bool):
    import diffusers
    import torch

    pipeline_map = {
        "WanPipeline": "WanPipeline",
        "HunyuanVideoPipeline": "HunyuanVideoPipeline",
        "LTXPipeline": "LTXPipeline",
        "CogVideoXPipeline": "CogVideoXPipeline",
    }
    pipeline_name = pipeline_map.get(pipeline_class, pipeline_class)
    pipeline_class_obj = getattr(diffusers, pipeline_name)

    device = get_torch_device()
    # bfloat16 is only reliable on CUDA; MPS uses float16 for inference,
    # CPU must use float32 (float16 is emulated and unreliable on CPU)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16

    pipeline = pipeline_class_obj.from_pretrained(model_id, torch_dtype=dtype)

    if enable_offload:
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            # enable_model_cpu_offload() is CUDA-only; fall back to direct device placement
            pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
    return pipeline


def load_reference_image(inputs: dict[str, Any], width: int, height: int):
    from io import BytesIO

    import requests
    from PIL import Image

    ref_path = inputs.get("reference_image_path")
    ref_url = inputs.get("reference_image_url")

    if ref_path:
        image = Image.open(ref_path).convert("RGB")
    elif ref_url:
        response = requests.get(ref_url, timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return ToolResult(
            success=False,
            error="image_to_video requires reference_image_url or reference_image_path",
        )

    return image.resize((width, height), Image.LANCZOS)


def generate_local_video(
    *,
    tool_name: str,
    variants: dict[str, dict[str, Any]],
    default_variant: str,
    inputs: dict[str, Any],
) -> ToolResult:
    import torch
    from diffusers.utils import export_to_video

    variant = inputs.get("model_variant", default_variant)
    if variant not in variants:
        return ToolResult(
            success=False,
            error=f"Unknown model_variant: {variant}. Available: {', '.join(sorted(variants))}",
        )

    meta = variants[variant]
    prompt = inputs["prompt"]
    operation = inputs.get("operation", "text_to_video")
    seed = inputs.get("seed")
    enable_offload = inputs.get("enable_model_offload", True)

    if operation == "image_to_video" and not meta.get("i2v"):
        return ToolResult(
            success=False,
            error=f"{meta['name']} does not support image_to_video.",
        )

    width = inputs.get("width", meta["default_width"])
    height = inputs.get("height", meta["default_height"])
    num_frames = inputs.get("num_frames", meta["default_num_frames"])
    fps = meta["fps"]
    model_id = meta.get("hf_i2v_id") if operation == "image_to_video" and meta.get("hf_i2v_id") else meta["hf_id"]
    pipeline = load_diffusers_pipeline(meta["pipeline_class"], model_id, enable_offload)

    generation_args: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "num_inference_steps": inputs.get("num_inference_steps", 30),
    }
    if seed is not None:
        generation_args["generator"] = torch.Generator(device="cpu").manual_seed(seed)
    if operation == "image_to_video":
        image = load_reference_image(inputs, width, height)
        if isinstance(image, ToolResult):
            return image
        generation_args["image"] = image
    if meta["pipeline_class"] == "CogVideoXPipeline":
        generation_args["negative_prompt"] = "worst quality, low quality, blurry, distorted, watermark"

    output = pipeline(**generation_args)
    frames = output.frames[0] if hasattr(output, "frames") else output.images

    output_path = Path(inputs.get("output_path", f"{tool_name}_{variant}.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=fps)

    return ToolResult(
        success=True,
        data={
            "provider": tool_name,
            "model_variant": variant,
            "provider_name": meta["name"],
            "mode": "local",
            "prompt": prompt,
            "model_id": model_id,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": round(num_frames / fps, 2),
            "operation": operation,
            "output": str(output_path),
            "format": "mp4",
            "license": meta["license"],
            **probe_output(output_path),
        },
        artifacts=[str(output_path)],
        seed=seed,
        model=model_id,
    )


def poll_heygen(execution_id: str, api_key: str, timeout: int = 600) -> str:
    import requests

    headers = {"X-Api-Key": api_key}
    url = f"https://api.heygen.com/v1/workflows/executions/{execution_id}"
    deadline = time.time() + timeout
    interval = 5.0

    while time.time() < deadline:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", {})
        status = data.get("status", "")

        if status == "completed":
            video_url = (
                data.get("output", {}).get("video", {}).get("video_url")
                or data.get("output", {}).get("video_url")
            )
            if video_url:
                return video_url
            raise RuntimeError(f"Completed but no video_url in output: {data}")

        if status in {"failed", "error"}:
            raise RuntimeError(f"HeyGen generation failed: {data.get('error', 'Unknown')}")

        time.sleep(min(interval, max(0.0, deadline - time.time())))
        interval = min(interval * 1.2, 30.0)

    raise TimeoutError(f"HeyGen execution {execution_id} timed out after {timeout}s")


def upload_image_fal(image_path: str) -> str:
    """Upload a local image to fal.ai storage and return a public URL."""
    import requests

    api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")
    if not api_key:
        raise RuntimeError("FAL_KEY or FAL_AI_API_KEY required for image upload")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    content_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        suffix.lstrip("."), "image/png"
    )

    # Initiate upload
    init_resp = requests.post(
        "https://rest.alpha.fal.ai/storage/upload/initiate",
        headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
        json={"content_type": content_type, "file_name": path.name},
        timeout=30,
    )
    init_resp.raise_for_status()
    data = init_resp.json()

    # Upload file content
    put_resp = requests.put(
        data["upload_url"],
        headers={"Content-Type": content_type},
        data=path.read_bytes(),
        timeout=60,
    )
    put_resp.raise_for_status()

    return data["file_url"]


def upload_image_heygen(image_path: str, api_key: str) -> str:
    """Upload a local image to HeyGen and return a public URL.

    Tries the v2 presigned-upload endpoint first, falls back to fal.ai storage.
    """
    import requests

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try HeyGen v2 presigned upload
    try:
        resp = requests.post(
            "https://api.heygen.com/v2/assets/upload",
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            json={"content_type": "image/png", "file_name": path.name},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            upload_url = data.get("upload_url")
            file_url = data.get("url") or data.get("file_url")
            if upload_url and file_url:
                put_resp = requests.put(
                    upload_url,
                    headers={"Content-Type": "image/png"},
                    data=path.read_bytes(),
                    timeout=60,
                )
                put_resp.raise_for_status()
                return file_url
    except Exception:
        pass

    # Fallback to fal.ai storage upload
    return upload_image_fal(image_path)


def generate_heygen_video(inputs: dict[str, Any]) -> ToolResult:
    import requests

    api_key = os.environ.get("HEYGEN_API_KEY")
    if not api_key:
        return ToolResult(success=False, error="HEYGEN_API_KEY not set.")

    provider = inputs.get("provider_variant", "veo_3_1")
    if provider not in HEYGEN_PROVIDERS:
        return ToolResult(
            success=False,
            error=f"Unknown provider_variant: {provider}. Available: {', '.join(sorted(HEYGEN_PROVIDERS))}",
        )

    prompt = inputs["prompt"]
    aspect_ratio = inputs.get("aspect_ratio", "16:9")
    operation = inputs.get("operation", "text_to_video")
    workflow_input: dict[str, Any] = {
        "prompt": prompt,
        "provider": provider,
        "aspect_ratio": aspect_ratio,
    }
    if operation == "image_to_video":
        ref_url = inputs.get("reference_image_url")
        ref_path = inputs.get("reference_image_path")
        if ref_path and not ref_url:
            ref_url = upload_image_heygen(ref_path, api_key)
        if not ref_url:
            return ToolResult(
                success=False,
                error="image_to_video requires reference_image_url or reference_image_path",
            )
        workflow_input["reference_image_url"] = ref_url

    response = requests.post(
        "https://api.heygen.com/v1/workflows/executions",
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json={"workflow_type": "GenerateVideoNode", "input": workflow_input},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    execution_id = payload.get("data", {}).get("execution_id")
    if not execution_id:
        return ToolResult(success=False, error=f"No execution_id in response: {payload}")

    video_url = poll_heygen(execution_id, api_key, timeout=600)
    output_path = Path(inputs.get("output_path", f"heygen_video_{execution_id}.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download = requests.get(video_url, timeout=120)
    download.raise_for_status()
    output_path.write_bytes(download.content)

    meta = HEYGEN_PROVIDERS[provider]
    return ToolResult(
        success=True,
        data={
            "provider": "heygen",
            "provider_variant": provider,
            "provider_name": meta["name"],
            "mode": "api",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "operation": operation,
            "execution_id": execution_id,
            "output": str(output_path),
            "format": "mp4",
        },
        artifacts=[str(output_path)],
        model=provider,
    )


def generate_ltx_modal_video(inputs: dict[str, Any]) -> ToolResult:
    import base64

    import requests

    endpoint_url = os.environ.get("MODAL_LTX2_ENDPOINT_URL")
    if not endpoint_url:
        return ToolResult(success=False, error="MODAL_LTX2_ENDPOINT_URL not set.")

    prompt = inputs["prompt"]
    operation = inputs.get("operation", "text_to_video")
    aspect = inputs.get("aspect_ratio", "16:9")
    width = inputs.get("width")
    height = inputs.get("height")
    if width is None or height is None:
        if aspect == "16:9":
            width, height = 1024, 576
        elif aspect == "9:16":
            width, height = 576, 1024
        else:
            width, height = 512, 512

    num_frames = inputs.get("num_frames", LTX2_FRAME_COUNTS.get(inputs.get("duration_hint", "5s"), 121))
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1) // 8) * 8 + 1

    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": 24,
        "steps": inputs.get("num_inference_steps", 30),
        "negative_prompt": "worst quality, low quality, blurry, distorted, watermark, text, logo",
    }
    if inputs.get("seed") is not None:
        payload["seed"] = inputs["seed"]

    if operation == "image_to_video":
        ref_path = inputs.get("reference_image_path")
        ref_url = inputs.get("reference_image_url")
        if ref_path:
            payload["input_image"] = base64.b64encode(Path(ref_path).read_bytes()).decode()
        elif ref_url:
            payload["input_image_url"] = ref_url
        else:
            return ToolResult(
                success=False,
                error="image_to_video requires reference_image_url or reference_image_path",
            )

    response = requests.post(endpoint_url, json=payload, timeout=300)
    response.raise_for_status()
    output_path = Path(inputs.get("output_path", "ltx_video_modal.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_type = response.headers.get("content-type", "")
    if "video" in content_type or "octet-stream" in content_type:
        output_path.write_bytes(response.content)
    else:
        response_payload = response.json()
        video_url = response_payload.get("video_url") or response_payload.get("url")
        if not video_url:
            return ToolResult(success=False, error=f"No video data in response: {response_payload}")
        download = requests.get(video_url, timeout=120)
        download.raise_for_status()
        output_path.write_bytes(download.content)

    return ToolResult(
        success=True,
        data={
            "provider": "ltx-modal",
            "provider_name": "LTX-2 (Modal)",
            "mode": "modal",
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": 24,
            "duration_seconds": round(num_frames / 24, 2),
            "operation": operation,
            "output": str(output_path),
            "format": "mp4",
        },
        artifacts=[str(output_path)],
        seed=inputs.get("seed"),
        model="ltx-2",
    )


def probe_output(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"file_size_bytes": path.stat().st_size}
    if not shutil.which("ffprobe"):
        return info

    import json

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            probe = json.loads(proc.stdout)
            fmt = probe.get("format", {})
            info["duration_seconds"] = float(fmt.get("duration", 0))
            info["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["video_width"] = int(stream.get("width", 0))
                    info["video_height"] = int(stream.get("height", 0))
                    info["video_codec"] = stream.get("codec_name", "")
                    break
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# RunningHub (https://www.runninghub.cn) — workflow-based image/video API
# Mirrors the TS submitRhTask / queryRhTask / uploadRhMedia / resolveMediaUrl /
# pollRhResult helpers in toonflow-runninghub-person.ts.
# ---------------------------------------------------------------------------

RUNNINGHUB_BASE_URL = "https://www.runninghub.cn"

# RunningHub enforces a per-API-key concurrency limit. The default is 1 — any
# simultaneous submission across this process (multiple agents, threads, async
# tasks) will hit errorCode=421 "queue limit reached". We serialize all
# RunningHub API calls (upload, submit, poll, query, download) through a
# module-level semaphore sized to match the server's allowance, so the caller
# never has to think about concurrency control.
#
# The semaphore is created lazily on first use so the env var is read after
# .env is loaded (tools that import _shared at module import time otherwise
# capture a default before the env file is parsed).
#
# Override via env var: RUNNINGHUB_CONCURRENCY=N
#   1 (default)  — strict serialization, fits any RunningHub key tier
#   >1           — if your key tier allows more, raise N. Holding the permit
#                  across the full poll/download cycle is still correct
#                  because the server-side queue depth is the bottleneck,
#                  not the wall time of a single in-flight task.
import threading as _threading
import os as _os
from contextlib import contextmanager as _contextmanager

_RH_CONCURRENCY = 1  # overwritten lazily in _get_rh_semaphore()
_RH_SEMAPHORE: _threading.Semaphore | None = None
_RH_SEMAPHORE_LOCK = _threading.Lock()  # protects lazy init


def _get_rh_semaphore() -> _threading.Semaphore:
    """Lazy-init the RunningHub semaphore from RUNNINGHUB_CONCURRENCY env var.

    Created on first call so .env is parsed first; module import happens
    earlier in some tool-loading paths.
    """
    global _RH_SEMAPHORE, _RH_CONCURRENCY
    if _RH_SEMAPHORE is None:
        with _RH_SEMAPHORE_LOCK:
            if _RH_SEMAPHORE is None:
                try:
                    n = int(_os.environ.get("RUNNINGHUB_CONCURRENCY", "1"))
                except ValueError:
                    n = 1
                if n < 1:
                    n = 1
                _RH_CONCURRENCY = n
                _RH_SEMAPHORE = _threading.Semaphore(n)
    return _RH_SEMAPHORE


# Backoff schedule for transient errors. When concurrency=1 the server-side
# queue can transiently look full while a prior task drains; longer waits
# between retries prevent hot-spinning against the limit. With concurrency>1
# the queue drains in parallel so backoff can be lighter, but we keep the
# same schedule as a safe default. Index = attempt number (0-based).
_RH_RETRY_BACKOFF = (5, 10, 20, 40, 60)


def _rh_call(func, *args, **kwargs):
    """Run a RunningHub API call under the configured concurrency limit.

    Use this wrapper at every entry point that talks to the RunningHub API
    (upload, submit, query, poll, download) so concurrent calls in the same
    process don't blow the server-side concurrency budget.
    """
    sem = _get_rh_semaphore()
    sem.acquire()
    try:
        return func(*args, **kwargs)
    finally:
        sem.release()


@_contextmanager
def _rh_permit():
    """Context manager that acquires/releases one RunningHub concurrency permit.

    Used by entry points that need to hold the permit across multiple internal
    calls (e.g. submit + the full poll+download cycle). For single-call use,
    `_rh_call` is simpler.
    """
    sem = _get_rh_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def runninghub_submit(
    workflow_id: str,
    node_info_list: list[dict[str, Any]],
    api_key: str,
    *,
    use_workflow_endpoint: bool = False,
) -> str:
    """Submit a RunningHub workflow task and return its task_id.

    Default endpoint: POST /openapi/v2/run/ai-app/{workflow_id}
    Workflow endpoint: POST /openapi/v2/run/workflow/{workflow_id}
    (used by ltx23_four_frames; payload includes addMetadata=true)

    Serialized through the RunningHub concurrency semaphore (sized from
    RUNNINGHUB_CONCURRENCY env var, default 1) — parallel submits in the
    same process return errorCode=421 if they exceed the limit.
    """
    with _rh_permit():
        return _runninghub_submit_locked(
            workflow_id, node_info_list, api_key,
            use_workflow_endpoint=use_workflow_endpoint,
        )


def _runninghub_submit_locked(
    workflow_id: str,
    node_info_list: list[dict[str, Any]],
    api_key: str,
    *,
    use_workflow_endpoint: bool = False,
) -> str:
    """Inner submit — must be called while holding _RH_GLOBAL_LOCK."""
    import requests

    path = "workflow" if use_workflow_endpoint else "ai-app"
    url = f"{RUNNINGHUB_BASE_URL}/openapi/v2/run/{path}/{workflow_id}"

    payload: dict[str, Any] = {
        "nodeInfoList": node_info_list,
        "instanceType": "default",
        "usePersonalQueue": "true",
    }
    if use_workflow_endpoint:
        payload["addMetadata"] = True

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    # Retry transient HTTP statuses (queue/full/inflight) before raising
    TRANSIENT_HTTP = {500, 502, 503, 504}
    if response.status_code in TRANSIENT_HTTP:
        raise RuntimeError(
            f"RunningHub submit transient HTTP {response.status_code} for workflow {workflow_id}"
        )
    response.raise_for_status()
    data = response.json()
    task_id = data.get("taskId")
    err_code = data.get("errorCode")
    if not task_id:
        # 421 queue limit / 1000 unknown - re-raise so caller can retry the whole submit
        msg = data.get("errorMessage", "")
        raise RuntimeError(f"RunningHub submit did not return taskId: {data} (errorCode={err_code})")
    return task_id


def runninghub_query(task_id: str, api_key: str) -> dict[str, Any]:
    """Query RunningHub task status. Returns the full response dict.

    Serialized through the RunningHub concurrency semaphore so we don't
    poll-flood the server while another submit is in flight.
    """
    with _rh_permit():
        return _runninghub_query_locked(task_id, api_key)


def _runninghub_query_locked(task_id: str, api_key: str) -> dict[str, Any]:
    """Inner query — must be called while holding _RH_GLOBAL_LOCK."""
    import requests

    response = requests.post(
        f"{RUNNINGHUB_BASE_URL}/openapi/v2/query",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"taskId": task_id},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def upload_image_runninghub(image_path: str, api_key: str) -> str:
    """Upload a local image (or media) file to RunningHub and return its download_url.

    Mirrors TS uploadRhMedia multipart construction. Used by image-edit and
    image-to-video workflows that need a hosted media URL.

    Serialized through the RunningHub concurrency semaphore.
    """
    with _rh_permit():
        return _upload_image_runninghub_locked(image_path, api_key)


def _upload_image_runninghub_locked(image_path: str, api_key: str) -> str:
    """Inner upload — must be called while holding _RH_GLOBAL_LOCK."""
    import requests

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower().lstrip(".")
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "mp3": "audio/mpeg",
    }
    mime = mime_map.get(suffix, "application/octet-stream")
    ext = suffix if suffix in {"png", "jpg", "jpeg", "webp", "mp4", "mp3"} else "png"

    boundary = f"----RHBoundary{int(time.time() * 1000):x}"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="upload_{int(time.time())}.{ext}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + path.read_bytes() + footer

    # Transient API error codes that warrant retry (queue/limit/auth blips).
    # 1 = "ApiKey verification error" (intermittent), 421 = queue limit, 500/502/503/504 = server.
    # RunningHub enforces concurrency=1 per API key, so even with the global
    # lock the server-side queue depth can transiently look full while a
    # previous task is still draining. Retry with longer backoff.
    TRANSIENT_HTTP = {500, 502, 503, 504}
    TRANSIENT_API_CODES = {1, 421}
    last_exc: Exception | None = None
    last_resp_data: dict | None = None

    # With concurrency=1, expect to wait through the queue. 5 attempts ×
    # 5/10/20/40/60s backoff ≈ 2-3 min before giving up.
    max_attempts = 5
    for attempt in range(max_attempts):
        response = requests.post(
            f"{RUNNINGHUB_BASE_URL}/openapi/v2/media/upload/binary",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            timeout=60,
        )
        # Retry on transient HTTP statuses (without raising_for_status)
        if response.status_code in TRANSIENT_HTTP:
            last_exc = RuntimeError(
                f"RunningHub media upload HTTP {response.status_code} (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(_RH_RETRY_BACKOFF[attempt])
            continue
        response.raise_for_status()
        data = response.json()
        last_resp_data = data
        api_code = data.get("code")
        download_url = (data.get("data") or {}).get("download_url") if isinstance(data.get("data"), dict) else None
        # Success path
        if api_code == 0 and download_url:
            return download_url
        # Transient API codes -> retry
        if api_code in TRANSIENT_API_CODES:
            last_exc = RuntimeError(
                f"RunningHub media upload transient error code={api_code} "
                f"msg={data.get('msg', '')[:120]} (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(_RH_RETRY_BACKOFF[attempt])
            continue
        # Permanent error -> raise immediately
        raise RuntimeError(f"RunningHub media upload failed: {data}")

    # All retries exhausted
    raise RuntimeError(
        f"RunningHub media upload failed after {max_attempts} attempts. Last error: {last_exc}. "
        f"Last response: {last_resp_data}"
    )


def runninghub_resolve_media(image_value: str, api_key: str) -> str:
    """Resolve a media reference to a URL.

    If image_value is already an http(s) URL, return it as-is. Otherwise treat
    it as a local path and upload via upload_image_runninghub.
    """
    if image_value.startswith(("http://", "https://")):
        return image_value
    return upload_image_runninghub(image_value, api_key)


def poll_runninghub(
    task_id: str,
    api_key: str,
    *,
    interval: float = 5.0,
    timeout: float = 600.0,
) -> tuple[str, bytes]:
    """Poll a RunningHub task until SUCCESS, then download the result bytes.

    Returns (download_url, file_bytes). Raises RuntimeError on FAILED/ERROR,
    TimeoutError on timeout. Mirrors TS pollRhResult, but returns bytes for
    direct file writes instead of base64.

    Serialized through the RunningHub concurrency semaphore for the full
    poll+download cycle so concurrent waiters don't all hit the server at
    the same interval.
    """
    with _rh_permit():
        return _poll_runninghub_locked(
            task_id, api_key,
            interval=interval, timeout=timeout,
        )


def _poll_runninghub_locked(
    task_id: str,
    api_key: str,
    *,
    interval: float = 5.0,
    timeout: float = 600.0,
) -> tuple[str, bytes]:
    """Inner poll — must be called while holding _RH_GLOBAL_LOCK."""
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        data = _runninghub_query_locked(task_id, api_key)
        status = data.get("status", "")
        if status == "SUCCESS":
            results = data.get("results") or []
            url = results[0].get("url") if results else None
            if not url:
                raise RuntimeError("RunningHub task completed but no result URL")
            dl = requests.get(url, timeout=120)
            dl.raise_for_status()
            return url, dl.content
        if status in ("FAILED", "ERROR"):
            msg = data.get("errorMessage") or data.get("errorCode") or status
            raise RuntimeError(f"RunningHub task {status}: {msg}")
        time.sleep(min(interval, max(0.0, deadline - time.time())))
    raise TimeoutError(f"RunningHub task {task_id} timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Grsai (https://grsai.dakka.com.cn) — nano-banana + gpt-image-2 generation API
# Endpoints: POST /v1/api/generate (submit), GET /v1/api/result?id=<task_id> (poll)
# ---------------------------------------------------------------------------

GRSAI_BASE_URL = "https://grsai.dakka.com.cn"


def grsai_generate(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST /v1/api/generate with the given payload. Returns the full response.

    `payload` should include: model, prompt, images (optional), aspectRatio
    (optional), imageSize (optional, nano-banana only), replyType (json/stream/async).

    Response shape: {id, status, results: [{url}], progress, error}
    """
    import requests

    response = requests.post(
        f"{GRSAI_BASE_URL}/v1/api/generate",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def grsai_query_result(task_id: str, api_key: str) -> dict[str, Any]:
    """GET /v1/api/result?id=<task_id>. Returns the full response dict."""
    import requests

    response = requests.get(
        f"{GRSAI_BASE_URL}/v1/api/result",
        params={"id": task_id},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def poll_grsai(
    task_id: str,
    api_key: str,
    *,
    interval: float = 5.0,
    timeout: float = 600.0,
) -> tuple[str, bytes]:
    """Poll a Grsai task until status=succeeded, then download the result bytes.

    Returns (download_url, file_bytes). Raises RuntimeError on violation/failed,
    TimeoutError on timeout. Status values: running, violation, succeeded, failed.
    """
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        data = grsai_query_result(task_id, api_key)
        status = data.get("status", "")
        if status == "succeeded":
            results = data.get("results") or []
            url = results[0].get("url") if results else None
            if not url:
                raise RuntimeError(f"Grsai task succeeded but no result URL: {data}")
            dl = requests.get(url, timeout=120)
            dl.raise_for_status()
            return url, dl.content
        if status in ("failed", "violation"):
            msg = data.get("error") or status
            raise RuntimeError(f"Grsai task {status}: {msg}")
        time.sleep(min(interval, max(0.0, deadline - time.time())))
    raise TimeoutError(f"Grsai task {task_id} timed out after {timeout}s")
