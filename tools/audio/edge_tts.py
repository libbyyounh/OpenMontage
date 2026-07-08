"""Microsoft edge-tts provider tool - free, no API key, Azure Neural voices.

Uses the Edge browser Read-Aloud endpoint via the `edge-tts` library.
Free and multilingual (300+ voices), but requires network and has no SLA.
"""

from __future__ import annotations

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
    ToolStatus,
    ToolTier,
)


EDGE_TTS_SETUP_OFFER = {
    "kind": "pip_install",
    "fix_complexity": "1-minute pip install (free, no API key)",
    "env_var": None,
    "what_it_unlocks": [
        "free multilingual TTS (300+ Azure Neural voices, incl. zh-CN-XiaoxiaoNeural)",
        "free fallback for paid TTS providers when no API keys are configured",
    ],
}


class EdgeTTS(BaseTool):
    name = "edge_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "edge"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = ["python:edge_tts"]
    install_instructions = (
        "Install edge-tts (free, no API key):\n"
        "  pip install edge-tts\n"
        "List voices:\n"
        "  edge-tts --list-voices\n"
        "Popular voices: zh-CN-XiaoxiaoNeural, en-US-AriaNeural, ja-JP-NanamiNeural"
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    agent_skills = ["edge-tts"]

    capabilities = [
        "text_to_speech",
        "free_generation",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "free": True,
        "ssml": False,
    }
    best_for = [
        "free cloud TTS with no API key",
        "multilingual narration including Mandarin Chinese",
        "default free fallback when paid TTS providers lack API keys",
    ]
    not_good_for = [
        "fully offline production (needs network)",
        "voice cloning",
        "commercial SLA / guaranteed availability (unofficial endpoint)",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {
                "type": "string",
                "default": "zh-CN-XiaoxiaoNeural",
                "description": (
                    "edge-tts voice name. Default zh-CN-XiaoxiaoNeural "
                    "(female, warm Mandarin). Examples: en-US-AriaNeural, "
                    "en-US-GuyNeural, zh-CN-YunxiNeural (male), ja-JP-NanamiNeural. "
                    "Run `edge-tts --list-voices` for the full list."
                ),
            },
            "rate": {
                "type": "string",
                "default": "+0%",
                "description": (
                    "SSML speaking rate, e.g. '-10%', '+20%'. A numeric multiplier "
                    "(1.2 -> '+20%') is accepted for tts_selector compatibility."
                ),
            },
            "volume": {
                "type": "string",
                "default": "+0%",
                "description": (
                    "SSML volume, e.g. '-20%'. A numeric multiplier is accepted "
                    "for tts_selector compatibility."
                ),
            },
            "pitch": {
                "type": "string",
                "default": "+0Hz",
                "description": (
                    "SSML pitch, e.g. '+5Hz', '-10Hz'. A numeric value is converted "
                    "to Hz for tts_selector compatibility."
                ),
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=2, retryable_errors=["timeout", "connection_error", "websocket"]
    )
    idempotency_key_fields = ["text", "voice", "rate", "volume", "pitch"]
    side_effects = [
        "writes audio file to output_path",
        "calls Microsoft Edge read-aloud endpoint",
    ]
    user_visible_verification = [
        "Listen to generated audio for intelligibility and tone"
    ]

    setup_offer = EDGE_TTS_SETUP_OFFER

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["setup_offer"] = self.setup_offer
        return info

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        # Scaffold: real implementation added in Task 2.
        return ToolResult(
            success=False, error="edge-tts execute not yet implemented"
        )
