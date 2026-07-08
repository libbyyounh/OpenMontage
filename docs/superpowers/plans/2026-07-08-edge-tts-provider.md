# Edge-TTS Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Microsoft `edge-tts` as a free, no-key TTS provider that auto-registers with the tool registry, routes through `tts_selector`, and becomes the preferred free fallback for paid TTS providers.

**Architecture:** A single `EdgeTTS(BaseTool)` in `tools/audio/edge_tts.py` calls the `edge-tts` async library via `asyncio.run(asyncio.wait_for(...))`. The registry auto-discovers it (`pkgutil.walk_packages`) and `tts_selector` auto-routes to it (`get_by_capability("tts")`) - no registry or selector code changes. Five paid TTS tools get `edge_tts` inserted before `piper_tts` in their `fallback_tools`.

**Tech Stack:** Python 3, `edge-tts>=6.1` (async, uses `aiohttp`), `asyncio`, OpenMontage `BaseTool`/`ToolResult`, `ffprobe` via `tools.analysis.audio_probe.probe_duration`, `pytest`.

## Global Constraints

- Branch: `feat/edge-tts-provider` (already created; spec committed at `29a4327`).
- Tool class: `EdgeTTS(BaseTool)`, `name="edge_tts"`, `provider="edge"`, `capability="tts"`, `runtime=API`, `estimate_cost=0.0` (free).
- `dependencies=["python:edge_tts"]` - no env var, no API key. `get_status()` uses BaseTool default (no override).
- Default voice: `zh-CN-XiaoxiaoNeural`. Standard scope: `text` + `voice` + `rate` + `volume` + `pitch` + `output_path`. No SSML, no `list_voices`.
- `edge_tts` is lazy-imported inside `_synthesize` so the module imports cleanly even when the package is absent (tool registers as UNAVAILABLE, matching `openai_tts`).
- Selector numeric params (`speed`/`speaking_rate` multiplier, `pitch` number) are coerced to SSML strings in `execute()`; native string values pass through.
- Fallback rule: insert `"edge_tts"` immediately before `"piper_tts"` in every paid TTS tool's `fallback_tools`.
- Tests are hermetic: network test gated by `OPENMONTAGE_NETWORK_TESTS=1`.
- Class names of paid TTS tools (verified): `OpenAITTS`, `ElevenLabsTTS`, `GoogleTTS`, `DashscopeTTS`, `DoubaoTTS`.

---

## File Structure

- **Create** `tools/audio/edge_tts.py` - `EdgeTTS(BaseTool)`. One responsibility: synthesize text to an MP3 via the edge-tts library and return a `ToolResult`.
- **Create** `.agents/skills/edge-tts/SKILL.md` - Layer 3 provider skill (voice list, SSML param format, selector coercion, limitations).
- **Create** `tests/test_edge_tts.py` - contract, registry, mocked-generate, empty-text, fallback-chain, and gated network tests.
- **Modify** `requirements.txt` - add `edge-tts>=6.1`.
- **Modify** `tools/audio/openai_tts.py` - `fallback` + `fallback_tools`.
- **Modify** `tools/audio/elevenlabs_tts.py` - `fallback_tools`.
- **Modify** `tools/audio/google_tts.py` - `fallback_tools`.
- **Modify** `tools/audio/dashscope_tts.py` - `fallback_tools`.
- **Modify** `tools/audio/doubao_tts.py` - `fallback_tools`.

---

### Task 1: Scaffold EdgeTTS provider (contract + registry + dep + skill)

**Files:**
- Create: `tools/audio/edge_tts.py`
- Create: `.agents/skills/edge-tts/SKILL.md`
- Modify: `requirements.txt`
- Test: `tests/test_edge_tts.py`

**Interfaces:**
- Consumes: `tools.base_tool.BaseTool` (and enums), `tools.analysis.audio_probe.probe_duration` (added in Task 2).
- Produces: `EdgeTTS` class with `name="edge_tts"`, `provider="edge"`, `capability="tts"`; `EDGE_TTS_SETUP_OFFER` constant; `get_info()` returns `setup_offer`. `execute()` is a stub in this task (real impl in Task 2).

- [ ] **Step 1: Write the failing contract + registry tests**

Create `tests/test_edge_tts.py`:

```python
"""Tests for the EdgeTTS provider tool."""

import os

import pytest


def test_edge_tts_contract():
    from tools.audio.edge_tts import EdgeTTS

    tool = EdgeTTS()
    assert tool.name == "edge_tts"
    assert tool.capability == "tts"
    assert tool.provider == "edge"
    assert tool.runtime.value == "api"
    assert tool.estimate_cost({"text": "hi"}) == 0.0
    assert (
        tool.input_schema["properties"]["voice"]["default"]
        == "zh-CN-XiaoxiaoNeural"
    )
    info = tool.get_info()
    assert info["setup_offer"]["kind"] == "pip_install"
    assert info["dependencies"] == ["python:edge_tts"]
    # edge_tts is installed (added to requirements.txt), so status is available.
    assert tool.get_status().value == "available"


def test_edge_tts_registry_discovery():
    from tools.tool_registry import registry

    registry.discover()
    assert "edge_tts" in registry.list_all()
    assert any(
        t.provider == "edge"
        for t in registry.get_by_capability("tts")
        if t.name == "edge_tts"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_edge_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.audio.edge_tts'` (or collection error for the import).

- [ ] **Step 3: Create `tools/audio/edge_tts.py` with the scaffold (stub execute)**

Create `tools/audio/edge_tts.py`:

```python
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
```

- [ ] **Step 4: Add `edge-tts` to `requirements.txt`**

In `requirements.txt`, replace:

```
openai>=2.44.0         # Videos API support for Sora 2
```

with:

```
openai>=2.44.0         # Videos API support for Sora 2
edge-tts>=6.1          # free Microsoft Edge TTS, no API key (zh-CN-XiaoxiaoNeural default voice)
```

- [ ] **Step 5: Create the Layer 3 skill `.agents/skills/edge-tts/SKILL.md`**

Create `.agents/skills/edge-tts/SKILL.md`:

```markdown
---
name: edge-tts
description: |
  Free Microsoft Edge Read-Aloud TTS. No API key, 300+ Azure Neural voices across 50+ locales (incl. high-quality Mandarin zh-CN-XiaoxiaoNeural). Use when: (1) free narration with no key, (2) multilingual voiceover incl. Chinese, (3) fallback when paid TTS providers lack keys. Needs network; unofficial endpoint, no SLA.
---

# edge-tts (free Microsoft Edge TTS)

`edge-tts` calls the Microsoft Edge browser's Read-Aloud endpoint. It is **free and
needs no API key**, but it **requires network** and uses an **unofficial endpoint
with no SLA** - treat it as a generous free default, not a commercial-grade API.

## Tool

`tools/audio/edge_tts.py` (`EdgeTTS`, `provider="edge"`, `capability="tts"`).
Auto-discovered by the registry and routable through `tts_selector`.

## Voice selection

Default: `zh-CN-XiaoxiaoNeural` (female, warm Mandarin - the most popular Chinese
neural voice).

Popular voices by locale:

| Locale | Female | Male |
|---|---|---|
| zh-CN | `zh-CN-XiaoxiaoNeural` | `zh-CN-YunxiNeural` |
| en-US | `en-US-AriaNeural` | `en-US-GuyNeural` |
| ja-JP | `ja-JP-NanamiNeural` | `ja-JP-KeitaNeural` |
| ko-KR | `ko-KR-SunHiNeural` | `ko-KR-InJoonNeural` |

Full list: `edge-tts --list-voices`.

## Rate / volume / pitch

These are SSML-style **strings**, not numbers:

- `rate`: `"+0%"` default; e.g. `"-10%"` slower, `"+20%"` faster.
- `volume`: `"+0%"` default; e.g. `"-20%"` quieter.
- `pitch`: `"+0Hz"` default; e.g. `"+5Hz"` higher, `"-10Hz"` lower.

**tts_selector compatibility:** the selector passes numeric `speed`/`speaking_rate`
(multiplier) and `pitch` (-50..50). `EdgeTTS.execute()` coerces them - `1.2` ->
`"+20%"`, `pitch=5` -> `"+5Hz"` - so edge-tts is transparently routable via the
selector. Pass native strings when calling `edge_tts` directly for precise control.

## Limitations

- Needs network (not offline - `piper_tts` is the offline fallback).
- No voice cloning.
- No SLA; the endpoint may rate-limit or change. For paid/SLA work, use ElevenLabs /
  Google TTS / OpenAI / DashScope / Doubao.
- No SSML passthrough in this tool (Standard scope).

## Positioning

edge-tts is the **preferred free fallback** for paid TTS providers. Every paid TTS
tool's `fallback_tools` lists `edge_tts` immediately before `piper_tts`, so a
missing-key paid provider falls through to edge-tts (online, free, high quality)
before piper (offline floor).
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_edge_tts.py -v`
Expected: PASS (`test_edge_tts_contract`, `test_edge_tts_registry_discovery`).

- [ ] **Step 7: Verify dep + skill are in place**

Run: `grep -n "edge-tts" requirements.txt && test -f .agents/skills/edge-tts/SKILL.md && echo OK`
Expected: prints the `edge-tts>=6.1` line, then `OK`.

- [ ] **Step 8: Commit**

```bash
git add tools/audio/edge_tts.py tests/test_edge_tts.py requirements.txt .agents/skills/edge-tts/SKILL.md
git commit -m "feat(tts): scaffold edge-tts provider (contract, dep, skill)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Implement EdgeTTS execution (mocked generate + empty-text)

**Files:**
- Modify: `tools/audio/edge_tts.py` (replace stub `execute` with real `execute` + helpers)
- Test: `tests/test_edge_tts.py`

**Interfaces:**
- Consumes: `edge_tts.Communicate(text, voice, rate=, volume=, pitch=).save(path)` (async); `tools.analysis.audio_probe.probe_duration(path) -> float | None`.
- Produces: `EdgeTTS.execute(inputs) -> ToolResult` with `data={provider, voice, rate, volume, pitch, text_length, audio_duration_seconds, output, format}`, `artifacts=[output_path]`, `model=voice`. Coerces numeric `speed`/`speaking_rate`/`pitch` to SSML strings.

- [ ] **Step 1: Write the failing execution tests**

Append to `tests/test_edge_tts.py`:

```python
class _FakeCommunicate:
    """Stand-in for edge_tts.Communicate; writes a fake mp3 body."""

    def __init__(self, text, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    async def save(self, path):
        from pathlib import Path

        Path(path).write_bytes(b"ID3\x03\x00\x00\x00fake-mp3-body")


def test_edge_tts_generate_mocked(monkeypatch, tmp_path):
    import edge_tts
    import tools.analysis.audio_probe as ap
    from tools.audio.edge_tts import EdgeTTS

    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    monkeypatch.setattr(ap, "probe_duration", lambda p: 1.23)

    tool = EdgeTTS()
    out = tmp_path / "out.mp3"
    result = tool.execute(
        {
            "text": "你好，世界",
            "voice": "zh-CN-XiaoxiaoNeural",
            "speed": 1.2,  # selector-style numeric -> "+20%"
            "pitch": 5,  # selector-style numeric -> "+5Hz"
            "output_path": str(out),
        }
    )
    assert result.success
    assert result.artifacts == [str(out)]
    assert result.data["format"] == "mp3"
    assert result.data["voice"] == "zh-CN-XiaoxiaoNeural"
    assert result.data["rate"] == "+20%"
    assert result.data["volume"] == "+0%"
    assert result.data["pitch"] == "+5Hz"
    assert result.data["audio_duration_seconds"] == 1.23
    assert result.data["text_length"] == len("你好，世界")
    assert result.cost_usd == 0.0
    assert result.model == "zh-CN-XiaoxiaoNeural"


def test_edge_tts_string_params_pass_through(monkeypatch, tmp_path):
    import edge_tts
    import tools.analysis.audio_probe as ap
    from tools.audio.edge_tts import EdgeTTS

    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    monkeypatch.setattr(ap, "probe_duration", lambda p: 0.5)

    tool = EdgeTTS()
    out = tmp_path / "str.mp3"
    result = tool.execute(
        {
            "text": "hello",
            "rate": "-10%",
            "volume": "-20%",
            "pitch": "+5Hz",
            "output_path": str(out),
        }
    )
    assert result.success
    assert result.data["rate"] == "-10%"
    assert result.data["volume"] == "-20%"
    assert result.data["pitch"] == "+5Hz"


def test_edge_tts_empty_text():
    from tools.audio.edge_tts import EdgeTTS

    tool = EdgeTTS()
    result = tool.execute({"text": "   "})
    assert not result.success
    assert "empty" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_edge_tts.py::test_edge_tts_generate_mocked tests/test_edge_tts.py::test_edge_tts_empty_text -v`
Expected: FAIL - `test_edge_tts_generate_mocked` asserts `result.success` but stub returns `success=False, error="edge-tts execute not yet implemented"`; `test_edge_tts_empty_text` likewise does not contain "empty".

- [ ] **Step 3: Replace the stub `execute` with the real implementation**

In `tools/audio/edge_tts.py`, replace this block:

```python
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        # Scaffold: real implementation added in Task 2.
        return ToolResult(
            success=False, error="edge-tts execute not yet implemented"
        )
```

with:

```python
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error="edge-tts not installed. " + self.install_instructions,
            )

        import time

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"edge-tts failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = 0.0
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        import asyncio
        from pathlib import Path

        from tools.analysis.audio_probe import probe_duration

        text = inputs.get("text", "")
        if not text.strip():
            return ToolResult(success=False, error="edge-tts: empty text")

        voice = inputs.get("voice", "zh-CN-XiaoxiaoNeural")
        rate = self._coerce_rate(inputs)
        volume = self._coerce_volume(inputs)
        pitch = self._coerce_pitch(inputs)

        output_path = Path(inputs.get("output_path", "edge_tts.mp3"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            asyncio.run(
                asyncio.wait_for(
                    self._synthesize(
                        text, voice, rate, volume, pitch, output_path
                    ),
                    timeout=300,
                )
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, error="edge-tts timed out after 300s"
            )

        if not output_path.exists():
            return ToolResult(
                success=False,
                error=f"edge-tts produced no output: {output_path}",
            )

        audio_duration = probe_duration(output_path)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "voice": voice,
                "rate": rate,
                "volume": volume,
                "pitch": pitch,
                "text_length": len(text),
                "audio_duration_seconds": (
                    round(audio_duration, 2) if audio_duration else None
                ),
                "output": str(output_path),
                "format": "mp3",
            },
            artifacts=[str(output_path)],
            model=voice,
        )

    @staticmethod
    def _coerce_rate(inputs: dict[str, Any]) -> str:
        raw = inputs.get("rate")
        if raw is None:
            for key in ("speed", "speaking_rate"):
                if inputs.get(key) is not None:
                    raw = inputs[key]
                    break
        if raw is None:
            return "+0%"
        if isinstance(raw, (int, float)):
            return f"{round((float(raw) - 1.0) * 100):+d}%"
        return str(raw)

    @staticmethod
    def _coerce_volume(inputs: dict[str, Any]) -> str:
        raw = inputs.get("volume")
        if raw is None:
            return "+0%"
        if isinstance(raw, (int, float)):
            return f"{round((float(raw) - 1.0) * 100):+d}%"
        return str(raw)

    @staticmethod
    def _coerce_pitch(inputs: dict[str, Any]) -> str:
        raw = inputs.get("pitch")
        if raw is None:
            return "+0Hz"
        if isinstance(raw, (int, float)):
            return f"{int(raw):+d}Hz"
        return str(raw)

    @staticmethod
    async def _synthesize(
        text: str,
        voice: str,
        rate: str,
        volume: str,
        pitch: str,
        output_path: Path,
    ) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            text, voice, rate=rate, volume=volume, pitch=pitch
        )
        await communicate.save(str(output_path))
```

Note: `import time` and `import asyncio` are done inside methods to keep the module's top-level imports unchanged from Task 1 (no need to edit the import block). `Path` is imported inside `_generate` for the same reason. `ToolStatus` is already imported at the top in Task 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_edge_tts.py -v`
Expected: PASS - all of `test_edge_tts_contract`, `test_edge_tts_registry_discovery`, `test_edge_tts_generate_mocked`, `test_edge_tts_string_params_pass_through`, `test_edge_tts_empty_text`.

- [ ] **Step 5: Commit**

```bash
git add tools/audio/edge_tts.py tests/test_edge_tts.py
git commit -m "feat(tts): implement edge-tts execution (async synthesize, param coerce)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Rewire paid TTS fallback chains (edge_tts before piper_tts)

**Files:**
- Modify: `tools/audio/openai_tts.py:41-42`
- Modify: `tools/audio/elevenlabs_tts.py:41-42`
- Modify: `tools/audio/google_tts.py:49-50`
- Modify: `tools/audio/dashscope_tts.py:46-51`
- Modify: `tools/audio/doubao_tts.py:43-44`
- Test: `tests/test_edge_tts.py`

**Interfaces:**
- Consumes: `EdgeTTS` exists and registers (Task 1).
- Produces: every paid TTS tool's `fallback_tools` lists `"edge_tts"` immediately before `"piper_tts"`. `find_fallback` returns the first AVAILABLE candidate, so a missing-key paid tool falls through to `edge_tts` (free, online) before `piper_tts` (offline floor).

- [ ] **Step 1: Write the failing fallback-chain test**

Append to `tests/test_edge_tts.py`:

```python
def test_paid_tts_fallbacks_prefer_edge_before_piper():
    from tools.audio.dashscope_tts import DashscopeTTS
    from tools.audio.doubao_tts import DoubaoTTS
    from tools.audio.elevenlabs_tts import ElevenLabsTTS
    from tools.audio.google_tts import GoogleTTS
    from tools.audio.openai_tts import OpenAITTS

    for cls in (OpenAITTS, ElevenLabsTTS, GoogleTTS, DashscopeTTS, DoubaoTTS):
        ft = cls.fallback_tools
        assert "edge_tts" in ft, f"{cls.name} missing edge_tts fallback"
        assert "piper_tts" in ft, f"{cls.name} missing piper_tts fallback"
        assert ft.index("edge_tts") < ft.index("piper_tts"), (
            f"{cls.name}: edge_tts must come before piper_tts"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_edge_tts.py::test_paid_tts_fallbacks_prefer_edge_before_piper -v`
Expected: FAIL - `AssertionError: openai_tts missing edge_tts fallback` (none of the five tools list `edge_tts` yet).

- [ ] **Step 3: Rewire `openai_tts.py`**

In `tools/audio/openai_tts.py`, replace:

```python
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
```

with:

```python
    fallback = "edge_tts"
    fallback_tools = ["edge_tts", "piper_tts"]
```

- [ ] **Step 4: Rewire `elevenlabs_tts.py`**

In `tools/audio/elevenlabs_tts.py`, replace:

```python
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "piper_tts"]
```

with:

```python
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "edge_tts", "piper_tts"]
```

- [ ] **Step 5: Rewire `google_tts.py`**

In `tools/audio/google_tts.py`, replace:

```python
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "elevenlabs_tts", "piper_tts"]
```

with:

```python
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "elevenlabs_tts", "edge_tts", "piper_tts"]
```

- [ ] **Step 6: Rewire `dashscope_tts.py`**

In `tools/audio/dashscope_tts.py`, replace:

```python
    fallback_tools = [
        "doubao_tts",
        "elevenlabs_tts",
        "openai_tts",
        "piper_tts",
    ]
```

with:

```python
    fallback_tools = [
        "doubao_tts",
        "elevenlabs_tts",
        "openai_tts",
        "edge_tts",
        "piper_tts",
    ]
```

(Leave its `fallback = "piper_tts"` line unchanged - only the list changes.)

- [ ] **Step 7: Rewire `doubao_tts.py`**

In `tools/audio/doubao_tts.py`, replace:

```python
    fallback = "google_tts"
    fallback_tools = ["google_tts", "elevenlabs_tts", "openai_tts", "piper_tts"]
```

with:

```python
    fallback = "google_tts"
    fallback_tools = ["google_tts", "elevenlabs_tts", "openai_tts", "edge_tts", "piper_tts"]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_edge_tts.py::test_paid_tts_fallbacks_prefer_edge_before_piper -v`
Expected: PASS.

- [ ] **Step 9: Run the full test file to confirm nothing regressed**

Run: `pytest tests/test_edge_tts.py -v`
Expected: PASS - all 6 tests.

- [ ] **Step 10: Commit**

```bash
git add tools/audio/openai_tts.py tools/audio/elevenlabs_tts.py tools/audio/google_tts.py tools/audio/dashscope_tts.py tools/audio/doubao_tts.py tests/test_edge_tts.py
git commit -m "feat(tts): make edge_tts the preferred free fallback before piper_tts

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Network integration test (gated) + preflight verification

**Files:**
- Test: `tests/test_edge_tts.py`

**Interfaces:**
- Consumes: the real `edge-tts` endpoint (network). `EdgeTTS.execute` (Task 2).
- Produces: a gated end-to-end test proving a real clip is generated; a preflight check proving the registry shows edge-tts as available.

- [ ] **Step 1: Add the gated network test**

Append to `tests/test_edge_tts.py`:

```python
@pytest.mark.skipif(
    not os.environ.get("OPENMONTAGE_NETWORK_TESTS"),
    reason="set OPENMONTAGE_NETWORK_TESTS=1 to run network TTS test",
)
def test_edge_tts_network_generate(tmp_path):
    from tools.audio.edge_tts import EdgeTTS

    tool = EdgeTTS()
    out = tmp_path / "net.mp3"
    result = tool.execute({"text": "你好，世界", "output_path": str(out)})
    assert result.success, result.error
    assert out.exists()
    assert (
        result.data["audio_duration_seconds"]
        and result.data["audio_duration_seconds"] > 0
    )
```

- [ ] **Step 2: Run the full suite (network test skipped by default)**

Run: `pytest tests/test_edge_tts.py -v`
Expected: PASS - 6 tests pass, `test_edge_tts_network_generate` SKIPPED.

- [ ] **Step 3: Optionally run the network test for real**

Run: `OPENMONTAGE_NETWORK_TESTS=1 pytest tests/test_edge_tts.py::test_edge_tts_network_generate -v`
Expected: PASS (requires network; the edge-tts endpoint returns a real MP3, `audio_duration_seconds > 0`). If the endpoint is rate-limiting or unreachable, this is an environment issue, not a code defect - note it and move on.

- [ ] **Step 4: Verify the preflight menu shows edge-tts as available**

Run:

```bash
python -c "
from tools.tool_registry import registry
registry.discover()
for t in registry.get_by_capability('tts'):
    print(t.name, t.provider, t.get_status().value)
"
```

Expected output includes: `edge_tts edge available` (alongside `piper_tts piper available` and the paid providers as `unavailable`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_tts.py
git commit -m "test(tts): add gated network test for edge-tts

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Standard scope (text+voice+rate/volume/pitch) -> Task 1 schema, Task 2 execution. ✅
- Default `zh-CN-XiaoxiaoNeural` -> Task 1 schema + contract test. ✅
- `runtime=API`, `estimate_cost=0.0`, `dependencies=["python:edge_tts"]`, no `get_status` override -> Task 1. ✅
- `setup_offer` + `get_info()` override -> Task 1 + contract test. ✅
- Direct async lib via `asyncio.run`, 300s `wait_for` -> Task 2. ✅
- Selector numeric coercion -> Task 2 `_coerce_*` + `test_edge_tts_generate_mocked` / `test_edge_tts_string_params_pass_through`. ✅
- Empty-text + timeout + no-output error handling -> Task 2 (`_generate`), empty-text test. (Timeout/no-output covered by code paths, not separate tests - acceptable for Standard scope.) ✅
- Fallback rewiring for all 5 paid tools -> Task 3 + test. ✅
- Layer 3 skill -> Task 1. ✅
- `requirements.txt` `edge-tts>=6.1` -> Task 1. ✅
- Tests: contract + mocked + gated network -> Tasks 1, 2, 4. ✅

**2. Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N". Every code step shows full code; every command shows expected output. ✅

**3. Type consistency:** `EdgeTTS` name/provider/capability consistent across tasks. `_coerce_rate`/`_coerce_volume`/`_coerce_pitch`/`_synthesize` signatures match between Task 2's Edit and the tests. `fallback_tools` entries use the string `"edge_tts"` (the tool `name`), matching how `find_fallback` looks them up by name. Class names `DashscopeTTS` (lowercase 's') match the verified source. ✅

No issues found.
