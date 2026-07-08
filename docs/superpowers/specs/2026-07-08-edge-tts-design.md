# Edge-TTS Provider Integration — Design

- **Date:** 2026-07-08
- **Topic:** Integrate Microsoft edge-tts as a free, no-key TTS provider in OpenMontage
- **Status:** Approved (brainstorming) → awaiting implementation plan
- **Scope:** Standard — text + voice + rate/volume/pitch + output_path, aligned with `tts_selector`

## 1. Goal

Add `edge-tts` (Microsoft Edge Read-Aloud endpoint) as a first-class TTS provider in
OpenMontage. It is **free, requires no API key, and supports 300+ Azure Neural voices
across 50+ locales** (including high-quality Mandarin). It becomes the default free
fallback for paid TTS providers when their API keys are not configured, replacing
`piper_tts` as the preferred fallback while `piper_tts` remains the offline floor.

## 2. Context (verified in repo today)

- TTS capability has 6 providers registered; only `piper_tts` (local) is currently
  available. Paid providers (`openai`, `elevenlabs`, `google_tts`, `dashscope`,
  `doubao`) are unavailable pending API keys.
- **Discovery is automatic.** `registry.discover()` walks `tools/` via
  `pkgutil.walk_packages`; `tts_selector._providers()` calls
  `registry.get_by_capability("tts")`. Dropping a `BaseTool` subclass with
  `capability="tts"` into `tools/audio/` is sufficient — **no registry or selector
  code changes required.**
- `edge_tts 6.1.12` is already installed locally.
- Reference implementations: `tools/audio/openai_tts.py` (API + streaming +
  `probe_duration`, lazy import) and `tools/audio/piper_tts.py` (local subprocess).
- `setup_offer` pattern: a class attribute + `get_info()` override surfaces an
  unavailable tool as a quick upgrade in the preflight menu (see
  `tools/graphics/comfyui_image.py`).
- Existing paid-TTS fallback chains all terminate in `piper_tts`.

## 3. Non-goals

- No SSML passthrough (Standard scope).
- No `list_voices` operation.
- No automatic language detection / locale-auto voice picking.
- No voice cloning (edge-tts does not support it).
- No retry loop inside `execute()` — `retry_policy` is declarative metadata only,
  matching `openai_tts`.

## 4. Architecture & File Layout

```
tools/audio/edge_tts.py            # EdgeTTS(BaseTool) — sole new code
.agents/skills/edge-tts/SKILL.md   # Layer 3 provider skill (lightweight)
tests/test_edge_tts.py             # contract + mocked generate tests
requirements.txt                   # +1 line: edge-tts>=6.1
```

**Discovery path (no code changes elsewhere):**
`registry.discover()` → imports `tools.audio.edge_tts` → registers `EdgeTTS` →
`tts_selector` picks it up via `get_by_capability("tts")` → appears in preflight
menu. With `edge_tts` installed, TTS availability goes from 1/6 to 2/6.

## 5. `EdgeTTS` Tool Contract

| Field | Value | Notes |
|---|---|---|
| `name` | `edge_tts` | |
| `provider` | `"edge"` | short name, matches tool name |
| `capability` / `tier` | `tts` / `VOICE` | aligns with other TTS tools |
| `runtime` | `API` | honest: it is a network service that happens to be free |
| `stability` / `determinism` | `EXPERIMENTAL` / `DETERMINISTIC` | new tool; deterministic per call |
| `dependencies` | `["python:edge_tts"]` | no env var, no key; `get_status()` uses BaseTool default — **no override** |
| `agent_skills` | `["edge-tts"]` | new Layer 3 skill |
| `fallback` / `fallback_tools` | `"piper_tts"` / `["piper_tts"]` | when edge-tts unavailable (no network), fall back to piper offline |
| `estimate_cost` | `0.0` | free |
| `resource_profile` | cpu=1, ram=256, vram=0, disk=50, **network_required=True** | |
| `retry_policy` | max_retries=2, retryable=`["timeout","connection_error","websocket"]` | declarative only |
| `idempotency_key_fields` | `["text","voice","rate","volume","pitch"]` | |
| `side_effects` | writes audio file; calls Microsoft Edge read-aloud endpoint | |
| `supports` | `{voice_cloning: False, multilingual: True, offline: False, native_audio: True, free: True, ssml: False}` | `free: True` is a signal flag |
| `setup_offer` | class attr + `get_info()` override | surfaces "pip install" upgrade when unavailable |

`best_for`:
- free cloud TTS with no API key
- multilingual narration including Mandarin Chinese
- default free fallback when paid TTS providers lack API keys

`not_good_for`:
- fully offline production (needs network)
- voice cloning
- commercial SLA / guaranteed availability (unofficial endpoint)

## 6. Input Schema & Selector Compatibility

```
text        (string, required)
voice       (string, default "zh-CN-XiaoxiaoNeural")
rate        (string, default "+0%")    # SSML, e.g. "-10%", "+20%"
volume      (string, default "+0%")
pitch       (string, default "+0Hz")   # SSML, e.g. "+5Hz"
output_path (string)
```

**Selector compatibility (coerce in `execute()`):** `tts_selector` passes numeric
params (`speaking_rate`/`speed` as multipliers, `pitch` as -50..50), but edge-tts
wants SSML strings. `execute()` coerces once:

- numeric `rate`/`speed`/`speaking_rate` multiplier → percent string
  (`1.2` → `"+20%"`, `0.9` → `"-10%"`)
- numeric `pitch` → Hz string (`5` → `"+5Hz"`)
- string values pass through unchanged

This makes edge-tts routable through `tts_selector` transparently without duplicating
parameters or breaking the selector contract.

## 7. Execution Flow

`execute(inputs)`:
1. `get_status()` (BaseTool default via `python:edge_tts` dep). Unavailable →
   `ToolResult(success=False, error="edge-tts not installed. " + install_instructions)`.
2. `start = time.time()`; `try _generate(inputs) except Exception → ToolResult(success=False, error=f"edge-tts failed: {exc}")`.
3. Backfill `duration_seconds` and `cost_usd=0.0`.

`_generate(inputs)` (lazy import `edge_tts` + `probe_duration`):
1. Empty-text guard → error if `text.strip()` is empty.
2. Read `voice` (default `zh-CN-XiaoxiaoNeural`), `rate`/`volume`/`pitch`; coerce
   numbers to SSML strings; accept `speaking_rate`/`speed` aliases.
3. `output_path = Path(inputs.get("output_path", "edge_tts.mp3"))`; `mkdir(parents=True, exist_ok=True)`.
4. `asyncio.run(asyncio.wait_for(self._synthesize(...), timeout=300))` — 300s ceiling
   prevents indefinite hangs (matches piper's 300s).
5. Verify output exists → `probe_duration(output_path)` for `audio_duration_seconds`.
6. Return `ToolResult(success=True, data={provider, voice, rate, volume, pitch,
   text_length, audio_duration_seconds, output, format:"mp3"}, artifacts=[str(output_path)], model=voice)`.

`_synthesize(text, voice, rate, volume, pitch, output_path)` (async):
```
communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
await communicate.save(str(output_path))
```

## 8. Error Handling

All failure modes are caught and returned as `ToolResult(success=False, error=...)` —
nothing propagates out of `execute()`:

- **Dependency missing** (`edge_tts` not installed) → install_instructions message.
- **Empty text** → explicit error.
- **Invalid voice name** → edge-tts raises; caught with a hint to check the voice name.
- **Network / WebSocket failure** → caught; `retry_policy` signals retryability to the
  orchestrator (no in-`execute` retry loop).
- **Timeout** (>300s) → `asyncio.TimeoutError` caught → error.

## 9. Fallback Rewiring

Insert `edge_tts` immediately before `piper_tts` in every paid tool's
`fallback_tools` list. `find_fallback` returns the first AVAILABLE candidate, so a
missing-key paid tool falls through to `edge_tts` (free, online) before `piper_tts`
(offline floor). Configured paid providers stay ahead of `edge_tts` so quality wins
when a key is present.

| Tool | Before | After |
|---|---|---|
| `openai_tts` | `["piper_tts"]`, fb=`piper_tts` | `["edge_tts","piper_tts"]`, fb=`edge_tts` |
| `elevenlabs_tts` | `["openai_tts","piper_tts"]` | `["openai_tts","edge_tts","piper_tts"]` (fb unchanged) |
| `google_tts` | `["openai_tts","elevenlabs_tts","piper_tts"]` | `["openai_tts","elevenlabs_tts","edge_tts","piper_tts"]` |
| `dashscope_tts` | `["doubao_tts","elevenlabs_tts","openai_tts","piper_tts"]` | `["doubao_tts","elevenlabs_tts","openai_tts","edge_tts","piper_tts"]` |
| `doubao_tts` | `["google_tts","elevenlabs_tts","openai_tts","piper_tts"]` | `["google_tts","elevenlabs_tts","openai_tts","edge_tts","piper_tts"]` |

## 10. Testing (`tests/test_edge_tts.py`)

No existing TTS tests in the repo; this establishes a lightweight pattern.

1. **Contract test** — tool registers; `capability="tts"`; `provider="edge"`;
   `get_status()==available` (edge_tts installed); `estimate_cost==0.0`; default
   voice present in `input_schema`.
2. **Mocked generate test** — monkeypatch `edge_tts.Communicate` to write a tiny mp3
   fixture; assert `ToolResult.success`, `artifacts` path, `format=="mp3"`, correct
   `data.voice`. No network.
3. **Network test (default skip)** — real generate of a short clip, assert
   `audio_duration_seconds > 0`. Runs only when `OPENMONTAGE_NETWORK_TESTS=1` is set,
   keeping CI hermetic.

## 11. Layer 3 Skill (`.agents/skills/edge-tts/SKILL.md`)

Lightweight (~60–100 lines):
- frontmatter `name: edge-tts` + description (free Edge read-aloud TTS, no key,
  300+ Azure Neural voices).
- Default voice and popular voices per locale.
- `rate`/`volume`/`pitch` SSML string format.
- Selector numeric → string coerce behavior.
- Known limitations: unofficial endpoint (no SLA), needs network, no cloning.
- Positioning: free fallback for paid TTS providers.

## 12. Dependencies

Add to `requirements.txt`: `edge-tts>=6.1` (`aiohttp` arrives as a transitive dep).
Lazy import inside `_generate` means the module imports cleanly even when the package
is absent — the tool registers as UNAVAILABLE without breaking discovery (same pattern
as `openai_tts`).

## 13. Known Limitations & Risks

- **`asyncio.run()` requires no running event loop.** All OpenMontage tools are
  `SYNC` and there is no async precedent in `tools/`, so this is safe today. If an
  async caller is ever introduced, this will need a loop-aware wrapper.
- **Unofficial endpoint.** edge-tts uses the Edge browser read-aloud service, which
  has no SLA and could rate-limit or change. Documented in the skill and
  `not_good_for`; `piper_tts` remains the offline fallback.
- **Runtime classification ambiguity.** `ToolRuntime` has no "free but networked"
  bucket; `API` is the honest choice (it is a cloud call), with `estimate_cost=0.0`
  and `supports.free=True` carrying the free-ness signal.

## 14. Open Questions

None. All design decisions resolved during brainstorming:
- Scope: Standard.
- Default voice: `zh-CN-XiaoxiaoNeural` (Chinese-first).
- Fallback role: edge-tts preferred free fallback, piper offline secondary.
- Implementation approach: A (direct async lib via `asyncio.run`).
