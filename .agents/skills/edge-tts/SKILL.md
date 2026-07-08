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
