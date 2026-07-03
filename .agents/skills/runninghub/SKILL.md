---
name: runninghub
description: RunningHub workflow-based image and video generation guide — covers the 9 workflow models (5 image + 4 video), their node schemas, async polling, prompt structure, and per-model best practices. Use before invoking runninghub_image or runninghub_video.
metadata:
  author: OpenMontage
  version: "1.0.0"
  tags: runninghub, image-generation, video-generation, chinese, workflow
---

# RunningHub

Use this skill when working with RunningHub workflow models in OpenMontage.
RunningHub is a Chinese workflow-based platform — each "model" is a pre-published
workflow with a fixed node schema. Tools submit a task, poll for completion, then
download the result.

## Authentication

- Env var: `RUNNINGHUB_API_KEY`
- Base URL: `https://www.runninghub.cn` (hardcoded in `tools/video/_shared.py`)
- Header: `Authorization: Bearer $RUNNINGHUB_API_KEY`
- Tools tolerate a `Bearer ` prefix on the key value (auto-stripped).

## Image Models (`runninghub_image`)

All 5 image workflows are exposed through one tool. Pick via the `model` parameter.

| Alias | Workflow ID | Modes | Best for |
|-------|-------------|-------|----------|
| `duanju` | 2052744677727715329 | text | 短剧 scenes, custom WxH (1K/2K/4K tier) |
| `zimage_portrait` | 2003681895185563650 | text | AI 短剧定妆照, photorealistic portrait |
| `qwen_image_t2i` | 1970396677775499266 | text | Qwen-image with `negative_prompt` + 7 aspect ratios |
| `qwen_image_edit` | 2029488621429989377 | singleImage | Image editing with reference + prompt |
| `zimage_8k` | 2058719340626796546 | text | 8K photorealistic direct output |

### Prompting tips (image)

- **All Chinese-native models** — Chinese prompts typically outperform English for 短剧/portrait workflows. Use Chinese for best results.
- **`duanju`**: dimensions come from `size` tier × `aspect_ratio`. Formula: `width = ratio_w × factor` where factor is 96 (1K), 128 (2K), 256 (4K). Result is clamped to [512, 4096] and rounded to multiple of 8. Use 1K for drafts, 2K for production, 4K only for hero stills.
- **`zimage_portrait`**: prompt-only. Tuned for AI 短剧定妆照 — describe character, wardrobe, lighting, scene. "超真实感" is the model's strength; lean into photorealistic language.
- **`qwen_image_t2i`**: use `negative_prompt` to suppress artifacts ("模糊, 低质量, 变形, 水印"). LoRA fields are pre-zeroed — do not try to enable them. Seed is randomized if not provided; supply a fixed `seed` for reproducibility. 7 aspect ratios supported: `1:1, 3:2, 2:3, 16:9, 9:16, 4:3, 3:4`.
- **`qwen_image_edit`**: requires `image_url` or `image_path` (local file is auto-uploaded). Describe only the intended change — the model preserves unspecified regions. Best for "change background", "alter wardrobe", "swap expression".
- **`zimage_8k`**: prompt-only, no size controls. Model decides dimensions. Best when you need maximum resolution and the brief allows the model to interpret composition.

## Video Models (`runninghub_video`)

All 4 video workflows are **image-to-video**. Text-to-video is not supported.

| Alias | Workflow ID | Endpoint | Durations | Resolutions | Notes |
|-------|-------------|----------|-----------|-------------|-------|
| `wan22` | 1956699246381469698 | ai-app | 5 | 480P | WAN2.2 official accelerated; fixed 848×480 |
| `ltx23` | 2029759632314474498 | ai-app | 5, 10 | 720P | Standard LTX2.3; computes WxH from aspect_ratio |
| `ltx23_long` | 2055155307592077313 | ai-app | 10 | 720P | Multi-shot segmented; same prompt → main + segment nodes |
| `ltx23_four_frames` | 2054820963426021378 | workflow | 5 | 720P | 4 reference images; uses `/run/workflow/` endpoint |

### Prompting tips (video)

- **`wan22`**: prompt is optional — motion-only generation works. 480P only, 5s only. Cheapest option; use for quick motion tests.
- **`ltx23`**: prompt recommended for scene control. `duration=10` doubles cost vs `5`. Resolution 720P, aspect_ratio `16:9` (1280×720) or `9:16` (720×1280).
- **`ltx23_long`**: the same prompt is sent to BOTH the main prompt node (620) AND the segmented prompt node (621) — this matches the upstream TS behavior. To get different prompts per segment, the schema would need extension (currently unified). Best for multi-shot continuity where the same direction applies throughout.
- **`ltx23_four_frames`**: provide 1–4 reference images via `reference_image_urls` or `reference_image_paths` (local files auto-uploaded). If fewer than 4, the last image is repeated to fill. Provide a `prompt` describing the desired motion/transition between frames.
- All video workflows poll every 5s with a 600s (10min) timeout.

## Async behavior

Every RunningHub call follows: **submit → poll → download**.

1. `POST /openapi/v2/run/ai-app/{workflow_id}` (or `/run/workflow/{workflow_id}` for four_frames) → returns `taskId`
2. `POST /openapi/v2/query` with `{taskId}` every 5s → status moves through `PENDING` → `SUCCESS` / `FAILED` / `ERROR`
3. On `SUCCESS`, fetch `results[0].url` and stream bytes directly to `output_path`

`tools/video/_shared.py` exposes `runninghub_submit`, `runninghub_query`, `poll_runninghub`, `upload_image_runninghub`, `runninghub_resolve_media`. Tools use them; agents don't call them directly.

## When to pick RunningHub

**Pick RunningHub when:**
- The brief is Chinese-content short-drama (短剧) — Z-image portrait models are specifically tuned for this.
- You need multi-frame silky transitions (4-image reference) — `ltx23_four_frames` is unique among OpenMontage providers.
- Cost-sensitive 480P video — `wan22` is cheaper than Seedance/Veo.
- The user explicitly asks for RunningHub or one of the workflow-specific aliases.

**Avoid RunningHub for:**
- Text-to-video (not supported — use `seedance_video` or `veo_video`).
- English-only prompts where FLUX/Seedance will outperform.
- Workflows needing seed-controlled reproducibility for video (RunningHub video workflows don't expose seed).
- 1080P+ video (max is 720P).

## Output convention

Both tools write the result file to `output_path` (default `runninghub_<model>.<ext>`) and return `ToolResult.data` with:
- `provider: "runninghub"`
- `model: <workflow_id>` and `model_alias: <alias>` for traceability
- `prompt`, `output`, `output_path`, `format`
- ffprobe metadata (file_size_bytes, duration_seconds for video, video_width/height/codec) via `probe_output()`

Cost is reported as `0.0` — RunningHub pricing isn't exposed in the source workflow definitions. If pricing becomes known, override `estimate_cost()` per model.
