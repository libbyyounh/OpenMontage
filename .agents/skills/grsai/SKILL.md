---
name: grsai
description: Grsai image generation guide covering 13 models (11 nano-banana + 2 gpt-image-2), aspectRatio/imageSize rules per model family, reference-image handling, sync vs async polling, and when to pick Grsai over other providers. Use before invoking grsai_image.
metadata:
  author: OpenMontage
  version: "1.0.0"
  tags: grsai, image-generation, nano-banana, gpt-image, chinese
---

# Grsai

Use this skill when working with Grsai image models in OpenMontage. Grsai
exposes 13 image models through one tool, all via the `/v1/api/generate`
endpoint on the 国内节点 (mainland China node).

## Authentication

- Env var: `GRSAI_API_KEY`
- Base URL: `https://grsai.dakka.com.cn` (国内节点, hardcoded)
- Header: `Authorization: Bearer $GRSAI_API_KEY`
- Tools tolerate a `Bearer ` prefix on the key value (auto-stripped).
- Get a key at https://grsai.ai/zh/dashboard/api-keys

## Models (`grsai_image`)

All 13 models are exposed through one tool. Pick via the `model` parameter.

### nano-banana series (11 models)

Google Gemini Image family. Use `image_size` (1K/2K/4K) + `aspect_ratio` (ratio).

| Model | Best for |
|-------|----------|
| `nano-banana` | baseline Gemini Image, general-purpose |
| `nano-banana-fast` | lowest latency, draft-quality |
| `nano-banana-2` | **default** — mainline v2, strong prompt adherence |
| `nano-banana-2-cl` | v2 with character-lock (consistent identity) |
| `nano-banana-2-2k-cl` | v2 character-lock at 2K |
| `nano-banana-2-4k-cl` | v2 character-lock at 4K |
| `nano-banana-pro` | pro tier, higher quality than v2 |
| `nano-banana-pro-vt` | pro with video-to-image variant |
| `nano-banana-pro-cl` | pro with character-lock |
| `nano-banana-pro-vip` | pro VIP, premium quality |
| `nano-banana-pro-4k-vip` | pro VIP at 4K — highest quality nano-banana |

### gpt-image-2 series (2 models)

GPT Image family. **No `image_size`** — `aspect_ratio` accepts ratio OR pixel value.

| Model | aspect_ratio accepts |
|-------|---------------------|
| `gpt-image-2` | ratio (`"16:9"`) or 1K pixel value (`"1024x1024"`) |
| `gpt-image-2-vip` | 1K–4K pixel value (`"1024x1024"`, `"2048x2048"`, `"3840x2160"`) — does NOT accept ratio |

### gpt-image-2-vip pixel-value constraints

When passing `"WxH"` to `gpt-image-2-vip`:
- Max edge ≤ 3840px
- Both edges must be multiples of 16
- Long-edge:short-edge ratio ≤ 3:1
- Total pixels between 655,360 and 8,294,400

Common gpt-image-2-vip pixel values (1K / 2K / 4K tiers):
- `1:1` → `1024x1024` / `2048x2048` / `2880x2880`
- `16:9` → `1280x720` / `2048x1152` / `3840x2160`
- `9:16` → `720x1280` / `1152x2048` / `2160x3840`
- `4:3` → `1152x864` / `2304x1728` / `3264x2448`
- `3:4` → `864x1152` / `1728x2304` / `2448x3264`

For full table, see the upstream API docs.

## Aspect ratios

Common ratios (all models except gpt-image-2-vip):
`auto, 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 5:4, 4:5, 21:9`

nano-banana-2 series **also** supports: `1:4, 4:1, 1:8, 8:1`

## Reference images

Pass reference images via `images` (array of base64 data URI or URL), `image_url`
(single URL), or `image_path` (single local file — auto-encoded to base64 data URI).
The tool merges all three into the API's `images` field. Use cases:

- **Image editing**: 1 reference image + prompt describing the change.
- **Style transfer**: 1 reference image + prompt describing desired output.
- **Multi-image composition**: multiple reference images + prompt describing how to combine.

## reply_type

- `json` (default) — synchronous; the API blocks until the image is ready (up to 300s).
  Use for typical workloads. Most nano-banana models return in <30s; 4K/vip models may take 60-120s.
- `async` — returns a task id immediately; the tool then polls `/v1/api/result?id=<task_id>`
  every 5s with a 600s timeout. Use for 4K jobs or when you want to bound the initial request latency.

## Prompting tips

- **nano-banana models respond well to natural language** — describe the scene, subject,
  lighting, mood. Avoid keyword stuffing.
- **gpt-image-2 models prefer structured prompts** — lead with subject, then style,
  then composition. Works well with both Chinese and English prompts.
- For `nano-banana-2-cl` (character lock): include a clear description of the character
  in the prompt; the model preserves identity across multiple generations with the same character description.
- For `nano-banana-pro-4k-vip`: use detailed, high-fidelity prompts — the model rewards
  specificity. Vague prompts waste the 4K resolution.
- All models support both Chinese and English prompts. Chinese often performs better
  for Chinese-cultural content (古风, 国潮, 短剧); English is fine for general subjects.

## When to pick Grsai

**Pick Grsai when:**
- The brief calls for Google Gemini Image (nano-banana) or GPT Image (gpt-image-2) aesthetics.
- You need character consistency across multiple images — `nano-banana-2-cl` / `nano-banana-pro-cl`.
- You need 4K stills — `nano-banana-pro-4k-vip` or `gpt-image-2-vip` at `3840x2160`.
- The user explicitly asks for Grsai or a nano-banana / gpt-image-2 model.

**Avoid Grsai for:**
- Local/offline generation (Grsai is API-only).
- Seed-controlled reproducibility (Grsai doesn't expose seed).
- When you need Stable Diffusion / FLUX-style fine-grained control (use `flux_image` or `comfyui_image`).

## Output convention

The tool writes the result to `output_path` (default `grsai_<model>.png`) and returns
`ToolResult.data` with:
- `provider: "grsai"`, `model: <model_alias>`
- `prompt`, `aspect_ratio`, `image_size` (nano-banana only)
- `output`, `output_path`, `format`
- ffprobe metadata (file_size_bytes, image dimensions if detectable) via `probe_output()`

Cost is reported as `0.0` — Grsai pricing isn't exposed in the API docs. If pricing
becomes known, override `estimate_cost()` per model.
