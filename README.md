# pi-text-model-vision

Give text-only models eyes in [Pi](https://github.com/earendil-works/pi-mono). `pi-text-model-vision` overrides the built-in `read` tool so a text-only main model can receive image observations from `opencode-go/mimo-v2.5`; it also lets the model inspect small details by cropping an original-resolution region with a normalized `box`. Ordinary text reads and image overviews retain Pi's native behavior.

The package registers one Pi-visible tool (`read`). It contains no skill, MCP server, Python runtime, or additional model-facing tool.

## Required opencode-go setup

The intended text-only workflow has a hard runtime prerequisite: Pi must already contain an authenticated, image-capable model at exactly:

```text
opencode-go/mimo-v2.5
```

This package does not install opencode-go, create credentials, or bundle account information. Configure opencode-go in Pi before installing the extension, then verify the model catalog:

```bash
pi --list-models opencode-go
```

The output must contain `mimo-v2.5` and declare image input support. In an interactive text-only session, the extension shows a UI warning when the model or authentication is unavailable; a visual `read` call also returns a direct error. Ordinary text reads continue to work.

## Install

Requirements:

- Pi `0.84.1` or newer behavior compatible with its public extension APIs.
- Node.js `>=22.19.0`.
- A configured `opencode-go/mimo-v2.5` as described above.

Install from npm and reload Pi:

```bash
pi install npm:pi-text-model-vision
```

Inside an already running interactive session:

```text
/reload
```

Pi warns that an extension overrides the built-in `read`; this is expected. Disable or remove any other extension that also overrides `read`, because the last loaded override wins.

## Use

Normal calls remain compatible with Pi's built-in schema:

```json
{
  "path": "src/index.ts",
  "offset": 1,
  "limit": 200
}
```

Read an image overview:

```json
{
  "path": "/absolute/path/to/screenshot.png"
}
```

Crop a region from the EXIF-oriented original image using normalized 0–1000 coordinates:

```json
{
  "path": "/absolute/path/to/screenshot.png",
  "box": [120, 180, 880, 720]
}
```

The crop is written as a lossless PNG under a private temporary directory matching `/tmp/pi-text-model-vision-*`, passed back through Pi's native image reader, and removed on session lifecycle cleanup. The result includes its temporary path; use that path for another focus round because each new `box` is relative to the currently observed image.

A typical loop is:

```text
read original overview
→ Mimo reports clarity and uncertain regions
→ main model selects a relevant bbox
→ read original with box
→ Mimo reinspects the source-detail crop
```

## Structured visual report

The nested Mimo request forces one private `report_visual_observation` function call. Its validated report contains:

- `conclusion`
- `visibleText`
- `spatialSummary`
- `clarity`: `clear`, `partial`, or `unreadable`
- `focusRequired`
- up to three `uncertainRegions`, each with bbox, legibility, reason, observed fragment, and answer relevance

Malformed reports are retried once. The main model receives the validated report as neutral JSON inside `<focused_vision_observation>`; no trust warning or crop instruction is injected into that observation.

## Direct attachments

Directly attached images are routed to Mimo before a turn when the current main model is text-only. Original-resolution cropping requires a readable local path; an attachment alone does not guarantee that the original file path is available.

## Configuration

Set environment variables before starting Pi:

- `PI_TEXT_MODEL_VISION_DISABLED=1` disables the extension.
- `PI_TEXT_MODEL_VISION_READ_OVERRIDE=0` keeps Pi's built-in `read` instead of registering the enhanced override.
- `PI_TEXT_MODEL_VISION_INPUT=0` disables direct-attachment routing.
- `PI_TEXT_MODEL_VISION_TOOL_RESULTS=0` disables image-result routing.
- `PI_TEXT_MODEL_VISION_SKIP_TOOLS` is a comma-separated list of tool names whose image results skip the vision sidecar (default `image_generate`). Set it to an empty string to analyze every tool result, including freshly generated images. Skipping `image_generate` avoids a nested vision call — up to `2 x TIMEOUT_MS` — blocking the agent loop right after a slow image generation finishes.
- `PI_TEXT_MODEL_VISION_MAX_IMAGES` controls uniform image sampling, from 1 to 32 (default 16).
- `PI_TEXT_MODEL_VISION_MAX_TOKENS` controls Mimo report output, from 256 to 16384 (default 4096).
- `PI_TEXT_MODEL_VISION_TIMEOUT_MS` controls Mimo timeout (default 180000).
- `PI_TEXT_MODEL_VISION_CACHE_ENTRIES` controls the session LRU, from 0 to 128 (default 32).
- `PI_TEXT_MODEL_VISION_REASONING_EFFORT` is `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` (default `low`).

The former `PI_FOCUSED_VISION_*` and `PI_VISION_ROUTER_*` routing names remain fallback aliases.

## Privacy and security boundary

Image preprocessing and cropping happen locally with Pi and `sharp`. Image bytes are then sent to the configured `opencode-go/mimo-v2.5` provider for interpretation. Do not use the extension for images that must not leave the machine.

The extension never reads or stores provider credentials itself. Pi's model registry resolves authentication and performs the nested request.

## Development

```bash
npm ci
npm run check
npm pack --dry-run
```

The focused test suite covers native text compatibility, EXIF-aware original cropping, temporary-file cleanup, forced structured reports, malformed-report retry, cache behavior, multimodal bypass, and nested usage accounting.

## License

MIT
