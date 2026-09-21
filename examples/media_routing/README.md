# Image and video routing through Switchyard

This example runs VANTAGE still-image and video requests through the HTTP server,
runner, core router, LLM client, and media crate. It records the actual upstream
payloads as hashes/counts/dimensions without saving media or credentials.

The [configuration](routes.toml) provides:

- `media/text-judge`: Cosmos Nano sees text, then selects Astra, Qwen, or Gemini.
- `media/vision-judge`: Cosmos Nano sees a 384px still or one 384px midpoint video
  frame, then selects an answer target.
- `media/gemini`: direct Gemini video route using Hub's `file` representation.
- `media/cosmos-native`: direct Cosmos Nano video route using `video_url`.

Astra receives six 640px frames across the full clip. Qwen/Cosmos receive native
video URLs or inline MP4. Gemini receives `file_id` for remote video and `file_data`
for inline video, with `format = "video/mp4"`. Still images reach answer models
unchanged. The same request can therefore give the judge a small preview while
preserving the answer's original media.

`libsy` does not decode media. Preparation runs in `libsy-llm-client`, using
`switchyard-media` and settings under each target's `[media]` table. With the HTTP
server, that client runs on the server host; with embedded Rust, it runs in the
calling application. See the [crate configuration reference](../../crates/switchyard-media/README.md)
for supported settings and limits.

## Run

Install FFmpeg and FFprobe on PATH. Build the server:

```bash
cargo build -p switchyard-server --locked
```

Download the two public samples pinned to one dataset revision:

```bash
curl -L --fail --output /tmp/vantage-pointing.jpg \
  'https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench/resolve/ad5297f645ba90830478a4c6a72a3a7ab077a2f7/data/pointing/images_annotated/000000_000000__largest_in_class_2.jpg'
curl -L --fail --output /tmp/vantage-video.mp4 \
  'https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench/resolve/ad5297f645ba90830478a4c6a72a3a7ab077a2f7/data/vqa/videos/drivesim___Collision___Real___Collision_4.mp4'
```

Set `NVIDIA_API_KEY` in the environment, then run the demo. This makes paid live
inference calls. `uv` installs the demo's Pillow dependency in an isolated environment.

```bash
uv run examples/media_routing/demo.py \
  --image /tmp/vantage-pointing.jpg \
  --video /tmp/vantage-video.mp4 \
  --output /tmp/media-routing-results.json
```

The demo starts a temporary Switchyard server and a loopback recording proxy.
The proxy forwards only Chat/Responses inference calls to Inference Hub. Public
media downloads go directly through the media crate's separate HTTP client.
Nine cases make 15 inference calls when every request succeeds without fallback.

## What is verified

Each routed case checks one judge call and one answer call, the model selected
by the judge, judge image count/dimensions, and unchanged answer media hashes.
For Astra video answers it checks six ordered sample positions spanning the clip.
Direct cases verify Gemini's remote/inline file forms and Cosmos's inline video.
No HTTP-success-only claim is used as proof of benchmark accuracy.

`judge_target_agreement` records whether Cosmos followed the intended prompt
category. This is reported separately from transport verification: the small
judge can choose an unexpected model or give an unreliable visual explanation.
The demo verifies that Switchyard follows the returned decision and sends the
configured media to that model. It does not measure VANTAGE task accuracy.

The output JSON records payload hashes, frame dimensions, selected models, and
verification results. Generated reports are not included in the repository.

## Limits

Frame sampling may miss brief events. Labels are requested seek positions, not
exact presentation timestamps. Resizing changes pixel coordinates; benchmark
pointing/bounding-box answers should retain full-size images or account for that
change in scoring. This demo leaves answer stills unchanged.

Same-model aliases with distinct media policies cannot be mixed in one route
because execution is keyed by model ID. The text/vision judge examples use separate
clients in separate routes. Judge and answer models within each route are distinct.

This is representative image/video routing, not a run of the full VANTAGE dataset.
Dataset task adapters, official scoring, model-specific context limits, and routing
quality evaluation remain separate work. Audio and real-time streaming video are
not processed by this crate.
