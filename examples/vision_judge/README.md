# Image routing with a Cosmos Nano judge

This example runs native Switchyard custom classification in two modes:

| Route | Judge input | Answer input |
|---|---|---|
| `vision/text-judge` | Text; `judge_max_images = 0` | Original text and image |
| `vision/image-judge` | Text and newest image; `judge_max_images = 1` | Original text and image |

Both routes use `nvidia/nvidia/cosmos3-nano-reasoner` as the judge. The custom
prompt asks it to select Cosmos Nano for descriptions and
`openai/openai/gpt-6-astra` for precise spatial tasks. Cosmos uses Chat
Completions; Astra uses Responses. The example does not assume either target's
accuracy or enforce the prompt's preference outside the judge.

The runner sends four requests through the native server. A temporary loopback
proxy forwards the real Inference Hub calls and records model IDs, image hashes,
verdicts, answers, usage, and per-call latency. It verifies the judge's image
count, the original answer image's hash, and agreement between the verdict,
upstream answer model, and response header. No credentials or image bytes are
written to the report. Live model calls require Inference Hub access and incur
inference usage.

## Run

From the repository root, with `NVIDIA_API_KEY` set in your environment:

```bash
cargo build -p switchyard-server
curl --fail --location --output /tmp/vantage-pointing.jpg \
  'https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench/resolve/ad5297f645ba90830478a4c6a72a3a7ab077a2f7/data/pointing/images_annotated/000000_000000__largest_in_class_2.jpg'
uv run python examples/vision_judge/demo.py \
  --image /tmp/vantage-pointing.jpg \
  --output /tmp/vision-judge-results.json
```

The sample comes from the image-only pointing task in
[PhysicalAI-VANTAGE-Bench](https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench),
revision `ad5297f645ba90830478a4c6a72a3a7ab077a2f7`, question
`000000__largest_in_class_2`: “Point to the largest car in the image.”
The second question is a scene-description prompt written for this demo.
The runner checks the pinned JPEG's SHA-256:
`04a9ef879f493b109f51527e72b73350b67927308819e8584a0be1ab28b019bd`.
No video extraction, image resizing, or dataset dependency is needed.

To use the routes directly without the recording proxy:

```bash
target/debug/switchyard-server --config examples/vision_judge/routes.toml \
  --host 127.0.0.1 --port 4000
```

Send ordinary OpenAI Chat requests with image content blocks to either route ID.
Independent image requests use `classify_trigger = "every_request"`; the existing
message-hash affinity fallback uses text only.

## Observed live run

[results.json](results.json) records a run on 2026-09-19 UTC. All four routed
requests and all eight upstream calls returned HTTP 200. In each case the
answer received the original image unchanged.

| Judge input | Task | Selected target | Judge images | Answer images | Judge seconds | Answer seconds |
|---|---|---|---:|---:|---:|---:|
| Text | Description | Cosmos Nano | 0 | 1 | 1.042 | 3.124 |
| Text | Pointing | Astra | 0 | 1 | 6.144 | 5.989 |
| Text + image | Description | Astra | 1 | 1 | 1.495 | 4.696 |
| Text + image | Pointing | Astra | 1 | 1 | 3.893 | 3.526 |

These are single-request timings, including network time, with different cache
conditions. They are not a latency benchmark or a cost comparison. Judge usage
is recorded separately from answer usage.

Cosmos Nano did not consistently follow the routing rubric: it chose Astra for
the image-based description and invented an image-specific rationale in the
text-only pointing case. Its pointing rationales also differed from Astra's
answer. The image-description rationale also misstated the requested task.
The integration checks passed, but judge calibration, prompt adherence,
and answer accuracy remain unvalidated. There is no gold-label scoring here.

Image dimensions/payload budgets, resizing, model modality declarations, and
production judge telemetry remain outside this example. The temporary recorder
is demo instrumentation, not part of the serving architecture.
