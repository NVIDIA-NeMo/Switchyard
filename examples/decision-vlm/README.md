# Local decision-model VLM routing

This example runs a [Decision 3.0 model](https://huggingface.co/collections/vllm-sr/decision-30)
as Switchyard's capability judge. Send a VQA question and images to
`/v1/decision`; Switchyard returns either the Cosmos or Astra destination. The
decision model reads the images. It returns choice probabilities rather than
generated text.

The caller then makes the answer call. This follows the VANTAGE POC's
`routed_vqa.py` flow, where a caller can send video to Cosmos and sampled frames
to Astra. This example stops after the decision and makes no answer-provider calls.

## Run locally

Run from the repository root. Python 3.12 is supported. Start with d3-edge for a
small CPU smoke test. The upstream runtime uses BF16 weights and its PyTorch
reference implementation on CPU. GPU latency needs a separate measurement.

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'transformers==5.17.0' accelerate safetensors pillow opencv-python-headless fastapi uvicorn einops

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "vllm-sr/d3-edge",
    revision="18d61df78684675463af9ae732872478845c5a79",
    local_dir="scratch/d3-edge",
)
PY

export DECISION_API_KEY=local-demo
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python scratch/d3-edge/d3_server.py \
  --model scratch/d3-edge --device cpu --name d3-edge --port 18231 --no-warmup
```

This runs the pinned model repository's own server and Python code. Its
`GET /health` endpoint reports supported modalities. For GPU deployment, install
matching CUDA builds of PyTorch and torchvision, then use `--device cuda:0`.
The optional `flash-linear-attention` package is for GPU use only.

In another terminal:

```sh
export DECISION_API_KEY=local-demo
cargo run -p switchyard-server --locked -- \
  --config examples/decision-vlm/routes.toml --host 127.0.0.1 --port 18232
```

In a third terminal, try one image or a short video:

```sh
.venv/bin/python examples/decision-vlm/route.py \
  --image scratch/d3-edge/assets/example-receipt.png \
  --question 'How was the order paid? A. Card B. Cash C. Gift card'

.venv/bin/python examples/decision-vlm/route.py \
  --video scratch/d3-edge/assets/example-video.mp4 --frames 4 \
  --category 'Motion Trajectory Estimation' \
  --question 'Which way does the square move? A. Left B. Right C. Still'
```

The output includes the chosen target, fallback, image count and wall time.
Repeat `--image` for multiple images. Omit media for a text-only control. The
script samples video frames in time order and resizes images to a 384-pixel
maximum edge by default. Use `--frames` and `--max-edge` to change this budget.
These are caller settings, not a native-video or media-processing API in Switchyard.

## Model choices

The published family includes d3-edge, d3-lite, d3-nano, d3-mini, d3-flash and d3,
with a shared System One HTTP format. Use the matching checkpoint and served name;
update `decision_targets.judge.id` in the TOML. Pin a model revision when
downloading it. Larger models need more memory. The full d3 checkpoint is about
52 GB before runtime memory.

| Model | Parameters including vision encoder |
| --- | ---: |
| d3-edge | 0.59B |
| d3-lite | 0.85B |
| d3-nano | 2.21B |
| d3-mini | 4.54B |
| d3-flash | 8.39B |
| d3 | 26.09B |

These sizes come from the linked collection's model cards. Local smoke checks
used d3-edge on CPU, d3-nano on CPU and GPU, and d3-flash on GPU. The other variants
were not deployed.

The upstream HTTP server accepts base64 image data URLs. Its Python API also
accepts remote URLs and local paths, but those forms are rejected by its HTTP
server. The example embeds media so the decision server does not fetch files.

## Evaluation limits

`cutoff = 0.4` and the candidate descriptions are illustrative. The score is the
judge's estimate that Astra succeeds while Cosmos fails, not a calibrated solve
probability. A higher cutoff sends fewer requests to Astra. This differs from
the threshold direction in the earlier VANTAGE text-classifier POC.

Choose descriptions and a cutoff from held-out paired model outcomes before
claiming accuracy or savings. Sparse frames may omit the evidence needed for
motion, counting or event-order questions. This example does not download or
evaluate VANTAGE and does not measure answer quality.

The demo sets `fail_open = false` so provider errors fail the request instead of
looking like a successful decision. Switchyard's default capable-tier fallback
is unchanged. Images in selected task messages and retained tool results are
passed in order; the original answer request is preserved. Other media types
are outside this example's tested scope.

The new `DecisionRequest.images` field defaults to empty when deserializing old
requests. Rust callers constructing the struct must add `images: Vec::new()`.
