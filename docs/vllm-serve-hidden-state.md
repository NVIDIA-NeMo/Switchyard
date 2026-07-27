# vLLM Hidden-State Serving for Local Models

This page captures operational notes for serving a local vLLM model with hidden-state extraction enabled. The hidden-state connector writes prefill activations to `.safetensors` files and returns the actual file path in `kv_transfer_params.hidden_states_path`.

Docker is not required by the protocol. Use Docker when you want a reproducible CUDA/vLLM runtime; use `vllm serve` directly when the local Python environment has a vLLM build that includes `extract_hidden_states` and `ExampleHiddenStatesConnector`.

## Docker launch

Pick one filesystem path for hidden states and mount it into the container. The container path used in `shared_storage_path` must be the same path clients pass as `kv_transfer_params.hidden_states_path`.

```bash
export HIDDEN_STATES_DIR=/tmp/vllm-hidden-states
export HF_CACHE_DIR=/tmp/vllm-hf-cache
mkdir -p "${HIDDEN_STATES_DIR}" "${HF_CACHE_DIR}"

docker run -d --name vllm_qwen35 \
  --gpus all \
  --ipc=host \
  -p 0.0.0.0:8000:8000 \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -v "${HIDDEN_STATES_DIR}:${HIDDEN_STATES_DIR}" \
  vllm/vllm-openai:latest-cu129 \
  Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable-chunked-prefill \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[39]}}}' \
  --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/vllm-hidden-states"}}'
```

Use host IPC for long-running Docker jobs. The default Docker IPC mode gives the
container a private 64 MiB `/dev/shm`, which can starve vLLM's tensor-parallel
shared-memory broadcast path while hidden-state extraction is enabled.

For `Qwen/Qwen3.6-35B-A3B`, layer `39` is the last hidden-state layer. To capture multiple layers, add each layer id to `eagle_aux_hidden_state_layer_ids`, for example `[0,1,2,39]`. Capturing all layers can make each probe much larger and may require a lower `--max-model-len` to leave enough KV-cache memory.

## Direct vLLM CLI launch

The direct CLI form serves the same model without Docker. There is no volume mount; `shared_storage_path` is a host path and clients must be able to read that same path.

```bash
export HIDDEN_STATES_DIR=/tmp/vllm-hidden-states
mkdir -p "${HIDDEN_STATES_DIR}"

vllm serve Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable-chunked-prefill \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[39]}}}' \
  --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/vllm-hidden-states"}}'
```

Use the direct CLI only after confirming your installed vLLM accepts both `--speculative-config '{"method":"extract_hidden_states",...}'` and `--kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector",...}'`. If those flags fail, use the known container image or install a vLLM build that contains the connector.

## Learned prefill-probe routing

The generic launch examples above capture only layer `39`. The learned
prefill-complexity checkpoint requires all 40 Qwen3.6 layers in ascending
order. For either launch method, replace
`eagle_aux_hidden_state_layer_ids` with:

```json
[
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
  20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
  30, 31, 32, 33, 34, 35, 36, 37, 38, 39
]
```

The configured layer list must match `extraction_layer_ids` in the
checkpoint's `router.json`. Switchyard rejects a checkpoint whose encoder,
layer order, hidden width, PCA shape, trunk shape, or output names do not
match the supported artifact contract.

The example route bundle is
[`benchmark/routing-profiles/prefill-probe-local.yaml`](../benchmark/routing-profiles/prefill-probe-local.yaml).
Point `PREFILL_ROUTER_CHECKPOINT_DIR` at the exported `inference_artifact`
directory containing `router.json` and `router.safetensors`; do not point it
at the parent training experiment. The checkpoint is external deployment
data and is not packaged with Switchyard.

```bash
export NVIDIA_API_KEY=nvapi-...
export VLLM_BASE_URL=http://127.0.0.1:8000
export HIDDEN_STATES_DIR=/tmp/vllm-hidden-states
export PREFILL_ROUTER_CHECKPOINT_DIR=/absolute/path/to/inference_artifact

switchyard --routing-profiles \
  benchmark/routing-profiles/prefill-probe-local.yaml -- serve --port 4000
```

The `probe` target is internal: it supplies hidden states but is not exposed
as a completion model. The `strong` and `weak` targets remain available as
direct completion choices alongside the virtual `prefill-complexity-router`
model.

### Probe input and feature pipeline

For each uncached task, Switchyard finds the first user message whose content
is a string. For a stock Terminus 2 prompt, it extracts the text between
`Task Description:\n` and `\n\nCurrent terminal state:\n`; otherwise it uses
the complete first-user string. System messages and subsequent turns are not
sent to the probe. The original conversation is preserved for the selected
completion model.

The probe returns a `hidden_states` tensor shaped
`[prompt_tokens, 40, 2048]`. Switchyard then:

1. Mean-pools over `prompt_tokens` independently for every layer.
2. Concatenates the 40 pooled vectors in ascending layer order, producing
   `40 * 2048 = 81,920` raw features.
3. Applies the checkpoint's fitted standard scaler and PCA-200 transform,
   producing one 200-dimensional feature block.
4. Runs each member of the five-model `200 -> 256 -> 128 -> 4` ensemble.
5. Applies independent sigmoid links and averages probabilities across the
   ensemble.
6. Reads only the configured `weak_checkpoint_head` and
   `strong_checkpoint_head`.

Switchyard does not train the router, fit PCA, update the checkpoint, or
perform online learning.

### Lambda-controlled routing

The policy first min-max normalizes `weak_cost` and `strong_cost` across the
two configured targets, then computes:

```text
weak_utility =
    lambda * P(weak correct)
    - (1 - lambda) * normalized_weak_cost

strong_utility =
    lambda * P(strong correct)
    - (1 - lambda) * normalized_strong_cost
```

A non-negative `weak_utility - strong_utility` selects weak; a negative
margin selects strong. `lambda` is the only continuous routing knob:
`lambda = 0` uses cost alone, while `lambda = 1` uses predicted correctness
alone. There is no configurable confidence threshold.

Successful decisions are cached in-process by the exact resolved probe text.
Probe, hidden-state, checkpoint, or scoring failures select strong and are not
cached, so a later request can retry the probe. `fallback_target_on_evict` is
separate: it selects the completion tier retried after a context-window
eviction.

## Verify one hidden-state file

Send one Chat Completions request with `max_tokens=1`. The probe should return a `kv_transfer_params.hidden_states_path` value that points at a `.safetensors` file.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Return one short sentence."}],
    "max_tokens": 1,
    "kv_transfer_params": {
      "hidden_states_path": "/tmp/vllm-hidden-states",
      "include_output_tokens": false
    }
  }'
```

Read the path from the response rather than assuming a filename. vLLM may choose the concrete safetensors file name.

```bash
uv run python - <<'PY_INNER'
from pathlib import Path
from safetensors import safe_open

path = Path("/tmp/vllm-hidden-states")
files = sorted(path.glob("*.safetensors"), key=lambda item: item.stat().st_mtime)
if not files:
    raise SystemExit("no safetensors files written")

with safe_open(files[-1], framework="numpy") as handle:
    for key in handle.keys():
        tensor = handle.get_tensor(key)
        print(files[-1], key, tensor.shape, tensor.dtype)
PY_INNER
```

## Troubleshooting

- `probe response missing kv_transfer_params`: the server is not running with `ExampleHiddenStatesConnector`, or the request did not include `kv_transfer_params`.
- `no safetensors files written`: check that `shared_storage_path` exists and is writable by the vLLM process.
- `No available shared memory broadcast block found` followed by `RPC call to sample_tokens timed out`: relaunch the Docker container with `--ipc=host`.
- Context-length startup errors from vLLM: lower `--max-model-len`, reduce the number of captured layers, or increase available GPU memory.
- Hidden-state extraction does not work with chunked prefill; keep `--no-enable-chunked-prefill` in the launch command.
