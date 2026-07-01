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
- Context-length startup errors from vLLM: lower `--max-model-len`, reduce the number of captured layers, or increase available GPU memory.
- Hidden-state extraction does not work with chunked prefill; keep `--no-enable-chunked-prefill` in the launch command.
