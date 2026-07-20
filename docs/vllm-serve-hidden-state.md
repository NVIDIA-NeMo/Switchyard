# vLLM Hidden-State Serving for Local Models

This page describes the hidden-state deployment contract for the learned
`prefill-probe` profile. The hidden-state connector writes prefill activations
to `.safetensors` files and returns the concrete file path in
`kv_transfer_params.hidden_states_path`.

The example profile is
`benchmark/routing-profiles/prefill-probe-local.yaml`. It separates three
roles:

- `probe`: `Qwen/Qwen3.6-35B-A3B`, used only to extract prompt hidden states;
- `weak`: the completion target mapped to the `nemotron-3-super` artifact head;
- `strong`: the completion target mapped to the `opus-4.7` artifact head.

The probe does not need to be one of the completion targets. Its model string
must instead match the artifact's `encoder` metadata exactly. Docker is not
required by the protocol. Use Docker for a reproducible CUDA/vLLM runtime, or
use `vllm serve` directly when the environment contains a build with
`extract_hidden_states` and `ExampleHiddenStatesConnector`.

## Router artifact contract

The router artifact is external deployment data. Switchyard does not package
or download it. Set `PREFILL_ROUTER_ARTIFACT_DIR` to a readable directory with:

```text
router.json
router.safetensors
```

At profile startup, Switchyard validates the artifact metadata, tensor names,
and tensor shapes. The current artifact contract uses Qwen layers `0..39` in
that exact order, hidden size 2048, and one feature block. For each prompt:

1. vLLM exports a `hidden_states` tensor shaped
   `[prompt_tokens, 40, 2048]`.
2. Switchyard averages over the prompt-token dimension independently for every
   layer and hidden dimension.
3. The 40 layer vectors are concatenated in layer order into one 81,920-value
   vector. This is one feature block, not one block per completion target.
4. The artifact's fitted scaler and PCA transform reduce that vector to 200
   values.
5. Five learned `200 -> 256 -> 128 -> 4` trunk members produce four
   correctness probabilities. Switchyard averages probabilities across the
   members and reads only the two heads named by `weak_checkpoint_head` and
   `strong_checkpoint_head`.

The scaler and PCA are fitted training artifacts, not fitted online. No
checkpoint training or online learning occurs in Switchyard.

## Routing policy

The profile min-max normalizes the two configured costs, then computes:

```text
weak_utility   = lambda * P(weak correct)   - (1 - lambda) * normalized_weak_cost
strong_utility = lambda * P(strong correct) - (1 - lambda) * normalized_strong_cost
margin         = weak_utility - strong_utility
```

A non-negative margin becomes public score `1.0` and selects weak. A negative
margin becomes public score `0.0` and selects strong. The learned profile's
threshold is fixed at `0.5`; tune routing only with `lambda`. The sample uses
`lambda: 0.5`, `weak_cost: 0`, and `strong_cost: 1`.

Successful decisions are cached by a hash of the resolved probe input: the
explicit benchmark input when present, otherwise the first string-valued user
instruction. A cache hit reuses the selected tier without another probe.
Probe, artifact, or scoring failures route to strong and are not cached, so a
later matching request retries the probe.

## Explicit probe input for Terminal-Bench

Ordinary clients do not need special request fields. When no override is
present, the profile scores and caches the first string-valued user message.
Terminal-Bench's stock Terminus 2 agent, however, places the raw task
instruction inside a larger first-user message containing its command protocol
and current terminal state. That wrapped message does not reproduce a router
checkpoint trained on the raw task instruction alone.

For prefill-probe benchmark runs, the repository provides
`benchmark/prefill_probe_terminus_2.py`. Its `PrefillProbeTerminus2` adapter adds
the exact `Terminus2.run()` instruction as this private top-level request field:

```json
{
  "_switchyard_prefill_probe_input": "<exact raw task instruction>"
}
```

The `prefill-probe` profile uses a non-empty string value for both scoring and
the decision-cache key. It removes the field before calling the selected
completion target, so that target still receives the unmodified Terminus
conversation. A present empty or non-string value is rejected as an invalid
request instead of silently falling back to the wrapped message.

Run Harbor from the repository root and select the adapter by import path:

```bash
uv run --no-sync harbor run \
  --agent-import-path benchmark.prefill_probe_terminus_2:PrefillProbeTerminus2 \
  --model openai/router \
  --path /path/to/terminal-bench-2-closed-book \
  --jobs-dir /path/to/jobs \
  --job-name prefill-probe-smoke \
  --n-tasks 1 \
  --n-concurrent 1 \
  --max-retries 0 \
  --yes
```

Replace the usual `--agent terminus-2` option with `--agent-import-path`; do
not pass both. Harbor gives a registered agent name precedence over a custom
import path. The adapter keeps the same explicit input in `extra_body` for all
LLM calls during the task while leaving normal prompt construction unchanged.

## Docker launch

Pick one filesystem path for hidden states and mount it into the container. The
path configured in vLLM's `shared_storage_path` must be the same absolute path
as the profile's `hidden_states_dir`. Both vLLM and Switchyard need access;
vLLM must create files there, and Switchyard must read, lock, and delete them.

The all-layer export was smoke-tested with vLLM
`0.23.1rc1.dev672+g93d8f834d`. The command below pins the tested container
digest and runs vLLM with the host user's UID and GID. Matching the host
identity is required because vLLM creates the hidden-state files with mode
`0600`; a container-owned file cannot be read or deleted by host-side
Switchyard.

```bash
export HIDDEN_STATES_DIR=/tmp/switchyard-hidden-states
export HF_CACHE_DIR=/tmp/vllm-hf-cache
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER="${USER:-switchyard}"
mkdir -p "${HIDDEN_STATES_DIR}" "${HF_CACHE_DIR}"

docker run -d --name vllm_qwen36 \
  --gpus all \
  --ipc=host \
  --user "${HOST_UID}:${HOST_GID}" \
  -e USER="${HOST_USER}" \
  -e LOGNAME="${HOST_USER}" \
  -e HOME=/tmp/vllm-home \
  -e HF_HOME=/model-cache \
  -p 0.0.0.0:8000:8000 \
  -v "${HF_CACHE_DIR}:/model-cache" \
  -v "${HIDDEN_STATES_DIR}:${HIDDEN_STATES_DIR}" \
  vllm/vllm-openai@sha256:3b7bb15f9f2b13f2f508d94d1900ea40b5be9f96d716ad977bcef742dac464bc \
  Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-num-seqs 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable-chunked-prefill \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]}}}' \
  --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/switchyard-hidden-states"}}'
```

Use host IPC for long-running Docker jobs. The default Docker IPC mode gives the
container a private 64 MiB `/dev/shm`, which can starve vLLM's tensor-parallel
shared-memory broadcast path while hidden-state extraction is enabled.

Do not shorten or reorder `eagle_aux_hidden_state_layer_ids` for this router
artifact. Switchyard validates the number and width of exported layers, but the
file format does not carry layer IDs for it to verify. A different order can
therefore produce a valid shape with incorrect features and routes.

## Direct vLLM CLI launch

The direct CLI form serves the same model without Docker. There is no volume mount; `shared_storage_path` is a host path and clients must be able to read that same path.

```bash
export HIDDEN_STATES_DIR=/tmp/switchyard-hidden-states
mkdir -p "${HIDDEN_STATES_DIR}"

vllm serve Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-num-seqs 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable-chunked-prefill \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]}}}' \
  --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/switchyard-hidden-states"}}'
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
      "hidden_states_path": "/tmp/switchyard-hidden-states",
      "include_output_tokens": false
    }
  }'
```

Read the path from the response rather than assuming a filename. vLLM may choose the concrete safetensors file name.

```bash
uv run --with safetensors python - <<'PY_INNER'
from pathlib import Path
from safetensors import safe_open

path = Path("/tmp/switchyard-hidden-states")
files = sorted(path.glob("*.safetensors"), key=lambda item: item.stat().st_mtime)
if not files:
    raise SystemExit("no safetensors files written")

with safe_open(files[-1], framework="numpy") as handle:
    for key in handle.keys():
        tensor = handle.get_tensor(key)
        print(files[-1], key, tensor.shape, tensor.dtype)
PY_INNER
```

The inspection command leaves the file in place. During normal routing,
Switchyard deletes a hidden-state file only after it has locked, read, and
successfully decoded it.

## Latency and capacity implications

Every uncached instruction adds a Qwen prefill, a 40-layer hidden-state export,
filesystem write/read, token mean pooling, scaler/PCA projection, and trunk
inference before the selected completion begins. Mean pooling does more CPU
work than selecting the last prompt token, but it does not add another model
forward pass because vLLM already exports the prompt activations. Exporting all
40 layers and moving the artifact through the filesystem are usually the larger
incremental costs.

Artifact size grows with prompt length. For BF16 Qwen features, the
`hidden_states` payload alone is approximately
`prompt_tokens * 40 * 2048 * 2` bytes. Budget shared-storage capacity and lower
`--max-model-len` if hidden-state extraction leaves insufficient KV-cache
memory.

## Troubleshooting

- `probe response missing kv_transfer_params`: the server is not running with `ExampleHiddenStatesConnector`, or the request did not include `kv_transfer_params`.
- `no safetensors files written`: check that `shared_storage_path` exists and is writable by the vLLM process.
- Artifact encoder mismatch: use the exact probe model named by `router.json`; changing probe checkpoints requires a matching exported router artifact.
- Layer-count or raw-feature-dimension mismatch: export all layers `0..39` in order and confirm the hidden width is 2048.
- `No available shared memory broadcast block found` followed by `RPC call to sample_tokens timed out`: relaunch the Docker container with `--ipc=host`.
- Context-length startup errors from vLLM: lower `--max-model-len` or increase available GPU memory. Do not reduce the captured layers without exporting a matching router artifact.
- Hidden-state extraction does not work with chunked prefill; keep `--no-enable-chunked-prefill` in the launch command.
