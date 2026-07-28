# vLLM Hidden-State Probe

The Rust server's `prefill_probe` route uses a small vLLM prefill request to extract prompt hidden
states, runs a learned checkpoint on CPU, and selects a weak or strong completion target. Use it
when a trained router can reduce completion cost enough to justify a separate probe-model prefill.

This route is available through `switchyard-server` TOML configuration. It is not a Python
route-bundle algorithm.

## Before you start

You need:

- A vLLM release with
  [hidden-state extraction](https://docs.vllm.ai/en/v0.24.0/features/speculative_decoding/extract_hidden_states/)
  and `ExampleHiddenStatesConnector`.
- A dedicated directory visible at the same path to vLLM and Switchyard. A RAM-backed filesystem
  such as `/dev/shm` avoids persistent disk I/O.
- An exported checkpoint directory containing `router.json` and `router.safetensors`.
- Strong and weak completion targets configured in the Rust server.

The checkpoint metadata's `encoder` must exactly match the configured probe model. Its extracted
layer count and hidden size must also match the tensors produced by vLLM.

## Start the probe endpoint

Create a directory used only for probe artifacts:

```bash
mkdir -p /dev/shm/switchyard-prefill
```

Start vLLM with the hidden-state extraction method and disk connector. Replace the model and layer
IDs with the values used to train the checkpoint:

```bash
vllm serve Qwen/Qwen3-8B \
  --speculative_config '{
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
      "hf_config": {
        "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]
      }
    }
  }' \
  --kv_transfer_config '{
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "shared_storage_path": "/dev/shm/switchyard-prefill",
      "use_synchronization_lock": true
    }
  }'
```

Keep `use_synchronization_lock` enabled. Switchyard acquires the companion `.lock` file before
reading, so it cannot parse a partially written safetensors artifact. Chunked prefill is
incompatible with vLLM hidden-state extraction and must be disabled.

Switchyard does not submit a custom output path. vLLM generates the filename under
`shared_storage_path` and returns it in `kv_transfer_params`, so
`allow_custom_save_path` can remain disabled.

## Configure the Rust server

Define the completion clients and targets as usual, then add the learned route:

```toml
schema_version = 1

[llm_clients.completions]
format = "openai_chat"
base_url = "https://completion-provider.example/v1"
api_key_env = "COMPLETION_API_KEY"

[targets.strong]
id = "provider/strong-model"
llm_client = "completions"

[targets.weak]
id = "provider/weak-model"
llm_client = "completions"

[routes.learned]
id = "switchyard/learned"
type = "prefill_probe"
strong_target = "strong"
weak_target = "weak"
probe_base_url = "http://127.0.0.1:8000/v1"
probe_model = "Qwen/Qwen3-8B"
hidden_states_dir = "/dev/shm/switchyard-prefill"
checkpoint_dir = "/opt/switchyard/router"
strong_checkpoint_head = "opus-4.7"
weak_checkpoint_head = "nemotron-3-super"
lambda = 0.75
weak_cost = 0.25
strong_cost = 1.0
probe_timeout_secs = 30.0
cache_capacity = 4096
```

Validate construction before binding a port:

```bash
cargo run -p switchyard-server -- --config routes.toml --dry-run
```

Then start the server:

```bash
export COMPLETION_API_KEY="..."
cargo run -p switchyard-server -- --config routes.toml
```

Clients select the route by sending `switchyard/learned` as the model.

## Tune the policy

The checkpoint produces a correctness probability for each named output head. The policy selects
the tier with the greater utility:

```text
utility = lambda * correctness_probability - (1 - lambda) * normalized_cost
```

`lambda` must be between `0.0` and `1.0`. At `1.0`, only predicted correctness matters; at `0.0`,
only configured cost matters. Equal utility selects weak.

With exactly two targets, costs are min-max normalized. Only their ordering matters, not the
magnitude of the difference:

- Set `weak_cost < strong_cost` when weak is cheaper.
- Set `weak_cost > strong_cost` when strong is cheaper.
- Equal costs remove the cost penalty from both tiers.

Costs must be finite and non-negative and use the same units. The checkpoint head names must be
distinct entries in `router.json`'s `output_names`.

`cache_capacity` bounds an in-memory LRU of successful decisions. Keys are process-randomized
hashes, so raw task text is not retained. Repeated identical tasks use the cached tier without
calling the probe endpoint. Probe and inference failures select strong and are not cached.

## Artifact lifecycle and failures

For each uncached task, Switchyard:

1. Sends one user message to the probe endpoint with `max_tokens = 1` and prompt-only hidden-state
   extraction.
2. Accepts only a returned `.safetensors` path inside `hidden_states_dir`.
3. Waits up to one second for vLLM's synchronization lock, then reads and validates
   `hidden_states` and optional `token_ids` tensors on Tokio's blocking pool.
4. Token-mean pools `[tokens, layers, hidden]` into one vector per layer.
5. Removes the safetensors file and its companion `.lock`, including when tensor parsing fails.

The HTTP request is bounded by `probe_timeout_secs`. A stale-file sweep runs before each probe and
removes unlocked `.safetensors` files older than five minutes from the dedicated directory. Do not
place unrelated safetensors files there.

The probe task text is not written to Switchyard logs. Treat the temporary hidden-state artifacts
as sensitive and restrict access to the shared directory.
