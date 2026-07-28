# Extending Server Configuration

## Add an LLM client and target

Define the upstream once under `llm_clients`, then reference it from targets.

```toml
[llm_clients.provider]
format = "openai_chat"
base_url = "https://example.com/v1"
api_key_env = "PROVIDER_API_KEY"

[targets.model]
id = "provider/model"
llm_client = "provider"
```

To support another wire format, add its `ClientFormat` variant and explicit construction match in
`src/config.rs`. Add a client type only when a second implementation exists.

## Add an algorithm

1. Implement and export the algorithm from `libsy`.
2. Add its TOML fields as a `RouteConfig` variant in `src/config.rs`.
3. Construct it in the `build_algorithm` match, resolving target names with `resolve_targets`.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.

## Configure a learned prefill-probe route

`prefill_probe` uses prompt hidden states from a separate vLLM endpoint to select one of two
completion targets:

```toml
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

The strong and weak values reference entries under `targets`. `probe_model` must match both the
model served by the probe endpoint and the checkpoint's `encoder` metadata. The checkpoint
directory must contain `router.json` and `router.safetensors`.

`probe_timeout_secs` defaults to 30 seconds and must be positive. `cache_capacity` defaults to
4096 successful task decisions and must be positive. Probe or checkpoint inference failures select
the strong target and are not cached.

See the
[vLLM hidden-state probe guide](../../docs/operations/vllm_hidden_state_probe.md)
for the required vLLM connector configuration, artifact lifecycle, and cost-policy semantics.
