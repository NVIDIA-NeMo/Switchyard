# Extending Server Configuration

## Add an LLM client and target

Define the upstream once under `llm_clients`, then reference it from targets.

```toml
[llm_clients.provider]
format = "openai_chat"
base_url = "https://example.com/v1"
api_key_env = "PROVIDER_API_KEY"
max_retries = 2

[targets.model]
id = "provider/model"
llm_client = "provider"
extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

`extra_body` is target-specific. It shallow-merges top-level provider options into
the outbound request, while explicit request fields win on conflicts.

The `chat_template_kwargs.enable_thinking` example is a provider/model-specific
vLLM option. It is not a portable Switchyard reasoning switch. Use it on a judge
target only when the upstream supports it and would otherwise return the verdict
outside normal assistant `content`.

To support another wire format, add its `ClientFormat` variant and explicit construction match in
`src/config.rs`. Add a client type only when a second implementation exists.

## Route a local Cosmos model as a tool

The demo-only `cosmos_media` client adapts vLLM-Omni's image endpoint to a normal routed model call.
It writes one PNG to `output_dir`, then returns its path as assistant text. Retries must be disabled
because generation has file-producing side effects.

```toml
[llm_clients.cosmos]
format = "cosmos_media"
base_url = "http://127.0.0.1:8000/v1"
max_retries = 0
output_dir = ".switchyard/media"

[targets.cosmos]
id = "nvidia/Cosmos3-Nano"
llm_client = "cosmos"

[routes.media]
id = "switchyard/media"
type = "model_as_tool"
primary_target = "frontier"
media_target = "cosmos"
tool_calling = true
```

`model_as_tool` appends a reserved `generate_media` function with one required string argument,
`prompt`. A matching tool call becomes a second libsy model call to the media target. Otherwise,
the primary response passes through unchanged. Python hosts serve both calls through the same
`Algorithm.run_stream()` interface.

## Add an algorithm

1. Implement and export the algorithm from `libsy`.
2. Add its TOML fields as an `AlgorithmConfig` variant in `src/config.rs`.
3. Construct it in the `build_algorithm` match, resolving target names with `resolve_targets`.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.
