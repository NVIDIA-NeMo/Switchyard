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
2. Add its TOML fields as an `AlgorithmConfig` variant in `src/config.rs`.
3. Construct it in the `build_algorithm` match, resolving target names with `resolve_targets`.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.
