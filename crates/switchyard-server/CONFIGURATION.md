# Extending Server Configuration

## Add an LLM client and target

Define the upstream once under `llm_clients`, then reference it from targets. `type` defaults to
`translating`.

```toml
[llm_clients.provider]
format = "openai_chat"
base_url = "https://example.com/v1"
api_key_env = "PROVIDER_API_KEY"

[targets.model]
id = "provider/model"
llm_client = "provider"
```

To support another client implementation or wire format, add the corresponding enum variant and
explicit construction match in `src/config.rs`.

## Add an algorithm

1. Implement and export the algorithm from `libsy`.
2. Add its TOML fields as an `AlgorithmConfig` variant in `src/config.rs`.
3. Construct it in the `build_algorithm` match, resolving target names with `resolve_targets`.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.
