# Extending Server Configuration

## Add a target

Add the exact upstream model ID under `targets`. The default translating HTTP client is used unless
`llm_client` names an entry under `llm_clients`.

```toml
[targets."provider/model"]
backend = { type = "openai_chat", base_url = "https://example.com/v1", api_key_env = "PROVIDER_API_KEY" }
```

No Rust change is needed unless the target requires a new backend type. In that case, add a
`BackendType` variant and its explicit `build_backend` match arm in `src/config.rs`.

## Add an algorithm

1. Implement and export the algorithm from `libsy`.
2. Add its TOML fields as an `AlgorithmConfig` variant in `src/config.rs`.
3. Construct it in the `build_algorithm` match, resolving target names with `resolve_targets`.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.
