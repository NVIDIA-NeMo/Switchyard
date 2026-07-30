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
extra_body = { service_tier = "priority" }

[targets.judge]
id = "provider/reasoning-model"
llm_client = "provider"
```

`extra_body` is target-specific. It shallow-merges top-level provider options into
the outbound request, while explicit request fields win on conflicts.

Any target a route consults as a judge — `judge_target`, `classifier_target`, or a
`stage_router` classifier — additionally gets `chat_template_kwargs.enable_thinking = false`,
because a verdict is a boolean the router parses: thinking buys no routing quality while
costing latency on the request path of every unlatched turn. Serving targets are left alone.
Setting `extra_body.chat_template_kwargs` yourself takes that key back, so
`{ enable_thinking = true }` opts a judge into thinking and any other spelling reaches the
provider unchanged.

It also keeps the judge's completion budget honest: the verdict is read from `content`, so a
judge that spends its budget thinking and is cut off before answering has no verdict to read
and fails open to the unescalated tier.

To support another wire format, add its `ClientFormat` variant and explicit construction match in
`src/config.rs`. Add a client type only when a second implementation exists.

## Add an algorithm

1. Implement and export the algorithm from `libsy`.
2. Add its TOML fields as an `AlgorithmConfig` variant in `src/config.rs`.
3. Construct it in the `build_algorithm` match, resolving target names with `resolve_targets`.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.
