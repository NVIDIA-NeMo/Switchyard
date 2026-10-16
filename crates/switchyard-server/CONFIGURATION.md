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
system_prompt = "Follow this model's deployment instructions."
extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

`system_prompt` is prepended when the target is a completion destination. Switchyard
prepares each fallback independently, so a failed target's prompt is not carried to the next one.

`extra_body` is target-specific. It shallow-merges top-level provider options into
the outbound request, while explicit request fields win on conflicts.

The `chat_template_kwargs.enable_thinking` example is a provider/model-specific
vLLM option. It is not a portable Switchyard reasoning switch. Use it on a judge
target only when the upstream supports it and would otherwise return the verdict
outside normal assistant `content`.

To support another wire format, add its `ClientFormat` variant and explicit construction match in
`../switchyard-runner/src/config.rs`. Add a client type only when a second implementation exists.

## Use a decision model as the capability judge

With `quality` and `economy` already defined as LLM targets:

```toml
[decision_clients.jev]
format = "system_one"
endpoint = "https://api.typesafe.ai/v1/systemone"
api_key_env = "TYPESAFE_API_KEY"
timeout_ms = 5000

[decision_targets.judge]
id = "jev-latest"
decision_client = "jev"

[routes.capability]
id = "switchyard/capability"
type = "llm_classifier"
mode = "capability"
classifier_target = "judge"
strong_target = "quality"
weak_target = "economy"

[routes.capability.decision]
cutoff = 0.4
candidates = { a = "quality", b = "economy" }
evidence = { candidates = { a = "Higher-quality model", b = "Lower-cost model" } }
```

- `endpoint` is the full System One URL. Set the named key environment variable
  before startup; caller credentials are not forwarded to the decision provider.
- Candidate labels must cover both tiers. Extra candidates add context, not routes.
  Supply evidence from your evaluations and choose a cutoff using those results.
- `decision.instructions` can replace the default relative-advantage instructions.
  It must preserve the meaning of `advantage` and `no_advantage`. Evidence is passed unchanged.
- The capable tier is selected only when its advantage score exceeds `cutoff`.
  Client failures and deadlines use the capable tier by default; set `fail_open = false`
  on the route to propagate those errors.
- Decision targets cannot serve final answers. Both normal requests and `/v1/decision`
  use this configuration. Omit `decision` to keep the existing LLM judge settings.

## Add an algorithm

1. Implement and export the algorithm from `libsy`.
2. Add its fields as an `AlgorithmSpec` variant in `../switchyard-runner/src/algorithm.rs`.
3. Construct it in that module's builder, resolving names through the caller-supplied target map.
4. Add a parsing test and an end-to-end server test when the algorithm makes LLM calls.
