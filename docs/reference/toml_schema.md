# TOML Schema

The native deployment file defines the LLM clients, targets, and routes that a
Switchyard server serves. It is read by `switchyard-server --config`.

Validate a file without starting the server:

```bash
switchyard-server --config routes.toml --dry-run
```

## Minimal Example

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "anthropic/claude-sonnet-4.5"
llm_client = "openrouter"

[routes.default]
id = "switchyard"
type = "passthrough"
target = "strong"
```

`schema_version` must be `1`. Table names under `llm_clients`, `targets`, and
`routes` are local references; clients send the route's `id` as the model name.

The optional top-level `fallback_client` names an entry under `[llm_clients]`. When set, any HTTP
method and path not implemented by Switchyard is forwarded through that client without translating
the path, query string, body, response, or model identifier. The fallback client does not need a
target. Only its `base_url` is used: caller end-to-end headers are forwarded, hop-by-hop headers are
removed, and configured API keys, extra headers, format, and retries are not applied. When omitted,
unmatched paths return `404`.

`schema_version`, `[targets]`, and `[routes]` must all be present, even when a
route reaches no upstream. A file without a `[targets]` table is rejected with
`missing field targets`; an empty `[targets]` table satisfies it.
`[llm_clients]` defaults to empty and may be omitted.

## `[llm_clients.<name>]`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `format` | Yes | — | `openai_chat`, `openai_responses`, or `anthropic_messages`. |
| `base_url` | Yes | — | Upstream base URL. |
| `api_key_env` | No | unset | Name of the environment variable holding the key. Omit to send no authentication. |
| `forward_auth` | No | `false` | Forward the caller's provider credential to this upstream. |
| `extra_headers` | No | `{}` | Custom HTTP headers sent to the model server. Set credentials with `api_key_env` or `forward_auth`; the server rejects headers owned by the selected auth mode. Header names are case-insensitive. |
| `max_retries` | No | `2` | Retry budget, `0`–`10`. |

The TOML never contains the secret itself. `api_key_env` names a variable that
must exist and be non-empty when the server loads.

Set `forward_auth = true` to use each caller's credential instead of a
server-owned key:

```toml
[llm_clients.claude]
format = "anthropic_messages"
base_url = "https://api.anthropic.com"
forward_auth = true
```

`forward_auth` cannot be combined with `api_key_env`. OpenAI clients forward
`authorization`, `chatgpt-account-id`, and `x-openai-fedramp`. Anthropic clients
forward `authorization` or `x-api-key`; for Claude subscription OAuth, they also
forward `oauth-*` values from `anthropic-beta` and remove all other inbound beta
values.

This setting gives `base_url` the caller's login. Enable it only when that
upstream should receive the credential, and use HTTPS unless the upstream runs
on loopback. Forwarding clients do not follow HTTP redirects. Check every
forwarding client used by a route, including classifier and judge targets. The
server rejects an Anthropic forwarding route called through an OpenAI endpoint,
or an OpenAI forwarding route called through an Anthropic endpoint, before it
calls an upstream.

## `[targets.<name>]`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `id` | Yes | — | Exact model ID sent upstream. |
| `llm_client` | Yes | — | Key under `[llm_clients]`. |
| `extra_body` | No | `{}` | Values merged into the upstream request when the request does not already set that key. |

## `[routes.<name>]`

Every route takes the common keys below, plus the keys for its type.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `id` | Yes | — | Public model ID that callers send in requests. |
| `type` | Yes | — | Routing algorithm for this route. |
| `context_window` | No | unset | Positive token count advertised for this route by `GET /v1/models`. Unset values appear as `null`. This does not enforce a request limit. |
| `tool_calling` | No | unset | Whether `GET /v1/models` advertises tool-calling support for this route. Unset values appear as `null`. |
| `reasoning` | No | unset | Whether `GET /v1/models` advertises reasoning support to Codex direct-provider discovery. Unset routes are advertised as non-reasoning. |
| `vision` | No | unset | Whether `GET /v1/models` advertises **image input** to Codex direct-provider discovery. Unset routes are advertised as text-only. This is not cosmetic: Codex reads `input_modalities` from the model card and, when it reads text-only, replaces an attached image with the text `image content omitted because you do not support image input` **before sending**, so a route whose target can see but which does not declare `vision = true` loses the image in the client. Declare it only when every target the route can select accepts images. |

### `noop`

Returns a buffered assistant response containing `OK` without calling an
upstream model. Use it for local smoke tests.

A noop-only deployment reaches no upstream but still needs the `[targets]`
table, which can be empty:

```toml
schema_version = 1

[targets]

[routes.smoke]
id = "noop-route"
type = "noop"
```

### `passthrough`

Sends parent requests to one target. It can also route delegated sub-agent work;
see [Sub-Agent-Aware Routing](../routing_algorithms/subagent_routing.md).

| Key | Required | Meaning |
|---|:---:|---|
| `target` | Yes | Target used for parent and harness-maintenance requests. |
| `subagents` | No | Nested `passthrough` or `llm_classifier` policy used only for delegated sub-agent work. Nested classifiers currently support only `mode = "custom"`. |

### `random`

Splits traffic across targets. See
[Random Routing](../routing_algorithms/random_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `targets` | Yes | — | Target names to choose from. |
| `weights` | No | equal | Finite, non-negative relative weights in `targets` order, with at least one positive value. Invalid weights are rejected at load time. |
| `seed` | No | unset | Reproduces the selection sequence. |

### `prefill_router`

Routes the latest non-empty user message with a checkpoint-backed prefill classifier. Build
`switchyard-server` with `--features prefill-router` and make the prefill router's Python
dependencies available in the active virtual environment.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `targets` | Yes | — | Target names in the exact order expected by the checkpoint outputs. |
| `checkpoint` | Yes | — | Path to the tensor-only router checkpoint. Relative paths use the server's working directory. |
| `device` | No | auto | PyTorch device used for encoder inference, such as `cpu`, `cuda`, or `cuda:0`. |
| `cache_dir` | No | unset | Directory where Hugging Face caches the downloaded encoder and tokenizer. |
| `max_length` | No | `2048` | Maximum tokenized encoder input length; longer prompts are truncated. |
| `batch_size` | No | `32` | Maximum prompts per encoder forward pass. |

```toml
[routes.prefill]
id = "switchyard/prefill"
type = "prefill_router"
targets = ["fast", "strong"]
checkpoint = "/models/router.pt"
```

### `llm_classifier`

Runs one of three judge-backed modes: `capability`, `escalation`, or `custom`.
`classifier_target` and `max_output_tokens` apply to all three.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `mode` | No | `capability` | Classifier behavior. Set it explicitly for new configurations. |
| `classifier_target` | Yes | — | Target the judge is called through. Not a routing destination. |
| `max_output_tokens` | No | `4096` | Maximum completion tokens for the judge verdict. Must be at least `1`. |
| `response_format_type` | No | `json_schema` | Structured-output mode for capability and escalation judges. Use `json_object` when the provider does not support JSON Schema; Switchyard adds the schema to the prompt and validates the verdict locally. Custom mode always uses its configured JSON Schema. |

Capability mode classifies before serving. See
[LLM Classifier Routing](../routing_algorithms/llm_classifier_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `strong_target` | Yes | — | Capable tier. |
| `weak_target` | Yes | — | Efficient tier. |
| `base_threshold` | Yes | — | Lowest solve probability that routes to the weak target. In `[0, 1]`. |
| `threshold_step` | No | `0.0` | Finite, non-negative amount added once for uncertain or unmatched verdicts and twice for unsupported verdicts. `base_threshold + 2 * threshold_step` must be at most `1`. |
| `classify_trigger` | No | `every_request` | When the judge runs. `every_request` judges every request, tool continuations included. `user_turn` judges each new user message and retains that target across intervening tool calls only when requests carry a session ID; without a session ID, it behaves like `every_request`. `new_session` judges once and reuses that target for the session. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message. Requires `classify_trigger = "new_session"`. |
| `recent_turn_window` | No | unset | When unset, the judge sees the opening task and latest user follow-up, when present. When set, it also sees trailing turns. |
| `prompt` | No | packaged prompt | Replaces the capability prompt. The packaged schema is sent separately as structured-output configuration. |

Escalation mode serves the weak target first and judges the completed turn. See
[Escalation-Router Routing](../routing_algorithms/escalation_router_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `strong_target` | Yes | — | Target used after the session latches. |
| `weak_target` | Yes | — | Target served before the latch. |
| `prompt` | No | packaged prompt | Replaces the trajectory-judge prompt. |
| `escalation.confirmations` | No | `2` | Consecutive escalate verdicts required to latch. Above `1` needs a session ID. |
| `escalation.recent_turn_window` | No | `28` | Trailing messages shown to the judge. |
| `escalation.window_message_chars` | No | `500` | Per-message cap inside that window. |

Existing configurations that contain `escalation` but omit `mode` remain valid.

Custom mode validates the judge's JSON against `response_schema`, resolves the
policy selector, and routes to any configured target label.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `targets` | Yes | — | Two or more target names available to the policy. |
| `default_target` | Yes | — | Target used when the judge fails or its verdict cannot be routed. |
| `prompt` | Yes | — | Judge system prompt. The configured inner schema is sent separately as structured-output configuration. |
| `response_schema` | Yes | — | Inner JSON Schema encoded as a TOML string. Switchyard adds the provider wrapper. |
| `policy` | Yes | — | Policy table. `target_selector` accepts a JSON Pointer such as `/decision/target`. |
| `policy.labels` | No | unset | Maps each verdict value onto a configured target name, so several verdicts may share one target. Every value must name a configured target and the table must not be empty. When set it is the whole verdict vocabulary; a verdict outside it falls back to `default_target`. |
| `classify_trigger` | No | `every_request` | When the judge runs. `every_request` judges every request, tool continuations included. `user_turn` judges each new user message and retains that target across intervening tool calls only when requests carry a session ID; without a session ID, it behaves like `every_request`. `new_session` judges once and reuses that target for the session. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message. Requires `classify_trigger = "new_session"`. |
| `recent_turn_window` | No | unset | When unset, the judge sees the opening task and latest user follow-up, when present. When set, it also sees trailing turns. |

Classifier prompts must not contain `{{RESPONSE_SCHEMA}}`. Switchyard supplies
the schema automatically: through the structured-output request in `json_schema`
mode, or in the prompt in `json_object` mode.

### `stage_router`

Scores tool signals to pick a tier per turn. See
[Stage-Router Routing](../routing_algorithms/stage_router_routing.md) for the
optional `handoff_notes` and `classifier` tables and for tuning.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `capable_target` | Yes | — | Capable tier. |
| `efficient_target` | Yes | — | Efficient tier. |
| `picker` | Yes | — | `efficient_first`, or `capable_first` (experimental, unbenchmarked). Tier used when the signals are not confident. |
| `confidence_threshold` | Yes | — | Corroboration a decisive pick needs. In `[0, 1]`. |
| `recent_turn_window` | No | `3` | Trailing tool results the signals are computed over. |
| `capable_system_prompt` | No | unset | System prompt handed to the capable tier. |
| `efficient_system_prompt` | No | unset | System prompt handed to the efficient tier. |
| `classifier.classify_trigger` | No | `every_request` | When the judge runs. See the `llm_classifier` route. `new_session` has no effect here. |
| `classifier.response_format_type` | No | `json_schema` | Structured-output mode for the optional classifier judge. Use `json_object` when the classifier provider does not support JSON Schema; Switchyard adds the schema to the prompt and validates the verdict locally. |
| `subagents` | No | unset | Nested `passthrough` or custom `llm_classifier` policy used only for delegated sub-agent work. See [Sub-Agent-Aware Routing](../routing_algorithms/subagent_routing.md). |

### `composite`

Composes other algorithms, letting one set another's
configuration. Today a classifier sets the tier a stage router falls open to when its own signals are not confident, leaving its scoring and escalation logic untouched. See
[Composite Routing](../routing_algorithms/composite_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `classifier.target` | Yes | — | Target the tier judge is called through. Not a routing destination. |
| `classifier.base_threshold` | Yes | — | `p_solve` floor that still routes to the efficient tier. In `[0, 1]`. |
| `classifier.classify_trigger` | Yes | — | `user_turn` re-picks the tier whenever the user speaks, `new_session` picks once and holds it. `every_request` is rejected here: a judge call per tool step is the cost this route exists to avoid. |
| `classifier.message_hash_fallback` | No | `false` | Retains the tier by hashing the first user message, for clients that send no session ID. Unlike the `llm_classifier` route, this works on either trigger. Conversations opening with the same text share a tier. |
| `stage.capable_target` | Yes | — | Capable tier. |
| `stage.efficient_target` | Yes | — | Efficient tier. |
| `stage.confidence_threshold` | Yes | — | Corroboration a decisive signal needs. In `[0, 1]`. |
| `stage.recent_turn_window` | No | `3` | Trailing tool results the signals are computed over. |
| `stage.capable_system_prompt` | No | unset | System prompt handed to the capable tier. |
| `stage.efficient_system_prompt` | No | unset | System prompt handed to the efficient tier. |
| `subagents` | No | unset | Nested policy used only for delegated sub-agent work. |

The tier is retained per session. A deployment that sends no session ID needs
`classifier.message_hash_fallback = true`, which keys on the first user message
instead. The stage table takes no `picker`: the classifier supplies that tier per turn. A turn the
classifier cannot reach falls open to the efficient tier. Leaving out
`classifier` is recommended: that judge runs ahead of the fall-open tier.

## Validation Errors

`--dry-run` prefixes configuration failures with
`invalid server config <path>:`. Within that wrapper, TOML deserialization
errors start with `failed to parse TOML:`, while errors from validating the
built configuration retain their inner message unchanged.

## Related Documentation

- [CLI Reference](../cli_reference.md)
- [Routing Overview](../routing_algorithms/overview.md)
