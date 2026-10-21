# Troubleshooting

Two tables. The first starts from what you see. The second lists settings that
load without error while disabling what you meant to configure. Each check
names where to look: a response header or body, a `/v1/stats` key, a routing
log field, a log line, or a `--dry-run` message.

## Symptoms

| Symptom | Check | Likely causes, most common first | Fix |
|---|---|---|---|
| The first request lands on the efficient model | `x-model-router-selected-model` on the response | Auto and Execution decide from tool results, and a request without tool history has none. This is by design. | Send a request with a critical tool error, as in [Verify routing is active](../getting_started.md#verify-routing-is-active). |
| Every request lands on one model | `curl -s localhost:4000/v1/stats | jq -c '.models | map_values(.calls)'` and `.algorithm_stats.stage_router.routing_decisions` | Chat-only traffic (`fall_open` only); both tiers name the same target or the same model ID; `capable_first` with a threshold the signals cannot reach; one tier is down and the other serves its fallbacks. | See the matching rows in [Silent misconfigurations](#silent-misconfigurations) and [When Auto does not route](../routing_algorithms/overview.md#when-auto-does-not-route). |
| The judge never changes the outcome | `switchyard_classifier_fail_open_total` on `/metrics`, and the log line `judge verdict unavailable; routing without one` with its `reason` | The provider rejects JSON Schema mode (`json_object` needed); the verdict is truncated by `max_output_tokens`; the judge model returns prose instead of JSON. | [Judge model compatibility](../routing_algorithms/llm_classifier_routing.md#judge-model-compatibility). |
| Escalation never latches | The log line `stateful escalation has no session ID; routing state will not persist` | The client sends no session ID. No escalation state survives a request without one, so the route never latches at any `confirmations`. The warning fires only when `confirmations` is 2 or more. | Send `x-switchyard-session-id`. See [Tuning options](../routing_algorithms/escalation_router_routing.md#tuning-options). |
| Requests hang or take much longer than the model needs | Server log for retries and cooldown; `/v1/stats` call counts per model | A judge or tier without `timeout_ms`; a judge call on every tool step; a dead tier retried before fallback. | Give the judge its own `[llm_clients]` entry with `timeout_ms`. On a Task or Custom `llm_classifier` route, set `classify_trigger = "user_turn"` and send a session ID. See the dead-tier row below. |
| HTTP 400 from the upstream when the client sends a thinking level | The upstream error body in the response | A Claude target on an OpenAI wire format behind a gateway. | [Thinking](../integrations/pi.md#thinking) and `omit_body_fields`. |
| HTTP 400 with code `context_length_exceeded` | The response body | Every target on the route overflowed. | [Context-Window Handling](context_window.md). |
| HTTP 400 on requests that contain images | The response body | The route sets `vision = false`. | Remove it, or declare `true` only when every target accepts images. See the [route keys](../reference/toml_schema.md#routesname). |
| The server does not start: `unknown field`, or `have no effect on the stage fallback judge` | The `--dry-run` message | An `auto` route with a tuning key; a typo; a key from another route type; `classifier.classify_trigger` other than the default, or `classifier.message_hash_fallback`, on a `stage_router` route. | [Auto](../reference/toml_schema.md#auto), the [stage_router keys](../reference/toml_schema.md#stage_router), and the [validation errors](../reference/toml_schema.md#validation-errors). |
| The server does not start: two targets `both name model` | The `--dry-run` message | Two targets share a model ID on one client with different `reasoning_effort`, `extra_body`, or `omit_body_fields`. | [Reasoning effort](../routing_algorithms/model_selection.md#reasoning-effort). |
| The client shows zero input tokens on `/v1/messages` | `prompt_tokens` in the routing log | The streamed `message_start` event reports zero usage when the target is on an OpenAI format. The real counts arrive in `message_delta`. | [Streamed token counts](../routing_algorithms/model_selection.md#pin-the-provider-and-endpoint). |
| Cost per run is higher than the single capable model | `cached_tokens` in the routing log on the turn after a model change; cost per arm | Each switch loses the prompt cache; the judge adds a call per turn; the efficient model is too weak and most turns escalate. | [Counter-examples](../routing_algorithms/model_selection.md#counter-examples) and the [inverted-pair check](../routing_algorithms/model_selection.md#check-for-an-inverted-pair). |
| The prompt cache is never read | `cached_tokens` stays `0` on repeated prompts | The gateway does not cache on that endpoint, or the prompt is too short to cache. | [Prompt caching](../integrations/pi.md#prompt-caching). |
| `/v1/stats` has no `tiers` key | The `/v1/stats` body | The Rust server reports decisions under `algorithm_stats.<algorithm>`. | Read `algorithm_stats.stage_router.routing_decisions`, as in [Verify routing is active](../getting_started.md#verify-routing-is-active). |
| The routing log does not say why a model was chosen | A routing-log record | The log records the served model and token counts only. | Use the decision counts in `/v1/stats`. For overrides, the `stage router override` log line also names the reason. See [Decision sources](../routing_algorithms/stage_router_routing.md#decision-sources). |
| Sub-agent requests ignore the `subagents` policy | The log line `this route has sub-agent routing but the calling harness does not send sub-agent identity`. It fires only for Claude Code versions known to lack child identity. For other harnesses, check whether delegated requests carry a session ID and an agent ID. | The harness sends no sub-agent identity, so delegated requests follow the parent route. | [Sub-Agent-Aware Routing](../routing_algorithms/subagent_routing.md). |

## Silent misconfigurations

These load and pass `--dry-run`. Most answer every request with HTTP 200; the
rows that return an error or hold a request open say so.

| Setting | Why it is silent | Warning | Fix |
|---|---|---|---|
| `picker = "capable_first"` with `confidence_threshold` at `0.88` or above | The signals reach at most 0.8741 toward efficient, so they never offload. Between `0.49` and `0.8741` only clean production offloads. | Startup: `uses picker "capable_first", which is experimental`. Nothing about the threshold. | Use `efficient_first`, or lower the threshold. See [Reachable scores per picker](../routing_algorithms/stage_router_routing.md#reachable-scores-per-picker). |
| `capable_target` and `efficient_target` name the same target, or two targets with the same model ID and identical settings | One model serves every decision. | None for the same target. Startup `reuses model id ...; only one target per id is kept` for two targets with one model ID. | Use two different models. See [When Auto does not route](../routing_algorithms/overview.md#when-auto-does-not-route). |
| `llm_classifier` with an `escalation` table, no `mode`, and `base_threshold`, `threshold_step`, `fail_open`, `recent_turn_window`, or `message_hash_fallback` | The route runs in escalation mode because `escalation` is present. The capability keys are accepted and ignored. | None. | Set `mode` explicitly. With `mode = "escalation"` the same keys become a load error. |
| `llm_classifier` capability with the default `classify_trigger = "every_request"` on a tool-calling agent | The judge runs on every tool continuation, so one task with twenty tool calls costs twenty-one judge calls. | None. | `classify_trigger = "user_turn"` with a session ID. See [When the judge runs](../routing_algorithms/llm_classifier_routing.md#when-the-judge-runs). |
| Escalation with clients that send no session ID | No escalation state survives a request without a session ID, so the route never latches. With `confirmations = 1` each judged turn can escalate on its own, at the price of a weak, a judge, and a strong call per turn, and the warning goes away, but nothing is retained. | Runtime, once per route, when `confirmations` is 2 or more: `stateful escalation has no session ID; routing state will not persist`. | Send `x-switchyard-session-id`. |
| `classify_trigger = "user_turn"` or `"new_session"` with clients that send no session ID and no `message_hash_fallback` | Nothing can be retained, so every request is judged. | Runtime, once per route: `affinity is enabled but this request carries no usable identity, so no affinity is applied`. | Send a session ID, or set `message_hash_fallback = true`. |
| `response_format_type = "json_schema"` (the default) with a provider that supports only JSON object mode | Every judge call is rejected by the provider. Task routes to the strong target and Execution falls open to the picker default, both with HTTP 200. Escalation stops the request and returns the provider's error on the first attempt; a 400 is not retried. | Runtime, per call on Task and Execution routes: `judge verdict unavailable; routing without one`, and `switchyard_classifier_fail_open_total` rises. Escalation routes return the error instead. | `response_format_type = "json_object"`. See [Judge model compatibility](../routing_algorithms/llm_classifier_routing.md#judge-model-compatibility). |
| A tier that refuses connections or returns HTTP 408, 429, or 5xx, with the default `max_retries = 2` | Each request retries that tier, then falls through to the route's other target, so one model serves everything, slowly. After the retries fail, a 5-second cooldown sends later requests straight to the other target. A tier that hangs behaves differently: without `timeout_ms` the request waits, and with it the request that reaches the deadline returns `504`. Only requests inside the cooldown that follows fall through. | Runtime log lines for the failed attempts and the fallback. | Fix the endpoint and watch `.models` call counts in `/v1/stats`. Set `timeout_ms` so a hang becomes a `504` instead of a wait. See [`[llm_clients.<name>]`](../reference/toml_schema.md#llm_clientsname). |
| A judge client without `timeout_ms` | A judge that accepts the request and never finishes holds the request open. | None. | Give the judge its own `[llm_clients]` entry with `timeout_ms`. See [Tuning options](../routing_algorithms/llm_classifier_routing.md#tuning-options). |
| `auto` or `stage_router` serving chat-only traffic | No tool results, so the signals abstain and every turn falls open to the efficient tier. | None. | Use Task or Composite. See [When routing will not help](../routing_algorithms/overview.md#when-routing-will-not-help). |
| `capable_first_threshold` on a standalone `stage_router` route | Only a composite classifier selects the tier that uses it. | None. | Use `confidence_threshold`, or a composite route. See the [knob reference](../routing_algorithms/stage_router_routing.md#knob-reference). |
| A custom `llm_classifier` with one routing target | Every judge call is paid for and no choice is made. | Startup: `custom classifier has only one routing target; judge calls add cost without a routing choice`. | Add a second target, or use `passthrough`. |

### Two of these rows, as files that pass `--dry-run`

The capability keys below are ignored because the route runs in escalation
mode:

```toml
[routes.agent]
id = "agent"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5          # ignored: the escalation table selects escalation mode
threshold_step = 0.1          # ignored
escalation = { confirmations = 2 }
```

Add `mode = "escalation"` and the same file fails to load with
`mode escalation cannot use capability routing settings`.

The threshold below can never be reached by the signals, so this route serves
the capable tier for every ordinary turn:

```toml
[routes.stage]
id = "stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "capable_first"
confidence_threshold = 0.9    # the signals reach 0.8741 at most toward efficient
```

## Related

- [Verify routing is active](../getting_started.md#verify-routing-is-active)
- [Tuning](../tuning/index.md)
- [Model Selection](../routing_algorithms/model_selection.md)
- [TOML Schema](../reference/toml_schema.md)
