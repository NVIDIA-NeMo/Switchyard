# switchyard-server

`switchyard-server` exposes libsy algorithms through OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages endpoints. A TOML file explicitly defines the LLM clients, targets, and
algorithm routes served by the process.

```toml
# routes.toml
schema_version = 1

[llm_clients.example]
format = "openai_chat"
base_url = "https://example.com/v1"
api_key_env = "API_KEY"
max_retries = 2

[targets.model_a]
id = "model/a"
llm_client = "example"
extra_body = { service_tier = "priority" }

[targets.model_b]
id = "model/b"
llm_client = "example"

[routes.general]
id = "switchyard/general"
type = "random"
targets = ["model_a", "model_b"]
weights = [1, 3]
seed = 42

[routes.classified]
id = "switchyard/classified"
type = "llm_classifier"
mode = "capability"
classifier_target = "model_a"
strong_target = "model_a"
weak_target = "model_b"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "model_a"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "model_a"
efficient_target = "model_b"
picker = "efficient_first"
confidence_threshold = 0.5
```

```bash
export API_KEY="..."
cargo install --locked switchyard-server
switchyard-server --config routes.toml
```

Ctrl+C and Unix `SIGTERM` stop new connections and allow active requests to drain for up to
`--shutdown-timeout` (30 seconds by default) before they are terminated.

The server logs exactly one structured terminal event per LLM request: successful responses at
`INFO`, 4xx responses at `WARN`, and 5xx responses at `ERROR`. Set
`RUST_LOG=switchyard_server=debug,libsy=debug` to include routing decisions and nested failure
details. A streaming failure is logged separately because it can occur after the response starts.

Target and route table names are local references. A target's `id` is the exact model ID sent
upstream, and a route's `id` is the model clients send to select that algorithm.

Each target references an entry under `llm_clients`. All configured clients use
`TranslatingLlmClient`; supported formats are `openai_chat`, `openai_responses`, and
`anthropic_messages`. Supported algorithms are `noop`, `random`, `passthrough`,
`llm_classifier`, and `stage_router`. The optional `prefill-router` feature also enables
`prefill_router`. An `api_key_env` value names an environment variable; the TOML never contains the
secret itself. If omitted, the client sends no authentication.
A client can set `forward_auth = true` instead of `api_key_env` to send the
caller's credential to the configured upstream. OpenAI clients forward
`authorization`, `chatgpt-account-id`, and `x-openai-fedramp`. Anthropic clients
forward `authorization` or `x-api-key`. Enable this only when every forwarding
client's `base_url` should receive the caller's login. A forwarding route must
be called through the matching provider API.
Target-level `extra_body` values are shallow-merged into the upstream request when
the request does not already contain that key.
`max_retries` defaults to `2` and applies to transport failures, timeouts, HTTP 408/429, and 5xx
responses.

Random-route `weights` are relative, follow target order, and do not need to sum to one. Omit them
for equal weighting. The optional `seed` reproduces the selection sequence for the same call order.

## Session routing log

Pass `--routing-log-file PATH` to append one JSON record after each completed routed response.
Streaming responses are recorded after the stream drains. Each record names the client-facing
`route_id`, routing `algorithm`, served `model`, tier, and token usage so routes sharing a backend
remain distinguishable. When enabled,
`GET /v1/routing/session-stats?session_id=ID` rescans the durable log and returns call and token
totals for that normalized session ID, normally supplied as `x-switchyard-session-id`, grouped by
served model. The legacy `proxy_x_session_id` remains a fallback when no normalized session ID is
present. The endpoint returns `404` when the session has no records and is not registered when
routing logging is disabled.

An `llm_classifier` route sends each task to `classifier_target` for a capability verdict, then
routes to `weak_target` or `strong_target`. Beyond the three targets it accepts these keys; only
`base_threshold` is required, and anything the judge cannot decide routes to `strong_target`:

| Key | Default | Meaning |
|---|---|---|
| `base_threshold` | *required* | Lowest solve probability that routes a task to `weak_target`. Raise it to send less traffic to the weak model. |
| `threshold_step` | `0.0` | Finite, non-negative amount added once for uncertain or unmatched verdicts and twice for unsupported verdicts. `base_threshold + 2 * threshold_step` must be at most `1`. |
| `classify_trigger` | `every_request` | When the judge runs. `every_request` judges every request including tool continuations, `user_turn` judges each new user message and holds that target across the tool calls between, `new_session` judges once and reuses that target for the session. |
| `message_hash_fallback` | `false` | Extends affinity to clients that send no session header, keying on the first user message. Requires `classify_trigger = "new_session"`. |

Session affinity retains a decision for the process lifetime, including a `strong_target`
fallback produced while the judge was unreachable. `message_hash_fallback` keys on request
content rather than a session id, so unrelated callers sending identical text share one
assignment.

A `stage_router` route scores tool-result and agent-progress signals from recent turns to pick a
tier per turn, without an extra classifier call on every turn. `capable_target`,
`efficient_target`, `picker` (`efficient_first` or `capable_first`), and `confidence_threshold`
are required. Optional handoff notes, per-tier system prompts, and a capability-judge fallback are
documented in [Stage-Router Routing](../../docs/routing_algorithms/stage_router_routing.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions |
| `POST` | `/v1/messages` | Anthropic Messages |
| `POST` | `/v1/responses` | OpenAI Responses |
| `POST` | `/v1/decision` | Resolve selected and fallback targets without a post-routing answer call |
| `POST` | `/v1/messages/count_tokens` | Token count from a route's Anthropic target |
| `POST` | `/v1/responses/input_tokens` | Token count from a route's OpenAI Responses target |
| `POST` | `/v1/responses/compact` | Compaction through a route's OpenAI Responses target |
| `ANY` | Any unmatched path | Raw forward through the optional `fallback_client` |
| `GET` | `/v1/models` | Routes served by this deployment |
| `GET` | `/v1/stats` | Per-model usage plus curated algorithm stats |
| `POST` | `/v1/stats/reset` | Clear accumulated stats |
| `GET` | `/metrics` | Prometheus text, see [Metrics](#metrics) |
| `GET` | `/health` | Liveness |

Requests name a route by its `id`, so `POST /v1/chat/completions` with `"model": "switchyard/general"`
routes through the `[routes.general]` entry above. Any of the three request formats can address any
route, and the server translates between them.

Set the top-level `fallback_client` to an entry under `[llm_clients]` to proxy any otherwise
unmatched method and path through that client. The fallback client does not need a target. Requests
and responses are forwarded without translation, including paths, query strings, bodies, and model
identifiers. The caller's end-to-end headers, including authorization, are forwarded; hop-by-hop
headers are removed, and the client's configured API key, extra headers, format, and retry policy
are not applied. Without `fallback_client`, unmatched paths return `404`.

The three model-bearing auxiliary endpoints above remain routed operations: they resolve the route
name to a compatible target, rewrite the upstream model ID, and use that target's configured client.

`POST /v1/decision` accepts `{"input_format": "openai_chat", "request": {...}}`, where the
nested request names the route in `model`. It executes required classifier or judge calls, then
returns the selected target and ordered fallbacks with their model, format, base URL, and
`extra_body`. It does not make a post-routing answer call. Routing-time calls still execute, and
response-dependent algorithms such as escalation and advisor routing may produce an answer while
deciding. When they do, the endpoint includes the buffered answer as `response`, encoded in
`input_format`; otherwise the field is omitted.

For `stage_router`, `algorithm_stats.stage_router` groups routing decisions by source and semantic
target and summarizes its score, confidence, and input-dimension histograms. These values reset
with `/v1/stats/reset`; the process-lifetime counters on `/metrics` remain cumulative.

Token counting selects an Anthropic-format completion target, preferring target names or model IDs
containing `opus`, `sonnet`, then `haiku`. Other ties preserve the route's target order.

## Metrics

`GET /metrics` exposes Prometheus text from the server's process-wide OpenTelemetry provider.
Routed-call compatibility metrics are:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard_build_info` | gauge | `version` | Constant `1` for this server version |
| `switchyard_total_requests` | gauge | none | Successful and failed final routed calls |
| `switchyard_total_errors` | gauge | none | Failed final routed calls |
| `switchyard_requests_total` | counter | `model` | Successful final routed calls |
| `switchyard_errors_total` | counter | `model` | Failed final routed calls |
| `switchyard_model_call_latency_ms` | histogram | `model` | Successful final routed-call latency |
| `switchyard_llm_calls_total` | counter | `algorithm`, `selected_model`, `outcome` | Logical routing and terminal model calls |
| `switchyard_llm_call_duration_ms` | histogram | `algorithm`, `selected_model`, `outcome` | Logical routing and terminal model-call latency |
| `switchyard_prompt_tokens_total` | counter | `model` | Input tokens, including cached and cache-creation tokens |
| `switchyard_completion_tokens_total` | counter | `model` | Output tokens |
| `switchyard_cached_tokens_total` | counter | `model` | Cached input tokens |
| `switchyard_cache_creation_tokens_total` | counter | `model` | Cache-creation input tokens |
| `switchyard_reasoning_tokens_total` | counter | `model` | Reasoning output tokens |
| `switchyard_total_latency_ms` | histogram | `model` | Full-turn latency for successful routed responses |
| `switchyard_routing_overhead_ms` | histogram | `algorithm` | Time spent producing the terminal routing outcome |
| `switchyard_classifier_fail_open_total` | counter | `judge_model`, `reason` | Judge failures that made a classifier route without a verdict |
| `switchyard_client_responses_total` | counter | `outcome` | Final LLM-route responses |
| `switchyard_upstream_attempts_total` | counter | `outcome`, `code` | Actual upstream HTTP attempts |
| `switchyard_router_retry_recovered_total` | counter | none | Upstream operations recovered by a retry |

`switchyard_classifier_fail_open_total` counts requests that still reached a target after the
judge call failed. `judge_model` names the configured judge target, and `reason` is one of eight
fixed error categories.

`switchyard_llm_calls_total` and `switchyard_llm_call_duration_ms` retain the logical call
boundary across routing: libsy records classifier and judge calls, while `libsy-llm-client`
records the terminal call when the routing outcome requires one. A response already produced by
routing is counted only by its original libsy call. Terminal fallback and backend retries remain
one logical call labeled with the algorithm-selected model.

`switchyard_total_latency_ms` observes an aggregate when it becomes available or a stream when it
ends cleanly. Its clock starts in a router-wide middleware, before the request body is read and
decoded, so it measures request ingress through response completion. It still excludes connection
accept and TLS handshake, which hyper completes before the server sees the request.

`switchyard_routing_overhead_ms` records the elapsed time in `run` from `run_started` until
`drive` returns a routing outcome. It includes algorithm execution and any classifier or judge
calls made while driving the algorithm, and it is recorded before a terminal answer call begins.
No duration for the request-serving call is subtracted. If routing itself produced the answer,
that call occurred inside the measured `drive` interval. The metric carries only `algorithm`,
since the duration describes the router rather than the target it chose; a routing failure before
an outcome records nothing. Its buckets start at 0.1 ms via a view in the server; the SDK defaults
start at 5 ms.

See [CONFIGURATION.md](CONFIGURATION.md) to add an LLM client, target, or algorithm.
