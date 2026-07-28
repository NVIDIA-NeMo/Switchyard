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
classifier_target = "model_a"
strong_target = "model_a"
weak_target = "model_b"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "model_a"
```

```bash
export API_KEY="..."
cargo run -p switchyard-server -- --config routes.toml
```

The server logs exactly one structured terminal event per LLM request: successful responses at
`INFO`, 4xx responses at `WARN`, and 5xx responses at `ERROR`. Set
`RUST_LOG=switchyard_server=debug,libsy=debug` to include routing decisions and nested failure
details. A streaming failure is logged separately because it can occur after the response starts.

Target and route table names are local references. A target's `id` is the exact model ID sent
upstream, and a route's `id` is the model clients send to select that algorithm.

Each target references an entry under `llm_clients`. All configured clients use
`TranslatingLlmClient`; supported formats are `openai_chat`, `openai_responses`, and
`anthropic_messages`. Supported algorithms are `noop`, `random`, `passthrough`, and
`llm_classifier`. An `api_key_env` value names an environment variable; the TOML
never contains the secret itself. If omitted, the client sends no authentication.
Target-level `extra_body` values are shallow-merged into the upstream request when
the request does not already contain that key.
`max_retries` defaults to `2` and applies to transport failures, timeouts, HTTP 408/429, and 5xx
responses.

Random-route `weights` are relative, follow target order, and do not need to sum to one. Omit them
for equal weighting. The optional `seed` reproduces the selection sequence for the same call order.

An `llm_classifier` route sends each task to `classifier_target` for a capability verdict, then
routes to `weak_target` or `strong_target`. Beyond the three targets it accepts these keys; only
`base_threshold` is required, and anything the judge cannot decide routes to `strong_target`:

| Key | Default | Meaning |
|---|---|---|
| `base_threshold` | *required* | Lowest solve probability that routes a task to `weak_target`. Raise it to send less traffic to the weak model. |
| `min_confidence` | `0.0` | Lowest judge confidence that permits weak routing. `0.0` disables the gate. |
| `capability_elevated_floor` | unset | Higher solve-probability floor applied only to tasks the judge marks uncertain, unsupported, or unmatched. Unset reuses `base_threshold`. |
| `session_affinity` | `false` | Reuses a session's first routing decision on later turns, so the judge is called once per session rather than once per turn. |
| `message_hash_fallback` | `false` | Extends affinity to clients that send no session header, keying on the first user message. Requires `session_affinity = true`. |

Session affinity retains a decision for the process lifetime, including a `strong_target`
fallback produced while the judge was unreachable. `message_hash_fallback` keys on request
content rather than a session id, so unrelated callers sending identical text share one
assignment.

## Metrics

`GET /metrics` exposes Prometheus text from the server's process-wide OpenTelemetry provider.
Routed-call compatibility metrics are:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard_build_info` | gauge | `version` | Constant `1` for this server version |
| `switchyard_total_requests` | gauge | none | Successful and failed final routed calls |
| `switchyard_total_errors` | gauge | none | Failed final routed calls |
| `switchyard_requests_total` | counter | `model`, optional `tier` | Successful final routed calls |
| `switchyard_errors_total` | counter | `model`, optional `tier` | Failed final routed calls |
| `switchyard_model_call_latency_ms` | histogram | `model`, optional `tier` | Successful final routed-call latency |
| `switchyard_prompt_tokens_total` | counter | `model`, optional `tier` | Input tokens, including cached and cache-creation tokens |
| `switchyard_completion_tokens_total` | counter | `model`, optional `tier` | Output tokens |
| `switchyard_cached_tokens_total` | counter | `model`, optional `tier` | Cached input tokens |
| `switchyard_cache_creation_tokens_total` | counter | `model`, optional `tier` | Cache-creation input tokens |
| `switchyard_reasoning_tokens_total` | counter | `model`, optional `tier` | Reasoning output tokens |
| `switchyard_total_latency_ms` | histogram | `model`, optional `tier` | Full-turn latency for successful routed responses |
| `switchyard_routing_overhead_ms` | histogram | `algorithm` | Algorithm run time minus the call that served it |
| `switchyard_client_responses_total` | counter | `outcome` | Final LLM-route responses |
| `switchyard_upstream_attempts_total` | counter | `outcome`, `code` | Actual upstream HTTP attempts |
| `switchyard_router_retry_recovered_total` | counter | none | Retry recoveries (currently always zero) |

The `tier` label is `strong` or `weak` for a distinguishable built-in LLM-classifier decision and
is omitted for untiered algorithms. Classifier calls are excluded from these families.

`switchyard_total_latency_ms` observes an aggregate when it becomes available or a stream when it
ends cleanly. Its clock starts in a router-wide middleware, before the request body is read and
decoded, so it covers the same span as the Python server's request-ingress-to-completion
measurement. It still excludes connection accept and TLS handshake, which hyper completes before
the server sees the request. The Rust server exports this metric as a histogram, while the Python
server exports its counterpart as a summary; this matches the existing histogram/summary difference
for model-call latency.

`switchyard_routing_overhead_ms` is what routing cost on top of the model call: the algorithm's run
time minus the call that served the request. Classifier calls are not subtracted, so an
LLM-classifier route reports its classification time here while `passthrough` and `random` report
the sub-millisecond cost of picking a target. It carries only `algorithm`, since the number
describes the router and not the target it chose, and a run that served nothing records nothing. Its
buckets start at 0.1 ms via a view in the server; the SDK defaults start at 5 ms.

Both clocks stop when the routed call resolves, which for a streamed response is when the stream
handle arrives rather than when the stream ends, so SSE relay time is in neither term. The Python
summary of the same name measures its total through stream completion, making its streaming values
mostly generation time.

See [CONFIGURATION.md](CONFIGURATION.md) to add an LLM client, target, or algorithm.
