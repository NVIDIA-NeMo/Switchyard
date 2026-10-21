# OpenTelemetry

Reference for the spans, attributes, and metrics emitted by libsy and its LLM client.

## Host setup

Your application installs a `tracing-opentelemetry` subscriber and global OTel meter
provider, and owns exporters, filtering, sampling, context propagation, and flushing.
Libsy installs none of these and sends no telemetry itself. Environment variables
alone do not enable collection. See [OTel Rust setup](https://opentelemetry.io/docs/languages/rust/)
and [OTLP configuration](https://opentelemetry.io/docs/specs/otel/protocol/exporter/).

## Rust metrics without a server

Enable the `metrics` feature on `switchyard-libsy` to collect and query metrics in
process. This needs no HTTP server, observer callback, or external collector.

```toml
switchyard-libsy = { version = "0.3", features = ["metrics"] }
```

Install the provider once, before running requests:

```rust
use switchyard_libsy::metrics::{MetricFilter, Metrics};

let metrics = Metrics::install()?;

// Run requests through switchyard_llm_client::run(..., None).await.
// Consume or drop response streams before querying their final measurements.

let turn = MetricFilter {
    session_id: Some("session-123"),
    agent_id: Some("worker-1"),
    agent_turn_idx: Some(3),
    ..MetricFilter::default()
};
let stats = metrics.stats(turn);
let prompt_tokens = stats.counter_total("switchyard_llm_call_prompt_tokens_total");
let text = metrics.prometheus(turn)?;

let all_stats = metrics.stats(MetricFilter::default());
let all_text = metrics.prometheus(MetricFilter::default())?;
```

Set `user_turn_idx` instead of `agent_turn_idx` to select all inference requests
in that user turn. Set both to require both to match. Unspecified fields match all
values. Series without the requested labels, including process-wide gauges, are
excluded from a filtered result. No matching data returns an empty snapshot;
unknown token counts remain absent.

`stats` returns a serializable `MetricsSnapshot` with every selected counter,
gauge, and histogram. Each series retains its name, labels, value, or histogram
count, sum, and buckets. Metric names use the same spelling as Prometheus.
`counter_total` sums a named counter across the selected series. `gather` exposes
the underlying Prometheus metric families. Queries are cumulative and do not
clear previously recorded values.

The Rust snapshot preserves metric series rather than copying the server's
`/v1/stats` model-summary JSON schema. `prometheus` provides the text representation
used by `/metrics`. Server-only measurements, such as HTTP ingress timing and
cache-eligibility estimates, require that server code to run. Library request
latency is available as `switchyard_inference_duration_ms`, from entry to `run`
through stream completion or drop.

`Metrics::new` creates a collector without installing its provider. Use
`opentelemetry::global::set_meter_provider(metrics.meter_provider().clone())` when
the host owns setup. `install` replaces any existing global meter provider.
Retain the handle while collecting. The default SDK limit is 2000 series per
instrument. For longer histories, use `Metrics::with_cardinality_limit(limit)`
and install its provider explicitly. Once the limit is reached, new series are
combined into an overflow series and lose their original request labels.

The client driver records `switchyard.llm_call.{input_tokens,prompt_tokens,
completion_tokens,total_tokens,cached_tokens,cache_creation_tokens,reasoning_tokens}`
for each call, with
`algorithm`, `selected_model`, and `call_kind` (`routing`, `answer`, or `decision`)
plus request identity. Input tokens exclude cached input. Prompt tokens include
uncached input, cache reads, and cache creation. Total tokens retain the provider
or codec total when available. These counters include routing responses even when they are
reused as the answer or discarded without replay. Available streamed answer usage
is recorded on stream completion or drop, including usage reported before a
stream error. Missing provider usage cannot be recovered. Hosts serving libsy's
offloaded calls without the client driver can use `metrics::record_usage` once
per call in the request's metric context.

The existing `switchyard.prompt_tokens`, `switchyard.completion_tokens`, and cache
and reasoning counters cover successful terminal answers. They are now emitted
by the client driver; the server uses those same counters without recording them
again. Use the `switchyard.llm_call.*` counters to include routing and decision
work. Reasoning tokens are a separate provider detail; do not add them to output
counts without checking whether the provider already includes them.

## Spans and attributes

| Span | Emitted by | Meaning |
|---|---|---|
| `libsy.run` | Libsy | One algorithm run, including routing-time work. OpenInference kind `CHAIN`. |
| `libsy.llm_call` | Libsy driver | Waiting for the host to fulfill an offloaded call. Includes host queueing. OpenInference kind `CHAIN`. |
| `libsy.decision_call` | Libsy driver | An offloaded Decision Model call, including host queueing. OpenInference kind `CHAIN`. |
| `libsy.client_call`, exported as `chat <model_id>` | LLM client driver | One candidate model call, including that candidate's retries. OTel kind `CLIENT`; OpenInference kind `LLM`. |

Hosts driving `run_stream` without the LLM client driver instrument their own model I/O.

`libsy.decision_call` records `algorithm`, `selected_model`, and terminal `outcome`.
Successful replies add available `input_tokens`, `output_tokens`, `total_tokens`,
`reasoning_tokens`, `gen_ai.response.id`, and `gen_ai.response.model` fields.
Unknown values are omitted; request content, answers, and error details are not recorded.

### Routing outcome fields

On `libsy.run`:

| Attribute | Type | Meaning / presence |
|---|---|---|
| `algorithm`, `switchyard.algorithm` | string | Name from `Algorithm::name()`. |
| `switchyard.route` | string | Inbound request model/route, when present. |
| `outcome` | string | `ok` or `error` when the algorithm task resolves. |
| `outcome_id` | string | Successful outcome's ID. `OutcomeMetadata::new` generates a UUIDv7. |
| `selected_model_ids` | string array | Successful outcome's selected model followed by ordered fallbacks. This is a plan, not proof that every model was called. |
| `session_id`, `session.id` | string | Request session ID, when supplied. Both names carry the same value. |
| `agent_id`, `task_id`, `task_kind`, `agent_role`, `correlation_id` | string | Corresponding request metadata, when supplied. |
| `evidence.source`, `evidence.verdict`, `evidence.trigger`, `evidence.reason_code` | string | Known string fields from outcome evidence, when present. |
| `evidence.score`, `evidence.confidence`, `evidence.threshold` | number | Known numeric fields from outcome evidence, when present. |

`RoutingOutcome.metadata` is available in Rust and Python. Its optional JSON evidence
exports only the listed keys with matching types; unknown keys and wrong types are
omitted. Evidence may be absent. Scores and confidence have algorithm-specific meanings.
Failed runs return typed errors, not outcome metadata; successful fail-open decisions
may carry a fixed `reason_code`.

### Model-call fields

The LLM client driver records these on `libsy.client_call`:

| Attribute | Type | Meaning / presence |
|---|---|---|
| `algorithm`, `switchyard.algorithm`, `selected_model` | string | Algorithm and candidate model ID. |
| `switchyard.candidate`, `switchyard.candidate_count` | integer | One-based candidate position and number of candidates. |
| `gen_ai.operation.name` | string | `chat`. |
| `gen_ai.request.model` | string | Requested model; the translating client records the upstream model name. |
| `gen_ai.request.stream` | boolean | Recorded as `true` for streaming requests; otherwise omitted. |
| `gen_ai.request.temperature`, `gen_ai.request.top_p` | number | Sampling values represented in the request IR, when set. |
| `gen_ai.request.top_k`, `gen_ai.request.max_tokens` | integer | Sampling/output limits, when set. |
| `gen_ai.request.reasoning.level`, `gen_ai.output.type` | string | Reasoning effort and recognized output type (`text` or `json`), when set. |
| `gen_ai.conversation.id` | string | Request session ID, when supplied. |
| `server.address`, `server.port` | string, integer | Upstream host and port, recorded by the translating client. |
| `gen_ai.response.id`, `gen_ai.response.model` | string | Values supplied by the upstream response. |
| `gen_ai.response.finish_reasons` | string array | Available normalized stop reasons. |
| `gen_ai.usage.input_tokens` | integer | Input tokens including cache reads and cache creation. |
| `gen_ai.usage.output_tokens` | integer | Output tokens. |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens` | integer | Cache-read and cache-creation input tokens. |
| `gen_ai.usage.reasoning.output_tokens` | integer | Reasoning output tokens. |
| `outcome` | string | `ok`, `error`, or `cancelled`. |
| `error.type`, `error` | string | Failure category/status and error description on this client span. |

`gen_ai.provider.name` is intentionally unset: an endpoint or model name does not
reliably identify the provider. Usage fields are omitted when unavailable, not
invented as zero. Available counts are capped at OTel's signed integer maximum.

## Metrics

Metrics use the `switchyard` meter scope. The tables use OTel instrument names.

Request-scoped metrics also carry `session_id`, `agent_id`, `agent_turn_idx`, and
`user_turn_idx` when a session ID is supplied. The tables below list the other labels.
Process-wide build and total gauges keep their existing labels.

Indices start at zero and are tracked per `(session_id, agent_id)` in the process:

- `agent_turn_idx` advances for each inbound inference request. Internal judge calls,
  answer candidates, retries, and stream events share that request's index.
- `user_turn_idx` advances when the supplied `turn_id` changes or a new trailing user
  input is observed. Tool-result-only messages keep the current user index. Without
  a turn ID, repeated identical input with identical history is treated as a retry.
- An absent agent ID uses `root`. Without a session ID, a supplied agent ID is still
  labeled, but turn indices are omitted.

Counters persist until process restart or `session_final = true`, which releases
all agents in the session after assigning the final request's indices. Concurrent
requests receive indices in the order they enter tracking. These are observed
request indices; joining an existing conversation starts at zero. Compacted or
partial histories can make user boundaries ambiguous without a supplied turn ID.

The LLM client driver and server establish the request context automatically.
Hosts serving `run_stream` directly get algorithm and offloaded-call labels;
custom host metrics can use `metric_context::for_request` and
`metric_context::attributes` to share them. Preserve the context while polling
futures and streams. Identity labels create a series per turn; exporter cardinality
limits still apply.

### Routing and client metrics

| Instrument | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard.runs` | Counter | `algorithm`, `outcome` | Completed algorithm tasks, including failures. |
| `switchyard.run_duration_ms` | Histogram | `algorithm`, `outcome` | Algorithm-task duration in milliseconds. |
| `switchyard.algorithms_in_flight` | UpDownCounter | `algorithm` | Active algorithm tasks; exported as a Prometheus gauge. |
| `switchyard.decisions` | Counter | `algorithm`, `selected_model` | Published routing decisions. |
| `switchyard.llm_calls` | Counter | `algorithm`, `selected_model`, `outcome` | Routing-time LLM calls and each terminal answer candidate, including failures. |
| `switchyard.llm_call_duration_ms` | Histogram | `algorithm`, `selected_model`, `outcome` | One duration sample in milliseconds per offloaded LLM call or terminal answer candidate, including failures; see streaming limits below. |
| `switchyard.decision_calls` | Counter | `algorithm`, `selected_model`, `outcome` | Offloaded Decision Model calls, recorded once on reply, failure, or unfulfilled drop. |
| `switchyard.decision_call_duration_ms` | Histogram | `algorithm`, `selected_model`, `outcome` | One duration sample in milliseconds per Decision Model call, including host queueing. |
| `switchyard.total_requests` | ObservableGauge | none | Process-wide total of successful and failed answer candidates after routing, including reused routing responses. |
| `switchyard.total_errors` | ObservableGauge | none | Process-wide total of failures after routing, including failed answer candidates and reused routing responses. |
| `switchyard.requests` | Counter | `model` | Successful answer candidates or reused routing responses, by model ID. |
| `switchyard.errors` | Counter | `model` | Failed answer candidates or reused routing responses, by model ID. |
| `switchyard.model_call_latency_ms` | Histogram | `model` | Successful answer-candidate duration in milliseconds, including that candidate's retries and stream consumption. Excludes routing time and reused routing responses. |
| `switchyard.routing_overhead_ms` | Histogram | `algorithm` | LLM client driver's time to obtain a successful routing outcome, including judge calls but excluding any subsequent answer call. |
| `switchyard.classifier_fail_open` | Counter | `judge_model`, `reason` | Judge failures that caused classification to proceed without a verdict. |
| `switchyard.upstream_attempts` | Counter | `outcome`, `code` | HTTP attempts, including retries, made by the translating client. |
| `switchyard.router_retry_recovered` | Counter | none | Upstream operations that succeeded after a retry. |

Algorithm/call `outcome` is `ok` or `error`. HTTP attempt `outcome` is `ok` for
2xx, `retryable_error` for 408/429/5xx or failures without a status, and
`other_error` otherwise. `code` is an allowlisted status, a status-class bucket,
or `none`. Classifier `reason` is `timeout`, `transport`, `upstream_5xx`,
`upstream_non_5xx`, `invalid_response`, `parse_error`, `client_error`, or `call_error`.

Client requests, routing decisions, model calls, and HTTP attempts have separate
counts. For `switchyard.llm_calls` and `switchyard.llm_call_duration_ms`, each
routing-time offloaded call and each terminal answer candidate contributes one
observation. HTTP retries stay within the same candidate's count and duration.
Reusing a response produced during routing keeps only its routing-time call observation.

For example, one buffered client request with no judge calls or retries, a failed
primary, and a successful fallback produces one client response, one routing decision,
two `llm_calls`, and two `llm_call_duration_ms` samples. The primary has
`selected_model=<primary>` and `outcome=error`; the fallback has
`selected_model=<fallback>` and `outcome=ok`. For terminal answer-candidate observations,
`selected_model` identifies the model called. For routing-time observations, it
identifies the initial routing candidate. For `switchyard.decisions`, it identifies
the initial routing selection.

The five request/error/latency instruments above record each answer candidate after
routing, not each HTTP attempt. A failed candidate followed by a successful fallback
adds two to `total_requests`, one to `total_errors`, one to `errors` for the failed
model, and one to `requests` for the successful model. Routing-time calls and failures
before a routing outcome do not contribute. A response reused from routing contributes
one to `total_requests` and one to either `requests` or `errors` (also `total_errors`
on failure), but no `model_call_latency_ms` sample. The `model` label is the
candidate's configured model ID (the serving model for a reused response).
For terminal answer candidates, it matches `selected_model` on `llm_calls` and
`llm_call_duration_ms`. The two total gauges are cumulative process-wide
compatibility values.

For streams, these five instruments record when the response stream ends or is dropped.
Stream errors count as failures. Dropping an unfinished stream also counts as a failure;
a terminal message permits a successful drop. Failed calls have no
`model_call_latency_ms` sample.

### Algorithm-specific metrics

Stage Router instruments use the prefix `switchyard.stage_router.`:

| Suffix | Type | Labels | Meaning |
|---|---|---|---|
| `routing_decisions` | Counter | `decision_source`, `target_name` | Choices by decision source and semantic target name. |
| `probability` | Histogram | none | Scorer's capable-model probability. |
| `confidence` | Histogram | none | Confidence used to resolve or defer a turn. |
| `severity` | Histogram | none | Tool-failure severity. |
| `spinning` | Histogram | none | Repeated unproductive tool activity. |
| `exploring` | Histogram | none | Exploratory tool activity. |
| `production_intensity` | Histogram | none | Production-oriented tool activity. |

These histograms contain unitless values from 0 to 1. They are recorded when tool
signals reach the scorer, including when it defers to a classifier. They are not
one sample per application request. They carry the current request identity labels.

Advisor Gate instruments use the prefix `switchyard.advisor_gate.`:

| Suffix | Type | Labels | Meaning |
|---|---|---|---|
| `reviews` | Counter | `verdict`, `trigger` | Review outcomes and what triggered them. |
| `consult_failures` | Counter | `reason` | Failed advisor consultations. |
| `discarded_turns` | Counter | none | Executor turns discarded after a redo verdict. |
| `discarded_tokens` | Counter | `kind` | Tokens in discarded turns; `kind` is `input`, `cached`, `cache_creation`, or `output`. |

## Limits

### Timing and streaming

- `libsy.run` may finish before the answer call. Nested algorithms have separate run spans.
- Decision calls record `error` when failed or dropped without a reply, including cancellation. Cancelled calls do not poll further host work. A response completed during a cancellation race may still record its result. These metrics work with any host and do not require a client observer.
- `libsy.llm_call` and routing-time call metrics end when the host fulfills the offloaded call. They include host queueing. The LLM client driver buffers routing streams before fulfilling the call. The span's `input_tokens`, `output_tokens`, `total_tokens`, and `reasoning_tokens` fields are buffered-response only.
- For terminal answer candidates, `switchyard.llm_calls` and `switchyard.llm_call_duration_ms` record when the response stream ends or is dropped. Duration includes that candidate's retries and stream consumption. Stream errors and unfinished drops record `outcome=error`; a terminal message permits a successful drop.
- `libsy.client_call` remains open while its stream is consumed. IDs, usage, and finish reasons update from normalized events. An unfinished stream dropped by its consumer records `cancelled`; a stream error records `error`.

### Data exposure

Built-in evidence excludes prompts, responses, and raw errors. Custom evidence is
checked for field names and types, not string contents or lengths. Client spans may
include error descriptions and upstream addresses; algorithm logs may include content
such as Advisor Gate's `reply_head`. Supplied session/correlation IDs are included.
Review collected data before exporting it outside your deployment.

Custom algorithms can record OTel instruments directly. Use the shared request identity labels for correlation. Keep other metric labels
to configured names and fixed categories, and omit user text.

### Source and stability

These are current implementation names, not a separately versioned telemetry schema.
Debug spans and log events are not a stable field contract.

- [Libsy spans, outcome projection, and metrics](../../crates/libsy/src/observability.rs)
- [Client span fields](../../crates/libsy-llm-client/src/run.rs) and [stream/usage observation](../../crates/libsy-llm-client/src/observability.rs)
- [Client metrics](../../crates/libsy-llm-client/src/metrics.rs)
- [Stage Router metrics](../../crates/libsy/src/algorithms/util/stage.rs) and [Advisor Gate metrics](../../crates/libsy/src/algorithms/advisor_gate/telemetry.rs)
