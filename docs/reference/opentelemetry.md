# OpenTelemetry

Libsy and its LLM client emit routing spans, model-call spans, and operational metrics.
Use traces to inspect one request and metrics to monitor traffic over time.
This page describes the current implementation, not a separate NVIDIA telemetry service.

## Collection and export

| Library | Behavior |
|---|---|
| `libsy` embedded in your application | Uses `tracing` and the global OTel meter provider. Your application installs the subscriber, provider, and exporters. Libsy installs no exporter and sends no telemetry itself. |
| `libsy-llm-client` | Adds model-call spans and metrics when driving an algorithm. Uses the same host-owned providers. |

Without a meter provider, library metrics are no-ops. Without a tracing subscriber,
spans are not collected. A logging-only subscriber does not export OTel traces.

Your application connects `tracing` to OTel through `tracing-opentelemetry` and
installs a global meter provider with its chosen metric readers/exporters.
Configure these before running algorithms. The host also owns filtering, sampling,
trace-context propagation, and flushing on shutdown.

Setting OTel environment variables alone does not install providers or exporters
in an embedded application. For an installed OTLP exporter, see the
[OTLP exporter configuration](https://opentelemetry.io/docs/specs/otel/protocol/exporter/).
Keep exporter credentials out of checked-in configuration.

## Traces

An algorithm run and the final model response have different lifetimes:
`libsy.run` can finish before the host makes the answer call.

| Span | Emitted by | Meaning |
|---|---|---|
| `libsy.run` | Libsy | One algorithm run, including routing-time work. OpenInference kind `CHAIN`. |
| `libsy.llm_call` | Libsy driver | Waiting for the host to fulfill an offloaded call. Includes host queueing. OpenInference kind `CHAIN`. |
| `libsy.client_call`, exported as `chat <model_id>` | LLM client driver | One candidate model call, including that candidate's retries. OTel kind `CLIENT`; OpenInference kind `LLM`. |

These are the main operational spans. Debug-level implementation spans and log
events are not a stable field contract. Hosts that drive `run_stream` themselves
must instrument their own model I/O; they do not get the LLM client driver's spans automatically.

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

`RoutingOutcome.metadata` is also available directly in Rust and Python. Its
evidence is optional JSON; OTel exports only the fields above with the expected
types. Unknown keys and wrong types are omitted. No evidence is valid for simple
decisions. Scores and confidence are algorithm-specific, not interchangeable.

Failed runs return typed errors and do not produce successful outcome metadata.
A successful fail-open decision can still carry a fixed `reason_code`.
Nested algorithm runs have their own run spans; one application request need not mean
one algorithm span.

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

For streaming responses, the client span stays alive while the stream is consumed.
Response IDs, usage, and finish reasons are recorded as normalized events arrive.
Dropping an unfinished stream records `cancelled`; stream errors record `error`.

The separate `libsy.llm_call` span ends when the host supplies a response handle.
Its `input_tokens`, `output_tokens`, `total_tokens`, and `reasoning_tokens` fields
are recorded only for buffered responses. Its duration is not full streaming latency.

## Metrics

Metrics use the `switchyard` meter scope. The tables use OTel instrument names.
With the default OTel Prometheus exporter naming, dots become underscores and
counters gain `_total`: `switchyard.runs` becomes `switchyard_runs_total`.
Histograms expose `_bucket`, `_sum`, and `_count` series. Your host chooses how
to expose or export the collected metrics.

### Routing and client metrics

| Instrument | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard.runs` | Counter | `algorithm`, `outcome` | Completed algorithm tasks, including failures. |
| `switchyard.run_duration_ms` | Histogram | `algorithm`, `outcome` | Algorithm-task duration in milliseconds. |
| `switchyard.algorithms_in_flight` | UpDownCounter | `algorithm` | Active algorithm tasks; exported as a Prometheus gauge. |
| `switchyard.decisions` | Counter | `algorithm`, `selected_model` | Published routing decisions. |
| `switchyard.llm_calls` | Counter | `algorithm`, `selected_model`, `outcome` | Logical offloaded and terminal model calls. |
| `switchyard.llm_call_duration_ms` | Histogram | `algorithm`, `selected_model`, `outcome` | Logical call duration in milliseconds, ending at response-handle availability for streams. |
| `switchyard.routing_overhead_ms` | Histogram | `algorithm` | LLM client driver's time to obtain a successful routing outcome, including judge calls but excluding any subsequent answer call. |
| `switchyard.classifier_fail_open` | Counter | `judge_model`, `reason` | Judge failures that caused classification to proceed without a verdict. |
| `switchyard.upstream_attempts` | Counter | `outcome`, `code` | HTTP attempts, including retries, made by the translating client. |
| `switchyard.router_retry_recovered` | Counter | none | Upstream operations that succeeded after a retry. |

Algorithm/call `outcome` is `ok` or `error`. HTTP attempt `outcome` is `ok` for
2xx, `retryable_error` for 408/429/5xx or failures without a status, and
`other_error` otherwise. `code` is an allowlisted status, a status-class bucket,
or `none`. Classifier `reason` is `timeout`, `transport`, `upstream_5xx`,
`upstream_non_5xx`, `invalid_response`, `parse_error`, `client_error`, or `call_error`.

Logical calls, candidate calls, and HTTP attempts are different counts. Candidate
fallbacks and HTTP retries remain within one logical call. A response produced
during routing is not counted again as a new terminal model call.

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
one sample per application request and are not split by route or session.

Advisor Gate instruments use the prefix `switchyard.advisor_gate.`:

| Suffix | Type | Labels | Meaning |
|---|---|---|---|
| `reviews` | Counter | `verdict`, `trigger` | Review outcomes and what triggered them. |
| `consult_failures` | Counter | `reason` | Failed advisor consultations. |
| `discarded_turns` | Counter | none | Executor turns discarded after a redo verdict. |
| `discarded_tokens` | Counter | `kind` | Tokens in discarded turns; `kind` is `input`, `cached`, `cache_creation`, or `output`. |

Algorithms can use OTel instruments directly. Use configured model/algorithm names
and fixed categories as metric labels, not request IDs, session IDs, or user text.

### Querying metrics

For example, inspect routing rate and latency with PromQL:

```promql
sum by (algorithm) (rate(switchyard_runs_total[5m]))

histogram_quantile(0.95,
  sum by (le, algorithm) (rate(switchyard_run_duration_ms_bucket[5m]))
)
```

Histogram percentiles are estimates from buckets, not exact per-request percentiles.

## Data boundaries

Built-in outcome evidence excludes prompts, responses, and raw errors. Its OTel
projection checks field names and types, not arbitrary string contents or lengths.
Do not place private data in custom evidence fields that are exported.

This is not a blanket redaction guarantee for all instrumentation. Client spans can
include error descriptions and upstream addresses. Existing algorithm logs can
include content, such as Advisor Gate's `reply_head`. Session and correlation IDs
are included when supplied. Review the full collected trace/log data before
exporting it outside your deployment.

## Source reference

These names describe the current source; they are not a separately versioned
telemetry schema. Check changes when upgrading.

- [Libsy spans, outcome projection, and metrics](../../crates/libsy/src/observability.rs)
- [Client span fields](../../crates/libsy-llm-client/src/run.rs) and [stream/usage observation](../../crates/libsy-llm-client/src/observability.rs)
- [Client metrics](../../crates/libsy-llm-client/src/metrics.rs)
- [Stage Router metrics](../../crates/libsy/src/algorithms/util/stage.rs) and [Advisor Gate metrics](../../crates/libsy/src/algorithms/advisor_gate/telemetry.rs)
