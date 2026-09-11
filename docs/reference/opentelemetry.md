# OpenTelemetry

Reference for the spans, attributes, and metrics emitted by libsy and its LLM client.

## Host setup

Your application installs a `tracing-opentelemetry` subscriber and global OTel meter
provider, and owns exporters, filtering, sampling, context propagation, and flushing.
Libsy installs none of these and sends no telemetry itself. Environment variables
alone do not enable collection. See [OTel Rust setup](https://opentelemetry.io/docs/languages/rust/)
and [OTLP configuration](https://opentelemetry.io/docs/specs/otel/protocol/exporter/).

## Spans and attributes

| Span | Emitted by | Meaning |
|---|---|---|
| `libsy.run` | Libsy | One algorithm run, including routing-time work. OpenInference kind `CHAIN`. |
| `libsy.llm_call` | Libsy driver | Waiting for the host to fulfill an offloaded call. Includes host queueing. OpenInference kind `CHAIN`. |
| `libsy.client_call`, exported as `chat <model_id>` | LLM client driver | One candidate model call, including that candidate's retries. OTel kind `CLIENT`; OpenInference kind `LLM`. |

Hosts driving `run_stream` without the LLM client driver instrument their own model I/O.

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

### Routing and client metrics

| Instrument | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard.runs` | Counter | `algorithm`, `outcome` | Completed algorithm tasks, including failures. |
| `switchyard.run_duration_ms` | Histogram | `algorithm`, `outcome` | Algorithm-task duration in milliseconds. |
| `switchyard.algorithms_in_flight` | UpDownCounter | `algorithm` | Active algorithm tasks; exported as a Prometheus gauge. |
| `switchyard.decisions` | Counter | `algorithm`, `selected_model` | Published routing decisions. |
| `switchyard.llm_calls` | Counter | `algorithm`, `selected_model`, `outcome` | Logical offloaded and terminal model calls. |
| `switchyard.llm_call_duration_ms` | Histogram | `algorithm`, `selected_model`, `outcome` | Logical call duration in milliseconds; see streaming limits below. |
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

## Limits

### Timing and streaming

- `libsy.run` may finish before the answer call. Nested algorithms have separate run spans.
- `libsy.llm_call` and logical call-duration metrics end at response-handle availability, not stream completion. The span's `input_tokens`, `output_tokens`, `total_tokens`, and `reasoning_tokens` fields are buffered-response only.
- `libsy.client_call` remains open while its stream is consumed. IDs, usage, and finish reasons update from normalized events. An unfinished stream dropped by its consumer records `cancelled`; a stream error records `error`.

### Data exposure

Built-in evidence excludes prompts, responses, and raw errors. Custom evidence is
checked for field names and types, not string contents or lengths. Client spans may
include error descriptions and upstream addresses; algorithm logs may include content
such as Advisor Gate's `reply_head`. Supplied session/correlation IDs are included.
Review collected data before exporting it outside your deployment.

Custom algorithms can record OTel instruments directly. Keep metric labels to
configured names and fixed categories, not request/session IDs or user text.

### Source and stability

These are current implementation names, not a separately versioned telemetry schema.
Debug spans and log events are not a stable field contract.

- [Libsy spans, outcome projection, and metrics](../../crates/libsy/src/observability.rs)
- [Client span fields](../../crates/libsy-llm-client/src/run.rs) and [stream/usage observation](../../crates/libsy-llm-client/src/observability.rs)
- [Client metrics](../../crates/libsy-llm-client/src/metrics.rs)
- [Stage Router metrics](../../crates/libsy/src/algorithms/util/stage.rs) and [Advisor Gate metrics](../../crates/libsy/src/algorithms/advisor_gate/telemetry.rs)
