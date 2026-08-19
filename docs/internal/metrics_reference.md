# Switchyard Metrics Reference

Operational reference for the Prometheus exposition served by a Switchyard
deployment. Pair with [`examples/prometheus/`](../../examples/prometheus/) for
a drop-in scrape config and starter alert rules.

## Endpoint

| Property | Value |
|---|---|
| Path | `GET /metrics` (HTTP path is `/metrics`, **not** `/v1/metrics`) |
| Content-Type | `text/plain; version=0.0.4; charset=utf-8` |
| Format | Prometheus text format 0.0.4 |
| Auth | None |
| Default scrape interval | 15s |

`GET /metrics` is served by the native Rust server.

A JSON summary of the same traffic lives at `GET /v1/stats`.

## Top-line gauges (no labels)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_total_requests` | gauge | Successful and failed routed model calls since process start. Classifier and judge calls are excluded; a context-window fallback can add another routed call. |
| `switchyard_total_errors` | gauge | Failed routed model calls since process start. |

## Run, decision, and call counters

Instrument names in the code use the OTel dotted form (`switchyard.runs`,
`switchyard.stage_router.score`, ...); the Prometheus exporter sanitizes the
dots to underscores and appends `_total` to counters, same as the other
families in this document.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_build_info{version}` | gauge | `1` at process start, with the server version as the `version` label. |
| `switchyard_runs_total{algorithm,outcome}` | counter | One per completed run of a routing algorithm; `outcome` is `ok` or `error`. |
| `switchyard_run_duration_ms{algorithm,outcome}` | histogram | Wall-clock duration of one routing-algorithm run, in milliseconds. |
| `switchyard_decisions_total{algorithm,selected_model}` | counter | One per run that resolves a target model. |
| `switchyard_llm_calls_total{algorithm,selected_model,outcome}` | counter | One per model call the router makes — algorithm-layer calls (classifier, judge, advisor) plus the terminal routed answer call. |
| `switchyard_llm_call_duration_ms{algorithm,selected_model,outcome}` | histogram | Wall-clock duration of one of those model calls, in milliseconds. |

`algorithm` is the routing algorithm's name (e.g. `stage_router`,
`llm_classifier`). `selected_model` is the model ID the call targeted — always
a real model, never `none`.

## Per-endpoint counters

The `model` label is the configured endpoint id (`openai/gpt-5.5`,
`azure_openai/gpt-5.5`, etc.).

The `tier` label is not exported on any of these families. The routing tier
(`strong`/`weak`) is a per-request routing decision and is recorded, when the
routing log is enabled, in the server's JSONL routing log (`tier` field) — not as
a Prometheus label.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_requests_total{model}` | counter | Successful routed model calls per endpoint. |
| `switchyard_errors_total{model}` | counter | Failed routed model calls per endpoint. |
| `switchyard_prompt_tokens_total{model}` | counter | Prompt-token billing per endpoint. |
| `switchyard_completion_tokens_total{model}` | counter | Completion-token usage per endpoint. |
| `switchyard_cached_tokens_total{model}` | counter | Cached prompt tokens per endpoint. |
| `switchyard_cache_creation_tokens_total{model}` | counter | Cache-creation tokens per endpoint. |
| `switchyard_reasoning_tokens_total{model}` | counter | Reasoning tokens per endpoint. |

## Per-endpoint latency histograms

Each histogram emits `_bucket`, `_sum`, and `_count` series. Use
`histogram_quantile` in PromQL to calculate a percentile.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_model_call_latency_ms{model}` | histogram | Successful final routed-call latency. |
| `switchyard_total_latency_ms{model}` | histogram | End-to-end latency for successful routed responses. For streaming responses this is full-turn time, **not** time-to-first-token. |

## Routing overhead (global, no model label)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_routing_overhead_ms{algorithm}` | histogram | Total run time minus the time spent in successful routed model calls, with overlapping hedged calls counted once. Includes classifier calls, failed routed attempts, target resolution, and decision publication; runs with no successful routed call are not recorded. Measured across the whole run, so it does not reconcile with `switchyard_run_duration_ms`, which times only the algorithm task. |

## Classifier fail-open counter

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_classifier_fail_open_total{judge_model,reason}` | counter | Judge failures that made a classifier route without a verdict. The caller's request can still succeed on the fallback target. |

`judge_model` is the configured judge target. `reason` is one of `timeout`, `transport`,
`upstream_5xx`, `upstream_non_5xx`, `invalid_response`, `parse_error`, `client_error`, or
`call_error`. The labels never include request or response text.

## Stage router metrics

Present when the routing algorithm is `stage_router`: one decision counter and
six distributions per turn.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_stage_router_routing_decisions_total{decision_source,target_name}` | counter | One per turn's final routing choice. `decision_source` is one of `override`, `tests_passed`, `dimensions`, `ambiguous`, `llm-classifier`, or `fall_open`; `target_name` is the model name the turn routed to (one of the router's two tier endpoints). |
| `switchyard_stage_router_score` | histogram | The stage scorer's signed routing score (positive favors the capable side, negative the efficient side). |
| `switchyard_stage_router_confidence` | histogram | The decision confidence used to resolve or defer the turn. |
| `switchyard_stage_router_severity` | histogram | Detected tool-failure severity for the turn. |
| `switchyard_stage_router_spinning` | histogram | Repeated unproductive tool activity for the turn. |
| `switchyard_stage_router_exploring` | histogram | Exploratory tool activity for the turn. |
| `switchyard_stage_router_production_intensity` | histogram | Production-oriented tool activity for the turn. |

The six histograms carry no labels. The score uses buckets from `-1` to `1`
in 0.25 steps; the other five use `[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]`.

## Advisor gate metrics

Present when the routing algorithm is `advisor_gate` (opt-in); otherwise
absent from the scrape.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_advisor_gate_reviews_total{verdict,trigger}` | counter | One per advisor-model consultation. `verdict` is `approve`, `redo`, or `unparseable`; `trigger` is `pattern`, `no_tool_call`, or `stall`. |
| `switchyard_advisor_gate_consult_failures_total{reason}` | counter | One per advisor call that failed before a verdict; `reason` from the same bounded set as the classifier fail-open reasons, minus `parse_error`. |
| `switchyard_advisor_gate_discarded_turns_total` | counter | One per executor turn discarded on a `redo` verdict. |
| `switchyard_advisor_gate_discarded_tokens_total{kind}` | counter | Tokens consumed by that discarded turn; `kind` is `input`, `cached`, `cache_creation`, or `output`. |

## Outcome counters for error-rate ratios

The `outcome` label takes exactly three values:

* `success` = HTTP 2xx
* `retryable_error` = HTTP 408, 429, any 5xx, or a failure before an HTTP status
* `other_error` = everything else (400, 401, 403, 422, …)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_client_responses_total{outcome}` | counter | HTTP responses returned to clients on the LLM-serving routes (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`). The denominator for the **router-served** error rate. |
| `switchyard_upstream_attempts_total{outcome,code}` | counter | Individual upstream call attempts. One client request can produce N attempts via retry. The denominator for the **direct-to-endpoint** baseline error rate. The `code` label carries the raw upstream HTTP status for plotting the error-code distribution (see below). |
| `switchyard_router_retry_recovered_total` | counter | Reserved retry-recovery counter. The current server exports it as zero. |

### The `code` label on `switchyard_upstream_attempts_total`

`code` is the raw upstream HTTP status as a string: `"200"`, `"429"`,
`"500"`, `"504"`, etc. Two special values:

* `code="none"`: a non-HTTP failure (network error, connection reset,
  pre-status timeout). The attempt never received a status line, so there
  is no code. These also count as `outcome="retryable_error"`.
* `code="4xx"` / `code="5xx"` / `code="1xx"` / `code="3xx"` / `code="other"`:
  an HTTP code outside the known-codes allowlist, clamped to its class so
  a misbehaving upstream cannot blow up label cardinality.

`outcome` is fully determined by `code`, so adding the label does not
multiply series. You get one series per distinct code either way. The
canonical codes (`200`, `404`, `429`, `500`, `504`, `none`) are seeded at `0` so
their time series exist from process start (a `rate()` over a never-seen
counter reads as "no data", not zero).

## Computing the success-criterion ratios

```promql
# Router error rate (the rate clients see)
router_error_rate =
  sum(rate(switchyard_client_responses_total{outcome="retryable_error"}[5m]))
  / sum(rate(switchyard_client_responses_total[5m]))

# Direct-endpoint error rate (what clients would have seen without the router)
direct_error_rate =
  sum(rate(switchyard_upstream_attempts_total{outcome="retryable_error"}[5m]))
  / sum(rate(switchyard_upstream_attempts_total[5m]))

# Headline metric: positive value means the router is reducing client errors
error_rate_reduction = direct_error_rate − router_error_rate

# Traffic share per endpoint
sum by (model) (rate(switchyard_requests_total[5m]))
  / ignoring(model) group_left sum(rate(switchyard_requests_total[5m]))

# Error-code distribution over time (stack the series in a Grafana time-series panel)
sum by (code) (rate(switchyard_upstream_attempts_total{code!="200"}[5m]))

# Same, as a 100%-stacked share rather than absolute rates
sum by (code) (rate(switchyard_upstream_attempts_total{code!="200"}[5m]))
  / ignoring(code) group_left
sum      (rate(switchyard_upstream_attempts_total{code!="200"}[5m]))
```

> **Note:** because `switchyard_upstream_attempts_total` now carries the
> `code` label, always wrap a bare selector in `sum()` (as the ratio
> queries above do) when you want a layer total. Otherwise the selector
> returns one series per code.

The ready-to-deploy alert rules implementing these expressions live in
[`examples/prometheus/switchyard.rules.yaml`](../../examples/prometheus/switchyard.rules.yaml).

## Cardinality

All labels are bounded enums. No per-request or per-user values escape
into label space.

| Label | Values | Where |
|---|---|---|
| `model` | One per configured endpoint, typically 2–6 per deployment. | All per-endpoint metrics. |
| `outcome` | Exactly 3: `success`, `retryable_error`, `other_error`. | Outcome counters |
| `code` | Bounded: the known-code allowlist (`200`, `400`, `401`, `403`, `404`, `408`, `409`, `422`, `429`, `500`, `502`, `503`, `504`), plus `none` and the per-class buckets `1xx`/`2xx`/`3xx`/`4xx`/`5xx`/`other`. About 20 values max. | `switchyard_upstream_attempts_total` |
| `le` | The configured histogram bucket boundaries. | Histogram buckets |
| `algorithm` | One stable value per configured algorithm. | Routing-overhead histogram, run and call counters |
| `decision_source` | Exactly 6: `override`, `tests_passed`, `dimensions`, `ambiguous`, `llm-classifier`, `fall_open`. | Stage-router decision counter |
| `target_name` | One per configured endpoint. | Stage-router decision counter |
| `selected_model` | One per configured endpoint (always a real model ID). | Decision and call counters |
| `reason` | Bounded error categories: `timeout`, `transport`, `upstream_5xx`, `upstream_non_5xx`, `invalid_response`, `parse_error`, `client_error`, `call_error` (consult failures use that set minus `parse_error`). | Classifier fail-open counter, advisor-gate consult-failure counter |
| `kind` | Exactly 4: `input`, `cached`, `cache_creation`, `output`. | Advisor-gate discarded-token counter |
| `judge_model` | One per configured judge target. | Classifier fail-open counter |
| `version` | One value: the server version. | `switchyard_build_info` |

## Triage cheatsheet

| Symptom on `/metrics` | Likely cause |
|---|---|
| `model="<unknown>"` rows appear | A routed-call observation did not include a selected model. |
| All counters at 0 after warm-up | Server just started with no traffic, or the scraper is hitting the wrong port. |
| `switchyard_routing_overhead_ms_count` stuck at `0` | No successful algorithm run has recorded a successful routed model call. |
| `switchyard_classifier_fail_open_total` rising | The judge target is failing or returning a response the classifier cannot parse. Check `judge_model` and `reason`. |
| `switchyard_client_responses_total{outcome="retryable_error"}` rising | Either the upstream is genuinely flaky, or retries are exhausting; compare client responses with retryable upstream attempts. |
