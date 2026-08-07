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

`GET /metrics` is served by the Python route-bundle server started with
`switchyard serve --routes PATH` and by the native Rust server.

A JSON summary of the same traffic lives at `GET /v1/stats`.

## Top-line gauges (no labels)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_total_requests` | gauge | Successful and failed routed model calls since process start. Classifier and judge calls are excluded; a context-window fallback can add another routed call. |
| `switchyard_total_errors` | gauge | Failed routed model calls since process start. |

## Per-endpoint counters

The `model` label is the configured endpoint id (`openai/gpt-5.5`,
`azure_openai/gpt-5.5`, etc.).

The `tier` label is optional. It is present when an algorithm defines a stable
tier for the selected model, such as `strong` or `weak` for a two-tier classifier.

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
canonical codes (`200`, `429`, `500`, `504`, `none`) are seeded at `0` so
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
| `algorithm` | One stable value per configured algorithm. | Routing-overhead histogram |
| `tier` | Small enumerated set, optional. | Per-endpoint counters and histograms on algorithms that supply it |
| `judge_model` | One per configured judge target. | Classifier fail-open counter |
| `reason` | Exactly 8 fixed error categories. | Classifier fail-open counter |

## Triage cheatsheet

| Symptom on `/metrics` | Likely cause |
|---|---|
| `model="<unknown>"` rows appear | A routed-call observation did not include a selected model. |
| All counters at 0 after warm-up | Server just started with no traffic, or the scraper is hitting the wrong port. |
| `switchyard_routing_overhead_ms_count` stuck at `0` | No successful algorithm run has recorded a successful routed model call. |
| `switchyard_classifier_fail_open_total` rising | The judge target is failing or returning a response the classifier cannot parse. Check `judge_model` and `reason`. |
| `switchyard_client_responses_total{outcome="retryable_error"}` rising | Either the upstream is genuinely flaky, or retries are exhausting; compare client responses with retryable upstream attempts. |
