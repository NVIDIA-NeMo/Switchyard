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
| Auth | None. Designed as a public scrape endpoint |
| Default scrape interval | 15s |

`GET /metrics` is served by the Python route-bundle server started with
`switchyard serve --routing-profiles PATH`.

A JSON variant of the same underlying data lives at `GET /v1/stats`, with
`GET /v1/routing/stats` as a backwards-compatible alias.

## Top-line gauges (no labels)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_total_requests` | gauge | Total upstream call attempts recorded since process start. **Per-attempt**, not per-client-request. One client request that retries-then-succeeds increments by 2. |
| `switchyard_total_errors` | gauge | Total upstream call attempts that errored, including those absorbed by retry. |

## Per-endpoint counters

The `model` label is the configured endpoint id (`openai/gpt-5.5`,
`azure_openai/gpt-5.5`, etc.).

The `tier` label is optional: present only when a routing factory
supplied one. The `random_routing` factory emits `tier="strong"|"weak"`.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_requests_total{model}` | counter | Successful upstream call attempts per endpoint. |
| `switchyard_errors_total{model}` | counter | Failed upstream call attempts per endpoint (any cause). |
| `switchyard_prompt_tokens_total{model}` | counter | Prompt-token billing per endpoint. |
| `switchyard_completion_tokens_total{model}` | counter | Completion-token usage per endpoint. |
| `switchyard_cached_tokens_total{model}` | counter | Cached prompt tokens per endpoint. |
| `switchyard_cache_creation_tokens_total{model}` | counter | Cache-creation tokens per endpoint. |
| `switchyard_reasoning_tokens_total{model}` | counter | Reasoning tokens per endpoint. |

## Per-endpoint latency summaries

Each summary emits `{quantile="0.5"}` and `{quantile="0.99"}` rows plus
`_sum` and `_count`.

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_model_call_latency_ms{model,quantile}` | summary | Upstream LLM call duration. |
| `switchyard_total_latency_ms{model,quantile}` | summary | End-to-end request latency (request entry → response complete). For streaming responses this is full-turn time, **not** time-to-first-token. |

## Routing overhead (global, no model label)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_routing_overhead_ms{quantile}` | summary | `total_latency_ms − backend_call_latency_ms`, across all calls. Bundles format translation, endpoint selection, body mutation, response wrapping, and retry wall time. |

Healthy traffic typically sits at p50 ≈ 0.4 ms, p99 ≈ 0.6 ms.

## Outcome counters for error-rate ratios

The `outcome` label takes exactly three values:

* `success` = HTTP 2xx
* `retryable_error` = HTTP 429 / 500 / 504
* `other_error` = everything else (400, 401, 403, 422, …)

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_client_responses_total{outcome}` | counter | HTTP responses returned to clients on the LLM-serving routes (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`). The denominator for the **router-served** error rate. |
| `switchyard_upstream_attempts_total{outcome,code}` | counter | Individual upstream call attempts. One client request can produce N attempts via retry. The denominator for the **direct-to-endpoint** baseline error rate. The `code` label carries the raw upstream HTTP status for plotting the error-code distribution (see below). |
| `switchyard_router_retry_recovered_total` | counter | Requests whose first upstream attempt failed but a subsequent attempt succeeded. This is direct evidence the routing logic rescued the request. |

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

# Live Endpoint Routing rescue rate
rate(switchyard_router_retry_recovered_total[5m])

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
| `quantile` | Exactly 2: `0.5`, `0.99`. | All summaries |
| `tier` | Small enumerated set, optional. | Per-endpoint counters/summaries on routing chains that supply it |

## Triage cheatsheet

| Symptom on `/metrics` | Likely cause |
|---|---|
| `model="<unknown>"` rows appear | The per-endpoint attribution wiring regressed. `ctx.selected_model` is not being set by the backend. |
| All counters at 0 after warm-up | Server just started with no traffic, or the scraper is hitting the wrong port. |
| `switchyard_routing_overhead_ms_count` stuck at `0` | The backend is not publishing `ctx.backend_call_latency_ms` (regression of the Python-backend routing-overhead wiring). |
| `switchyard_client_responses_total{outcome="retryable_error"}` rising | Either the upstream is genuinely flaky, or retries are exhausting (compare with `switchyard_router_retry_recovered_total` rate). |
