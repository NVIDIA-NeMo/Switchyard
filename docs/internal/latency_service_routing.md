# Latency Service Routing

The **latency-aware router** (`type: latency_service`) routes each request to one of
several equivalent endpoints, biasing traffic toward whichever endpoint a central
**Latency Service** most recently observed as healthy and fast. It is the production
routing policy for OpenRouter-backed deployments, where a separate Latency Service
owns heartbeat probing and statistical latency profiling, and Switchyard consumes its
verdicts to make per-request steering decisions.

This document explains how the router works, the design choices behind it, and how to
configure and operate it.

## When to use it

Use `latency_service` when you have **multiple interchangeable endpoints** serving the
same logical model (e.g. the same model behind different providers or regions) and you
want Switchyard to:

- bias traffic toward the lowest-latency healthy endpoint,
- automatically steer away from degraded endpoints, and
- retry on a different endpoint when one fails, without the client ever seeing it.

If you only have a single backend, use `passthrough`. If you want a fixed weighted
strong/weak split for benchmarking, use
[Random Routing](../routing_algorithms/random_routing.md).

## Architecture

The router is implemented as a single custom `LLMBackend`,
`LatencyServiceLLMBackend` (`switchyard/lib/backends/latency_service_llm_backend.py`),
rather than a monolithic routing strategy. It slots into the standard four-role chain:

```
[RequestProcessor*] → LatencyServiceLLMBackend → [ResponseProcessor*] → TranslationEngine
```

The backend owns four things:

1. **A pool of `OpenAILLMClient` instances**, one per configured endpoint, keyed by the
   endpoint's `model` ID.
2. **A thread-safe in-memory health cache** (`dict[str, EndpointHealth]` guarded by a
   `threading.Lock`).
3. **A background `HealthPoller` daemon thread**
   (`switchyard/lib/backends/health_poller.py`) that refreshes the cache from the
   Latency Service.
4. **Endpoint-selection + retry logic** on the request hot path.

Inbound format translation (Anthropic Messages / OpenAI Responses → OpenAI Chat) is
delegated to the Rust `TranslationEngine`, so the backend transparently accepts any
inbound wire format and only ever speaks OpenAI Chat Completions upstream.

### Two-clock design: poll loop vs. hot path

The defining property of this router is that **health polling and request serving run on
separate clocks**:

- The **`HealthPoller` daemon thread** is the only thing that ever talks to the Latency
  Service. It polls on a fixed interval and *writes* verdicts into the cache.
- The **request hot path** only ever *reads* the cache (under the lock). It never makes a
  network call to the Latency Service.

Health awareness adds **zero per-request latency**. Selection is a dictionary
read and a weighted random pick. The poller is a plain daemon thread (not an asyncio
task), so its lifecycle is trivial. It starts when the backend is constructed and dies
with the process. Its synchronous `httpx.Client` never contends with the event loop
serving requests.

## The health poller

`HealthPoller` polls `{latency_service_url}/v1/endpoints/health` every `poll_interval_s`
seconds (default 10s), passing the configured endpoint IDs as `endpoint_ids` query
params. The Latency Service responds with a verdict and a latency sample per endpoint:

```json
{
  "endpoint_health": {
    "openai/gpt-5.5":       {"status": "healthy",  "last_latency_ms": 420.0},
    "azure_openai/gpt-5.5": {"status": "degraded", "last_latency_ms": 980.0}
  }
}
```

Each cache entry is an `EndpointHealth` snapshot carrying the discrete `status` and the
most recent `last_latency_ms`. The status enum is the **shared contract** between
Switchyard and the Latency Service: both sides must agree on these three string values:

| Status | Meaning |
|---|---|
| `healthy` | Endpoint is up and within latency expectations. |
| `degraded` | Endpoint is reachable but slow or partially failing. |
| `unknown` | No current verdict (warming up, or fallback after a poll failure). |

### Failure handling: fall back to UNKNOWN, not stale data

If a poll fails for any reason (DNS, network, timeout, non-2xx, or a schema mismatch on
the response), the poller **resets every endpoint in the cache to `UNKNOWN`** rather than
leaving stale verdicts in place. With every endpoint at `UNKNOWN`, the router degrades to
uniform random routing across the full pool. This is the safe default when health
information is unavailable. Acting on stale "healthy" data after the Latency Service has
gone dark would be strictly worse.

`last_latency_ms` is `None` before the first successful poll and whenever the Latency
Service reports it as null.

## Endpoint selection

On each request, `_select_endpoint()` picks a `model` ID from the cache in two steps:

### 1. Tier preference: `HEALTHY > UNKNOWN > DEGRADED`

Endpoints are bucketed by status, and the router picks from the **best non-empty tier**.
`UNKNOWN` is deliberately preferred over `DEGRADED`: an endpoint we have no verdict for is
a better bet than one the Latency Service explicitly flagged as degraded.

### 2. Within a tier: inverse-latency weighted random

Within the chosen tier, `_pick_by_latency()` selects an endpoint with probability
proportional to `1 / last_latency_ms`, so an endpoint the Latency Service recently saw
at 200 ms receives roughly twice the traffic of one at 400 ms. This is **weighted random,
not always-pick-the-fastest**, which spreads load and avoids stampeding a single endpoint.

The picker falls back to **uniform random** within the tier when:

- there is only one candidate, or
- any candidate's `last_latency_ms` is unknown (`None`), or
- any sample is non-positive (defensive: the Latency Service should never report ≤ 0).

This keeps behavior predictable while the poller is warming up or when the upstream
reports nulls.

### 3. Optional: session affinity (sticky routing)

By default the router picks per **turn**. Every request runs the tier + latency steps
above independently. For a multi-turn conversation this interleaves endpoints, which lets
each endpoint's upstream prompt/KV cache lapse and forces expensive cache *re-writes* on
the next turn that lands there.

Set `session_affinity: true` to route per **conversation** instead. The latency-aware
picker decides only the **first** turn, then pins that conversation to the endpoint
that served it. Every later turn reuses that endpoint (the affinity fast-path bypasses
tier + latency selection entirely), so the upstream cache stays warm for the life of
the conversation.

- **Conversation identity.** A streaming proxy holds no session ID. The client re-sends
  the full history each turn. `session_key_from_body()`
  (`switchyard/lib/session_key.py`) derives a stable key by hashing the stable harness
  prefix made from the system prompt + the first user message. Later turns only append, so
  every turn of one conversation hashes the same; distinct conversations differ.
- **Health overrides the pin.** A pin is reused only while its endpoint is `HEALTHY` or
  `UNKNOWN`. If it goes `DEGRADED` (or the call fails this request), the next turn
  re-routes through normal selection and re-pins to whatever serves it. Locality yields
  to health, so a session is never funneled into a failing endpoint.
- **Bounded memory.** Pins live in an in-process LRU map (the L1) capped at
  `affinity_max_sessions` (default `10_000`); the least-recently-used pin is evicted past
  the cap.
- **Cross-worker stickiness (optional shared store).** The L1 map is per process, so a
  multi-worker/multi-pod deployment may pin the same conversation differently per worker,
  and pins are lost on pod churn. Set `affinity_store: "redis"` (with `affinity_store_url`)
  to add a shared, persistent L2: on an L1 miss the pin is read through from the store and
  warmed back into L1, and every pin is written through to it. Pins then survive across
  replicas and pod restarts (bounded by `affinity_store_ttl_seconds`, refreshed on each
  write). The L2 is **best-effort** — an error or timeout falls back to L1 / unpinned and
  never fails a request; fail-open operations are counted on
  `switchyard_affinity_l2_errors_total`, and a circuit breaker (3 consecutive failures →
  skip the store for 10 s, then probe; `switchyard_affinity_l2_breaker_open` gauge) keeps
  a store outage from taxing every request with timeout waits. The store connection is released by the backend's
  `shutdown()` (the server lifespan teardown hook awaits it). Requires the
  `switchyard[affinity-redis]` extra. (Alternatively, front the workers with a
  session-affinity load balancer.)

The `switchyard.route_decision` span tags each pick with `switchyard.affinity_hit` so you
can measure the warm-reuse rate. Affinity is **off by default**; existing per-turn latency
routing is unchanged unless opted in.

#### Deploying the shared store: the operational contract

What an operator needs to know to provision and run the Redis L2. Everything below is
verified against the implementation (`RedisPinStore`, `SessionAffinity`).

**Topology.** A **standalone Redis endpoint only** (the client is
`redis.asyncio.from_url`): a single instance, a primary endpoint of a replicated setup,
or a managed single-endpoint tier. **Redis Cluster and Sentinel URLs are not
supported** — cluster `MOVED` redirects surface as fail-open errors, meaning requests
keep succeeding while stickiness silently degrades. A sustained
`switchyard_affinity_l2_errors_total` rate from the first request is the
mis-provisioning signal; watch it when first enabling the store.

**Auth and TLS.** URL-driven: `rediss://` for TLS, credentials inside the URL. Route
YAML expands `${ENV}` references in every string value, so keep the URL in a secret and
configure `affinity_store_url: "${AFFINITY_REDIS_URL}"` rather than an inline literal.

**What is stored.** Key = `affinity_key_prefix` + a 16-hex-char hash of the
conversation's stable prefix (system prompt + first user message); value = the pinned
endpoint id string. **No prompt text, message content, or user identifier leaves the
process** — a pin is ~50–100 bytes.

**Traffic and sizing.** One `GET` per request whose conversation isn't in the local L1
(first turn on a pod), and one `SET` (which refreshes the TTL) per successful request.
Working-set memory ≈ active conversations × ~150 B, bounded by the sliding
`affinity_store_ttl_seconds` (default 1 h). **Persistence (RDB/AOF) and HA are optional**:
losing the store only resets stickiness — conversations re-route cold and re-pin — it
never affects correctness. `maxmemory-policy allkeys-lru` is the recommended eviction
policy; an evicted pin costs one cold re-route.

**Failure envelope.** Every store operation is bounded by a 0.1 s socket/connect
timeout (a colocated Redis answers in well under 1 ms, so this is ~100× headroom) and
fails open — requests never fail because of the store. A **circuit breaker** caps the
outage tax: after 3 consecutive failures, L2 operations are skipped without a network
attempt for a 10 s cooldown, then one operation probes; success closes the breaker,
failure re-arms it. Worst case per pod during an outage is therefore a brief window of
up to ~0.2 s added per request (one `GET` on the L1 miss + one `SET` after success)
before the breaker opens, then ~zero added latency with one 0.1 s probe per 10 s —
stickiness degrades to per-pod L1 throughout. Alert on
`switchyard_affinity_l2_breaker_open == 1` (the outage signal) and on
`rate(switchyard_affinity_l2_errors_total[5m])` (degradation without a full trip —
note the rate drops to ~0.1/s per pod while the breaker is open, since only probes
fail). Pod readiness (`/health`, `is_ready()`) never depends on Redis.

**Isolation.** Deployments or routes sharing one Redis should set distinct
`affinity_key_prefix` values (or separate logical databases). A pin is honored whenever
its stored endpoint id exists in the reading route's endpoint set, so shared prefixes
cross-pollinate stickiness between routes that reuse endpoint ids.

**Config changes, upgrades, rollback.** A persisted pin whose endpoint id is no longer
in the route's endpoint list is ignored gracefully (the next turn re-routes and
re-pins) — shrinking or renaming the endpoint set needs no store cleanup. The
conversation-key derivation is not a cross-version contract: a Switchyard upgrade may
invalidate existing pins (one-time cold re-route, self-healing). To roll back, set
`affinity_store: "memory"`; stale keys expire via TTL, and `FLUSHDB` is always safe
(stickiness reset only).

The deterministic LLM-classifier route also supports `affinity_warmup_turns` to delay
when a tier becomes sticky. Latency-service routing does **not** use that knob. Its
affinity objective is endpoint cache reuse, so it pins after the first successful
endpoint selection.

## Retry and failover

A single client request may try up to `1 + max_retries` endpoints (default `max_retries`
is 2, so 3 attempts). On each attempt:

1. Select an endpoint. If selection returns one already tried this request (and untried
   endpoints remain), pick uniformly at random from the **untried** set. Dedup prevents
   wasting an attempt re-hitting an endpoint that just failed.
2. Stamp the endpoint's `upstream_model` into `body["model"]` and call the upstream.
3. On a **transient** failure such as `429` (rate limit), `408` (request timeout), any
   `5xx`, or a network / pre-status / SDK error, log it, record the error in stats, increment
   the outcome metric, and continue to the next endpoint. These are exactly the faults a
   health-aware router should absorb by trying a different endpoint.
4. On a **4xx client error** (`400`, `401`, `403`, `404`, `409`, `413`, `415`, `422`, …)
   the loop **fails fast**: it does *not* retry. The request itself is malformed or
   unauthorized, so every replica rejects the same payload identically; retrying only adds
   latency before surfacing the same status. (Context-window overflow surfaces as a `400`
   here and is also passed straight through. The `latency_service` route does not
   participate in the chain-level evict-and-retry described in
   [Context-Window Handling](../operations/context_window.md).)

If an attempt succeeds **after** at least one failure, the router records a
`retry_recovered` outcome. This is direct evidence the steering logic rescued a request that
would otherwise have failed.

Whenever the request ends on an upstream HTTP error, either a fail-fast 4xx or an
exhausted retryable error (e.g. a `503` after retries are exhausted), the backend records the
upstream status code and body on `ctx` so the endpoint passes the real upstream status
through to the client rather than masking it as a generic `500`.

## Configuration

`LatencyServiceBackendConfig` owns the router's policy, retry, and optional
session-affinity settings. In CLI YAML, configure the Python
`type: latency_service` router in a `routes:` bundle; there are no
policy-specific CLI flags. Run it with:

```bash
switchyard --routing-profiles my_routes.yaml -- serve --port 4100
```

The Rust `type: latency-service` profile loaded by `switchyard serve --config`
is a separate schema and does not yet expose `session_affinity` or
`affinity_max_sessions`.

### Config schema

`LatencyServiceBackendConfig`
(`switchyard/lib/config/latency_service_backend_config.py`):

| Field | Default | Purpose |
|---|---|---|
| `latency_service_url` | `""` | Base URL of the Latency Service. |
| `endpoints` | `[]` | The endpoints to route across (see below). At least one is required. |
| `poll_interval_s` | `10.0` | How often the background poller refreshes health. |
| `poll_timeout_s` | `5.0` | Timeout for each health API call. |
| `max_retries` | `2` | Extra endpoints to try on failure (total attempts = `1 + max_retries`). |
| `credential_policy` | `"configured_endpoint"` | Credential precedence for upstream calls. The default uses each endpoint's configured `api_key`; set `"caller_override"` only for BYO-key deployments where caller headers should replace endpoint keys. |
| `session_affinity` | `false` | Pin each conversation to one endpoint after the first turn (sticky routing: keeps the upstream cache warm). See [Endpoint selection §3](#3-optional-session-affinity-sticky-routing). |
| `affinity_max_sessions` | `10_000` | LRU cap on the in-process pin map (L1) when `session_affinity` is on. |
| `affinity_store` | `"memory"` | Shared L2 pin store behind the L1. `"memory"` = per-process only; `"redis"` shares pins across workers/pods and persists them across pod churn (requires `session_affinity: true`, `affinity_store_url`, and the `affinity-redis` extra). |
| `affinity_store_url` | `None` | Connection URL for the shared store (e.g. `redis://host:6379/0`). Required when `affinity_store` is `"redis"`. |
| `affinity_store_ttl_seconds` | `3_600` | Expiry for a shared pin; refreshed on each write, so an active conversation slides its TTL. |
| `affinity_key_prefix` | `"swyd:pin:"` | Namespace prefix for shared-store keys. |
| `enable_stats` | `true` | Wire the stats processors + `/metrics` exposition. |

Each `LatencyServiceEndpoint`:

| Field | Required | Purpose |
|---|---|---|
| `model` | yes | **Latency Service lookup key**: must be unique across the endpoint list. Also the value stamped into `body["model"]` unless `upstream_model` is set. |
| `upstream_model` | no | Override for `body["model"]` sent upstream. Defaults to `model`. |
| `api_key` | no | API key for the backing LLM API. This is the default upstream credential. |
| `base_url` | no | Base URL for the backing LLM API (include `/v1`). |
| `timeout` | no | Per-request timeout in seconds. |

### Why two `model` fields?

The Latency Service registers each endpoint under an ID *it* picked (e.g.
`openrouter-primary/openai/gpt-4o`). The upstream gateway may expect the OpenRouter
model name for the same destination (e.g. `openai/gpt-4o`). So:

- **`model`** is the Latency Service lookup key (and the metrics label).
- **`upstream_model`** is what gets stamped into `body["model"]` on the outbound call.

When they're equal, `upstream_model` can be omitted.

### Example

Create an OpenRouter key at <https://openrouter.ai/> and export it as
`OPENROUTER_API_KEY` before running this example.

```yaml
routes:
  gpt-4o:
    type: latency_service
    latency_service_url: https://latency-service.example.com
    poll_interval_s: 10.0
    poll_timeout_s: 5.0
    max_retries: 2
    endpoints:
      - model: openrouter-primary/openai/gpt-4o  # Latency Service endpoint_id
        upstream_model: openai/gpt-4o            # OpenRouter model name to call
        base_url: https://openrouter.ai/api/v1
        api_key: ${OPENROUTER_API_KEY}
      - model: openrouter-backup/openai/gpt-4o
        upstream_model: openai/gpt-4o
        base_url: https://openrouter.ai/api/v1
        api_key: ${OPENROUTER_API_KEY}
```

A runnable version of this config, with extensive inline notes, lives at
[`experiments/latency_routing.yaml`](../../experiments/latency_routing.yaml).
A production-style Kubernetes manifest lives at
[`examples/k8s/latency-aware-deployment.yaml`](../../examples/k8s/latency-aware-deployment.yaml).

### BYO-key (multi-tenant) mode

By default, caller `Authorization: Bearer <key>` and `x-api-key` headers do not replace
the configured endpoint credentials. This keeps server-owned latency-service deployments
from accidentally forwarding a client placeholder or unrelated proxy credential upstream.

For a multi-tenant BYO-key deployment, set `credential_policy: caller_override` on the
latency-service route. In that mode, a non-empty caller key is forwarded as the upstream
credential for that request, and the endpoint `api_key` is used only when the caller omits
the header. Credential precedence is **`Authorization` > `x-api-key` > endpoint
`api_key`**.

## Response fidelity (Responses API)

For an endpoint configured `request_type: openai_responses`, the backend calls the
upstream `/v1/responses` surface and returns the upstream payload **verbatim** in both
modes:

- **Non-streaming** calls return the upstream's exact JSON body — the raw HTTP response
  is used instead of round-tripping through the OpenAI SDK's typed model.
- **Streaming** calls forward the upstream's SSE frames as **verbatim strings**
  (`RawSSEFrameStream` in `switchyard/lib/llm_client.py`): no typed-event parse happens,
  so per-event provider extras, explicit nulls, event names, and comment keep-alives all
  survive. Frames are byte-equivalent modulo CRLF → LF line normalization. Usage
  accounting still works — the stats layers parse the frame's `data:` payload to find
  `response.completed` usage without altering what flows to the client.

When the inbound request is also Responses-format, the terminal translation
short-circuits (same format in and out), so the client receives the upstream payload
as-is. **Cross-format** calls (e.g. a chat client served by a Responses endpoint, or
vice versa) are *re-synthesized* by the translation engine and are inherently lossy —
fields the upstream never produced in the target format cannot be invented. For
full-fidelity Responses routing, configure every endpoint on the route with
`request_type: openai_responses`.

### Field-fidelity matrix (Responses API)

How each field class fares per path. "Exact" means byte-equivalent to the upstream
(streaming: modulo CRLF normalization); "synthesized" means the translation engine
rebuilds the payload from its neutral representation and only mapped fields survive.

| Field class (examples) | Same-format, non-streaming | Same-format, streaming | Cross-format (either direction) |
|---|---|---|---|
| Standard scalars (`id`, `model`, `status`, `created_at`) | exact | exact | synthesized (`id`/`model` mapped; `created_at` → chat `created` defaults to `0`) |
| Sampling/config echoes (`temperature`, `top_p`, `store`, `text`) | exact | exact | dropped (no chat-format equivalent) |
| `reasoning` config + explicit-`null` fields | exact (nulls preserved) | exact (nulls preserved) | dropped |
| `usage` (incl. `*_tokens_details`) | exact | exact | token counts + reasoning detail mapped; cached-token detail drops |
| Output content (`output[]`, text deltas) | exact | exact | mapped (text/tool calls translate; unmappable block types drop) |
| Provider extras (unknown keys, e.g. Azure `content_filter_results`) | exact | exact | dropped |
| SSE event names / keep-alive comments | n/a | exact | re-framed to the target format's event contract |
| `metadata`, `previous_response_id` | exact | exact | dropped |

Wire-level proofs live in
`tests/test_upstream_error_passthrough.py`
(`test_responses_body_returned_exactly_on_the_wire`,
`test_responses_stream_forwarded_verbatim_on_the_wire`,
`test_azure_flavored_responses_stream_preserved_on_the_wire`) against
OpenAI-shaped and Azure-shaped samples; validation against live production
captures remains a deployment-side step.

## Failure-source annotation

When a request fails, the error response carries two headers so clients and
observability can tell **which layer** the failure came from without parsing bodies:

| Header | Values | Meaning |
|---|---|---|
| `x-switchyard-error-source` | `switchyard` \| `provider` | `provider`: an upstream LLM failure passed through (HTTP error, or the 500 rendered for a status-less network fault). `switchyard`: this proxy rejected or failed the request itself — credential-policy 401 (`missing_caller_api_key`), translation 400 (`invalid_value`), model-not-found 404, context-window 400, body validation 400, unexpected internal 500. |
| `x-switchyard-upstream-model` | model id | The model actually attempted upstream when the failure happened. Present only when a routing selection took place. |

The provider error **body** still passes through verbatim — the annotation is
header-only, deliberately, so the transparent-proxy contract on bodies is preserved.
The same vocabulary appears as `switchyard.error_source` / `switchyard.upstream_model`
tags on failed `switchyard.upstream_attempt` spans and as `error_source` /
`upstream_model` fields on the per-event error log (below).

Layering note: Switchyard can only distinguish *itself* from *its upstreams*. A proxy in
front of Switchyard (e.g. LiteLLM) should tag its own failures the same way and
propagate these headers from below. Mid-stream failures after HTTP 200 has been
committed cannot carry headers and are not annotated today.

## Observability

When `enable_stats` is `true` (the default), `LatencyServiceProfileConfig` wires a
`StatsRequestProcessor` + `StatsResponseProcessor` pair sharing one `StatsAccumulator`.
Because `LatencyServiceLLMBackend` is a Python-only backend, profile assembly cannot wrap
it with the Rust-native `StatsLlmBackend`; instead it records success / error /
call-latency in-place into the same accumulator the response processor records token usage
into. It also sets `ctx.selected_model` and `ctx.backend_call_latency_ms` so per-endpoint
attribution and routing-overhead figures land correctly on `/metrics`.

The backend publishes its in-memory health cache and poll-loop health to `/metrics`:

| Metric | Type | Meaning |
|---|---|---|
| `switchyard_endpoint_status{model,status}` | gauge | Current verdict per endpoint; exactly one status row per model is `1`. |
| `switchyard_endpoint_last_latency_ms{model}` | gauge | Last latency sample per endpoint. Absent when the upstream reported null. |
| `switchyard_latency_service_poll_ok` | gauge | `1` iff the most recent poll succeeded. |
| `switchyard_latency_service_poll_age_seconds` | gauge | Seconds since the last successful poll. Absent before the first success. |
| `switchyard_latency_service_polls_total` | counter | Total successful polls. |
| `switchyard_latency_service_poll_failures_total` | counter | Total failed polls; each resets the cache to `unknown`. |
| `switchyard_affinity_hits_total` | counter | Turns served by an existing session-affinity pin (warm reuse). **Emitted only when `session_affinity` is on.** |
| `switchyard_affinity_misses_total` | counter | First/unpinnable turns routed by the latency-aware picker while affinity was on. Reuse rate = `hits / (hits + misses)`. **Emitted only when `session_affinity` is on.** |
| `switchyard_affinity_l2_hits_total` | counter | Pins resolved from the shared (L2) store after an in-process miss — cross-worker/churn warm reuse. **Emitted only when a shared affinity store is configured.** |
| `switchyard_affinity_l2_errors_total` | counter | Shared-store get/put operations that failed open; routing fell back to in-process pins. Alert on a sustained rate — the store is degraded while requests keep succeeding. **Emitted only when a shared affinity store is configured.** |

See the [Metrics Reference](metrics_reference.md) for full label semantics, PromQL recipes
(traffic share per endpoint, retry-rescue rate), and a triage cheatsheet.

### Expected routing overhead

`switchyard_routing_overhead_ms{quantile}` is the source of truth for what Switchyard
itself adds to a request: per request it records `total_latency − backend_call_latency`,
which bundles format translation, endpoint selection, body mutation, response wrapping,
and retry wall time — everything except the upstream call.

Reference figures, measured on an idle dev box over loopback (stub upstream + stub
Latency Service, non-streaming `/v1/chat/completions`, sequential; commit `0289a415`):

| Figure | Value |
|---|---|
| `switchyard_routing_overhead_ms{quantile="0.5"}` | ~0.3 ms |
| `switchyard_routing_overhead_ms{quantile="0.99"}` | ~7 ms |
| Mean overhead (`_sum / _count`) | ~2 ms |
| Client-observed added wall time vs calling the upstream directly | ~2 ms mean |

Expected order of magnitude is therefore **single-digit milliseconds**. When an
end-to-end comparison ("direct provider call" vs "through the stack") shows a much
larger delta — e.g. ~100 ms — the difference lives **outside** this metric:

* the extra network hop and TLS handshake from the client to the Switchyard pod,
* any fronting layer (LiteLLM / inference-gateway) in front of Switchyard,
* per-pod queueing under concurrency — a single event loop serializes CPU work, so
  client wall time grows with in-flight load while per-request overhead does not
  (at loopback concurrency 10 the same box shows ~22 ms client p50 with an unchanged
  overhead summary),
* upstream time-to-first-token variance between the two measurement runs.

Attribute before optimizing: read the deployment's own summary first.

```promql
# Median / p99 Switchyard-internal overhead
switchyard_routing_overhead_ms{quantile="0.5"}
switchyard_routing_overhead_ms{quantile="0.99"}

# Mean over a window
rate(switchyard_routing_overhead_ms_sum[5m]) / rate(switchyard_routing_overhead_ms_count[5m])
```

Two caveats. For streaming responses `switchyard_total_latency_ms` (and hence the
overhead summary) measures the **full turn**, not time-to-first-token — TTFT is what
interactive users feel, so don't read a long generation as router overhead. And the
summary is per-process: in a multi-pod deployment each pod reports its own.

The repeatable harness is the CI perf benchmark (`.github/workflows/perf.yml`): aiperf
against a no-op route measures the full server path with zero backend cost, non-streaming
and streaming, at configurable concurrency.

### Per-event error log (Loki)

`/metrics` answers *how many* of each error code, but Prometheus stores aggregates. It
carries no per-event timestamps. For audit, replay, or a "what failed and when" panel,
each failed upstream attempt also emits one structured JSON line on the
`switchyard.upstream_errors` logger
(`switchyard/lib/endpoints/upstream_error_log.py`):

```json
{"event":"upstream_attempt_failed","timestamp":"2026-05-29T12:00:00+00:00","model":"openai/gpt-5.5","upstream_model":"gpt-5.5","attempt":1,"status_code":429,"code":"429","outcome":"retryable_error","error_source":"provider","error_type":"APIStatusError","error":"..."}
```

`code` and `outcome` mirror the labels on `switchyard_upstream_attempts_total`, so the log
joins cleanly to the counter; `status_code` is `null` (and `code="none"`) for a non-HTTP
failure. Because the deployment uses plain-text logging, the message *is* the JSON object.
A Loki query parses it with `| json` and zero formatter config:

```logql
{app="switchyard"} | json | event="upstream_attempt_failed"
```

`is_ready()` returns `True` once the poller has completed at least one successful poll,
which makes it useful as a readiness gate during startup.

## Testing the router

Three tiers, cheapest first. The `switchyard-latency-router-e2e` skill carries the
runnable commands and helper scripts for each.

1. **Offline unit suite** (no keys): `tests/test_latency_service_*.py`,
   `tests/test_session_key.py`, `tests/test_outcome_metrics.py`, and
   `tests/test_upstream_error_passthrough.py`: full coverage of selection, tiering,
   inverse-latency weighting, retry/fail-fast, session affinity, the health poller,
   metrics, and spans.
2. **In-repo integration** (`tests/e2e/test_latency_service_llm_backend.py`, marked
   `integration`): an in-process mock Latency Service feeds verdicts while the LLM
   calls hit OpenRouter. Requires `OPENROUTER_API_KEY` (skips without it).
3. **Real-world**: point a real backend at a mock or live Latency Service and confirm
   it routes. A mock LS pointed at `openrouter.ai` proves the
   poll → tier → weighting path with no API key — a keyless dispatch returns `401`,
   which confirms traffic reached the host.

## Source map

| File | Responsibility |
|---|---|
| `switchyard/lib/backends/latency_service_llm_backend.py` | Selection, session-affinity pins, retry, stats/metrics emission, per-attempt error log. |
| `switchyard/lib/session_key.py` | `session_key_from_body()`: stable per-conversation key for session affinity. |
| `switchyard/lib/session_affinity.py` | `SessionAffinity`: L1 (in-process LRU) + optional best-effort L2 pin store. |
| `switchyard/lib/affinity_pin_store.py` | `AffinityPinStore`: async protocol for a shared/persistent L2 store. |
| `switchyard/lib/redis_pin_store.py` | `RedisPinStore`: Redis-backed shared L2 (lazy `redis`, `affinity-redis` extra). |
| `switchyard/lib/endpoints/upstream_error_log.py` | Structured per-attempt failure log (`switchyard.upstream_errors`, Loki). |
| `switchyard/lib/backends/health_poller.py` | `EndpointHealthStatus`, `EndpointHealth`, `HealthPoller` daemon. |
| `switchyard/lib/config/latency_service_backend_config.py` | `LatencyServiceEndpoint`, `LatencyServiceBackendConfig`. |
| `switchyard/lib/profiles/latency_service.py` | `LatencyServiceProfileConfig`: profile assembly + stats wiring. |
| `tests/test_latency_service_llm_backend.py` | Selection, retry, tier-preference tests. |
| `tests/test_latency_service_health_metrics.py` | Poller and metrics tests. |
