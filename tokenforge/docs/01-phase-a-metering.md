# Phase A — Metering overlay

**Zero changes to Switchyard.** This is the Nutanix / Mirantis PoC deliverable
and the NVIDIA demo asset. It runs against public code and representative data,
so it needs no NDA.

## Run it

```bash
# 1. TokenForge intake receiver
cd tokenforge/prototype
pip install fastapi uvicorn httpx pydantic
uvicorn tokenforge_m360.intake_receiver:app --port 9900

# 2. Switchyard, pointed at it
export SWITCHYARD_INTAKE_TARGET_URL=http://localhost:9900/v1/intake/switchyard
export NIM_API_KEY=...  NIM_BASE_URL=https://integrate.api.nvidia.com/v1
switchyard serve --routing-profiles tokenforge/config/route.tokenforge.yaml --port 8080
```

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-switchyard-intake-enabled: true' \
  -H 'x-tokenforge-tenant-id: acct_8821' \
  -H 'x-tokenforge-contract-id: ctr_4410' \
  -H 'x-tokenforge-cost-center: eng-platform' \
  -H 'x-switchyard-tier: weak' \
  -d '{"model":"tf-tiered","messages":[{"role":"user","content":"hello"}]}'
```

Then:

```bash
curl -s localhost:9900/v1/margin | jq        # per-tenant revenue, cost, margin
curl -s localhost:9900/v1/reconcile | jq     # drift vs Switchyard's own counters
```

Note the intake sink is **opt-in per request** — either
`x-switchyard-intake-enabled: true` or body `store=true`. TokenForge Edge sets
this header unconditionally so no billable request escapes metering.

## What the wire looks like

```
Client ──▶ TokenForge Edge ──▶ Switchyard ──▶ NIM / vLLM / OpenAI
              │  stamps x-tokenforge-*            │
              │  sets intake-enabled: true        │
              │                                   │ async, off response path
              └────── RevenueOS ◀── TokenForge Core ◀┘
                      rated usage        rate card + margin
```

## Field mapping — intake record → metered usage event

| Intake field | Meter / field | Note |
| --- | --- | --- |
| `prompt_tokens` | `ai.tokens.input` | `cached_tokens` is a **subset** of this |
| `completion_tokens` | `ai.tokens.output` | |
| `cached_tokens` | `ai.tokens.cached_read` | provider prompt-cache read |
| `cache_creation_tokens` | `ai.tokens.cache_write` | |
| `cost_usd` | `cost.cost_usd` | **supplier cost, never a customer price** |
| `cost_details.{base_input,cached_input,cache_write}` | `cost.cost_details` | |
| `router_type`, `routed_to` | `routing.*` | `routed_to == "strong"` → `ai.router.escalation` |
| `router_correlation_id` | `routing.correlation_id` | join key for spend attribution |
| `x-switchyard-agent-id` / `-parent-agent-id` | `session.*` | builds the spend tree |
| `x-tokenforge-tenant-id` | `tenant_id` | **absent → quarantine, never guess** |

## Reconciliation — mandatory

The intake sink is fire-and-forget with a bounded queue; `on_queue_full` may drop
records. It is a telemetry stream, not a system of record. Controls implemented
in `intake_receiver.py`:

| Control | Mechanism |
| --- | --- |
| Idempotency | `event_id = sha256(request_id \| correlation_id \| created_at)` |
| Drift detection | `/v1/reconcile` compares intake token sums to `/v1/stats` |
| Drift thresholds | warn ≥ 0.5%, **block invoicing** ≥ 2% |
| Zero-token guard | success + `completion_tokens == 0` → quarantine (known 0.1.0 defect on Codex Responses tasks) |
| Unpriced model | quarantine — an unpriced model is a revenue leak, not a free request |
| Unattributed request | quarantine — also a security finding in multi-tenant deployments |
| Audit chain | raw record persisted immutably **before** rating |

Cross-check sources on the Switchyard side:

- `GET /v1/stats` (alias `/v1/routing/stats`) — JSON aggregates.
- `GET /metrics` — Prometheus. Labels are only `model` and optional `tier`;
  **there is no cost metric and no tenant label**. Unauthenticated by design, so
  scrape only from inside the trust boundary.
- `GET /v1/routing/session-stats?session_id=…` — per-session detail. Also
  unauthenticated and returns arbitrary session data by id; never expose it.

## Cost-aware routing, for free

`x-switchyard-tier` is read by Switchyard's shipped `header_routing` profile.
TokenForge Edge can therefore downgrade a near-cap tenant from `strong` to
`weak` **in Phase A, with no Switchyard code and no processor**. This is the
"smart routing" lever from the Nutanix Token Optimization deck, working today.

## Not satisfied by Switchyard

Switchyard has **no semantic or response caching of any kind** — no embeddings,
no vector store, no similarity matching. `cached_tokens` is pass-through of the
*provider's* prompt cache and `session_cache.py` is an LRU pin store. The
"Cache & shaping — prefix, semantic" box in the current architecture diagram
must be built in TokenForge or sourced elsewhere. **Do not claim it in customer
material.**
