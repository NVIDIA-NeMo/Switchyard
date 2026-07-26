# TokenForge × NVIDIA Switchyard — Integration Design Specification

**Status:** Draft v0.1 — design spec, not a validated architecture
**Date:** 2026-07-26
**Owner:** Product Architecture, Monetize360.AI
**Upstream:** [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) @ `main` (`64eaa61`)

---

## 1. Executive summary

Switchyard is NVIDIA's open-source LLM traffic proxy: it routes requests across
providers, translates between OpenAI / Anthropic / Responses API formats, and
emits per-request token and cost telemetry.

Switchyard is a **routing fabric**, not a gateway. Verified against the source at
`main`, it deliberately ships without:

| Absent capability | Evidence |
| --- | --- |
| Authentication / authorization | `docs/cli_reference.md`: "the proxy ignores inbound auth." `/metrics` is unauthenticated by design. |
| Multi-tenancy | No `tenant`, `org`, `project`, or `customer` field in config, `ProxyContext`, or Prometheus labels. `user_id` is an anonymous per-machine id from `~/.switchyard/user_id`. |
| Quota / budget enforcement | No rate limiter, no counter store, no denial path. |
| Cost-aware routing | Routers select on probability, LLM classification, tool-call signals, or latency health — never price. No cost metric in Prometheus. |
| Configurable rate card | `MODEL_PRICING` is a hardcoded `dict` in `switchyard/lib/cost_estimator.py`. |
| Chargeback, invoicing, settlement | Out of scope for the project by design. |

**This is the wedge, not a gap.** Every one of those absences is a line item in
the Monetize360 platform. The integration thesis:

> **Switchyard is the fabric. TokenForge is the control plane. RevenueOS is the
> commerce layer.** Switchyard decides *which model*; TokenForge decides
> *whether, for whom, and at what cost*; RevenueOS turns the result into a
> rated charge, an invoice, and a settlement.

Switchyard already ships three seams built for precisely this handoff — an
intake webhook carrying `cost_usd`, a request-processor pre-flight hook, and
routing-decision correlation ids stamped for downstream spend joins. Phase A
uses only the first and requires **zero code changes to Switchyard**.

---

## 2. Nomenclature

Canonical stack, aligned to the Monetize360 GTM taxonomy:

```
┌──────────────────────────────────────────────────────────────────┐
│ RevenueOS         commerce layer                                 │
│                   catalog · rate cards · rating · billing ·      │
│                   entitlements · settlement · ASC 606 rev-rec    │
├──────────────────────────────────────────────────────────────────┤
│ Mbrix             composable build layer (no-code canvas)        │
│ M360 Agents       30+ governed Q2C agents                        │
├──────────────────────────────────────────────────────────────────┤
│ TokenForge        AI token control plane   ◀── THIS SPEC         │
│                   identity · policy & quota · budget enforcement │
│                   cost-aware routing · guardrails · attribution  │
├──────────────────────────────────────────────────────────────────┤
│ Switchyard        routing fabric (NVIDIA OSS)                    │
│                   protocol translation · tier routing · telemetry│
├──────────────────────────────────────────────────────────────────┤
│ NIM · Nemotron · vLLM · Ollama · OpenAI · Anthropic              │
└──────────────────────────────────────────────────────────────────┘
```

`TokenForge` supersedes the working name *Tokenomix*. Internally the function is
still FinOps. Tagline: **"TokenForge — the control plane for your AI Token
Factory."**

Note a live taxonomy conflict inherited from the source corpus: **NCP** means
*NVIDIA Cloud Partner* in the Nutanix documents and *Non-Cloud Provider* in the
NVIDIA GTM playbook. This spec uses **NCP = NVIDIA Cloud Partner** throughout.

---

## 3. Verified integration surface

Everything in this section was read from the repository. Nothing is inferred.

### 3.1 Seam 1 — Intake sink (the metering tap)

Per-request records are POSTed asynchronously **off the response path**, with a
bounded queue and retries. Config lives in `IntakeSinkConfig`
(`switchyard/lib/config/intake_sink_config.py`, an 8-line re-export of the Rust
type): `intake_base_url`, `workspace`, `user_id`, `api_key`,
`target: IntakeTarget { url, format, authenticated }`, `max_queue_size`,
`request_timeout_s`, `max_retries`, `on_queue_full`, `capture_content`.

The decisive flag: **`--intake-target-url` / `$SWITCHYARD_INTAKE_TARGET_URL`
redirects the entire record stream to any endpoint you own.**

Payload fields relevant to rating:

```
request, response, provider: "switchyard", user_id, created_at
router_type, routed_to
prompt_tokens, completion_tokens, cached_tokens, cache_creation_tokens
cost_usd, cost_input_usd, cost_output_usd
cost_details: { base_input, cached_input, cache_write }
```

Opt-in per request via body `store=true` or header
`x-switchyard-intake-enabled: true`.

> **Constraint:** fire-and-forget. `on_queue_full` may drop records. Acceptable
> for dashboards; **not** acceptable as the sole system of record for an
> invoice.
>
> **REVISED —** intake is no longer the metering source of record. TokenForge
> Edge meters synchronously from the response path, which is lossless by
> construction; intake became enrichment and cross-check. See
> [`03-metering-integrity.md`](03-metering-integrity.md), which supersedes §6.3.

### 3.2 Seam 2 — Request processor (the enforcement point)

Duck-typed, no base class. Required shape:

```python
async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest
```

Runs **before** the upstream call. It can read the redacted header map and
`_caller_api_key` from `ctx.metadata`, rewrite the request, or raise to block.
Injected as the `pre_routing_request_processors` kwarg to the table builder —
**never declarable in route YAML** (`RouteBundle` docstring: "route YAML never
declares those processors itself").

> **Constraint:** any exception raised is flattened to
> `SwitchyardProcessorError(str(error))`, so HTTP status is not directly
> controllable. Workaround: stamp `_upstream_http_status` and
> `CTX_ERROR_SOURCE` on the context. See §7.2 — this is our first upstream PR.

### 3.3 Seam 3 — Identity headers (already present)

Switchyard reads a rich identity header set. Credential precedence:
`x-switchyard-api-key` → `Authorization: Bearer` → `x-api-key` (all three
redacted to `[REDACTED]` in the retained map). Session/agent identity:

```
x-switchyard-session-id | -agent-id | -parent-agent-id | -is-subagent
                        | -agent-kind | -agent-role | -task-id | -task-kind
                        | -turn-id | -request-id | -session-final
```

Plus vendor aliases (`x-claude-code-session-id`, `x-nemo-relay-session-id`,
`x-dynamo-session-id`). Routing: `x-switchyard-tier`, `x-switchyard-trial-id`.

The **agent hierarchy headers are the sleeper asset**: `-agent-id`,
`-parent-agent-id`, `-is-subagent` let us build a *spend tree* for a multi-agent
run — attributing a sub-agent's tokens to the parent task and the parent task to
a tenant contract. No commercial gateway does this today.

> **Constraint:** there is no tenant header. We introduce
> `x-tokenforge-tenant-id` / `-contract-id` / `-cost-center` (§5.2). Because the
> Rust server captures every inbound header verbatim into
> `Metadata::http_headers`, custom headers survive without a patch.

### 3.4 Seam 4 — Route-selection attribution

`CTX_ROUTE_SELECTION` carries `{router_model, router_strategy,
router_selected_endpoint, router_selected_model, router_selected_provider,
router_correlation_id}`, stamped upstream as `x-litellm-spend-logs-metadata` and
returned as `x-switchyard-*` response headers. **This exists specifically so a
front proxy can join its own spend row on the correlation id** — purpose-built
for our use case.

### 3.5 Seam 5 — Programmatic route table

A custom router **cannot be named from YAML**: route type resolution is a closed
hardcoded alias dict, `_PROFILE_CONFIGS` is dormant (every shipped config uses
`register=False`), and `pyproject.toml` declares no entry points. Three
programmatic paths remain:

1. Construct a `RouteBundle` directly and skip the YAML parser (explicitly
   supported: "launchers skip the parser and construct a `RouteBundle`
   directly").
2. `RouteTable.register(model, chain_runtime, metadata=..., default=...)`.
3. Implement the `Profile` protocol — three async methods, `run`, `process`,
   `rprocess` — and wrap in `ProfileSwitchyard(MyConfig(...).build())`.

We use (3) for the TokenForge routing profile and (1) for programmatic
deployment. Making the plugin registry live is our second upstream PR (§7.2).

### 3.6 Which server to build against

Two unrelated implementations with incompatible configs:

| | Python `switchyard serve` | Rust `switchyard-server` |
| --- | --- | --- |
| Config | YAML bundle (`--routing-profiles`) | TOML (`--config`) |
| Route types | 9 (`model`, `passthrough`, `random_routing`, `noop`, `deterministic`, `stage_router`, `escalation_router`, `plan_execute`, `latency_service`) | 3 (`noop`, `random`, `llm_classifier`) |
| Stats / `/metrics` / intake / cost | **yes** | none |
| TLS | no | yes |

**Design against the Python FastAPI server.** It is the only one with the
metering surface. TLS termination is ours (§4.2) — the Rust server's TLS is not
a reason to switch.

---

## 4. Target architecture

### 4.1 Component view

```
                      ┌────────────────────────────────────────┐
   Apps · Agents      │  TokenForge Edge (new)                 │
   RAG services  ────▶│  ├── AuthN/Z · API-key → tenant        │
   Ext. customers     │  ├── Budget preflight (Redis counters) │
                      │  ├── Header stamping (tenant/contract) │
                      │  └── TLS · rate limit · audit log      │
                      └────────────────┬───────────────────────┘
                                       │ x-tokenforge-* headers
                                       ▼
                      ┌────────────────────────────────────────┐
                      │  Switchyard (NVIDIA OSS, unmodified)   │
                      │  ├── pre_routing_request_processors:   │
                      │  │     TokenForgePolicyProcessor  ◀────┼── Phase B
                      │  ├── route table (tier routing)        │
                      │  ├── protocol translation              │
                      │  └── intake sink ──────────────────────┼──┐
                      └────────────────┬───────────────────────┘  │
                                       ▼                          │
                        NIM · Nemotron · vLLM · OpenAI · Anthropic│
                                                                  │
   ┌──────────────────────────────────────────────────────────────┘
   ▼
┌────────────────────────────────────────────────────────────────┐
│  TokenForge Core (new)                                         │
│  ├── /v1/intake/switchyard   ← intake-target-url               │
│  ├── /v1/policy/decide       ← Phase-B preflight (p99 < 15ms)  │
│  ├── Mediation: normalize · dedupe · tenant-attribute          │
│  ├── Rate card store (versioned, negotiated rates)             │
│  ├── Budget ledger (Redis hot + Postgres durable)              │
│  ├── Spend-tree builder (agent-id → parent → task → contract)  │
│  └── Reconciliation vs /v1/stats + /metrics                    │
└─────────────────────────────┬──────────────────────────────────┘
                              ▼  rated usage events
┌────────────────────────────────────────────────────────────────┐
│  RevenueOS   meters → rate plans → charges → wallet drawdown   │
│              → invoice → settlement → ASC 606                  │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Why an Edge component and not only a processor

Enforcement must happen at the *earliest* point for three reasons:

1. **Cost avoidance.** A `deterministic` route calls a classifier LLM *before*
   tier selection. A budget denial inside a request processor may already have
   burned classifier tokens. Denying at the edge burns nothing.
2. **HTTP fidelity.** A processor exception is flattened to
   `SwitchyardProcessorError` — we cannot cleanly return `402 Payment Required`
   or `429` with a `Retry-After`. The edge can.
3. **Auth is genuinely absent.** Switchyard never validates a key against an
   allowlist; `credential_policy: caller_required` only rejects a *missing* key,
   and `/metrics` is unauthenticated. Exposing Switchyard directly to external
   customers is not defensible.

The Phase-B processor is still valuable as **defense in depth** and as the hook
for cost-aware routing decisions that need post-classification context.

---

## 5. Data model

### 5.1 Tenant identity resolution

```
inbound API key  ──▶  TokenForge key registry
                      ├── tenant_id        (M360 billing account)
                      ├── contract_id      (RevenueOS contract)
                      ├── rate_card_id     (versioned, negotiated)
                      ├── cost_center      (chargeback allocation)
                      ├── route_tag        online|enterprise|marketplace|partner
                      ├── entitlements     allowed routes, models, max tier
                      └── budget_refs      [wallet_id, monthly_cap, daily_cap]
```

`route_tag` carries straight through from the Mirantis onboarding design — it
drives price-model selection, marketplace settlement, and partner payouts.

### 5.2 Headers TokenForge stamps

| Header | Purpose |
| --- | --- |
| `x-tokenforge-tenant-id` | Billing account |
| `x-tokenforge-contract-id` | RevenueOS contract for rate-card selection |
| `x-tokenforge-rate-card-id` | Pinned rate-card version (rating determinism) |
| `x-tokenforge-cost-center` | Chargeback / department allocation |
| `x-tokenforge-budget-state` | `ok` \| `warn` \| `throttle` — advisory to the router |
| `x-tokenforge-decision-id` | Joins the preflight decision to the intake record |
| `x-switchyard-tier` | **Existing Switchyard header** — our cost-aware tier hint |

`x-switchyard-tier` is the elegant part: the shipped `header_routing` profile
already routes on it, so TokenForge can downgrade a near-cap tenant from
`strong` to `weak` **with no Switchyard code at all**.

### 5.3 Metered usage event (TokenForge → RevenueOS)

```jsonc
{
  "event_id": "tfe_01J...",              // idempotency key
  "source": "switchyard",
  "decision_id": "tfd_01J...",           // joins to preflight decision
  "correlation_id": "...",               // Switchyard router_correlation_id
  "occurred_at": "2026-07-26T14:03:11Z",
  "tenant_id": "acct_8821",
  "contract_id": "ctr_4410",
  "cost_center": "eng-platform",
  "route_tag": "enterprise",
  "session": { "session_id": "...", "task_id": "...",
               "agent_id": "...", "parent_agent_id": "...",
               "is_subagent": true, "turn_id": "..." },
  "routing": { "route": "coding-agent", "router_type": "deterministic",
               "routed_to": "strong", "selected_model": "nvidia/nemotron-...",
               "selected_provider": "nim", "tier": "strong" },
  "meters": {
    "prompt_tokens": 18422, "completion_tokens": 1180,
    "cached_tokens": 16000, "cache_creation_tokens": 0,
    "reasoning_tokens": 0, "inference_requests": 1,
    "gateway_transactions": 1, "agent_session_seconds": 0
  },
  "cost": {                              // SUPPLIER cost — what we pay
    "source": "switchyard_intake",
    "cost_usd": 0.0231,
    "cost_input_usd": 0.0175, "cost_output_usd": 0.0056,
    "cost_details": { "base_input": 0.0023, "cached_input": 0.0152,
                      "cache_write": 0.0 }
  },
  "price": {                             // CUSTOMER price — what we charge
    "rate_card_id": "rc_gpu_ai_v7",
    "resolved_by": "tokenforge",
    "amount_usd": 0.0520,
    "margin_usd": 0.0289, "margin_pct": 55.6
  },
  "integrity": { "reconciled": false, "reconcile_batch": null }
}
```

**The cost/price split is the core modelling decision.** Switchyard's `cost_usd`
is *supplier cost* from its hardcoded `MODEL_PRICING`. It must never be shown to
a customer as a price. TokenForge owns the customer-facing rate card, and
`margin_usd` on every single request is what makes the Nutanix deck's
"2.25× margin per token" claim auditable rather than illustrative.

### 5.4 Meter catalog

Reuses the existing M360 AI meters so RevenueOS needs no new primitives:

| Meter | Unit | Source |
| --- | --- | --- |
| `ai.tokens.input` | token | intake `prompt_tokens` |
| `ai.tokens.output` | token | intake `completion_tokens` |
| `ai.tokens.cached_read` | token | intake `cached_tokens` |
| `ai.tokens.cache_write` | token | intake `cache_creation_tokens` |
| `ai.tokens.reasoning` | token | Prometheus `..._reasoning_tokens_total` |
| `ai.inference.request` | request | 1 per intake record |
| `ai.gateway.transaction` | txn | 1 per gateway call |
| `ai.agent.session` | second | derived from session-final turn spans |
| `ai.router.escalation` | event | `routed_to == "strong"` |
| `ai.cache.hit` | event | Phase C — priced at a discount, never at full token price |

Note `ai.router.escalation`: a *billable governance event*. Escalation to the
expensive tier is exactly what a FinOps buyer wants priced, tiered, and capped.

---

## 6. Phase A — metering overlay (zero Switchyard changes)

**Goal:** billing-ready, tenant-attributed, margin-annotated token usage in
days. This is the Nutanix / Mirantis PoC deliverable and the NVIDIA demo asset.

### 6.1 Deployment

```bash
export SWITCHYARD_INTAKE_TARGET_URL=https://tokenforge.m360.ai/v1/intake/switchyard
switchyard serve --routing-profiles config/route.tokenforge.yaml --port 8080
```

That is the entire integration. See [`01-phase-a-metering.md`](01-phase-a-metering.md)
for the receiver implementation and [`../config/route.tokenforge.yaml`](../config/route.tokenforge.yaml).

### 6.2 Flow

1. Client calls TokenForge Edge with its M360 API key.
2. Edge resolves tenant → stamps `x-tokenforge-*`, sets
   `x-switchyard-intake-enabled: true`, forwards to Switchyard.
3. Switchyard routes, translates, calls the backend, returns the response.
4. Switchyard POSTs the intake record to TokenForge Core.
5. Core joins on `decision_id`, applies the rate card, computes margin, emits a
   rated usage event to RevenueOS.
6. RevenueOS rates against the contract, draws down the wallet, invoices.

### 6.3 Reconciliation (mandatory — do not skip)

The intake sink is lossy by design and there is a **known 0.1.0 defect**: Codex
Responses-API tasks may record `0` token usage in `/v1/stats`. Controls:

- **Triangulate.** Scrape `/metrics` (`switchyard_prompt_tokens_total`,
  `..._completion_tokens_total`, `switchyard_requests_total{model,tier}`) on a
  fixed interval and poll `/v1/stats`. Compare aggregate token counts against
  the sum of intake records per window.
- **Alert on drift** above 0.5%; **block invoice generation** above 2%.
- **Zero-token guard.** Any record with `completion_tokens == 0` and a non-error
  response is quarantined, not rated.
- **Idempotency.** `event_id` derived from
  `sha256(request_id | correlation_id | created_at)` — intake retries must not
  double-bill.
- **Audit chain.** Persist the raw intake record immutably before rating.
  RevenueOS must be able to walk source record → meter → tenant → rate card →
  charge → invoice line.

The `/metrics` endpoint is unauthenticated and has **no tenant label** — scrape
it only from inside the trust boundary, and never expose it.

---

## 7. Phase B — enforcement and cost-aware routing

### 7.1 TokenForgePolicyProcessor

A request processor injected via `pre_routing_request_processors`. Contract:

```python
async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest
```

Responsibilities, in order:

1. Read tenant context from `ctx.metadata` headers (already stamped by Edge).
2. Call `POST /v1/policy/decide` — hot path, hard 15 ms timeout,
   **fail-open by default** (configurable; sovereign deployments fail closed).
3. On `deny`, raise with `_upstream_http_status = 402` stamped on the context.
4. On `throttle`, rewrite `x-switchyard-tier` to `weak` and/or clamp
   `max_tokens` in `request.body`.
5. On `allow`, stamp `x-tokenforge-decision-id` for the intake join.

Reference implementation: [`../prototype/tokenforge_m360/policy_processor.py`](../prototype/tokenforge_m360/policy_processor.py).

`request.body` is an untyped dict — `max_tokens` clamping is a direct mutation
via `replace_body()`. There is no typed `messages` field; messages are
`request.body["messages"]`.

### 7.2 Upstream contributions to NVIDIA

Four PRs, ordered by acceptance likelihood. Each is generally useful, not
M360-specific — this is how we earn standing in the NVIDIA ecosystem rather than
maintaining a fork.

| # | PR | Rationale |
| --- | --- | --- |
| 1 | **Typed processor rejection.** Let a processor raise a `SwitchyardPolicyError(status, code, message)` that survives to the HTTP envelope instead of flattening to `SwitchyardProcessorError`. | Unblocks every policy/guardrail integration, not just ours. Small, surgical. |
| 2 | **Externalize `MODEL_PRICING`.** Load the rate table from YAML/env, keeping the current dict as the default. | Negotiated NCP rates differ from list price; a hardcoded dict makes cost telemetry wrong for every enterprise. Clear community win. |
| 3 | **Activate the profile registry.** Honour `_PROFILE_CONFIGS` on the YAML path so a route `type:` can resolve a registered custom profile. | The registry already exists and is dormant. Turns "patch four dicts in `route_bundle.py`" into a supported extension path. |
| 4 | **Cost-aware router (`type: cost_router`).** A tier picker that consults an external decision endpoint or a local price table. | Natural next router after `latency_service` — same shape, different objective. Directly serves NVIDIA's Token Factory narrative. |

Until #1 and #3 land, Phase B runs via the programmatic `RouteBundle` path
(§3.5) — no fork of Switchyard required, only our own launcher.

### 7.3 Cost-aware routing — the five levers

Mapping the Nutanix Token Optimization levers onto real Switchyard mechanics:

| Lever | Switchyard mechanism | Status |
| --- | --- | --- |
| Smart routing | `x-switchyard-tier` header, `header_routing` profile | **Works today**, zero code |
| Dynamic pricing | TokenForge rate card by tier/latency/off-peak | TokenForge-side only |
| Margin guardrails | Preflight deny/throttle + per-request `margin_pct` | Phase B |
| Right-size & batch | Clamp `max_tokens`; **no batching primitive exists** | Partial — batching is TokenForge-side |
| Fill idle GPU | Route low-priority work to a NIM endpoint | Needs capacity signal; NCP-specific |

> **Honest gap (now resolved as a decision —** see
> [`04-caching-decision.md`](04-caching-decision.md)**):** Switchyard has **no
> semantic or response caching whatsoever.**
> Zero embeddings, no vector store, no similarity matching. `cached_tokens`
> metrics are pass-through of the *provider's* prompt cache, and
> `session_cache.py` is an LRU pin store. The "Cache & shaping — prefix,
> semantic" box in the current M360 architecture diagram is **not** satisfied by
> Switchyard and must be built in TokenForge or sourced elsewhere. Do not claim
> it in customer material.

---

## 8. Deployment topologies

| Topology | Shape | Fit |
| --- | --- | --- |
| **Sidecar** | Edge + Switchyard + processor in one pod; Core central | Single-tenant enterprise, FSI |
| **Central gateway** | Shared Switchyard fleet, tenant by header | NCP / neocloud multi-tenant |
| **Air-gapped** | Full stack on-prem, NIM backends only, no egress | Sovereign AI |

Air-gapped notes: intake target points at an in-cluster Core; rate cards ship as
signed config; the reconciliation scrape stays inside the boundary. Switchyard's
`ddtrace` tracing (**Datadog only — no OpenTelemetry**) must be disabled, which
means TokenForge owns tracing for sovereign deployments.

Non-negotiables carried from the M360 platform: SOC 2 Type 2, RBAC, SSO,
AES-256 at rest, TLS 1.2/1.3 in motion, immutable audit, EU data residency,
ASC 606 / IFRS 15, TM Forum alignment.

---

## 9. Phasing

| Phase | Scope | Exit criterion |
| --- | --- | --- |
| **A0 — Spike** | Intake receiver + one route + margin dashboard | Rated, tenant-attributed token usage from a live Switchyard, end to end |
| **A1 — PoC** | Reconciliation, rate-card store, RevenueOS event emission, spend tree | Auditable chain: intake record → meter → tenant → rate card → charge → invoice line, with < 0.5% drift |
| **B0 — Enforcement** | Edge authn/z, budget ledger, preflight, `x-switchyard-tier` downgrade | A near-cap tenant is demonstrably downgraded, then denied with `402`, having burned no classifier tokens |
| **B1 — Upstream** | PRs 1–3; cost-aware router prototype | At least one PR merged into NVIDIA-NeMo/Switchyard |
| **C — Productize** | Mbrix templates, M360 Agents (budget-guardian, margin-watch), marketplace listing | Reference-sellable by Nutanix / Mirantis / NCPs |

A0 and A1 need no NVIDIA engagement and no NDA — they run against public code
and representative data, which is exactly the pre-NDA PoC posture the Nutanix
playbook calls for.

---

## 10. Open questions

1. **NeMo Relay Switchyard Decision API.** NVIDIA documents a Relay-side
   decision plugin (`decision_api_url`, `decision_profile_id`,
   `decision_timeout_millis` default 25, `mode: enforce|observe_only`) and a
   `/v1/atof/events` history endpoint. **None of it exists in the OSS repo** —
   grep for `atof` and `decision` returns nothing, and the JSON contract is
   unpublished. If TokenForge can implement that contract, it becomes a *drop-in
   NVIDIA-sanctioned decision service* — the single highest-leverage unknown in
   this design. **Action: get the contract from NVIDIA.**
2. **Streaming attribution.** Response processors see one opaque
   `ChatResponse`; a stream is an opaque `ChatResponseStream` inside it, with no
   chunk hook. Confirm intake token counts are complete for SSE responses before
   billing streamed traffic.
3. **Agent-session metering.** `agent_session_seconds` needs a definition of
   session end. `x-switchyard-session-final` exists — is it reliably sent by
   Claude Code / Codex / Relay clients, or must we infer by timeout?
4. **Rate-card versioning under replay.** If a rate card changes mid-cycle, does
   a late-arriving intake record rate at receipt-time or occurrence-time? (Design
   position: occurrence-time, via `rate_card_id` pinned at preflight — hence
   `x-tokenforge-rate-card-id`.)
5. **NCP terminology.** Reconcile *NVIDIA Cloud Partner* vs *Non-Cloud Provider*
   before any customer-facing document ships.

---

## 11. Appendix — assets in this branch

| Path | Contents |
| --- | --- |
| `tokenforge/docs/00-design-spec.md` | This document |
| `tokenforge/docs/01-phase-a-metering.md` | Phase A implementation guide |
| `tokenforge/docs/02-phase-b-enforcement.md` | Phase B policy design + decision API contract |
| `tokenforge/docs/03-metering-integrity.md` | **Supersedes §6.3** — response-path meter, three-meter reconciliation |
| `tokenforge/docs/04-caching-decision.md` | Prefix vs semantic caching: build/source/defer decision |
| `tokenforge/demo/` | Runnable Phase A0/B0 demo (stdlib only) |
| `tokenforge/config/route.tokenforge.yaml` | Switchyard route bundle for the PoC |
| `tokenforge/prototype/tokenforge_m360/intake_receiver.py` | FastAPI intake sink + rating |
| `tokenforge/prototype/tokenforge_m360/policy_processor.py` | Phase-B request processor |
| `tokenforge/prototype/tokenforge_m360/rate_card.py` | Versioned rate card + margin calc |
| `tokenforge/prototype/tokenforge_m360/models.py` | Metered usage event schema |
