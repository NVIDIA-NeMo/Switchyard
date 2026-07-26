# Phase B — Enforcement, decision API, and upstream contributions

Phase A observes. Phase B **acts**: it denies, throttles, and routes on cost.
That is the half competitors cannot copy from telemetry alone.

## 1. Where enforcement lives

Two enforcement points, deliberately:

| | TokenForge Edge | `TokenForgePolicyProcessor` |
| --- | --- | --- |
| Position | before Switchyard | inside Switchyard's request pipeline |
| Can return clean `402` / `429` | **yes** | no (see §3) |
| Burns classifier tokens on denial | no | **yes**, on `deterministic` routes |
| Sees post-classification context | no | yes |
| Role | **primary** | defense in depth + cost-aware tier decisions |

The classifier ordering is the deciding argument. On a `deterministic` route
Switchyard calls the classifier LLM *before* tier selection, so a processor-level
denial has already spent money. Deny at the Edge; shape at the processor.

## 2. Decision API contract

`POST /v1/policy/decide` — hot path, hard 15 ms timeout.

```jsonc
// request
{
  "decision_id": "tfd_01J...",
  "tenant_id": "acct_8821",
  "contract_id": "ctr_4410",
  "route": "tf-escalating",
  "requested_tier": "strong",
  "session_id": "...", "agent_id": "...", "is_subagent": "true"
}
```

```jsonc
// response
{
  "decision_id": "tfd_01J...",
  "action": "allow",          // allow | throttle | deny
  "rate_card_id": "rc_poc_v1",// PINNED here -> occurrence-time rating
  "force_tier": null,         // throttle: "weak"
  "max_tokens_cap": null,     // throttle: clamp on request body
  "status": null,             // deny: 402 | 429
  "code": null,               // deny: budget_exceeded | not_entitled | rate_limited
  "budget_state": "ok",       // ok | warn | throttle | deny
  "remaining_usd": 412.55
}
```

Design positions:

- **`fail_open` is configurable, and the default differs by topology.** Shared
  NCP gateways fail open (availability wins). Sovereign and FSI deployments fail
  **closed** — an unreachable control plane must not mean ungoverned spend.
- **`rate_card_id` is pinned at decision time**, echoed back on the intake
  record via `x-tokenforge-rate-card-id`. A late-arriving usage record then rates
  at *occurrence* time, which is what makes a mid-cycle price change safe.
- **Budget ledger:** Redis for hot counters (atomic decrement on decide),
  Postgres for the durable ledger, reconciled against actual intake spend. The
  preflight *reserves*; the intake record *settles*.

### The NVIDIA-shaped opportunity

NVIDIA documents a **NeMo Relay Switchyard Decision API** — Relay calls out for a
routing decision (`decision_api_url`, `decision_profile_id`,
`decision_timeout_millis` default 25, `mode: enforce|observe_only`) and posts
history to `/v1/atof/events`.

**None of it exists in the OSS repo** — grep for `atof` and `decision` returns
nothing, and the JSON contract is unpublished. But the shape is unmistakably the
same as the contract above, including the sub-25 ms budget and the
enforce/observe split.

> **If TokenForge implements that contract, it becomes a drop-in
> NVIDIA-sanctioned decision service.** This is the single highest-leverage
> unknown in the design. **Action: get the contract from NVIDIA.**

## 3. The processor constraint, and how we work around it

`switchyard_rust/core.py` wraps every processor exception:

```python
try:
    current = await process(ctx, current)
except Exception as error:
    raise _processor_error(error) from error   # -> SwitchyardProcessorError
```

So a raised `TokenForgePolicyError(402, …)` loses its status. Two consequences:

1. **Workaround (today):** stamp `_upstream_http_status` and `_error_source` on
   `ctx.metadata` before raising. Fragile — it depends on internal key names.
2. **Also note** the 500 path renders `repr(exc)[:200]`, which leaks Python
   exception class names to clients. Never put tenant identifiers or budget
   figures in a policy exception message.

Both are fixed by upstream PR #1.

## 4. Upstream contributions

Ordered by acceptance likelihood. Each is generally useful rather than
M360-specific — this is how we earn standing in the NVIDIA ecosystem instead of
carrying a fork.

### PR 1 — Typed processor rejection
Let a processor raise `SwitchyardPolicyError(status, code, message)` that
survives to the HTTP envelope instead of flattening to
`SwitchyardProcessorError`. Small, surgical, unblocks every policy and guardrail
integration. **Highest priority.**

### PR 2 — Externalize `MODEL_PRICING`
Load the price table from YAML/env, keeping the current dict as default. Today
it is hardcoded in `switchyard/lib/cost_estimator.py`, so `cost_usd` is simply
wrong for any deployment with negotiated rates — which is every NCP and every
enterprise. Clear community win.

### PR 3 — Activate the profile registry
`_PROFILE_CONFIGS` and `@profile_config(..., register=True)` already exist but
are dormant: every shipped config uses `register=False`, `route_bundle.py` never
calls `lookup_profile_config`, and route types resolve through a closed
hardcoded alias dict. Honouring the registry on the YAML path turns "patch four
dicts in `route_bundle.py`" into a supported extension path.

### PR 4 — Cost-aware router (`type: cost_router`)
A tier picker that consults an external decision endpoint or a local price
table. Same shape as the existing `latency_service` router, different objective.
Directly serves NVIDIA's own Token Factory narrative — Switchyard has routers
for probability, classification, signals and latency, but **none for price**.

Until PRs 1 and 3 land, Phase B runs via the programmatic `RouteBundle` path —
launchers are explicitly supported in constructing a `RouteBundle` directly,
so **no fork of Switchyard is required**, only our own launcher.

## 5. Open items

1. **Streaming attribution.** Response processors are chunk-blind: they run once,
   before the `StreamingResponse` exists, and receive one `ChatResponse` wrapping
   an unconsumed stream. There is an opt-in tap —
   `attach_final_response_callback(response, served_model=…, callback=…)` fires
   when the stream drains — which is the mechanism to verify SSE token counts are
   complete before billing streamed traffic.
2. **Agent-session metering.** `ai.agent.session` needs a session-end signal.
   `x-switchyard-session-final` exists (plus `x-dynamo-session-final`) — confirm
   whether real clients send it reliably, or infer by timeout.
3. **Cross-format streaming is limited.** Streaming is same-format passthrough
   today; Anthropic/Responses inbound against a non-matching backend raises
   `NotImplementedError`. Constrain PoC routes accordingly.
