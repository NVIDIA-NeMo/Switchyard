# Metering integrity — correcting the intake-as-meter design

**This document revises §6.3 of the design spec.** The original design made the
intake sink the metering source of record, with reconciliation bolted on to
detect loss. That was the wrong shape, and the pushback was correct.

## 1. The flaw

Switchyard's intake sink is:

- **asynchronous** — POSTed off the response path,
- **bounded** — `max_queue_size` with an `on_queue_full` policy that can drop,
- **fire-and-forget** — a failed POST after `max_retries` is gone,
- **opt-in** — per request, via header or body `store=true`.

A revenue system of record can be none of those things. "Reconcile and block
invoicing above 2% drift" is a *detection* control dressed up as a *correctness*
control. It tells you that you lost money; it does not stop you losing it. And
the 2% threshold is arbitrary — at scale, 0.5% of a large token bill is real
money, and blocking the invoice run is an operational failure, not a fix.

Worse, it makes an availability property of a telemetry queue into a determinant
of revenue accuracy. That is a dependency no auditor should accept.

## 2. The correction: meter on the response path

TokenForge Edge is **already in the request path** — it has to be, because
Switchyard has no authentication and no tenant concept, so something must sit in
front to resolve an API key to a tenant. Since the Edge is a reverse proxy, it is
also in the *response* path. So it can read `usage` directly out of the response
body, synchronously, before returning to the client.

That meter is **lossless by construction**: if the client got a billable
response, the Edge saw it. There is no queue to overflow.

| | Edge meter | Intake sink | `/v1/stats` + `/metrics` |
| --- | --- | --- | --- |
| Timing | synchronous, in-path | async, off-path | scrape |
| Loss mode | none — same lifetime as the response | queue drop, retry exhaustion | counter reset |
| Tenant attribution | **native** (Edge owns identity) | via our stamped headers | **none** |
| Role | **invoice basis** | enrichment + cross-check | aggregate cross-check |

**Intake stops being the meter and becomes enrichment.** That is genuinely
valuable, because it carries three things the response body does not:

1. `cost_usd` / `cost_details` — Switchyard's supplier-cost calculation.
2. The routing decision — `router_type`, `routed_to`, `router_selected_model`,
   `router_correlation_id`.
3. The cache breakdown — `cached_tokens`, `cache_creation_tokens`.

So the two meters are complementary, not redundant, and each covers the other's
blind spot. Implemented in
[`../demo/tokenforge_edge.py`](../demo/tokenforge_edge.py) (`_post_edge_meter`)
and [`../demo/tokenforge_core.py`](../demo/tokenforge_core.py)
(`POST /v1/meter/edge`).

### What the demo shows

```
edge_settled          35 reqs     497183 tok   basis=invoice
intake_async          35 reqs     497183 tok   basis=enrichment
switchyard_stats      36 reqs     514864 tok   basis=cross-check
intake loss vs settled meter : 0.00%   verdict=OK
```

The `switchyard_stats` divergence is **expected and explainable**, which is the
point of having three meters:

- **+1 request** — the unauthenticated bypass request in scenario 5. Someone
  called Switchyard directly, Switchyard served it (it has no auth), and Core
  quarantined the record for having no tenant. The count difference is the
  security finding.
- **+17,681 tokens** — classifier-LLM spend on `deterministic` routes. The Edge
  cannot see it (it is an internal Switchyard call), but it is real cost that
  must be attributed or margin is overstated. Core folds it in from the intake
  record's `classifier_usage`.

A single-meter design cannot distinguish "we lost records" from "these two
numbers legitimately differ." A three-meter design can, and that is what makes
the drift number actionable instead of alarming.

## 3. Residual gaps — stated plainly

### 3.1 Reserve vs settle

The demo ends with Northwind at **100.4%** of a $0.35 cap. That overshoot is not
a bug, it is the structural gap in any preflight-enforced budget: the decision to
allow happens *before* the tokens are known, so the last permitted request can
cross the cap.

Mitigations, in order of preference:

1. **Reserve a worst-case estimate at preflight**, settle the difference on the
   response-path meter. Reserve `max_tokens × strong-tier output rate`, so the
   ledger is conservative and the overshoot becomes an *under*-shoot.
2. **Soft-stop margin.** Deny at 97% rather than 100%, sized to the largest
   single-request cost on the contract's rate card.
3. **Contractual framing.** Make the cap a *commit + overage* construct rather
   than a hard stop — which is what RevenueOS already models, and what most
   enterprise customers actually want.

Option 1 plus option 3 is the recommendation. A hard cap that can be exceeded by
one request is worse than an overage clause that is honest about it.

### 3.2 Streaming

For SSE responses the usage block is not in a JSON body the Edge can simply read.
The Edge must either accumulate chunks and parse the terminal usage frame, or
ensure `stream_options: {"include_usage": true}` is set on the upstream request.

**Unverified:** whether Switchyard passes `stream_options` through, whether it
injects `include_usage` itself, and whether usage survives its format
translation. The research pass on this was cut short. **This must be verified
before billing any streamed traffic** — it is the single largest open risk in the
metering design, because agentic clients stream by default.

Note also that Switchyard's response processors are chunk-blind (they run once,
before the `StreamingResponse` exists, receiving one `ChatResponse` wrapping an
unconsumed stream), but there is an opt-in tap —
`attach_final_response_callback(response, served_model=…, callback=…)` — which
fires when the stream drains. That is the mechanism if the Edge cannot get usage
itself.

### 3.3 The zero-token defect

Switchyard 0.1.0 has a known issue where Codex Responses-API tasks may record
`0` token usage in `/v1/stats`.

**Hypothesis (unverified):** the defect is in Switchyard's own accounting layer,
not in the upstream provider response — in which case an Edge reading `usage`
from the response body is **immune to it**, and moving the meter to the response
path fixes the defect rather than merely detecting it.

If instead the upstream response genuinely carries no usage for that path, no
proxy can meter it and the only honest options are to (a) exclude that path from
token-based pricing and price it per-request, or (b) block it at the Edge for
billable tenants. The zero-token quarantine in Core stays either way.

**Action: verify which layer the defect lives in.** It changes the answer
materially.

### 3.4 Hardening the intake sink anyway

Even as enrichment, intake loss should be minimised, and this is configuration
rather than architecture:

- Raise `max_queue_size` well above peak concurrency × retry latency.
- Set `on_queue_full` to a blocking/backpressure policy **if one exists** — the
  `IntakeQueueFullPolicy` enum's variants and default were not verified. If the
  only policy is drop, that is worth an upstream PR.
- Confirm whether there is a drain/flush on `SIGTERM`. If queued records are lost
  on rolling restarts, enrichment gaps will correlate with deploys — which looks
  exactly like a billing anomaly.

## 4. Revised controls

| Control | Purpose | Status |
| --- | --- | --- |
| Response-path Edge meter | **invoice basis**, lossless | implemented |
| Idempotency via `event_id` | intake retries must not double-bill | implemented |
| Three-meter reconciliation | distinguish loss from legitimate divergence | implemented |
| Quarantine: no tenant | revenue leak + security finding | implemented |
| Quarantine: zero tokens | metering hole, not a free request | implemented |
| Quarantine: unpriced model | revenue leak | implemented |
| Reserve-then-settle ledger | close the cap-overshoot gap | designed, §3.1 |
| Streaming usage verification | unblock billing streamed traffic | **open, §3.2** |
| Zero-token root-cause | may be fixed outright by the Edge meter | **open, §3.3** |
| Immutable raw-record store | audit chain source → invoice line | designed |

The invoice-blocking drift threshold stays, but its role changes: it is now a
tripwire on *enrichment* completeness, not a guard on billing correctness.
Billing correctness is structural.
