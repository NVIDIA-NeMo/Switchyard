# TokenForge × Switchyard — runnable demo

```bash
python3 tokenforge/demo/run_demo.py
```

Add `--serve` to keep the services up for live poking. **Stdlib only** — no pip
install, no venv, no Rust toolchain, Python 3.9+. It runs on a locked-down laptop
or inside an air-gapped environment, which is the point.

## What starts

| Service | Port | Role |
| --- | --- | --- |
| Switchyard stand-in | 8080 | routing fabric — no auth, no tenancy, no budget |
| TokenForge Core | 9900 | intake, rating, margin, reconciliation, spend tree |
| TokenForge Edge | 9000 | authN/Z, budget preflight, tier shaping, settled meter |

## Honest disclosure about the stand-in

`switchyard_sim.py` is **not Switchyard.** Real Switchyard needs Python 3.12+ and
a Rust toolchain, and `nemo-switchyard` is not published to PyPI, so it cannot run
on an arbitrary machine.

The stand-in reproduces Switchyard's **observable contract**, read from the source
at `main`: the five endpoints plus `/v1/stats` and `/metrics`, the credential
precedence and `[REDACTED]` header retention, `x-switchyard-tier` routing, the
intake sink's opt-in and payload shape, `MODEL_PRICING` verbatim, and the
Prometheus metric names and content type. It also faithfully reproduces the
*absences* — no auth, no tenancy, no budget, no caching.

**Nothing in TokenForge Core or Edge is simulated.** They are real
implementations that consume only Switchyard's public contract, and
`rate_card.py` is imported unchanged from `../prototype/`. Swapping the stand-in
for the real proxy is two commands and zero TokenForge code changes:

```bash
export SWITCHYARD_INTAKE_TARGET_URL=http://127.0.0.1:9900/v1/intake/switchyard
switchyard serve --routing-profiles tokenforge/config/route.tokenforge.yaml --port 8080
```

## Scenarios

| # | Scenario | Design claim it proves |
| --- | --- | --- |
| 1 | Baseline enterprise traffic | tenant-attributed rating, per-request margin |
| 2 | Multi-agent run, 1 parent + 3 sub-agents | spend tree rolls up to the parent task |
| 3 | Budget walk: ok → warn → throttle → deny | Phase B enforcement, clean `402` |
| 4 | Entitlement ceiling | `strong` request forced to `weak`; route `403` |
| 5 | Forged key + direct-to-Switchyard bypass | `401` at Edge; bypass is served then quarantined |

## Reading the output

Three numbers matter.

**Margin by model.** The cheap tier carries the higher multiple (~3.7× vs ~2.3×).
That is what makes a budget-driven downgrade *profitable* rather than merely
cheaper — the customer pays less and margin percentage goes up.

**Budget overshoot.** Northwind lands slightly above 100% of its cap. Not a bug:
preflight decides before token counts are known, so the last permitted request can
cross. This is the reserve-vs-settle gap, with mitigations in
[`../docs/03-metering-integrity.md`](../docs/03-metering-integrity.md) §3.1.

**Three-meter divergence.** `switchyard_stats` reports more requests and more
tokens than the settled meter, and both differences are explainable:

- **+1 request** — the scenario-5 bypass. Switchyard served an unauthenticated
  request; Core quarantined it for having no tenant. The delta *is* the security
  finding.
- **+~17k tokens** — classifier-LLM spend on `deterministic` routes. Invisible to
  the Edge, real cost, folded in from the intake record or margin is overstated.

A single-meter design cannot tell "we lost records" from "these numbers
legitimately differ." That is why there are three.

## Endpoints to poke

```
GET  :9900/v1/margin                 per-tenant and per-model revenue, cost, margin
GET  :9900/v1/reconcile              three-meter integrity report
GET  :9900/v1/spend-tree/{task_id}   agent-hierarchy spend rollup
GET  :9900/v1/events                 raw rated events
POST :9900/v1/meter/edge             settled response-path meter
GET  :9000/v1/budgets                budget state per tenant
GET  :9000/v1/decisions              every allow/throttle/deny decision
POST :9000/v1/policy/decide          the decision API contract
GET  :8080/metrics                   Switchyard Prometheus (no cost, no tenant label)
GET  :8080/v1/stats                  Switchyard aggregate counters
```

## Demo tenants

| Key | Tenant | Cap | Max tier | Purpose |
| --- | --- | --- | --- | --- |
| `m360_key_acme` | Acme FSI | $5.00 | strong | baseline + spend tree |
| `m360_key_northwind` | Northwind Telco | $0.35 | strong | budget walk to denial |
| `m360_key_sovereign` | Sovereign AI Cloud | $5.00 | **weak** | entitlement ceiling |

## Dashboard

`run_demo.py` writes `dashboard.html` — self-contained, light and dark, stat tiles
plus horizontal bars for margin and budget state. Open it directly in a browser.
