# TokenForge × NVIDIA Switchyard

Design specification and prototype assets for integrating **Monetize360
TokenForge** (AI token control plane + FinOps) and **RevenueOS** (commerce layer)
with [NVIDIA Switchyard](https://github.com/NVIDIA-NeMo/Switchyard).

> **Status:** Draft v0.1 design spec. Not a validated architecture.

## The thesis in one paragraph

Switchyard is a **routing fabric**, not a gateway. Verified against the source at
`main`, it ships with no authentication ("the proxy ignores inbound auth"), no
multi-tenancy, no quota or budget enforcement, no cost-aware routing, and a
hardcoded price table. Every one of those absences is a Monetize360 line item.
Switchyard decides *which model*; **TokenForge** decides *whether, for whom, and
at what cost*; **RevenueOS** turns the result into a rated charge, an invoice,
and a settlement.

```
RevenueOS   ── commerce: catalog · rating · billing · entitlements · settlement
Mbrix       ── composable build layer · M360 Agents
TokenForge  ── control plane: identity · policy & quota · budget · attribution
Switchyard  ── fabric: protocol translation · tier routing · telemetry
NIM · Nemotron · vLLM · OpenAI · Anthropic
```

## Documents

| Document | Contents |
| --- | --- |
| [00-design-spec.md](docs/00-design-spec.md) | **Start here.** Verified integration surface, target architecture, data model, phasing, open questions |
| [01-phase-a-metering.md](docs/01-phase-a-metering.md) | Zero-code metering overlay — runnable PoC |
| [02-phase-b-enforcement.md](docs/02-phase-b-enforcement.md) | Policy decision API, enforcement design, 4 upstream PRs |
| [03-metering-integrity.md](docs/03-metering-integrity.md) | **Supersedes spec §6.3** — why the meter moved to the response path |
| [04-caching-decision.md](docs/04-caching-decision.md) | Prefix vs semantic caching: source, defer, and how to price a cache hit |

## Prototype

| Path | Contents |
| --- | --- |
| [`config/route.tokenforge.yaml`](config/route.tokenforge.yaml) | Switchyard route bundle (verified schema) |
| [`prototype/tokenforge_m360/models.py`](prototype/tokenforge_m360/models.py) | Metered usage event — the TokenForge → RevenueOS contract |
| [`prototype/tokenforge_m360/rate_card.py`](prototype/tokenforge_m360/rate_card.py) | Versioned rate card, cost/price split, margin |
| [`prototype/tokenforge_m360/intake_receiver.py`](prototype/tokenforge_m360/intake_receiver.py) | Phase A intake sink, rating, reconciliation, spend tree |
| [`prototype/tokenforge_m360/policy_processor.py`](prototype/tokenforge_m360/policy_processor.py) | Phase B enforcement processor |

## Runnable demo

```bash
python3 tokenforge/demo/run_demo.py
```

Stdlib only — no pip install, no venv, no Rust toolchain, Python 3.9+. Starts a
Switchyard stand-in (:8080), TokenForge Core (:9900) and TokenForge Edge (:9000),
drives five scenarios, prints a report and writes `tokenforge/demo/dashboard.html`.
See [demo/README.md](demo/README.md).

## Quick start against real Switchyard (Phase A)

```bash
pip install fastapi uvicorn httpx pydantic
uvicorn tokenforge_m360.intake_receiver:app --port 9900   # from prototype/
```

```bash
export SWITCHYARD_INTAKE_TARGET_URL=http://localhost:9900/v1/intake/switchyard
switchyard serve --routing-profiles tokenforge/config/route.tokenforge.yaml
```

That is the entire Phase A integration — no Switchyard code changes.

## Three findings that shaped the design

1. **`--intake-target-url` is the metering tap.** Redirects per-request records —
   `cost_usd`, full token breakdown, routing decision — to any endpoint we own,
   asynchronously off the response path. Zero code changes.
2. **`x-switchyard-tier` gives us cost-aware routing today.** Switchyard's
   shipped `header_routing` profile already routes on it, so a near-cap tenant
   can be downgraded `strong` → `weak` with no Switchyard code at all.
3. **The agent hierarchy headers are the sleeper asset.**
   `x-switchyard-agent-id` / `-parent-agent-id` / `-is-subagent` let us build a
   **spend tree** for a multi-agent run — a sub-agent's tokens roll up to the
   parent task, and the task to a tenant contract. Nothing in the market does
   this.

## Two things Switchyard does *not* do

Stated plainly so no customer material overclaims:

- **No caching.** No semantic cache, no response cache, no embeddings, no vector
  store. `cached_tokens` is pass-through of the *provider's* prompt cache. The
  "Cache & shaping" box in the M360 architecture diagram is **not** satisfied by
  Switchyard — resolved as source-prefix / defer-semantic in
  [04-caching-decision.md](docs/04-caching-decision.md).
- **No auth, anywhere.** Every route is open, including `POST /v1/stats/reset`
  and `/metrics`. Switchyard must never be exposed directly to external
  customers.
