"""TokenForge Edge — authN/Z, budget preflight, tier shaping, response-path meter.

Everything Switchyard deliberately omits happens here. The Edge is the only
component clients ever reach; Switchyard sits behind it and is never exposed.

Four jobs:

  1. **Authenticate.** Resolve an M360 API key to a tenant, contract, rate card,
     cost centre and entitlements. Switchyard never validates a key -- it only
     harvests and forwards it.
  2. **Preflight budget.** Decide allow / throttle / deny BEFORE Switchyard sees
     the request. This ordering is deliberate: on a `deterministic` route
     Switchyard calls a classifier LLM *before* tier selection, so a denial
     inside a Switchyard request processor has already burned classifier tokens.
     Denying here burns nothing.
  3. **Shape.** On throttle, rewrite `x-switchyard-tier` to `weak` and clamp
     `max_tokens`. Switchyard's shipped `header_routing` profile already honours
     that header, so cost-aware routing needs no Switchyard code at all.
  4. **Meter on the response path.** Read `usage` straight out of the response
     body and post it to Core as an independent meter.

Job 4 is the important one architecturally. The intake sink is asynchronous and
lossy by design (bounded queue, drop-on-full), so it cannot be the sole system of
record for an invoice. The Edge is synchronous and in-path, so its meter is
lossless by construction. Intake then becomes *enrichment* -- it carries
`cost_usd` and the routing decision, which the response body does not -- and the
two meters cross-check each other.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional, Tuple

from _http import Ctx, Service, post_json

SWITCHYARD = "http://127.0.0.1:8080"
CORE = "http://127.0.0.1:9900"

service = Service("tokenforge-edge", 9000)
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Key registry. Production: Postgres + Redis cache, rotated keys, RBAC.
# ---------------------------------------------------------------------------

TENANTS: Dict[str, Dict[str, Any]] = {
    "m360_key_acme": {
        "tenant_id": "acct_8821", "name": "Acme FSI",
        "contract_id": "ctr_4410", "rate_card_id": "rc_poc_v1",
        "cost_center": "eng-platform", "route_tag": "enterprise",
        "monthly_cap_usd": 5.00, "allowed_routes": ["tf-tiered", "tf-escalating", "tf-nemotron"],
        "max_tier": "strong",
    },
    "m360_key_northwind": {
        "tenant_id": "acct_9107", "name": "Northwind Telco",
        "contract_id": "ctr_5522", "rate_card_id": "rc_poc_v1",
        "cost_center": "ai-products", "route_tag": "partner",
        # Deliberately tight so the demo walks ok -> warn -> throttle -> deny.
        "monthly_cap_usd": 0.35, "allowed_routes": ["tf-tiered", "tf-escalating"],
        "max_tier": "strong",
    },
    "m360_key_sovereign": {
        "tenant_id": "acct_7003", "name": "Sovereign AI Cloud",
        "contract_id": "ctr_6001", "rate_card_id": "rc_poc_v1",
        "cost_center": "gov-inference", "route_tag": "marketplace",
        "monthly_cap_usd": 5.00, "allowed_routes": ["tf-nemotron"],
        # Entitled to the cheap tier only -- an entitlement, not a budget state.
        "max_tier": "weak",
    },
}

# Budget ledger. Production: Redis for hot counters (atomic decrement at
# preflight), Postgres for the durable ledger, reconciled against actual spend.
# The preflight RESERVES; the settled meter SETTLES.
SPEND: Dict[str, float] = {}
DECISIONS: list = []

WARN_AT = 0.60      # of cap
THROTTLE_AT = 0.85
DENY_AT = 1.00


def _tenant_for_key(api_key: Optional[str]) -> Optional[dict]:
    return TENANTS.get(api_key or "")


def _budget_state(tenant: dict) -> Tuple[str, float, float]:
    with _lock:
        spent = SPEND.get(tenant["tenant_id"], 0.0)
    cap = tenant["monthly_cap_usd"]
    used = spent / cap if cap else 0.0
    if used >= DENY_AT:
        return "deny", spent, cap
    if used >= THROTTLE_AT:
        return "throttle", spent, cap
    if used >= WARN_AT:
        return "warn", spent, cap
    return "ok", spent, cap


def record_spend(tenant_id: str, amount_usd: float) -> None:
    with _lock:
        SPEND[tenant_id] = SPEND.get(tenant_id, 0.0) + amount_usd


# ---------------------------------------------------------------------------
# Decision API — the contract TokenForge would expose to NeMo Relay's
# Switchyard Decision plugin (decision_api_url / mode: enforce|observe_only).
# ---------------------------------------------------------------------------


@service.route("POST", "/v1/policy/decide")
def decide(ctx: Ctx):
    tenant = _tenant_for_key(ctx.body.get("api_key"))
    decision_id = "tfd_" + uuid.uuid4().hex[:24]
    if tenant is None:
        return 200, {"decision_id": decision_id, "action": "deny",
                     "status": 401, "code": "invalid_key"}
    return 200, _decide_for(tenant, ctx.body.get("route"),
                            ctx.body.get("requested_tier"), decision_id)


def _decide_for(tenant: dict, route: Optional[str], requested_tier: Optional[str],
                decision_id: str) -> dict:
    if route not in tenant["allowed_routes"]:
        return {"decision_id": decision_id, "action": "deny", "status": 403,
                "code": "not_entitled",
                "message": "route %s not entitled for contract %s" % (route, tenant["contract_id"]),
                "budget_state": "ok"}

    state, spent, cap = _budget_state(tenant)
    base = {
        "decision_id": decision_id,
        "rate_card_id": tenant["rate_card_id"],   # PINNED -> occurrence-time rating
        "budget_state": state,
        "spent_usd": round(spent, 6),
        "cap_usd": cap,
        "remaining_usd": round(max(cap - spent, 0.0), 6),
    }

    if state == "deny":
        base.update({"action": "deny", "status": 402, "code": "budget_exceeded",
                     "message": "monthly token budget of $%.2f exhausted" % cap})
        return base

    # Entitlement ceiling is independent of budget state.
    tier = requested_tier or "weak"
    if tenant["max_tier"] == "weak" and tier == "strong":
        base.update({"action": "throttle", "force_tier": "weak",
                     "reason": "entitlement_ceiling"})
        return base

    if state == "throttle":
        base.update({"action": "throttle", "force_tier": "weak",
                     "max_tokens_cap": 512, "reason": "budget_throttle"})
        return base

    base["action"] = "allow"
    base["force_tier"] = tier
    return base


# ---------------------------------------------------------------------------
# The client-facing proxy
# ---------------------------------------------------------------------------


@service.route("POST", "/v1/chat/completions")
def chat(ctx: Ctx):
    api_key = _extract_key(ctx)
    tenant = _tenant_for_key(api_key)
    if tenant is None:
        # Switchyard would have served this request. The Edge is the only reason
        # an invalid key fails.
        return 401, {"error": {"message": "invalid API key", "type": "authentication_error",
                               "code": "invalid_api_key"}}

    route = ctx.body.get("model")
    requested_tier = ctx.header("x-tokenforge-requested-tier") or "weak"
    decision_id = "tfd_" + uuid.uuid4().hex[:24]
    decision = _decide_for(tenant, route, requested_tier, decision_id)
    with _lock:
        DECISIONS.append(dict(decision, tenant_id=tenant["tenant_id"], route=route))

    if decision["action"] == "deny":
        # A clean, correctly-typed HTTP status -- which a Switchyard request
        # processor cannot produce today, because every processor exception is
        # flattened to SwitchyardProcessorError. See upstream PR #1.
        return decision["status"], {"error": {
            "message": decision.get("message", "denied"),
            "type": "budget_error", "code": decision["code"],
            "budget_state": decision["budget_state"],
            "spent_usd": decision.get("spent_usd"),
            "cap_usd": decision.get("cap_usd"),
        }}

    body = dict(ctx.body)
    tier = decision.get("force_tier", requested_tier)
    cap = decision.get("max_tokens_cap")
    if cap:
        current = body.get("max_tokens")
        body["max_tokens"] = min(current, cap) if isinstance(current, int) else cap

    headers = {
        # Tenant attribution. Switchyard has no tenant concept, so these are
        # ours -- and they survive because it retains every inbound header.
        "x-tokenforge-tenant-id": tenant["tenant_id"],
        "x-tokenforge-contract-id": tenant["contract_id"],
        "x-tokenforge-rate-card-id": tenant["rate_card_id"],
        "x-tokenforge-cost-center": tenant["cost_center"],
        "x-tokenforge-route-tag": tenant["route_tag"],
        "x-tokenforge-decision-id": decision["decision_id"],
        "x-tokenforge-budget-state": decision["budget_state"],
        # Switchyard's OWN header, honoured by its shipped header_routing
        # profile: cost-aware routing with zero Switchyard code.
        "x-switchyard-tier": tier,
        # Never let a billable request escape metering.
        "x-switchyard-intake-enabled": "true",
        # Pass agent-hierarchy headers straight through -- they build the spend tree.
        "x-switchyard-session-id": ctx.header("x-switchyard-session-id"),
        "x-switchyard-task-id": ctx.header("x-switchyard-task-id"),
        "x-switchyard-agent-id": ctx.header("x-switchyard-agent-id"),
        "x-switchyard-parent-agent-id": ctx.header("x-switchyard-parent-agent-id"),
        "x-switchyard-is-subagent": ctx.header("x-switchyard-is-subagent"),
    }

    status, response = post_json(SWITCHYARD + "/v1/chat/completions", body, headers)

    # --- Response-path meter ------------------------------------------------
    # Synchronous and in-path, therefore lossless. This is the settled meter;
    # the async intake record is enrichment and cross-check, not the source of
    # truth. It also sidesteps the known 0.1.0 zero-token defect, which lives in
    # Switchyard's own accounting rather than in the upstream response.
    if status == 200 and isinstance(response, dict):
        usage = response.get("usage") or {}
        _post_edge_meter(tenant, decision, route, tier, usage, response)

    return status, response


def _post_edge_meter(tenant: dict, decision: dict, route: Optional[str],
                     tier: str, usage: dict, response: dict) -> None:
    payload = {
        "tenant_id": tenant["tenant_id"],
        "contract_id": tenant["contract_id"],
        "rate_card_id": tenant["rate_card_id"],
        "decision_id": decision["decision_id"],
        "route": route,
        "tier": tier,
        "served_model": response.get("model"),
        "correlation_id": (response.get("_switchyard") or {}).get(
            "x-switchyard-router-correlation-id"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }
    try:
        _, result = post_json(CORE + "/v1/meter/edge", payload)
        if isinstance(result, dict) and result.get("price_usd"):
            # Settle the budget against the rated price.
            record_spend(tenant["tenant_id"], float(result["price_usd"]))
    except Exception:  # noqa: BLE001 - never fail a served request on metering
        pass


def _extract_key(ctx: Ctx) -> Optional[str]:
    auth = ctx.header("authorization")
    if auth:
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value:
            return value
    return ctx.header("x-api-key")


@service.route("GET", "/v1/budgets")
def budgets(ctx: Ctx):
    rows = []
    for key, tenant in TENANTS.items():
        state, spent, cap = _budget_state(tenant)
        rows.append({
            "tenant_id": tenant["tenant_id"], "name": tenant["name"],
            "route_tag": tenant["route_tag"], "max_tier": tenant["max_tier"],
            "spent_usd": round(spent, 6), "cap_usd": cap,
            "used_pct": round(spent / cap * 100, 1) if cap else 0.0,
            "budget_state": state,
        })
    return 200, {"tenants": rows}


@service.route("GET", "/v1/decisions")
def decisions(ctx: Ctx):
    with _lock:
        return 200, {"count": len(DECISIONS), "decisions": DECISIONS[-200:]}


@service.route("GET", "/health")
def health(ctx: Ctx):
    return 200, {"status": "ok"}
