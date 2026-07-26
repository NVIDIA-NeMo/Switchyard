"""TokenForge Core — intake, rating, reconciliation, reporting.

This is real product code, not a simulation. It is the stdlib port of
`prototype/tokenforge_m360/intake_receiver.py` (which uses FastAPI/pydantic and
needs Python 3.12), and it imports the *actual* rate-card and margin logic from
`prototype/tokenforge_m360/rate_card.py` unchanged.

Because it only consumes Switchyard's intake contract, swapping the stand-in for
the real proxy requires no change here at all.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))

from tokenforge_m360.rate_card import POC_RATE_CARD, compute_margin  # noqa: E402

from _http import Ctx, Service, get_json  # noqa: E402

SWITCHYARD_BASE = os.environ.get("SWITCHYARD_BASE_URL", "http://127.0.0.1:8080")
DRIFT_WARN_PCT = 0.5
DRIFT_BLOCK_PCT = 2.0

service = Service("tokenforge-core", 9900)

_lock = threading.Lock()
EVENTS: Dict[str, dict] = {}          # async intake records (enrichment)
EDGE_METER: Dict[str, dict] = {}      # synchronous response-path meter (settled)
QUARANTINE: List[dict] = []


def _first(headers: Dict[str, Any], name: str) -> Optional[str]:
    """Switchyard retains headers as dict[str, list[str]], lowercased."""
    value = headers.get(name)
    if isinstance(value, list):
        return value[0].strip() if value else None
    return value.strip() if isinstance(value, str) else None


def _as_bool(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def derive_event_id(request_id, correlation_id, created_at) -> str:
    """Deterministic idempotency key.

    The intake sink retries, so a natural key is the only thing between a retry
    and a double-billed request.
    """
    material = "|".join(str(p or "") for p in (request_id, correlation_id, created_at))
    return "tfe_" + hashlib.sha256(material.encode()).hexdigest()[:32]


@service.route("POST", "/v1/intake/switchyard")
def intake(ctx: Ctx):
    record = ctx.body
    headers = record.get("request_headers") or {}
    routing = record.get("routing") or {}
    usage = record.get("usage") or {}

    tenant_id = _first(headers, "x-tokenforge-tenant-id")
    if not tenant_id:
        # Bypassed TokenForge Edge. Quarantine rather than guess: an
        # unattributed token is a revenue leak, and in a multi-tenant
        # deployment it is also a security finding.
        _quarantine("missing_tenant", {"record_model": record.get("model")})
        return 202, {"status": "quarantined", "reason": "missing_tenant"}

    event_id = derive_event_id(
        record.get("request_id"),
        routing.get("router_correlation_id"),
        record.get("created_at"),
    )
    with _lock:
        if event_id in EVENTS:
            return 200, {"status": "duplicate", "event_id": event_id}

    routed_to = routing.get("routed_to")
    served_model = routing.get("router_selected_model") or ""
    escalated = routed_to == "strong"

    meters = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(usage.get("cached_tokens") or 0),
        "cache_creation_tokens": int(usage.get("cache_creation_tokens") or 0),
        "router_escalations": 1 if escalated else 0,
    }
    # A classifier call on a `deterministic` route is real spend the customer
    # never asked for. Attribute it, or margin is overstated.
    classifier = record.get("classifier_usage") or {}
    if classifier:
        meters["classifier_prompt_tokens"] = int(classifier.get("prompt_tokens") or 0)
        meters["classifier_completion_tokens"] = int(classifier.get("completion_tokens") or 0)

    cost = {
        "cost_usd": float(record.get("cost_usd") or 0.0),
        "cost_input_usd": float(record.get("cost_input_usd") or 0.0),
        "cost_output_usd": float(record.get("cost_output_usd") or 0.0),
        "cost_details": record.get("cost_details") or {},
    }

    event = {
        "event_id": event_id,
        "source": "switchyard",
        "decision_id": _first(headers, "x-tokenforge-decision-id"),
        "occurred_at": record.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "contract_id": _first(headers, "x-tokenforge-contract-id"),
        "cost_center": _first(headers, "x-tokenforge-cost-center"),
        "route_tag": _first(headers, "x-tokenforge-route-tag"),
        "session": {
            "session_id": _first(headers, "x-switchyard-session-id"),
            "task_id": _first(headers, "x-switchyard-task-id"),
            "agent_id": _first(headers, "x-switchyard-agent-id"),
            "parent_agent_id": _first(headers, "x-switchyard-parent-agent-id"),
            "is_subagent": _as_bool(_first(headers, "x-switchyard-is-subagent")),
        },
        "routing": {
            "route": record.get("model"),
            "router_type": routing.get("router_type"),
            "routed_to": routed_to,
            "selected_model": served_model,
            "selected_provider": routing.get("router_selected_provider"),
            "tier": _first(headers, "x-switchyard-tier"),
            "correlation_id": routing.get("router_correlation_id"),
        },
        "meters": meters,
        "cost": cost,
        "price": None,
        "integrity": {"reconciled": False, "quarantined": False, "quarantine_reason": None},
    }

    # --- Zero-token guard -------------------------------------------------
    # Known 0.1.0 defect: Codex Responses-API tasks may record 0 token usage.
    # A successful response with no completion tokens is a metering hole.
    if meters["completion_tokens"] == 0 and not record.get("error"):
        event["integrity"] = {"reconciled": False, "quarantined": True,
                              "quarantine_reason": "zero_completion_tokens"}
        with _lock:
            EVENTS[event_id] = event
        _quarantine("zero_completion_tokens", {"event_id": event_id})
        return 202, {"status": "quarantined", "event_id": event_id}

    # --- Rating -----------------------------------------------------------
    card = POC_RATE_CARD
    if not card.is_priced(served_model):
        event["integrity"] = {"reconciled": False, "quarantined": True,
                              "quarantine_reason": "unpriced_model:" + served_model}
        with _lock:
            EVENTS[event_id] = event
        _quarantine("unpriced_model", {"model": served_model})
        return 202, {"status": "quarantined", "event_id": event_id}

    amount = card.price_tokens(
        served_model,
        prompt_tokens=meters["prompt_tokens"],
        completion_tokens=meters["completion_tokens"],
        cached_tokens=meters["cached_tokens"],
        cache_creation_tokens=meters["cache_creation_tokens"],
        tier=event["routing"]["tier"],
        escalated=escalated,
    )
    margin_usd, margin_pct = compute_margin(amount, cost["cost_usd"])
    event["price"] = {
        "rate_card_id": card.rate_card_id,
        "amount_usd": amount,
        "margin_usd": margin_usd,
        "margin_pct": margin_pct,
    }

    with _lock:
        EVENTS[event_id] = event
    return 200, {"status": "rated", "event_id": event_id, "margin_pct": margin_pct}


def _quarantine(reason: str, detail: dict) -> None:
    with _lock:
        QUARANTINE.append(dict(detail, reason=reason))


# ---------------------------------------------------------------------------
# Response-path meter (the settled meter)
# ---------------------------------------------------------------------------


@service.route("POST", "/v1/meter/edge")
def meter_edge(ctx: Ctx):
    """Settled meter, posted synchronously by TokenForge Edge.

    The Edge sits in the response path, so this meter is lossless by
    construction -- unlike the async intake sink, which has a bounded queue and
    may drop on full. This is the system of record for an invoice; the intake
    record enriches it with supplier `cost_usd` and the routing decision, which
    the response body does not carry.
    """
    payload = ctx.body
    served_model = payload.get("served_model") or ""
    card = POC_RATE_CARD

    price = card.price_tokens(
        served_model,
        prompt_tokens=int(payload.get("prompt_tokens") or 0),
        completion_tokens=int(payload.get("completion_tokens") or 0),
        # The response body reports no cache breakdown -- see the honest
        # trade-off in docs/03-metering-integrity.md. Cache detail comes from
        # the intake record and is folded in at reconciliation.
        cached_tokens=0,
        tier=payload.get("tier"),
        escalated=payload.get("tier") == "strong",
    ) if card.is_priced(served_model) else 0.0

    key = payload.get("correlation_id") or payload.get("decision_id")
    with _lock:
        EDGE_METER[key] = {
            "tenant_id": payload.get("tenant_id"),
            "contract_id": payload.get("contract_id"),
            "decision_id": payload.get("decision_id"),
            "route": payload.get("route"),
            "tier": payload.get("tier"),
            "served_model": served_model,
            "prompt_tokens": int(payload.get("prompt_tokens") or 0),
            "completion_tokens": int(payload.get("completion_tokens") or 0),
            "price_usd": price,
        }
    return 200, {"status": "metered", "price_usd": price}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@service.route("GET", "/v1/margin")
def margin(ctx: Ctx):
    tenants: Dict[str, dict] = {}
    models: Dict[str, dict] = {}
    with _lock:
        events = list(EVENTS.values())

    for event in events:
        price = event.get("price")
        if not price:
            continue
        row = tenants.setdefault(event["tenant_id"], _blank_row())
        _accumulate(row, event, price)
        model_row = models.setdefault(event["routing"]["selected_model"], _blank_row())
        _accumulate(model_row, event, price)

    for row in list(tenants.values()) + list(models.values()):
        _finalize(row)
    return 200, {"tenants": tenants, "models": models}


def _blank_row() -> dict:
    return {"revenue_usd": 0.0, "cost_usd": 0.0, "margin_usd": 0.0,
            "requests": 0, "tokens": 0, "escalations": 0}


def _accumulate(row: dict, event: dict, price: dict) -> None:
    row["revenue_usd"] += price["amount_usd"]
    row["cost_usd"] += event["cost"]["cost_usd"]
    row["margin_usd"] += price["margin_usd"]
    row["requests"] += 1
    row["tokens"] += event["meters"]["prompt_tokens"] + event["meters"]["completion_tokens"]
    row["escalations"] += event["meters"]["router_escalations"]


def _finalize(row: dict) -> None:
    for key in ("revenue_usd", "cost_usd", "margin_usd"):
        row[key] = round(row[key], 6)
    row["margin_pct"] = round(row["margin_usd"] / row["revenue_usd"] * 100, 2) if row["revenue_usd"] else 0.0
    row["margin_multiple"] = round(row["revenue_usd"] / row["cost_usd"], 2) if row["cost_usd"] else 0.0


@service.route("GET", "/v1/reconcile")
def reconcile(ctx: Ctx):
    """Triangulate intake totals against Switchyard's own counters.

    Mandatory before invoicing: the intake sink is fire-and-forget with a bounded
    queue and `on_queue_full` may drop records. `/metrics` carries no tenant
    label, so this is an aggregate cross-check only -- and it is unauthenticated,
    so scrape it from inside the trust boundary.
    """
    try:
        stats = get_json(SWITCHYARD_BASE + "/v1/stats")
    except Exception as error:  # noqa: BLE001
        return 503, {"error": "switchyard unreachable: %r" % error}

    sy_total = int(stats.get("prompt_tokens", 0)) + int(stats.get("completion_tokens", 0))
    with _lock:
        events = list(EVENTS.values())
    tf_total = sum(
        e["meters"]["prompt_tokens"] + e["meters"]["completion_tokens"]
        + e["meters"].get("classifier_prompt_tokens", 0)
        + e["meters"].get("classifier_completion_tokens", 0)
        for e in events
    )

    with _lock:
        edge_rows = list(EDGE_METER.values())
    edge_total = sum(r["prompt_tokens"] + r["completion_tokens"] for r in edge_rows)

    # Three independent meters. The Edge meter is the invoice basis because it is
    # synchronous and in-path; intake is async and lossy; /v1/stats is
    # Switchyard's own aggregate. Divergence between them is the signal.
    intake_loss = (round((len(edge_rows) - len(events)) / len(edge_rows) * 100, 4)
                   if edge_rows else 0.0)
    drift = round(abs(sy_total - tf_total) / sy_total * 100, 4) if sy_total else 0.0
    verdict = ("block_invoicing" if intake_loss >= DRIFT_BLOCK_PCT
               else "warn" if intake_loss >= DRIFT_WARN_PCT else "ok")

    return 200, {
        "meters": {
            "edge_settled": {"requests": len(edge_rows), "tokens": edge_total,
                             "basis": "invoice", "lossless": True},
            "intake_async": {"requests": len(events), "tokens": tf_total,
                             "basis": "enrichment", "lossless": False},
            "switchyard_stats": {"requests": stats.get("total_requests"),
                                 "tokens": sy_total, "basis": "cross-check"},
        },
        # Requests the Edge served but no intake record arrived for -- the
        # revenue that would have been silently missing in an intake-only design.
        "intake_loss_pct": intake_loss,
        "stats_vs_intake_drift_pct": drift,
        "verdict": verdict,
        "drift_warn_pct": DRIFT_WARN_PCT,
        "drift_block_pct": DRIFT_BLOCK_PCT,
        "quarantined": len(QUARANTINE),
        "quarantine_reasons": _reason_counts(),
        "note": ("switchyard_stats includes classifier-LLM tokens on "
                 "deterministic routes that the Edge meter cannot see -- expected "
                 "divergence, not loss."),
    }


def _reason_counts() -> dict:
    counts: Dict[str, int] = {}
    with _lock:
        for item in QUARANTINE:
            counts[item["reason"]] = counts.get(item["reason"], 0) + 1
    return counts


@service.prefix("GET", "/v1/spend-tree/")
def spend_tree(ctx: Ctx):
    """Roll a multi-agent run's spend up the agent hierarchy.

    Switchyard already carries agent-id / parent-agent-id / is-subagent. Nothing
    else in the market attributes a sub-agent's tokens to its parent task and the
    task to a tenant contract.
    """
    task_id = ctx.path.rsplit("/", 1)[-1].split("?")[0]
    nodes: Dict[str, dict] = {}
    with _lock:
        events = list(EVENTS.values())

    for event in events:
        session = event["session"]
        price = event.get("price")
        if session.get("task_id") != task_id or not price:
            continue
        agent = session.get("agent_id") or "root"
        node = nodes.setdefault(agent, {
            "agent_id": agent,
            "parent_agent_id": session.get("parent_agent_id"),
            "is_subagent": session.get("is_subagent", False),
            "revenue_usd": 0.0, "cost_usd": 0.0, "tokens": 0, "requests": 0,
        })
        node["revenue_usd"] = round(node["revenue_usd"] + price["amount_usd"], 6)
        node["cost_usd"] = round(node["cost_usd"] + event["cost"]["cost_usd"], 6)
        node["tokens"] += event["meters"]["prompt_tokens"] + event["meters"]["completion_tokens"]
        node["requests"] += 1

    return 200, {"task_id": task_id, "nodes": sorted(
        nodes.values(), key=lambda n: (n["is_subagent"], n["agent_id"]))}


@service.route("GET", "/v1/events")
def events(ctx: Ctx):
    with _lock:
        return 200, {"count": len(EVENTS), "events": list(EVENTS.values())[:200]}


@service.route("GET", "/health")
def health(ctx: Ctx):
    return 200, {"status": "ok"}
