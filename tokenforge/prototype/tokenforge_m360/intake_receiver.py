"""TokenForge intake receiver -- Phase A, the zero-code integration.

Point Switchyard at this service and per-request cost telemetry starts flowing
with no changes to Switchyard itself:

    export SWITCHYARD_INTAKE_TARGET_URL=http://localhost:9900/v1/intake/switchyard
    switchyard serve --routing-profiles tokenforge/config/route.tokenforge.yaml

Run this receiver:

    uvicorn tokenforge_m360.intake_receiver:app --port 9900

Switchyard POSTs records asynchronously off the response path with a bounded
queue and retries (`IntakeSinkConfig`). Two consequences drive this design:

  * Retries mean we MUST be idempotent -> `derive_event_id`.
  * `on_queue_full` may DROP records, so this is not a system of record on its
    own -> `/v1/reconcile` triangulates against `/metrics` and `/v1/stats`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .models import (
    CustomerPrice,
    Integrity,
    Meters,
    MeteredUsageEvent,
    RoutingRef,
    SessionRef,
    SupplierCost,
    derive_event_id,
)
from .rate_card import POC_RATE_CARD, compute_margin

log = logging.getLogger("tokenforge.intake")

app = FastAPI(title="TokenForge Intake", version="0.1.0")

SWITCHYARD_BASE = os.environ.get("SWITCHYARD_BASE_URL", "http://localhost:8080")
DRIFT_WARN_PCT = float(os.environ.get("TOKENFORGE_DRIFT_WARN_PCT", "0.5"))
DRIFT_BLOCK_PCT = float(os.environ.get("TOKENFORGE_DRIFT_BLOCK_PCT", "2.0"))

# PoC stores. Production: Postgres for durable events, Redis for hot counters.
_EVENTS: dict[str, MeteredUsageEvent] = {}
_QUARANTINE: list[dict[str, Any]] = []


def _first(headers: dict[str, Any], name: str) -> str | None:
    """Switchyard retains headers as dict[str, list[str]] (lowercased)."""
    value = headers.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value if isinstance(value, str) else None


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


@app.post("/v1/intake/switchyard")
async def receive(request: Request) -> JSONResponse:
    record = await request.json()

    headers = record.get("request_headers") or record.get("headers") or {}
    routing = record.get("routing") or record
    usage = record.get("usage") or record

    tenant_id = _first(headers, "x-tokenforge-tenant-id")
    if not tenant_id:
        # No tenant attribution means the request bypassed TokenForge Edge.
        # Quarantine rather than guess -- an unattributed token is a revenue leak
        # and, in a multi-tenant deployment, a security finding.
        _QUARANTINE.append({"reason": "missing_tenant", "record": record})
        log.warning("quarantined unattributed intake record")
        return JSONResponse({"status": "quarantined", "reason": "missing_tenant"}, 202)

    event_id = derive_event_id(
        record.get("request_id") or _first(headers, "x-request-id"),
        routing.get("router_correlation_id") or routing.get("correlation_id"),
        record.get("created_at"),
    )
    if event_id in _EVENTS:
        # Intake retry. Already rated -- acknowledge without double-billing.
        return JSONResponse({"status": "duplicate", "event_id": event_id}, 200)

    routed_to = routing.get("routed_to")
    selected_model = (
        routing.get("router_selected_model")
        or routing.get("selected_model")
        or record.get("model")
        or ""
    )
    escalated = routed_to == "strong"

    meters = Meters(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(usage.get("cached_tokens") or 0),
        cache_creation_tokens=int(usage.get("cache_creation_tokens") or 0),
        router_escalations=1 if escalated else 0,
    )

    cost = SupplierCost(
        cost_usd=float(record.get("cost_usd") or 0.0),
        cost_input_usd=float(record.get("cost_input_usd") or 0.0),
        cost_output_usd=float(record.get("cost_output_usd") or 0.0),
        cost_details=record.get("cost_details") or {},
    )

    event = MeteredUsageEvent(
        event_id=event_id,
        decision_id=_first(headers, "x-tokenforge-decision-id"),
        occurred_at=_parse_ts(record.get("created_at")),
        tenant_id=tenant_id,
        contract_id=_first(headers, "x-tokenforge-contract-id"),
        cost_center=_first(headers, "x-tokenforge-cost-center"),
        route_tag=_first(headers, "x-tokenforge-route-tag"),  # type: ignore[arg-type]
        session=SessionRef(
            session_id=_first(headers, "x-switchyard-session-id"),
            task_id=_first(headers, "x-switchyard-task-id"),
            turn_id=_first(headers, "x-switchyard-turn-id"),
            agent_id=_first(headers, "x-switchyard-agent-id"),
            parent_agent_id=_first(headers, "x-switchyard-parent-agent-id"),
            is_subagent=_as_bool(_first(headers, "x-switchyard-is-subagent")),
            session_final=_as_bool(_first(headers, "x-switchyard-session-final")),
        ),
        routing=RoutingRef(
            route=record.get("model") or "unknown",
            router_type=routing.get("router_type"),
            routed_to=routed_to,
            selected_model=selected_model,
            selected_provider=routing.get("router_selected_provider"),
            tier=_first(headers, "x-switchyard-tier"),
            correlation_id=routing.get("router_correlation_id"),
        ),
        meters=meters,
        cost=cost,
    )

    # --- Zero-token guard -------------------------------------------------
    # Known 0.1.0 defect: Codex Responses-API tasks may record 0 token usage.
    # A successful response with zero completion tokens is a metering hole, not
    # a free request.
    if meters.completion_tokens == 0 and not record.get("error"):
        event.integrity = Integrity(
            quarantined=True, quarantine_reason="zero_completion_tokens"
        )
        _EVENTS[event_id] = event
        _QUARANTINE.append({"reason": "zero_completion_tokens", "event_id": event_id})
        return JSONResponse({"status": "quarantined", "event_id": event_id}, 202)

    # --- Rating -----------------------------------------------------------
    card = _resolve_rate_card(_first(headers, "x-tokenforge-rate-card-id"))
    if not card.is_priced(selected_model):
        event.integrity = Integrity(
            quarantined=True,
            quarantine_reason=f"unpriced_model:{selected_model}",
        )
        _EVENTS[event_id] = event
        _QUARANTINE.append({"reason": "unpriced_model", "model": selected_model})
        return JSONResponse({"status": "quarantined", "event_id": event_id}, 202)

    amount = card.price_tokens(
        selected_model,
        prompt_tokens=meters.prompt_tokens,
        completion_tokens=meters.completion_tokens,
        cached_tokens=meters.cached_tokens,
        cache_creation_tokens=meters.cache_creation_tokens,
        tier=event.routing.tier,
        escalated=escalated,
    )
    margin_usd, margin_pct = compute_margin(amount, cost.cost_usd)
    event.price = CustomerPrice(
        rate_card_id=card.rate_card_id,
        amount_usd=amount,
        margin_usd=margin_usd,
        margin_pct=margin_pct,
    )

    _EVENTS[event_id] = event
    await _emit_to_revenueos(event)
    return JSONResponse(
        {"status": "rated", "event_id": event_id, "margin_pct": margin_pct}, 200
    )


@app.get("/v1/reconcile")
async def reconcile() -> dict[str, Any]:
    """Triangulate intake totals against Switchyard's own counters.

    The intake sink is lossy by design, so this is mandatory before invoicing.
    `/metrics` carries no tenant label -- it is an aggregate cross-check only,
    and must be scraped from inside the trust boundary since it is
    unauthenticated by design.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        stats = (await client.get(f"{SWITCHYARD_BASE}/v1/stats")).json()

    sy_prompt = int(stats.get("prompt_tokens") or 0)
    sy_completion = int(stats.get("completion_tokens") or 0)
    tf_prompt = sum(e.meters.prompt_tokens for e in _EVENTS.values())
    tf_completion = sum(e.meters.completion_tokens for e in _EVENTS.values())

    drift = _drift_pct(sy_prompt + sy_completion, tf_prompt + tf_completion)
    if drift >= DRIFT_BLOCK_PCT:
        verdict = "block_invoicing"
    elif drift >= DRIFT_WARN_PCT:
        verdict = "warn"
    else:
        verdict = "ok"

    return {
        "switchyard": {"prompt_tokens": sy_prompt, "completion_tokens": sy_completion},
        "tokenforge": {"prompt_tokens": tf_prompt, "completion_tokens": tf_completion},
        "drift_pct": drift,
        "verdict": verdict,
        "quarantined": len(_QUARANTINE),
    }


@app.get("/v1/margin")
async def margin_summary() -> dict[str, Any]:
    """Per-tenant revenue and margin -- the PoC 'prove margin' deliverable."""
    by_tenant: dict[str, dict[str, float]] = {}
    for event in _EVENTS.values():
        if event.price is None:
            continue
        row = by_tenant.setdefault(
            event.tenant_id,
            {"revenue_usd": 0.0, "cost_usd": 0.0, "margin_usd": 0.0, "requests": 0.0},
        )
        row["revenue_usd"] += event.price.amount_usd
        row["cost_usd"] += event.cost.cost_usd
        row["margin_usd"] += event.price.margin_usd
        row["requests"] += 1

    for row in by_tenant.values():
        revenue = row["revenue_usd"]
        row["margin_pct"] = round((row["margin_usd"] / revenue) * 100, 2) if revenue else 0.0
    return {"tenants": by_tenant}


@app.get("/v1/spend-tree/{task_id}")
async def spend_tree(task_id: str) -> dict[str, Any]:
    """Roll a multi-agent run's spend up the agent hierarchy.

    Switchyard already carries `x-switchyard-agent-id` / `-parent-agent-id` /
    `-is-subagent`. Nothing else in the market attributes a sub-agent's tokens
    to its parent task and the task to a tenant contract.
    """
    nodes: dict[str, dict[str, Any]] = {}
    for event in _EVENTS.values():
        if event.session.task_id != task_id or event.price is None:
            continue
        agent = event.session.agent_id or "root"
        node = nodes.setdefault(
            agent,
            {
                "agent_id": agent,
                "parent_agent_id": event.session.parent_agent_id,
                "is_subagent": event.session.is_subagent,
                "revenue_usd": 0.0,
                "cost_usd": 0.0,
                "tokens": 0,
            },
        )
        node["revenue_usd"] += event.price.amount_usd
        node["cost_usd"] += event.cost.cost_usd
        node["tokens"] += event.meters.prompt_tokens + event.meters.completion_tokens

    return {"task_id": task_id, "nodes": list(nodes.values())}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _resolve_rate_card(rate_card_id: str | None):
    # PoC: single card. Production: versioned lookup pinned at preflight so a
    # late-arriving record rates at occurrence time, not receipt time.
    return POC_RATE_CARD


def _parse_ts(raw: Any) -> datetime:
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _drift_pct(expected: int, actual: int) -> float:
    if expected == 0:
        return 0.0
    return round(abs(expected - actual) / expected * 100.0, 4)


async def _emit_to_revenueos(event: MeteredUsageEvent) -> None:
    """Forward the rated event to RevenueOS ingestion.

    PoC logs only. Production posts to the RevenueOS usage-ingestion API with
    `event_id` as the idempotency key, then RevenueOS handles rate-plan
    selection, wallet drawdown, invoicing and ASC 606 recognition.
    """
    log.info(
        "rated event=%s tenant=%s model=%s price=%.6f cost=%.6f margin=%.2f%%",
        event.event_id,
        event.tenant_id,
        event.routing.selected_model,
        event.price.amount_usd if event.price else 0.0,
        event.cost.cost_usd,
        event.price.margin_pct if event.price else 0.0,
    )
