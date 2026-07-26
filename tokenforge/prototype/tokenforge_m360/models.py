"""TokenForge metered-usage event schema.

The contract between Switchyard's intake sink and RevenueOS rating. Field names
under `cost` mirror Switchyard's intake payload verbatim so the mapping stays
auditable; `price` is TokenForge-owned and never sourced from Switchyard.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

RouteTag = Literal["online", "enterprise", "marketplace", "partner"]
BudgetState = Literal["ok", "warn", "throttle", "deny"]


class SessionRef(BaseModel):
    """Agent hierarchy, lifted from Switchyard's `x-switchyard-*` headers.

    `agent_id` / `parent_agent_id` / `is_subagent` are what let us build a spend
    tree for a multi-agent run: a sub-agent's tokens roll up to the parent task,
    and the task rolls up to a tenant contract.
    """

    session_id: str | None = None
    task_id: str | None = None
    turn_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    is_subagent: bool = False
    session_final: bool = False


class RoutingRef(BaseModel):
    """Switchyard's routing decision, for attribution and escalation metering."""

    route: str
    router_type: str | None = None
    routed_to: str | None = None          # "strong" | "weak" | endpoint id
    selected_model: str | None = None
    selected_provider: str | None = None
    tier: str | None = None
    correlation_id: str | None = None     # router_correlation_id


class Meters(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    inference_requests: int = 1
    gateway_transactions: int = 1
    agent_session_seconds: float = 0.0
    router_escalations: int = 0


class SupplierCost(BaseModel):
    """What WE pay. Sourced from Switchyard's hardcoded MODEL_PRICING.

    Never surface these figures to a customer as a price -- see CustomerPrice.
    """

    source: str = "switchyard_intake"
    cost_usd: float = 0.0
    cost_input_usd: float = 0.0
    cost_output_usd: float = 0.0
    cost_details: dict[str, float] = Field(default_factory=dict)


class CustomerPrice(BaseModel):
    """What we CHARGE. TokenForge-owned, resolved against a pinned rate card."""

    rate_card_id: str
    resolved_by: str = "tokenforge"
    amount_usd: float = 0.0
    margin_usd: float = 0.0
    margin_pct: float = 0.0


class Integrity(BaseModel):
    """Reconciliation state. RevenueOS must not invoice an unreconciled event."""

    reconciled: bool = False
    reconcile_batch: str | None = None
    quarantined: bool = False
    quarantine_reason: str | None = None


class MeteredUsageEvent(BaseModel):
    event_id: str
    source: str = "switchyard"
    decision_id: str | None = None
    occurred_at: datetime

    tenant_id: str
    contract_id: str | None = None
    cost_center: str | None = None
    route_tag: RouteTag | None = None

    session: SessionRef = Field(default_factory=SessionRef)
    routing: RoutingRef
    meters: Meters = Field(default_factory=Meters)
    cost: SupplierCost = Field(default_factory=SupplierCost)
    price: CustomerPrice | None = None
    integrity: Integrity = Field(default_factory=Integrity)


def derive_event_id(
    request_id: str | None,
    correlation_id: str | None,
    created_at: str | datetime | None,
) -> str:
    """Deterministic idempotency key.

    The intake sink retries on failure (`max_retries`), so a natural key is the
    only thing standing between a retry and a double-billed request.
    """
    if isinstance(created_at, datetime):
        created_at = created_at.astimezone(timezone.utc).isoformat()
    material = "|".join(str(part or "") for part in (request_id, correlation_id, created_at))
    return "tfe_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
