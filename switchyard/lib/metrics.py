# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry metric instruments and record helpers for Switchyard.

All metric names live here, in one place, so the Prometheus exposition surface is
easy to audit. Instrument names use the OTel dotted form (``switchyard.requests``);
the Prometheus exporter sanitizes ``.``→``_`` and appends ``_total`` to monotonic
counters, reproducing the historical metric names. **No instrument sets a `unit`:**
the Prometheus exporter would otherwise append a unit suffix and turn
``switchyard.total_latency_ms`` into ``..._ms_ms``. Units are baked into the name
(``_ms``, ``_usd``) instead.

Every ``record_*`` helper is a no-op when observability is disabled (no meter), so
call sites stay unconditional and the default install pays no overhead.

Label cardinality is intentionally bounded: ``model``, ``tier``, ``router``,
``role`` are small fixed sets; nothing per-request (request id, prompt text) is
ever an attribute.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from opentelemetry.metrics import Observation

# Module state. Built once from the process meter (or a test meter). All access
# goes through `_state()`, which lazily initializes from observability.
_meter: Any = None
_meter_resolved: bool = False
_instruments: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Outcome classification — shared by the client-outcome middleware, the
# upstream-attempt accounting, and the latency-service backend. These are pure
# helpers (no instrument state) so they work regardless of whether
# observability is enabled.
# ---------------------------------------------------------------------------

OutcomeBucket = Literal["success", "retryable_error", "other_error"]

#: HTTP status codes the success criterion counts as router-rescuable errors.
RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 504})

#: Status codes emitted verbatim as the ``code`` label. Anything else is
#: clamped to its class (``"4xx"`` / ``"5xx"`` / …) so a misbehaving upstream
#: cannot inflate label cardinality.
KNOWN_STATUS_CODES: frozenset[int] = frozenset(
    {200, 400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}
)

#: ``code`` label value for a non-HTTP failure (network error, pre-status
#: timeout) — the request never received a status line, so there is no code.
NO_STATUS_CODE: str = "none"


def classify(status_code: int) -> OutcomeBucket:
    """Map an HTTP status code to its outcome bucket.

    2xx → ``success``; 429 / 500 / 504 → ``retryable_error``; everything else
    → ``other_error``.
    """
    if 200 <= status_code < 300:
        return "success"
    if status_code in RETRYABLE_STATUSES:
        return "retryable_error"
    return "other_error"


def code_label(status_code: int | None) -> str:
    """Render the ``code`` label for one upstream attempt.

    ``None`` (non-HTTP failure) → :data:`NO_STATUS_CODE`; a known code is
    emitted verbatim; any other HTTP code is clamped to its class.
    """
    if status_code is None:
        return NO_STATUS_CODE
    if status_code in KNOWN_STATUS_CODES:
        return str(status_code)
    if 100 <= status_code < 600:
        return f"{status_code // 100}xx"
    return "other"


# ---------------------------------------------------------------------------
# Latency-service observable-gauge source. The latency-service backend registers
# a callable that returns a snapshot of its poll/health/affinity state; the
# observable-gauge callbacks below read it on each scrape. Counters (polls,
# poll_failures, affinity hits/misses) are surfaced as observable gauges over
# the backend's own cumulative integers so they need no per-event recording.
# ---------------------------------------------------------------------------

#: Snapshot the latency-service source returns. Keys are optional; callbacks
#: skip missing ones.
LatencyServiceSnapshot = dict[str, Any]

_latency_service_source: Callable[[], LatencyServiceSnapshot] | None = None


def register_latency_service_source(
    source: Callable[[], LatencyServiceSnapshot] | None,
) -> None:
    """Register (or clear) the latency-service state source for the gauges.

    Single-process server: one latency-service backend at a time owns the
    source. Passing ``None`` clears it (called from the backend's ``shutdown``).
    """
    global _latency_service_source
    _latency_service_source = source

# Running totals exposed as gauges to preserve the historical
# `switchyard_total_requests` / `switchyard_total_errors` series. Counters
# (`switchyard.requests` etc.) already carry the per-model breakdown; these
# mirror the old top-level totals for dashboards that join on them.
_total_requests: int = 0
_total_errors: int = 0


def _build_info_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from importlib.metadata import version as _pkg_version

    from opentelemetry.metrics import Observation

    try:
        ver = _pkg_version("switchyard")
    except Exception:
        ver = "unknown"
    return [Observation(1, {"version": ver})]


def _total_requests_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    return [Observation(_total_requests, {})]


def _total_errors_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    return [Observation(_total_errors, {})]


def _ls_snapshot() -> LatencyServiceSnapshot:
    """Return the current latency-service snapshot, or empty when unregistered."""
    if _latency_service_source is None:
        return {}
    try:
        return _latency_service_source()
    except Exception:  # pragma: no cover - defensive; never break a scrape
        return {}


def _endpoint_status_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    obs: list[Observation] = []
    for model_id, current in snap.get("endpoint_status", {}).items():
        for status in ("healthy", "degraded", "unknown"):
            value = 1 if status == current else 0
            obs.append(Observation(value, {"model": model_id, "status": status}))
    return obs


def _endpoint_last_latency_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    return [
        Observation(latency_ms, {"model": model_id})
        for model_id, latency_ms in snap.get("endpoint_last_latency_ms", {}).items()
        if latency_ms is not None
    ]


def _poll_ok_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    if "poll_ok" not in snap:
        return []
    return [Observation(1 if snap["poll_ok"] else 0, {})]


def _poll_age_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    age = snap.get("poll_age_seconds")
    if age is None:
        return []
    return [Observation(age, {})]


def _polls_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    if "polls" not in snap:
        return []
    return [Observation(snap["polls"], {})]


def _poll_failures_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    if "poll_failures" not in snap:
        return []
    return [Observation(snap["poll_failures"], {})]


def _affinity_hits_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    if "affinity_hits" not in snap:
        return []
    return [Observation(snap["affinity_hits"], {})]


def _affinity_misses_callback(_options: Any) -> list[Observation]:  # pragma: no cover
    from opentelemetry.metrics import Observation

    snap = _ls_snapshot()
    if "affinity_misses" not in snap:
        return []
    return [Observation(snap["affinity_misses"], {})]


def _build_instruments(meter: Any) -> dict[str, Any]:
    """Create every instrument against *meter*. Called once per meter."""
    inst: dict[str, Any] = {}

    # Counters → Prometheus `_total`.
    inst["requests"] = meter.create_counter("switchyard.requests")
    inst["errors"] = meter.create_counter("switchyard.errors")
    for token in (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
    ):
        inst[token] = meter.create_counter(f"switchyard.{token}")
    inst["routing_decisions"] = meter.create_counter("switchyard.routing_decisions")
    inst["client_responses"] = meter.create_counter("switchyard.client_responses")
    inst["upstream_attempts"] = meter.create_counter("switchyard.upstream_attempts")
    inst["retry_recovered"] = meter.create_counter("switchyard.retry_recovered")
    # Histograms → Prometheus `_bucket`/`_sum`/`_count`.
    inst["model_call_latency_ms"] = meter.create_histogram("switchyard.model_call_latency_ms")
    inst["total_latency_ms"] = meter.create_histogram("switchyard.total_latency_ms")
    inst["routing_overhead_ms"] = meter.create_histogram("switchyard.routing_overhead_ms")
    inst["ttft_ms"] = meter.create_histogram("switchyard.ttft_ms")
    inst["cost_usd"] = meter.create_histogram("switchyard.cost_usd")

    # Observable gauges for build info and running totals.
    meter.create_observable_gauge("switchyard.build_info", callbacks=[_build_info_callback])
    meter.create_observable_gauge(
        "switchyard.total_requests", callbacks=[_total_requests_callback]
    )
    meter.create_observable_gauge("switchyard.total_errors", callbacks=[_total_errors_callback])

    # Latency-service health/affinity surface — observed from the backend's
    # state on each scrape. Status / latency / poll_ok / poll_age are gauges;
    # polls / poll_failures / affinity hits+misses are cumulative observable
    # counters so the Prometheus exporter renders them with the historical
    # `_total` suffix.
    meter.create_observable_gauge(
        "switchyard.endpoint_status", callbacks=[_endpoint_status_callback]
    )
    meter.create_observable_gauge(
        "switchyard.endpoint_last_latency_ms",
        callbacks=[_endpoint_last_latency_callback],
    )
    meter.create_observable_gauge(
        "switchyard.latency_service_poll_ok", callbacks=[_poll_ok_callback]
    )
    meter.create_observable_gauge(
        "switchyard.latency_service_poll_age_seconds", callbacks=[_poll_age_callback]
    )
    meter.create_observable_counter(
        "switchyard.latency_service_polls", callbacks=[_polls_callback]
    )
    meter.create_observable_counter(
        "switchyard.latency_service_poll_failures", callbacks=[_poll_failures_callback]
    )
    meter.create_observable_counter(
        "switchyard.affinity_hits", callbacks=[_affinity_hits_callback]
    )
    meter.create_observable_counter(
        "switchyard.affinity_misses", callbacks=[_affinity_misses_callback]
    )

    return inst


def _state() -> dict[str, Any] | None:
    """Return the instrument cache, building it lazily from the process meter."""
    global _meter, _meter_resolved, _instruments
    if _instruments:
        return _instruments
    if not _meter_resolved:
        from switchyard.lib import observability

        _meter = observability.get_meter()
        _meter_resolved = True
    if _meter is None:
        return None
    _instruments = _build_instruments(_meter)
    return _instruments


def ensure_instruments() -> None:
    """Force instrument creation so observable gauges report on the next scrape.

    Instruments are otherwise built lazily on the first ``record_*`` call; the
    ``/metrics`` handler calls this so pull-only series (build info, latency-
    service health gauges) appear even before any request has been served.
    """
    _state()


def _attrs(**kwargs: Any) -> dict[str, Any]:
    """Build an attribute dict, dropping ``None`` values."""
    return {k: v for k, v in kwargs.items() if v is not None}


def record_request(*, model: str, tier: str | None, router: str | None) -> None:
    """Count one served request (and bump the global request total)."""
    global _total_requests
    inst = _state()
    if inst is None:
        return
    _total_requests += 1
    inst["requests"].add(1, _attrs(model=model, tier=tier, router=router))


def record_error(*, model: str, tier: str | None, status: str | None = None) -> None:
    """Count one backend error (and bump the global error total)."""
    global _total_errors
    inst = _state()
    if inst is None:
        return
    _total_errors += 1
    inst["errors"].add(1, _attrs(model=model, tier=tier, status=status))


def record_tokens(
    *,
    model: str,
    tier: str | None,
    prompt: int = 0,
    completion: int = 0,
    cached: int = 0,
    cache_creation: int = 0,
    reasoning: int = 0,
) -> None:
    """Record token counters for one response. Zero values are skipped."""
    inst = _state()
    if inst is None:
        return
    attrs = _attrs(model=model, tier=tier)
    for name, value in (
        ("prompt_tokens", prompt),
        ("completion_tokens", completion),
        ("cached_tokens", cached),
        ("cache_creation_tokens", cache_creation),
        ("reasoning_tokens", reasoning),
    ):
        if value:
            inst[name].add(value, attrs)


def record_latencies(
    *,
    model: str,
    tier: str | None,
    router: str | None,
    model_call_ms: float | None = None,
    total_ms: float | None = None,
    routing_overhead_ms: float | None = None,
) -> None:
    """Record the latency histograms for one request. ``None`` values are skipped."""
    inst = _state()
    if inst is None:
        return
    model_attrs = _attrs(model=model, tier=tier)
    if model_call_ms is not None:
        inst["model_call_latency_ms"].record(model_call_ms, model_attrs)
    if total_ms is not None:
        inst["total_latency_ms"].record(total_ms, model_attrs)
    if routing_overhead_ms is not None:
        inst["routing_overhead_ms"].record(routing_overhead_ms, _attrs(router=router))


def record_cost(
    *, model: str, tier: str | None, role: str, kind: str, cost_usd: float
) -> None:
    """Record one dollar-cost sample. ``role`` ∈ routed|classifier|planner."""
    inst = _state()
    if inst is None:
        return
    inst["cost_usd"].record(cost_usd, _attrs(model=model, tier=tier, role=role, kind=kind))


def record_ttft(*, model: str, tier: str | None, ttft_ms: float) -> None:
    """Record time-to-first-token for one streaming response."""
    inst = _state()
    if inst is None:
        return
    inst["ttft_ms"].record(ttft_ms, _attrs(model=model, tier=tier))


def record_routing_decision(
    *, router: str, source: str | None, tier: str | None
) -> None:
    """Count one routing decision by strategy, decision source, and chosen tier."""
    inst = _state()
    if inst is None:
        return
    inst["routing_decisions"].add(1, _attrs(router=router, source=source, tier=tier))


def record_client_response(*, outcome: str) -> None:
    """Count one client-facing response outcome (success/retryable/other)."""
    inst = _state()
    if inst is None:
        return
    inst["client_responses"].add(1, {"outcome": outcome})


def record_upstream_attempt(*, outcome: str, code: str) -> None:
    """Count one upstream attempt by outcome and status code."""
    inst = _state()
    if inst is None:
        return
    inst["upstream_attempts"].add(1, {"outcome": outcome, "code": code})


def record_retry_recovered() -> None:
    """Count one request recovered by retrying after an upstream failure."""
    inst = _state()
    if inst is None:
        return
    inst["retry_recovered"].add(1, {})


def reset_for_test(meter: Any) -> None:
    """Rebind the meter and clear instruments. Test seam only."""
    global _meter, _meter_resolved, _instruments, _total_requests, _total_errors
    global _latency_service_source
    _meter = meter
    _meter_resolved = True
    _instruments = {}
    _total_requests = 0
    _total_errors = 0
    _latency_service_source = None
