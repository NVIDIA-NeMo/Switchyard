# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry span helpers for the proxy routing path.

Replaces the old ddtrace ``tracing.py``. Each helper opens a span on the process
tracer, or yields a no-op when observability is disabled — so call sites stay
unconditional and the default install pays no overhead.

``current_traceparent`` serializes the active span context as a W3C
``traceparent`` string for injection into ``ProxyContext`` so the Rust extension
can parent its spans under the Python span (one distributed trace).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

# Bound lazily from observability; `_resolved` guards the one-time lookup so a
# disabled SDK isn't re-checked on every span.
_tracer: Any = None
_resolved: bool = False


def _get_tracer() -> Any:
    global _tracer, _resolved
    if _resolved:
        return _tracer
    from switchyard.lib import observability

    _tracer = observability.get_tracer()
    _resolved = True
    return _tracer


def _set_attrs(span: Any, attributes: Mapping[str, Any]) -> None:
    """Set each non-``None`` attribute on *span* (``None`` signals 'unavailable')."""
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


@contextmanager
def request_span(name: str, attributes: Mapping[str, Any]) -> Iterator[Any]:
    """Open the request root span (or yield ``None`` when disabled)."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        _set_attrs(span, attributes)
        yield span


@contextmanager
def stage_span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
    """Open a child span for one chain stage (or yield ``None`` when disabled)."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            _set_attrs(span, attributes)
        yield span


@contextmanager
def route_decision_span(
    *,
    router: str,
    tier: str | None = None,
    selected_model: str | None = None,
    selected_target: str | None = None,
    **extra: Any,
) -> Iterator[Any]:
    """Open the ``switchyard.route_decision`` span with the routing attributes.

    Extra strategy-specific signals (``draw``, ``score``, ``threshold``,
    ``source`` …) are passed as keyword args; ``None`` values are dropped.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    attributes: dict[str, Any] = {
        "router": router,
        "tier": tier,
        "selected_model": selected_model,
        "selected_target": selected_target,
        **extra,
    }
    with tracer.start_as_current_span("switchyard.route_decision") as span:
        _set_attrs(span, attributes)
        yield span


def current_traceparent() -> str | None:
    """Return the active span context as a W3C ``traceparent`` string, or ``None``.

    ``None`` when observability is disabled or no span is active. Used to hand the
    trace context to the Rust extension across the FFI boundary.
    """
    if _get_tracer() is None:
        return None
    from opentelemetry.propagate import inject

    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


def add_span_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    """Add an event to the current span if one is recording (else no-op)."""
    if _get_tracer() is None:
        return
    from opentelemetry import trace

    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.add_event(name, {k: v for k, v in (attributes or {}).items() if v is not None})


def reset_for_test(tracer: Any) -> None:
    """Rebind the tracer. Test seam only."""
    global _tracer, _resolved
    _tracer = tracer
    _resolved = True
