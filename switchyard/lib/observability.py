# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry SDK bootstrap for Switchyard.

Owns the process-wide OTel providers used by the proxy:

- a ``TracerProvider`` exporting spans over OTLP, and
- a ``MeterProvider`` holding a ``PrometheusMetricReader`` (pull, serves
  ``/metrics``) and a ``PeriodicExportingMetricReader`` (OTLP push).

Both providers share one ``Resource`` (``service.name`` / ``service.version`` /
``service.instance.id``); the same ``service.instance.id`` is handed to the Rust
trace SDK so Python and Rust spans correlate as one service. The global
propagator is W3C TraceContext so the ``traceparent`` injected into
``ProxyContext`` round-trips across the FFI boundary.

The whole module is a **no-op** when the OTel packages are not installed (the
``otel`` extra), when ``OTEL_SDK_DISABLED`` is truthy, or when no OTLP endpoint
is configured. In that state every accessor returns ``None``/``False`` and the
hot path pays no overhead — matching the default (non-observability) install.
"""

from __future__ import annotations

import os
import uuid
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # avoid importing the (optional) OTel API at module load
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

# Module-level state. Guarded by the fact that init is single-threaded at
# startup; idempotency makes repeat calls cheap and safe.
_initialized: bool = False
_enabled: bool = False
_instance_id: str | None = None
_tracer: Any = None
_meter: Any = None
_prometheus_reader: Any = None
_prometheus_registry: Any = None

#: Instrumentation scope name used for the tracer and meter.
_SCOPE = "switchyard"


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _service_name() -> str:
    return os.environ.get("OTEL_SERVICE_NAME", "switchyard")


def _service_version() -> str:
    try:
        return _pkg_version("switchyard")
    except Exception:
        return "unknown"


def init_observability() -> bool:
    """Initialize the OTel providers once. Return whether observability is enabled.

    No-ops (returns ``False``) when ``OTEL_SDK_DISABLED`` is truthy, when the OTel
    packages are not importable, or when no OTLP endpoint is configured. Safe to
    call repeatedly; subsequent calls return the established state without
    rebuilding providers.
    """
    global _initialized, _enabled, _instance_id, _tracer, _meter
    global _prometheus_reader, _prometheus_registry

    if _initialized:
        return _enabled
    _initialized = True

    if _truthy(os.environ.get("OTEL_SDK_DISABLED")):
        return False
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        # OTLP is the primary sink; without an endpoint we stay dark rather than
        # spin up exporters that would only error on every flush.
        return False

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
        from prometheus_client import CollectorRegistry
    except Exception:
        # otel extra not installed → stay a no-op.
        return False

    _instance_id = str(uuid.uuid4())
    resource = Resource.create(
        {
            "service.name": _service_name(),
            "service.version": _service_version(),
            "service.instance.id": _instance_id,
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # Own a dedicated registry rather than the process-global default so the
    # scrape surface is self-contained (and tests stay isolated).
    _prometheus_registry = CollectorRegistry()
    _prometheus_reader = PrometheusMetricReader(registry=_prometheus_registry)
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            _prometheus_reader,
            PeriodicExportingMetricReader(OTLPMetricExporter()),
        ],
    )
    metrics.set_meter_provider(meter_provider)

    set_global_textmap(TraceContextTextMapPropagator())

    _tracer = trace.get_tracer(_SCOPE, _service_version())
    _meter = metrics.get_meter(_SCOPE, _service_version())

    # Best-effort: light up the Rust trace SDK with the shared identity so Rust
    # spans join the same trace. Absent/older bindings simply skip this.
    try:
        import switchyard_rust

        init_tracing = getattr(switchyard_rust, "init_tracing", None)
        if init_tracing is not None:
            init_tracing(
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
                _service_name(),
                _service_version(),
                _instance_id,
            )
    except Exception:
        # Rust tracing is additive; never let it block Python observability.
        pass

    _enabled = True
    return True


def is_enabled() -> bool:
    """Return whether observability was successfully initialized and enabled."""
    return _enabled


def service_instance_id() -> str | None:
    """Return the shared ``service.instance.id``, or ``None`` when disabled."""
    return _instance_id


def get_tracer() -> Tracer | None:
    """Return the process tracer, or ``None`` when observability is disabled."""
    return cast("Tracer | None", _tracer)


def get_meter() -> Meter | None:
    """Return the process meter, or ``None`` when observability is disabled."""
    return cast("Meter | None", _meter)


def prometheus_reader() -> Any:
    """Return the ``PrometheusMetricReader`` backing ``/metrics`` (or ``None``)."""
    return _prometheus_reader


def prometheus_registry() -> Any:
    """Return the dedicated Prometheus ``CollectorRegistry`` (or ``None``)."""
    return _prometheus_registry


def reset_for_test() -> None:
    """Reset module state. Test seam only — not part of the public contract."""
    global _initialized, _enabled, _instance_id, _tracer, _meter
    global _prometheus_reader, _prometheus_registry
    _initialized = False
    _enabled = False
    _instance_id = None
    _tracer = None
    _meter = None
    _prometheus_reader = None
    _prometheus_registry = None
