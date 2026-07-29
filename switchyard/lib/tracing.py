# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Datadog (ddtrace) spans for the proxy routing path.

Switchyard runs inside a proxy that owns the Datadog APM trace. These helpers
add child spans and tags around the route-decision and upstream-attempt blocks
so routing behaviour is visible in APM.

``ddtrace`` is an **optional** dependency (the ``tracing`` extra). When it is
not installed the helpers are complete no-ops with no overhead, so the default
install and non-Datadog deployments are unaffected. When it is installed, the
spans created here nest under whatever span the surrounding proxy already has
active, so they appear inline in the proxy's trace.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

try:  # ddtrace >= 2.0 exposes the global tracer here
    from ddtrace.trace import tracer as _dd_tracer
except Exception:  # pragma: no cover - import shape varies / absent in dev
    try:  # ddtrace < 2.0 fallback
        from ddtrace import tracer as _dd_tracer
    except Exception:
        _dd_tracer = None


@runtime_checkable
class Span(Protocol):
    """Minimal span surface the routing instrumentation relies on."""

    def set_tag(self, key: str, value: Any) -> None: ...


class _NoopSpan:
    """Span stand-in used when no tracer is available; drops every tag."""

    def set_tag(self, key: str, value: Any) -> None:
        return None


_NOOP_SPAN = _NoopSpan()


@contextmanager
def routing_span(name: str) -> Iterator[Span]:
    """Open a child span named *name*, or yield a no-op span when ddtrace is absent.

    The span nests under whatever span the surrounding proxy has active, so the
    routing instrumentation shows up inline in the proxy's Datadog trace and is
    finished automatically when the ``with`` block exits.
    """
    if _dd_tracer is None:
        yield _NOOP_SPAN
        return
    with _dd_tracer.trace(name) as span:
        yield span


def set_tags(span: Span, tags: Mapping[str, Any]) -> None:
    """Set each non-``None`` tag on *span*.

    ``None`` values are skipped so an unavailable signal (e.g. a signal that
    has not been recorded yet) leaves no empty/misleading tag on the span.
    """
    for key, value in tags.items():
        if value is not None:
            span.set_tag(key, value)
