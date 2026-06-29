# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GET /metrics`` served from the OpenTelemetry Prometheus reader.

The OTel ``PrometheusMetricReader`` registers the proxy's instruments into a
dedicated ``CollectorRegistry`` owned by :mod:`switchyard.lib.observability`.
This endpoint renders that registry as Prometheus text exposition. When
observability is disabled the registry is ``None`` and the endpoint returns an
empty exposition with a 200 so scrapers see a valid (empty) surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import Response

from switchyard.lib import metrics, observability

if TYPE_CHECKING:
    from fastapi import FastAPI

#: Prometheus text exposition format 0.0.4 content-type.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def register_metrics_endpoint(app: FastAPI) -> None:
    """Mount ``GET /metrics`` rendering the OTel Prometheus registry."""
    router = APIRouter()

    async def get_metrics() -> Response:
        """Render the OTel Prometheus registry as text exposition."""
        registry = observability.prometheus_registry()
        if registry is None:
            return Response(content="", media_type=PROMETHEUS_CONTENT_TYPE)
        # Build instruments so pull-only series (build info, latency-service
        # health gauges) render even before the first request is served.
        metrics.ensure_instruments()
        from prometheus_client import generate_latest

        return Response(
            content=generate_latest(registry),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    router.get("/metrics")(get_metrics)
    app.include_router(router, tags=["Metrics"])
