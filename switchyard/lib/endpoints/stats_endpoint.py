# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI endpoint module exposing stats over HTTP.

Serves three paths off the same shared stats source:

- ``GET /v1/stats`` — native JSON snapshot.
- ``GET /v1/routing/stats`` — alias of ``/v1/stats`` for backwards compat.
- ``GET /metrics`` — Prometheus text-format exposition rendered from the
  same snapshot via :func:`switchyard.lib.prometheus_exposition.render_prometheus`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import Response

from switchyard.lib.endpoints import outcome_metrics
from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.endpoints.prometheus_emitter import render as render_extra_metrics
from switchyard.lib.prometheus_exposition import render_prometheus
from switchyard.lib.stats_accumulator import StatsAccumulator

if TYPE_CHECKING:
    from fastapi import FastAPI


# Prometheus text exposition format 0.0.4 content-type.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class StatsEndpoint(NemoSwitchyardEndpoint):
    """Exposes stats via ``GET /v1/stats``, ``/v1/routing/stats`` alias,
    and ``GET /metrics`` (Prometheus exposition).

    Contributed automatically by :class:`StatsResponseProcessor.get_endpoint`
    — no manual wiring required.

    The ``/v1/routing/stats`` alias exists so existing consumers of the
    historical endpoint path (``benchmark/run_terminal_bench_harbor.sh``, external
    dashboards) work against passthrough without any config change.

    ``/metrics`` renders the same underlying snapshot for Prometheus scrapers;
    JSON behavior on ``/v1/stats`` is untouched.
    """

    register_once = True

    def __init__(self, stats: StatsAccumulator) -> None:
        self._stats = stats

    def register(self, app: FastAPI) -> None:
        routes = APIRouter()
        stats = self._stats

        async def get_stats() -> dict[str, Any]:
            """Snapshot of per-model request / token / latency / cost stats."""
            return await stats.snapshot()

        async def reset_stats() -> dict[str, str]:
            """Zero all stats counters."""
            await stats.reset()
            return {"status": "reset"}

        async def get_metrics() -> Response:
            """Prometheus text-format exposition of the shared stats snapshot.

            Components that own non-request-derived state (Latency Service
            verdicts, poll-loop health) contribute extra lines via
            :mod:`switchyard.lib.endpoints.prometheus_emitter` so a single
            ``/metrics`` scrape carries both surfaces.
            """
            snapshot = await stats.snapshot()
            outcome_block = "\n".join(outcome_metrics.render_lines()) + "\n"
            return Response(
                content=(
                    render_prometheus(snapshot)
                    + outcome_block
                    + render_extra_metrics()
                ),
                media_type=PROMETHEUS_CONTENT_TYPE,
            )

        # native path.
        routes.get("/v1/stats")(get_stats)
        routes.post("/v1/stats/reset")(reset_stats)
        # Compatibility alias.
        routes.get("/v1/routing/stats")(get_stats)
        routes.post("/v1/routing/stats/reset")(reset_stats)

        app.include_router(routes, tags=["Stats"])

        # Prometheus exposition lives at the conventional ``/metrics`` path,
        # untagged from the JSON Stats routes so scraper discovery / OpenAPI
        # consumers see them separately.
        metrics_router = APIRouter()
        metrics_router.get("/metrics")(get_metrics)
        app.include_router(metrics_router, tags=["Metrics"])
