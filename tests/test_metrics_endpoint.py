# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end FastAPI test that ``StatsEndpoint`` serves both
``/v1/stats`` (existing JSON) and ``/metrics`` (Prometheus exposition)
off the same shared :class:`StatsAccumulator`.

Pins the contract the ticket calls out: existing ``/v1/stats`` behavior
stays intact, ``/metrics`` returns Prometheus text-format with the core
metric names.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client.parser import text_string_to_metric_families

from switchyard.lib.endpoints.stats_endpoint import PROMETHEUS_CONTENT_TYPE, StatsEndpoint
from switchyard.lib.stats_accumulator import StatsAccumulator


@pytest.fixture
def stats() -> StatsAccumulator:
    return StatsAccumulator()


@pytest.fixture
async def client(stats: StatsAccumulator) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    StatsEndpoint(stats).register(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_v1_stats_returns_existing_json_shape(
    client: httpx.AsyncClient, stats: StatsAccumulator
) -> None:
    await stats.record_success(model="m", backend_latency_ms=10.0)
    await stats.record_usage(
        model="m", prompt_tokens=5, completion_tokens=3, total_latency_ms=20.0
    )

    resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    body = resp.json()
    # Existing schema is unchanged — keep the contract surface visible.
    assert body["total_requests"] == 1
    assert body["models"]["m"]["calls"] == 1
    assert body["models"]["m"]["prompt_tokens"] == 5
    assert body["models"]["m"]["max_prompt_tokens"] == 5
    assert "cost_estimate" in body


async def test_metrics_returns_prometheus_exposition(
    client: httpx.AsyncClient, stats: StatsAccumulator
) -> None:
    await stats.record_success(model="strong/m", backend_latency_ms=42.5, tier="strong")
    await stats.record_usage(
        model="strong/m",
        prompt_tokens=120,
        completion_tokens=30,
        total_latency_ms=88.0,
        routing_overhead_ms=8.0,
        tier="strong",
    )

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == PROMETHEUS_CONTENT_TYPE

    body = resp.text
    # Core metric headers the ticket calls out.
    for line in (
        "# TYPE switchyard_requests_total counter",
        "# TYPE switchyard_errors_total counter",
        "# TYPE switchyard_model_call_latency_ms summary",
        "# TYPE switchyard_total_latency_ms summary",
        "# TYPE switchyard_routing_overhead_ms summary",
    ):
        assert line in body, f"missing exposition line: {line}"

    # Selected model/tier counter sample lands with the expected label set.
    assert 'switchyard_requests_total{model="strong/m",tier="strong"} 1' in body
    assert (
        'switchyard_model_call_latency_ms_count{model="strong/m",tier="strong"} 1'
        in body
    )


async def test_metrics_output_round_trips_through_official_prometheus_parser(
    client: httpx.AsyncClient, stats: StatsAccumulator
) -> None:
    """Spec compliance gate: ``prometheus_client.parser`` is the reference
    parser used by every real scraper. Anything it accepts, Prometheus will."""
    await stats.record_success(model="openai/gpt-5.2", backend_latency_ms=42.5, tier="strong")
    await stats.record_error(model="anth/claude", tier="weak")
    await stats.record_success(model="anth/claude", backend_latency_ms=5.0, tier="weak")
    await stats.record_usage(
        model="openai/gpt-5.2",
        prompt_tokens=120,
        completion_tokens=30,
        total_latency_ms=88.0,
        routing_overhead_ms=8.0,
        tier="strong",
    )
    await stats.record_usage(
        model="anth/claude",
        prompt_tokens=40,
        completion_tokens=5,
        total_latency_ms=15.0,
        routing_overhead_ms=3.0,
        tier="weak",
    )

    resp = await client.get("/metrics")
    assert resp.status_code == 200

    families = {f.name: f for f in text_string_to_metric_families(resp.text)}

    # prometheus-client strips the ``_total`` suffix from counter family names
    # but preserves it on individual samples — assert against the family form.
    expected = {
        "switchyard_total_requests": "gauge",
        "switchyard_total_errors": "gauge",
        "switchyard_requests": "counter",
        "switchyard_errors": "counter",
        "switchyard_prompt_tokens": "counter",
        "switchyard_completion_tokens": "counter",
        "switchyard_cached_tokens": "counter",
        "switchyard_model_call_latency_ms": "summary",
        "switchyard_total_latency_ms": "summary",
        "switchyard_routing_overhead_ms": "summary",
    }
    for name, kind in expected.items():
        assert name in families, f"family {name} missing from parsed output"
        assert families[name].type == kind, f"family {name} parsed as {families[name].type}"

    # Counter values survive the parse round-trip with the right labels.
    req_samples = {
        (s.labels["model"], s.labels["tier"]): s.value
        for s in families["switchyard_requests"].samples
    }
    assert req_samples[("openai/gpt-5.2", "strong")] == 1
    assert req_samples[("anth/claude", "weak")] == 1

    # Summary sum aggregates across all observations (8 + 3 = 11 ms overhead).
    overhead = families["switchyard_routing_overhead_ms"]
    overhead_sum = next(s.value for s in overhead.samples if s.name.endswith("_sum"))
    assert overhead_sum == 11.0
