# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end FastAPI test that ``GET /metrics`` serves the OTel Prometheus
exposition rendered from :mod:`switchyard.lib.observability`'s dedicated
registry.

When observability is disabled the endpoint returns an empty 200 exposition;
when enabled it renders the OTel instrument surface (``switchyard_requests_total``
etc.) recorded via :mod:`switchyard.lib.metrics`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from switchyard.lib import metrics, observability
from switchyard.lib.endpoints.metrics_endpoint import (
    PROMETHEUS_CONTENT_TYPE,
    register_metrics_endpoint,
)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[CollectorRegistry]:
    """Enable observability against an isolated Prometheus registry + meter."""
    reg = CollectorRegistry()
    reader = PrometheusMetricReader(registry=reg)
    provider = MeterProvider(metric_readers=[reader])
    metrics.reset_for_test(provider.get_meter("switchyard"))
    monkeypatch.setattr(observability, "prometheus_registry", lambda: reg)
    monkeypatch.setattr(observability, "is_enabled", lambda: True)
    try:
        yield reg
    finally:
        metrics.reset_for_test(None)


@pytest.fixture
async def client(registry: CollectorRegistry) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    register_metrics_endpoint(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_metrics_returns_empty_exposition_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observability, "prometheus_registry", lambda: None)
    app = FastAPI()
    register_metrics_endpoint(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == PROMETHEUS_CONTENT_TYPE
    assert resp.text == ""


async def test_metrics_returns_prometheus_exposition(
    client: httpx.AsyncClient,
) -> None:
    metrics.record_request(model="strong/m", tier="strong", router="random")
    metrics.record_tokens(model="strong/m", tier="strong", prompt=120, completion=30)
    metrics.record_latencies(
        model="strong/m", tier="strong", router="random",
        model_call_ms=42.5, total_ms=88.0, routing_overhead_ms=8.0,
    )

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == PROMETHEUS_CONTENT_TYPE

    body = resp.text
    # Counter sample lands with the expected label set (Prometheus `_total`).
    assert "switchyard_requests_total{" in body
    assert 'model="strong/m"' in body and 'tier="strong"' in body
    # Latency histograms render as `_bucket`/`_sum`/`_count`, never `_ms_ms`.
    assert "switchyard_model_call_latency_ms_count" in body
    assert "_ms_ms" not in body


async def test_metrics_output_round_trips_through_official_prometheus_parser(
    client: httpx.AsyncClient,
) -> None:
    """Spec compliance gate: ``prometheus_client.parser`` is the reference
    parser used by every real scraper. Anything it accepts, Prometheus will."""
    metrics.record_request(model="openai/gpt-5.2", tier="strong", router="random")
    metrics.record_error(model="anth/claude", tier="weak")
    metrics.record_request(model="anth/claude", tier="weak", router="random")
    metrics.record_tokens(model="openai/gpt-5.2", tier="strong", prompt=120, completion=30)
    metrics.record_latencies(
        model="openai/gpt-5.2", tier="strong", router="random",
        model_call_ms=42.5, total_ms=88.0, routing_overhead_ms=8.0,
    )
    metrics.record_latencies(
        model="anth/claude", tier="weak", router="random",
        model_call_ms=5.0, total_ms=15.0, routing_overhead_ms=3.0,
    )

    resp = await client.get("/metrics")
    assert resp.status_code == 200

    families: dict[str, Any] = {
        f.name: f for f in text_string_to_metric_families(resp.text)
    }

    expected = {
        "switchyard_total_requests": "gauge",
        "switchyard_total_errors": "gauge",
        "switchyard_requests": "counter",
        "switchyard_errors": "counter",
        "switchyard_prompt_tokens": "counter",
        "switchyard_completion_tokens": "counter",
        "switchyard_model_call_latency_ms": "histogram",
        "switchyard_total_latency_ms": "histogram",
        "switchyard_routing_overhead_ms": "histogram",
    }
    for name, kind in expected.items():
        assert name in families, f"family {name} missing from parsed output"
        assert families[name].type == kind, f"family {name} parsed as {families[name].type}"

    req_samples = {
        (s.labels["model"], s.labels["tier"]): s.value
        for s in families["switchyard_requests"].samples
    }
    assert req_samples[("openai/gpt-5.2", "strong")] == 1
    assert req_samples[("anth/claude", "weak")] == 1

    # Histogram sum aggregates across observations (8 + 3 = 11 ms overhead).
    overhead = families["switchyard_routing_overhead_ms"]
    overhead_sum = next(s.value for s in overhead.samples if s.name.endswith("_sum"))
    assert overhead_sum == 11.0
