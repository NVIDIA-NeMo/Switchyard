# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OTel metric instruments and record helpers (switchyard.lib.metrics)."""

from __future__ import annotations

import importlib

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def _metrics_with_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("switchyard")
    metrics = importlib.import_module("switchyard.lib.metrics")
    metrics.reset_for_test(meter)
    return metrics, reader


def _metric_names(reader) -> set[str]:
    names: set[str] = set()
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.add(metric.name)
    return names


def test_record_helpers_emit_expected_instruments():
    metrics, reader = _metrics_with_reader()

    metrics.record_request(model="m", tier="strong", router="random")
    metrics.record_tokens(
        model="m",
        tier="strong",
        prompt=10,
        completion=5,
        cached=2,
        cache_creation=1,
        reasoning=3,
    )
    metrics.record_latencies(
        model="m",
        tier="strong",
        router="random",
        model_call_ms=10.0,
        total_ms=12.0,
        routing_overhead_ms=2.0,
    )
    metrics.record_cost(model="m", tier="strong", role="routed", kind="input", cost_usd=0.01)
    metrics.record_ttft(model="m", tier="strong", ttft_ms=8.0)
    metrics.record_routing_decision(router="random", source="coin", tier="strong")

    names = _metric_names(reader)
    # Instrument names use OTel dotted form; the SDK exposes them verbatim.
    expected = {
        "switchyard.requests",
        "switchyard.prompt_tokens",
        "switchyard.completion_tokens",
        "switchyard.cached_tokens",
        "switchyard.cache_creation_tokens",
        "switchyard.reasoning_tokens",
        "switchyard.model_call_latency_ms",
        "switchyard.total_latency_ms",
        "switchyard.routing_overhead_ms",
        "switchyard.cost_usd",
        "switchyard.ttft_ms",
        "switchyard.routing_decisions",
        "switchyard.build_info",
        "switchyard.total_requests",
    }
    missing = expected - names
    assert not missing, f"missing instruments: {missing}"


def test_no_unit_suffix_in_prometheus_render():
    # The Prometheus exporter appends a unit suffix only when the instrument has a
    # `unit`. None of our instruments set one, so latency names stay `_ms` (not `_ms_ms`).
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    reader = PrometheusMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("switchyard")
    metrics = importlib.import_module("switchyard.lib.metrics")
    metrics.reset_for_test(meter)
    metrics.record_latencies(
        model="m", tier="strong", router="random",
        model_call_ms=10.0, total_ms=12.0, routing_overhead_ms=2.0,
    )

    import prometheus_client

    out = prometheus_client.generate_latest().decode()
    assert "switchyard_total_latency_ms_bucket" in out
    assert "_ms_ms" not in out
    assert "switchyard_requests_total" not in out  # not recorded in this test


def test_record_helpers_noop_without_meter():
    metrics = importlib.import_module("switchyard.lib.metrics")
    metrics.reset_for_test(None)
    # Must not raise when observability is disabled.
    metrics.record_request(model="m", tier=None, router="random")
    metrics.record_ttft(model="m", tier=None, ttft_ms=1.0)
