# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate that a latency-service-routed chain exposes ``/metrics``.

``/metrics`` is mounted unconditionally and served from the OTel Prometheus
registry; it returns 200 for every chain shape (empty exposition when
observability is disabled).
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry

from switchyard.lib import metrics, observability
from switchyard.lib.backends.health_poller import HealthPoller
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.server.switchyard_app import build_switchyard_app


@pytest.fixture
def _observability(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    reg = CollectorRegistry()
    reader = PrometheusMetricReader(registry=reg)
    provider = MeterProvider(metric_readers=[reader])
    metrics.reset_for_test(provider.get_meter("switchyard"))
    monkeypatch.setattr(observability, "prometheus_registry", lambda: reg)
    monkeypatch.setattr(observability, "is_enabled", lambda: True)
    monkeypatch.setattr(observability, "init_observability", lambda: True)
    try:
        yield
    finally:
        metrics.reset_for_test(None)


def _config() -> LatencyServiceBackendConfig:
    return LatencyServiceBackendConfig(
        latency_service_url="http://latency-service.test:8080",
        endpoints=[
            LatencyServiceEndpoint(
                model="model-A",
                api_key="test-key",
                base_url="http://llm.test/v1",
            ),
        ],
    )


def _build_app():
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            switchyard = ProfileSwitchyard(
                LatencyServiceProfileConfig.from_config(_config())
                .build()
                .with_runtime_components()
            )
            return build_switchyard_app(switchyard)


def test_recipe_app_exposes_metrics_endpoint(_observability: None) -> None:
    """Latency-service deployments must serve ``/metrics`` (OTel exposition)."""
    app = _build_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # The latency-service health surface is published via OTel observable gauges.
    assert "switchyard_endpoint_status" in resp.text


def test_route_bundle_latency_service_exposes_metrics(_observability: None) -> None:
    """Deployment path (YAML bundle) must also surface ``/metrics``."""
    from switchyard.cli.route_bundle import build_route_bundle_table

    bundle = {
        "routes": {
            "ls-route": {
                "type": "latency_service",
                "latency_service_url": "http://latency-service.test:8080",
                "endpoints": [
                    {
                        "model": "model-A",
                        "api_key": "test-key",
                        "base_url": "http://llm.test/v1",
                    },
                ],
            },
        },
    }

    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            table = build_route_bundle_table(bundle)
            app = build_switchyard_app(table)

            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "switchyard_endpoint_status" in resp.text
