# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gates for latency-service routing-state metrics (OTel).

The latency-service backend contributes per-endpoint verdict gauges and
poll-loop health gauges/counters to the OTel meter via observable callbacks
reading a snapshot it exposes. These tests assert both the snapshot source and
the rendered OTel Prometheus surface.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry

from switchyard.lib import metrics, observability
from switchyard.lib.backends.health_poller import (
    EndpointHealth,
    EndpointHealthStatus,
    HealthPoller,
)
from switchyard.lib.backends.latency_service_llm_backend import (
    LatencyServiceLLMBackend,
)
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.server.switchyard_app import build_switchyard_app


@pytest.fixture(autouse=True)
def _clear_source() -> Iterator[None]:
    metrics.register_latency_service_source(None)
    yield
    metrics.register_latency_service_source(None)


def _config(*models: str) -> LatencyServiceBackendConfig:
    return LatencyServiceBackendConfig(
        latency_service_url="http://latency.test:8080",
        endpoints=[
            LatencyServiceEndpoint(
                model=model,
                base_url=f"http://llm-{model}.test/v1",
                api_key="test-key",
            )
            for model in models
        ],
    )


def _latency_service_switchyard(
    config: LatencyServiceBackendConfig,
) -> ProfileSwitchyard:
    """Build the latency-service profile-backed serving adapter."""
    return ProfileSwitchyard(
        LatencyServiceProfileConfig.from_config(config)
        .build()
        .with_runtime_components()
    )


def _make_backend(config: LatencyServiceBackendConfig) -> LatencyServiceLLMBackend:
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"):
            return LatencyServiceLLMBackend(config)


def _snapshot(backend: LatencyServiceLLMBackend) -> dict[str, Any]:
    """Read the backend's OTel metrics snapshot (what the gauges observe)."""
    return backend._metrics_snapshot()


class TestEndpointStatusSnapshot:
    def test_one_entry_per_endpoint(self) -> None:
        backend = _make_backend(_config("model-A", "model-B"))
        with backend._cache_lock:
            backend._health_cache["model-A"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY, 100.0,
            )
            backend._health_cache["model-B"] = EndpointHealth(
                EndpointHealthStatus.DEGRADED, 800.0,
            )

        snap = _snapshot(backend)
        assert snap["endpoint_status"] == {"model-A": "healthy", "model-B": "degraded"}

    def test_unknown_default_before_first_poll(self) -> None:
        backend = _make_backend(_config("model-A"))
        assert _snapshot(backend)["endpoint_status"] == {"model-A": "unknown"}


class TestEndpointLatencySnapshot:
    def test_latency_present_only_when_sampled(self) -> None:
        backend = _make_backend(_config("with-sample", "no-sample"))
        with backend._cache_lock:
            backend._health_cache["with-sample"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY, 250.5,
            )
            backend._health_cache["no-sample"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY, None,
            )

        latencies = _snapshot(backend)["endpoint_last_latency_ms"]
        assert latencies["with-sample"] == 250.5
        assert latencies["no-sample"] is None


class TestPollHealthSnapshot:
    def test_before_first_poll_signals_never_polled(self) -> None:
        backend = _make_backend(_config("model-A"))
        snap = _snapshot(backend)
        assert snap["poll_ok"] is False
        assert "poll_age_seconds" not in snap
        assert snap["polls"] == 0
        assert snap["poll_failures"] == 0

    def test_after_successful_poll_reports_age_and_ok(self) -> None:
        backend = _make_backend(_config("model-A"))
        backend._poller._poll_count = 3
        backend._poller._last_poll_ok = True
        backend._poller._last_success_at = time.monotonic() - 1.0

        snap = _snapshot(backend)
        assert snap["poll_ok"] is True
        assert snap["polls"] == 3
        assert snap["poll_age_seconds"] > 0

    def test_poll_failure_flips_ok_to_false(self) -> None:
        backend = _make_backend(_config("model-A"))
        backend._poller._poll_count = 5
        backend._poller._last_success_at = time.monotonic() - 1.0
        backend._poller._poll_failures = 1
        backend._poller._last_poll_ok = False

        snap = _snapshot(backend)
        assert snap["poll_ok"] is False
        assert snap["poll_failures"] == 1


class TestSourceLifecycle:
    def test_construction_registers_source(self) -> None:
        backend = _make_backend(_config("model-A"))
        assert metrics._latency_service_source is not None
        backend.shutdown()
        assert metrics._latency_service_source is None

    def test_shutdown_idempotent(self) -> None:
        backend = _make_backend(_config("model-A"))
        backend.shutdown()
        backend.shutdown()
        assert metrics._latency_service_source is None


class TestRoutingOverhead:
    async def test_metrics_record_routing_overhead_for_python_backend(self) -> None:
        """A request through the latency-service chain records routing_overhead_ms."""
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        from switchyard_rust.core import ChatRequest

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        metrics.reset_for_test(provider.get_meter("switchyard"))
        try:
            with patch(
                "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
            ) as mock_cls:
                mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
                with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
                    switchyard = _latency_service_switchyard(
                        LatencyServiceBackendConfig(
                            latency_service_url="http://latency.test:8080",
                            endpoints=[
                                LatencyServiceEndpoint(
                                    model="model-A",
                                    api_key="test-key",
                                    base_url="http://llm.test/v1",
                                ),
                            ],
                        )
                    )
                    backend = next(
                        component
                        for component in switchyard.iter_components()
                        if hasattr(component, "_clients")
                    )
                    from openai.types.chat import ChatCompletion
                    from openai.types.chat.chat_completion import Choice
                    from openai.types.chat.chat_completion_message import (
                        ChatCompletionMessage,
                    )

                    completion = ChatCompletion(
                        id="cmpl-test",
                        object="chat.completion",
                        created=1700000000,
                        model="model-A",
                        choices=[
                            Choice(
                                index=0,
                                message=ChatCompletionMessage(role="assistant", content="ok"),
                                finish_reason="stop",
                            )
                        ],
                    )
                    backend._clients["model-A"].acompletion = AsyncMock(return_value=completion)

                    await switchyard.call(ChatRequest.openai_chat({
                        "model": "model-A",
                        "messages": [{"role": "user", "content": "hi"}],
                    }))

            count = 0
            for rm in reader.get_metrics_data().resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        if metric.name == "switchyard.routing_overhead_ms":
                            count += sum(p.count for p in metric.data.data_points)
            assert count >= 1
        finally:
            metrics.reset_for_test(None)


class TestEndToEnd:
    def test_metrics_endpoint_includes_health_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The full chain's /metrics carries the latency-service health surface."""
        reg = CollectorRegistry()
        reader = PrometheusMetricReader(registry=reg)
        provider = MeterProvider(metric_readers=[reader])
        metrics.reset_for_test(provider.get_meter("switchyard"))
        monkeypatch.setattr(observability, "prometheus_registry", lambda: reg)
        monkeypatch.setattr(observability, "is_enabled", lambda: True)
        monkeypatch.setattr(observability, "init_observability", lambda: True)
        try:
            with patch(
                "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
            ) as mock_cls:
                mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
                with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
                    switchyard = _latency_service_switchyard(
                        LatencyServiceBackendConfig(
                            latency_service_url="http://latency.test:8080",
                            endpoints=[
                                LatencyServiceEndpoint(
                                    model="model-A",
                                    api_key="test-key",
                                    base_url="http://llm.test/v1",
                                ),
                            ],
                        )
                    )
                    app = build_switchyard_app(switchyard)
                    with TestClient(app, raise_server_exceptions=False) as client:
                        resp = client.get("/metrics")

            assert resp.status_code == 200
            body = resp.text
            assert "switchyard_endpoint_status" in body
            assert "switchyard_latency_service_poll_ok" in body
            assert "switchyard_latency_service_polls_total" in body
        finally:
            metrics.reset_for_test(None)
