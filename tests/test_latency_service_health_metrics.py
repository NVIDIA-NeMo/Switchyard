# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gates for latency-service routing-state metrics.

The Prometheus exposition rendered from the accumulator covers request
flow but says nothing about *why* a request landed on a given endpoint.
The latency-service backend now contributes per-endpoint verdict gauges
and poll-loop health gauges so dashboards can see the routing inputs,
not just outputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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
from switchyard.lib.endpoints import prometheus_emitter
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.server.switchyard_app import build_switchyard_app


@pytest.fixture(autouse=True)
def _clean_table():
    prometheus_emitter._clear_for_tests()
    yield
    prometheus_emitter._clear_for_tests()


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
        .with_runtime_components(enable_stats=config.enable_stats)
    )


def _make_backend(config: LatencyServiceBackendConfig) -> LatencyServiceLLMBackend:
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"):
            return LatencyServiceLLMBackend(config)


def _emit(backend: LatencyServiceLLMBackend) -> str:
    """Render this backend's emitter output as a single string."""
    return "\n".join(backend._render_prometheus_lines())


class TestEndpointStatusGauge:
    def test_emits_one_row_per_status_per_endpoint(self) -> None:
        """Three-state encoding lets `sum by (status)` give a clean histogram."""
        backend = _make_backend(_config("model-A", "model-B"))
        with backend._cache_lock:
            backend._health_cache["model-A"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY, 100.0,
            )
            backend._health_cache["model-B"] = EndpointHealth(
                EndpointHealthStatus.DEGRADED, 800.0,
            )

        out = _emit(backend)
        assert 'switchyard_endpoint_status{model="model-A",status="healthy"} 1' in out
        assert 'switchyard_endpoint_status{model="model-A",status="degraded"} 0' in out
        assert 'switchyard_endpoint_status{model="model-A",status="unknown"} 0' in out
        assert 'switchyard_endpoint_status{model="model-B",status="degraded"} 1' in out
        assert 'switchyard_endpoint_status{model="model-B",status="healthy"} 0' in out

    def test_unknown_default_before_first_poll(self) -> None:
        """New backends start with every endpoint UNKNOWN."""
        backend = _make_backend(_config("model-A"))
        out = _emit(backend)
        assert 'switchyard_endpoint_status{model="model-A",status="unknown"} 1' in out
        assert 'switchyard_endpoint_status{model="model-A",status="healthy"} 0' in out


class TestEndpointLatencyGauge:
    def test_emits_only_for_endpoints_with_sample(self) -> None:
        """Absence is meaningful: don't emit a zero where no sample exists."""
        backend = _make_backend(_config("with-sample", "no-sample"))
        with backend._cache_lock:
            backend._health_cache["with-sample"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY, 250.5,
            )
            backend._health_cache["no-sample"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY, None,
            )

        out = _emit(backend)
        assert (
            'switchyard_endpoint_last_latency_ms{model="with-sample"} 250.5' in out
        )
        assert 'switchyard_endpoint_last_latency_ms{model="no-sample"}' not in out


class TestPollHealthGauges:
    def test_before_first_poll_signals_never_polled(self) -> None:
        """poll_ok=0 + no poll_age line lets a scraper detect "never polled"."""
        backend = _make_backend(_config("model-A"))
        out = _emit(backend)
        assert "switchyard_latency_service_poll_ok 0" in out
        assert "switchyard_latency_service_poll_age_seconds" not in out.replace(
            "# HELP switchyard_latency_service_poll_age_seconds", ""
        ).replace(
            "# TYPE switchyard_latency_service_poll_age_seconds", ""
        )
        assert "switchyard_latency_service_polls_total 0" in out
        assert "switchyard_latency_service_poll_failures_total 0" in out

    def test_after_successful_poll_emits_age_and_ok(self) -> None:
        backend = _make_backend(_config("model-A"))
        backend._poller._poll_count = 3
        backend._poller._last_poll_ok = True
        backend._poller._last_success_at = __import__("time").monotonic() - 1.0

        out = _emit(backend)
        assert "switchyard_latency_service_poll_ok 1" in out
        assert "switchyard_latency_service_polls_total 3" in out
        # Age value is positive — exact value depends on test wall time.
        age_lines = [
            line for line in out.splitlines()
            if line.startswith("switchyard_latency_service_poll_age_seconds ")
        ]
        assert len(age_lines) == 1
        age_value = float(age_lines[0].split()[-1])
        assert age_value > 0

    def test_poll_failure_flips_ok_to_zero(self) -> None:
        """Even with prior successes, a recorded failure flips poll_ok off
        so dashboards alarm on the *latest* poll result, not history."""
        backend = _make_backend(_config("model-A"))
        backend._poller._poll_count = 5
        backend._poller._last_success_at = __import__("time").monotonic() - 1.0
        backend._poller._poll_failures = 1
        backend._poller._last_poll_ok = False

        out = _emit(backend)
        assert "switchyard_latency_service_poll_ok 0" in out
        assert "switchyard_latency_service_poll_failures_total 1" in out

    def test_success_after_prior_failure_emits_ok(self) -> None:
        """Prior failures are historical counters; poll_ok tracks latest outcome."""
        backend = _make_backend(_config("model-A"))
        backend._poller._poll_count = 5
        backend._poller._poll_failures = 1
        backend._poller._last_poll_ok = True
        backend._poller._last_success_at = __import__("time").monotonic() - 1.0

        out = _emit(backend)
        assert "switchyard_latency_service_poll_ok 1" in out
        assert "switchyard_latency_service_poll_failures_total 1" in out


class TestEmitterLifecycle:
    async def test_construction_registers_emitter(self) -> None:
        assert prometheus_emitter._EMITTERS == []
        backend = _make_backend(_config("model-A"))
        assert len(prometheus_emitter._EMITTERS) == 1
        # Confirm table output includes this backend's lines.
        assert "switchyard_endpoint_status" in prometheus_emitter.render()
        await backend.shutdown()
        assert prometheus_emitter._EMITTERS == []

    async def test_shutdown_idempotent(self) -> None:
        """Lifespan tear-down may call shutdown more than once."""
        backend = _make_backend(_config("model-A"))
        await backend.shutdown()
        await backend.shutdown()
        assert prometheus_emitter._EMITTERS == []


class TestRoutingOverhead:
    async def test_metrics_record_routing_overhead_for_python_backend(self) -> None:
        """End-to-end: a request through the latency-service chain must
        publish ``switchyard_routing_overhead_ms`` with non-zero count.

        Before ``ctx.backend_call_latency_ms`` existed, the Rust response
        processor had no backend-latency reading to subtract from total,
        and the summary stayed at ``count=0`` even under heavy load.
        """
        from unittest.mock import AsyncMock

        from openai.types.chat import ChatCompletion
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        from switchyard_rust.core import ChatRequest

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

                app = build_switchyard_app(switchyard)
                with TestClient(app, raise_server_exceptions=False) as client:
                    metrics = client.get("/metrics")

        assert metrics.status_code == 200
        body = metrics.text
        count_line = next(
            line for line in body.splitlines()
            if line.startswith("switchyard_routing_overhead_ms_count")
        )
        count = int(count_line.split()[-1])
        assert count >= 1, f"expected non-zero routing_overhead samples, body=\n{body}"


class TestEndToEnd:
    def test_metrics_endpoint_includes_health_lines(self) -> None:
        """Wire the full chain via the recipe and verify /metrics carries
        the new lines on top of the standard accumulator exposition."""
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
                    metrics = client.get("/metrics")

        assert metrics.status_code == 200
        body = metrics.text
        # Both surfaces coexist on the same scrape.
        assert "switchyard_requests_total" in body
        assert "switchyard_endpoint_status" in body
        assert "switchyard_latency_service_poll_ok" in body
        assert "switchyard_latency_service_polls_total" in body
