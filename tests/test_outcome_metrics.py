# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit + integration tests for the OTel outcome counters."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from prometheus_client import CollectorRegistry

from switchyard.lib import metrics, observability
from switchyard.lib.backends.health_poller import (
    EndpointHealth,
    EndpointHealthStatus,
    HealthPoller,
)
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.endpoints.upstream_error import (
    record_upstream_attempt_failure,
    record_upstream_attempt_success,
)
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.lib.proxy_context import (
    CTX_UPSTREAM_ATTEMPTS_RECORDED,
    CTX_UPSTREAM_HTTP_STATUS,
    ProxyContext,
)
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest, SwitchyardUpstreamError


@pytest.fixture
def meter() -> Iterator[InMemoryMetricReader]:
    """Bind an in-memory meter so the record helpers emit observable points."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.reset_for_test(provider.get_meter("switchyard"))
    try:
        yield reader
    finally:
        metrics.reset_for_test(None)


def _counter_value(reader: Any, name: str, **want: str) -> int:
    """Sum points of *name* whose attributes match the wanted subset."""
    total = 0
    data = reader.get_metrics_data()
    if data is None:
        return 0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    if all(point.attributes.get(k) == v for k, v in want.items()):
                        total += point.value
    return total


def _latency_service_switchyard(
    config: LatencyServiceBackendConfig,
) -> ProfileSwitchyard:
    """Build the latency-service profile-backed serving adapter."""
    return ProfileSwitchyard(
        LatencyServiceProfileConfig.from_config(config)
        .build()
        .with_runtime_components()
    )


# ---------------------------------------------------------------------------
# Classification (pure helpers, now on switchyard.lib.metrics)
# ---------------------------------------------------------------------------


class TestClassify:
    @pytest.mark.parametrize("code", [200, 201, 204, 299])
    def test_2xx_is_success(self, code: int) -> None:
        assert metrics.classify(code) == "success"

    @pytest.mark.parametrize("code", [429, 500, 504])
    def test_spec_codes_are_retryable_error(self, code: int) -> None:
        """Exactly the codes the success criterion lists count as retryable."""
        assert metrics.classify(code) == "retryable_error"

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 502, 503])
    def test_other_codes_are_other_error(self, code: int) -> None:
        """Bad-payload / bad-key / non-spec 5xx fall outside the criterion."""
        assert metrics.classify(code) == "other_error"


class TestCodeLabel:
    def test_none_is_the_no_status_sentinel(self) -> None:
        """Non-HTTP failures have no status line → the ``none`` sentinel."""
        assert metrics.code_label(None) == metrics.NO_STATUS_CODE
        assert metrics.code_label(None) == "none"

    @pytest.mark.parametrize("code", sorted(metrics.KNOWN_STATUS_CODES))
    def test_known_codes_emitted_verbatim(self, code: int) -> None:
        assert metrics.code_label(code) == str(code)

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(418, "4xx"), (451, "4xx"), (599, "5xx"), (100, "1xx"), (302, "3xx")],
    )
    def test_unknown_codes_clamp_to_class(self, code: int, expected: str) -> None:
        """An oddball upstream code collapses to its class, bounding cardinality."""
        assert metrics.code_label(code) == expected

    @pytest.mark.parametrize("code", [0, 99, 600, 700])
    def test_out_of_range_codes_clamp_to_other(self, code: int) -> None:
        assert metrics.code_label(code) == "other"


# ---------------------------------------------------------------------------
# Latency-service backend wiring
# ---------------------------------------------------------------------------


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
        max_retries=2,
    )


def _build_backend(config: LatencyServiceBackendConfig):
    from switchyard.lib.backends.latency_service_llm_backend import (
        LatencyServiceLLMBackend,
    )

    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"):
            return LatencyServiceLLMBackend(config)


def _make_completion() -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-test",
        object="chat.completion",
        created=1700000000,
        model="any",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
    )


def _api_status_error(status_code: int) -> openai.APIStatusError:
    """Build a synthetic APIStatusError carrying the given status code."""
    import httpx

    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://llm.test/v1/chat/completions"),
        json={"error": {"message": "synthetic"}},
    )
    return openai.APIStatusError(
        "synthetic", response=response, body={"error": "synthetic"}
    )


class TestBackendCounters:
    async def test_success_records_one_attempt_success(self, meter: InMemoryMetricReader) -> None:
        backend = _build_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        await backend.call(ProxyContext(), ChatRequest.openai_chat({
            "model": "model-A",
            "messages": [{"role": "user", "content": "hi"}],
        }))

        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="success", code="200"
        ) == 1
        assert _counter_value(meter, "switchyard.retry_recovered") == 0

    async def test_429_then_success_increments_recovered(self, meter: InMemoryMetricReader) -> None:
        """First attempt 429, retry succeeds — the steering signal we care about."""
        backend = _build_backend(_config("model-A", "model-B"))
        with backend._cache_lock:
            backend._health_cache["model-A"] = EndpointHealth(
                status=EndpointHealthStatus.HEALTHY,
            )
            backend._health_cache["model-B"] = EndpointHealth(
                status=EndpointHealthStatus.UNKNOWN,
            )
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(429),
        )
        backend._clients["model-B"].acompletion = AsyncMock(
            return_value=_make_completion(),
        )
        with backend._cache_lock:
            backend._health_cache["model-A"] = EndpointHealth(
                EndpointHealthStatus.HEALTHY,
                10.0,
            )
            backend._health_cache["model-B"] = EndpointHealth(
                EndpointHealthStatus.DEGRADED,
                100.0,
            )

        await backend.call(ProxyContext(), ChatRequest.openai_chat({
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        }))

        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="retryable_error", code="429"
        ) == 1
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="success", code="200"
        ) == 1
        assert _counter_value(meter, "switchyard.retry_recovered") == 1

    async def test_401_does_not_count_as_retryable(self, meter: InMemoryMetricReader) -> None:
        """A 401 (bad key) is ``other_error`` and is not retried — fail fast."""
        backend = _build_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(401),
        )

        with pytest.raises(openai.APIStatusError):
            await backend.call(ProxyContext(), ChatRequest.openai_chat({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
            }))

        assert backend._clients["model-A"].acompletion.call_count == 1
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="other_error", code="401"
        ) == 1
        assert _counter_value(meter, "switchyard.retry_recovered") == 0

    async def test_network_error_counts_as_retryable(self, meter: InMemoryMetricReader) -> None:
        """Non-HTTP exceptions (network, pre-status timeout) map to retryable_error."""
        backend = _build_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=RuntimeError("connection refused"),
        )

        with pytest.raises(RuntimeError):
            await backend.call(ProxyContext(), ChatRequest.openai_chat({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
            }))

        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="retryable_error", code="none"
        ) == 3

    async def test_failure_emits_structured_error_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every failed attempt emits one per-event structured log for Loki."""
        backend = _build_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(429),
        )

        with caplog.at_level(logging.WARNING, logger="switchyard.upstream_errors"):
            with pytest.raises(openai.APIStatusError):
                await backend.call(ProxyContext(), ChatRequest.openai_chat({
                    "model": "x",
                    "messages": [{"role": "user", "content": "hi"}],
                }))

        records = [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == "switchyard.upstream_errors"
        ]
        assert len(records) == 3
        assert all(r["event"] == "upstream_attempt_failed" for r in records)
        assert all(r["status_code"] == 429 and r["code"] == "429" for r in records)
        assert all(r["model"] == "model-A" for r in records)
        assert [r["attempt"] for r in records] == [1, 2, 3]


# ---------------------------------------------------------------------------
# FastAPI middleware — client-side counter (served on /metrics)
# ---------------------------------------------------------------------------


class TestClientResponseMiddleware:
    @pytest.fixture(autouse=True)
    def _observability(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        reg = CollectorRegistry()
        reader = PrometheusMetricReader(registry=reg)
        provider = MeterProvider(metric_readers=[reader])
        metrics.reset_for_test(provider.get_meter("switchyard"))
        monkeypatch.setattr(observability, "prometheus_registry", lambda: reg)
        monkeypatch.setattr(observability, "is_enabled", lambda: True)
        # init_observability is called by build_switchyard_app; no-op it so it
        # doesn't rebuild providers over our test meter.
        monkeypatch.setattr(observability, "init_observability", lambda: True)
        try:
            yield
        finally:
            metrics.reset_for_test(None)

    def test_successful_chat_completion_counts_as_success(self) -> None:
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
                    component for component in switchyard.iter_components()
                    if hasattr(component, "_clients")
                )
                backend._clients["model-A"].acompletion = AsyncMock(
                    return_value=_make_completion(),
                )

                app = build_switchyard_app(switchyard)
                with TestClient(app, raise_server_exceptions=False) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "model-A",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    assert response.status_code == 200
                    exposition = client.get("/metrics").text

        assert 'switchyard_client_responses_total{outcome="success"} 1' in exposition

    def test_metrics_route_is_not_counted_as_client_response(self) -> None:
        """Only the LLM routes feed the client outcome counter."""
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
                    for _ in range(5):
                        client.get("/metrics")
                        client.get("/health")
                        client.get("/v1/models")
                    exposition = client.get("/metrics").text

        # No client_responses series exists at all (never incremented).
        assert 'switchyard_client_responses_total{outcome="success"}' not in exposition


# ---------------------------------------------------------------------------
# Endpoint-layer fallback — wires the upstream-attempt counter for backends
# (Rust native / passthrough / multi) that issue one attempt and can't reach
# the metrics helpers themselves.
# ---------------------------------------------------------------------------


class TestEndpointUpstreamAttemptFallback:
    def test_success_records_one_200(self, meter: InMemoryMetricReader) -> None:
        record_upstream_attempt_success(ProxyContext())
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="success", code="200"
        ) == 1

    def test_rust_upstream_http_error_records_its_status(
        self, meter: InMemoryMetricReader
    ) -> None:
        """A Rust backend's typed ``SwitchyardUpstreamError.status_code`` is used."""
        exc = SwitchyardUpstreamError("boom")
        exc.status_code = 500
        record_upstream_attempt_failure(ProxyContext(), exc)
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="retryable_error", code="500"
        ) == 1

    def test_python_backend_ctx_status_takes_priority(
        self, meter: InMemoryMetricReader
    ) -> None:
        """A Python backend's stashed ctx status is recorded even without a typed exc."""
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 401
        record_upstream_attempt_failure(ctx, RuntimeError("opaque"))
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="other_error", code="401"
        ) == 1

    def test_status_less_upstream_error_is_retryable_none(
        self, meter: InMemoryMetricReader
    ) -> None:
        """An upstream failure with no HTTP status (network) maps to code=none."""
        record_upstream_attempt_failure(ProxyContext(), SwitchyardUpstreamError("conn reset"))
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="retryable_error", code="none"
        ) == 1

    def test_internal_error_is_not_an_upstream_attempt(
        self, meter: InMemoryMetricReader
    ) -> None:
        """A non-upstream chain failure (e.g. translation/processor) records nothing."""
        record_upstream_attempt_failure(ProxyContext(), ValueError("internal bug"))
        assert _counter_value(meter, "switchyard.upstream_attempts") == 0

    def test_dedup_flag_suppresses_fallback(self, meter: InMemoryMetricReader) -> None:
        """A backend that records its own attempts opts the endpoint out."""
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_ATTEMPTS_RECORDED] = True
        record_upstream_attempt_success(ctx)
        record_upstream_attempt_failure(ctx, SwitchyardUpstreamError("boom"))
        assert _counter_value(meter, "switchyard.upstream_attempts") == 0

    async def test_latency_service_backend_sets_dedup_flag(
        self, meter: InMemoryMetricReader
    ) -> None:
        """The latency-service backend claims attempt accounting on its ctx."""
        backend = _build_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(return_value=_make_completion())
        ctx = ProxyContext()
        await backend.call(ctx, ChatRequest.openai_chat({
            "model": "model-A",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        assert ctx.metadata.get(CTX_UPSTREAM_ATTEMPTS_RECORDED) is True
        assert _counter_value(
            meter, "switchyard.upstream_attempts", outcome="success", code="200"
        ) == 1
