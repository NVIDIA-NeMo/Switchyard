# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datadog routing spans/tags emitted by ``LatencyServiceLLMBackend``.

The backend emits a ``switchyard.route_decision`` span around endpoint selection
and a ``switchyard.upstream_attempt`` span around each upstream call. These tests
install a fake tracer (so no ddtrace install is needed) and assert the tag set.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from switchyard.lib import tracing
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
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

# ---------------------------------------------------------------------------
# Fake tracer (stands in for ddtrace so no install is required)
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tags: dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def trace(self, name: str) -> _FakeSpan:
        span = _FakeSpan(name)
        self.spans.append(span)
        return span

    def named(self, name: str) -> list[_FakeSpan]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def tracer(monkeypatch: pytest.MonkeyPatch) -> _FakeTracer:
    fake = _FakeTracer()
    monkeypatch.setattr(tracing, "_dd_tracer", fake)
    return fake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(*models: str, **kwargs: Any) -> LatencyServiceBackendConfig:
    return LatencyServiceBackendConfig(
        latency_service_url="http://ls.test:8080",
        endpoints=[
            LatencyServiceEndpoint(model=m, base_url=f"http://{m}.test", api_key="k")
            for m in models
        ],
        **kwargs,
    )


def _make_backend(config: LatencyServiceBackendConfig) -> LatencyServiceLLMBackend:
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient"
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock()
        with patch.object(HealthPoller, "start"):
            return LatencyServiceLLMBackend(config)


def _set_health(
    backend: LatencyServiceLLMBackend,
    mapping: dict[str, EndpointHealthStatus | EndpointHealth],
) -> None:
    with backend._cache_lock:
        for mid, value in mapping.items():
            if isinstance(value, EndpointHealthStatus):
                value = EndpointHealth(status=value)
            backend._health_cache[mid] = value


def _completion(content: str = "ok") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="m",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def _request(**overrides: Any) -> ChatRequest:
    body: dict[str, Any] = {"model": "incoming", "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return ChatRequest.openai_chat(body)  # type: ignore[arg-type]


def _api_status_error(status_code: int) -> openai.APIStatusError:
    response = httpx.Response(
        status_code, request=httpx.Request("POST", "http://upstream.test/v1/chat/completions")
    )
    return openai.APIStatusError("upstream error", response=response, body=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_success_emits_route_and_attempt_spans(tracer: _FakeTracer) -> None:
    backend = _make_backend(_config("model-A"))
    _set_health(backend, {"model-A": EndpointHealth(EndpointHealthStatus.HEALTHY, 50.0)})
    backend._clients["model-A"].acompletion = AsyncMock(return_value=_completion())

    await backend.call(ProxyContext(), _request(model="incoming"))

    route = tracer.named("switchyard.route_decision")
    attempt = tracer.named("switchyard.upstream_attempt")
    assert len(route) == 1
    assert len(attempt) == 1

    # poll-age is absent (poller never recorded a success) — None tags are skipped.
    assert route[0].tags == {
        "switchyard.model": "incoming",
        "switchyard.candidate_endpoints": "model-A",
        "switchyard.selected_endpoint": "model-A",
        "switchyard.was_fastest_selected": True,
        "switchyard.affinity_hit": False,
    }

    assert attempt[0].tags == {
        "switchyard.model": "incoming",
        "switchyard.selected_endpoint": "model-A",
        "switchyard.retry_count": 0,
        "switchyard.outcome": "success",
        "switchyard.upstream_status_code": 200,
    }


async def test_poll_age_tag_present_after_a_poll(tracer: _FakeTracer) -> None:
    backend = _make_backend(_config("model-A"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._poller._last_success_at = time.monotonic()
    backend._clients["model-A"].acompletion = AsyncMock(return_value=_completion())

    await backend.call(ProxyContext(), _request())

    route = tracer.named("switchyard.route_decision")[0]
    assert route.tags["switchyard.latency_service_poll_age_ms"] >= 0.0


async def test_api_status_error_tags(tracer: _FakeTracer) -> None:
    backend = _make_backend(_config("model-A"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._clients["model-A"].acompletion = AsyncMock(
        side_effect=_api_status_error(429)
    )

    with pytest.raises(openai.APIStatusError):
        await backend.call(ProxyContext(), _request())

    attempt = tracer.named("switchyard.upstream_attempt")[0]
    assert attempt.tags["switchyard.upstream_status_code"] == 429
    assert attempt.tags["switchyard.outcome"] == "retryable_error"
    assert attempt.tags["switchyard.error_code"] == "429"
    # Failure-source annotation (SWITCH-882): the attempt failed upstream.
    assert attempt.tags["switchyard.error_source"] == "provider"
    assert attempt.tags["switchyard.upstream_model"] == "model-A"


async def test_generic_error_tags(tracer: _FakeTracer) -> None:
    backend = _make_backend(_config("model-A"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._clients["model-A"].acompletion = AsyncMock(side_effect=RuntimeError("down"))

    with pytest.raises(RuntimeError, match="down"):
        await backend.call(ProxyContext(), _request())

    attempt = tracer.named("switchyard.upstream_attempt")[0]
    assert attempt.tags["switchyard.outcome"] == "retryable_error"
    assert attempt.tags["switchyard.error_code"] == "none"
    # A non-HTTP failure has no status line.
    assert "switchyard.upstream_status_code" not in attempt.tags
    # A network-level fault is still an upstream-side failure (SWITCH-882).
    assert attempt.tags["switchyard.error_source"] == "provider"
    assert attempt.tags["switchyard.upstream_model"] == "model-A"


async def test_retry_count_is_sequential_and_ends_success(tracer: _FakeTracer) -> None:
    backend = _make_backend(_config("model-A", "model-B", max_retries=1))
    _set_health(backend, {
        "model-A": EndpointHealthStatus.HEALTHY,
        "model-B": EndpointHealthStatus.HEALTHY,
    })
    backend._clients["model-A"].acompletion = AsyncMock(side_effect=RuntimeError("down"))
    backend._clients["model-B"].acompletion = AsyncMock(return_value=_completion())

    await backend.call(ProxyContext(), _request())

    attempts = tracer.named("switchyard.upstream_attempt")
    # retry_count is the 0-based attempt index, monotonically increasing.
    assert [a.tags["switchyard.retry_count"] for a in attempts] == list(range(len(attempts)))
    assert attempts[-1].tags["switchyard.outcome"] == "success"


async def test_was_fastest_false_when_latency_unknown(tracer: _FakeTracer) -> None:
    backend = _make_backend(_config("model-A", "model-B"))
    # Both HEALTHY but no latency samples -> uniform pick, "fastest" undefined.
    _set_health(backend, {
        "model-A": EndpointHealthStatus.HEALTHY,
        "model-B": EndpointHealthStatus.HEALTHY,
    })
    selected = {"model-A", "model-B"}
    backend._clients["model-A"].acompletion = AsyncMock(return_value=_completion())
    backend._clients["model-B"].acompletion = AsyncMock(return_value=_completion())

    await backend.call(ProxyContext(), _request())

    route = tracer.named("switchyard.route_decision")[0]
    assert route.tags["switchyard.was_fastest_selected"] is False
    assert route.tags["switchyard.selected_endpoint"] in selected


async def test_call_works_without_tracer() -> None:
    # No ``tracer`` fixture: _dd_tracer is None (ddtrace absent in dev), so the
    # spans are no-ops and call() must proceed normally.
    backend = _make_backend(_config("model-A"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._clients["model-A"].acompletion = AsyncMock(return_value=_completion())

    ctx = ProxyContext()
    await backend.call(ctx, _request())
    assert ctx.selected_model == "model-A"
