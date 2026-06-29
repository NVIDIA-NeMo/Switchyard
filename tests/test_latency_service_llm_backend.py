# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`LatencyServiceLLMBackend` (usage case)."""

import random
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import ValidationError

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
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponseType,
    response_type_matches,
)
from switchyard_rust.translation import TranslationEngine

LATENCY_SERVICE_URL = "http://latency-service.test:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ep(
    model: str,
    base_url: str | None = None,
    request_type: str = "openai_chat",
) -> LatencyServiceEndpoint:
    return LatencyServiceEndpoint(
        model=model,
        base_url=base_url or f"http://llm-{model}.test",
        api_key="test-key",
        request_type=request_type,  # type: ignore[arg-type]
    )


def _config(*models: str, **kwargs) -> LatencyServiceBackendConfig:
    request_type = str(kwargs.pop("request_type", "openai_chat"))
    return LatencyServiceBackendConfig(
        latency_service_url=LATENCY_SERVICE_URL,
        endpoints=[_ep(m, request_type=request_type) for m in models],
        **kwargs,
    )


def _make_backend(config: LatencyServiceBackendConfig) -> LatencyServiceLLMBackend:
    """Construct a backend with a mocked OpenAILLMClient and stopped poller."""
    with patch(
        "switchyard.lib.backends"
        ".latency_service_llm_backend.OpenAILLMClient"
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(
            name=f"client-{kw.get('base_url')}"
        )
        with patch.object(HealthPoller, "start"):
            backend = LatencyServiceLLMBackend(config)
    return backend


def _set_health(
    backend: LatencyServiceLLMBackend,
    health_map: dict[str, EndpointHealthStatus | EndpointHealth],
) -> None:
    """Write directly to the backend's health cache for deterministic tests.

    Accepts either a bare ``EndpointHealthStatus`` (auto-wrapped with no
    latency sample) or a full ``EndpointHealth`` snapshot.
    """
    with backend._cache_lock:
        for model_id, value in health_map.items():
            if isinstance(value, EndpointHealthStatus):
                value = EndpointHealth(status=value)
            backend._health_cache[model_id] = value


def _make_completion(
    *, model: str = "test-model", content: str = "hello",
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def _openai_request(**overrides) -> ChatRequest:
    body: dict = {
        "model": "incoming-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return ChatRequest.openai_chat(body)  # type: ignore[arg-type]


def _api_status_error(status_code: int) -> openai.APIStatusError:
    """Synthetic APIStatusError carrying the given upstream status code."""
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://llm.test/v1/chat/completions"),
        json={"error": {"message": "synthetic"}},
    )
    return openai.APIStatusError(
        "synthetic", response=response, body={"error": "synthetic"}
    )


# ---------------------------------------------------------------------------
# Health poller helpers
# ---------------------------------------------------------------------------


def _make_poller(
    model_ids: list[str],
    health_cache: dict[str, EndpointHealth],
    poll_interval_s: float = 100.0,
) -> HealthPoller:
    cache_lock = threading.Lock()
    return HealthPoller(
        latency_service_url=LATENCY_SERVICE_URL,
        model_ids=model_ids,
        health_cache=health_cache,
        cache_lock=cache_lock,
        poll_interval_s=poll_interval_s,
        poll_timeout_s=5.0,
    )


def _health_response(status_code: int, **kwargs) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("GET", LATENCY_SERVICE_URL + "/v1/endpoints/health"),
        **kwargs,
    )


def _mock_health_response(poller: HealthPoller, response: httpx.Response) -> None:
    mock_client = MagicMock()
    mock_client.get.return_value = response
    poller._http_client = mock_client


def _run_one_poll(poller: HealthPoller) -> None:
    """Execute exactly one poll iteration and then exit the run loop."""
    original_wait = poller._stop_event.wait
    call_count = 0

    def _wait_then_stop(timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            poller._stop_event.set()
        return original_wait(timeout=0)

    poller._stop_event.wait = _wait_then_stop  # type: ignore[method-assign]
    poller.run()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_supported_request_types(self):
        backend = _make_backend(_config("model-A"))
        assert backend.supported_request_types == [ChatRequestType.OPENAI_CHAT]

    def test_responses_endpoint_supported_request_types(self):
        backend = _make_backend(_config("model-A", request_type="openai_responses"))
        assert backend.supported_request_types == [ChatRequestType.OPENAI_RESPONSES]

    def test_mixed_endpoint_supported_request_types_are_stable(self):
        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[
                _ep("chat-model"),
                _ep("responses-model", request_type="openai_responses"),
            ],
        )
        backend = _make_backend(config)
        assert backend.supported_request_types == [
            ChatRequestType.OPENAI_CHAT,
            ChatRequestType.OPENAI_RESPONSES,
        ]

    def test_request_type_aliases_normalize(self):
        assert LatencyServiceEndpoint(model="m", request_type="chat").request_type == "openai_chat"
        assert (
            LatencyServiceEndpoint(model="m", request_type="responses").request_type
            == "openai_responses"
        )

    def test_unknown_request_type_raises(self):
        with pytest.raises(ValidationError):
            LatencyServiceEndpoint(model="m", request_type="anthropic")  # type: ignore[arg-type]

    def test_unknown_backend_config_field_raises(self):
        with pytest.raises(ValidationError):
            LatencyServiceBackendConfig(
                latency_service_url=LATENCY_SERVICE_URL,
                endpoints=[_ep("model-A")],
                credential_polciy="caller_override",
            )

    def test_no_endpoints_raises(self):
        with pytest.raises(ValueError, match="At least one endpoint"):
            _make_backend(_config())

    def test_single_endpoint_ok(self):
        backend = _make_backend(_config("model-A"))
        assert "model-A" in backend._clients

    def test_missing_model_raises(self):
        # dataclass enforces ``model`` as required, but an empty string
        # should still be rejected by the backend as a routing key.
        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[LatencyServiceEndpoint(model="")],
        )
        with pytest.raises(ValueError, match="must have a 'model' field"):
            _make_backend(config)

    def test_duplicate_model_raises(self):
        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[_ep("model-A"), _ep("model-A", base_url="http://other")],
        )
        with pytest.raises(ValueError, match="Duplicate model ID"):
            _make_backend(config)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class TestClientConstruction:
    def test_disables_sdk_retries(self):
        """The SDK retry layer is disabled so the health-aware ``call`` loop is
        the single source of retries.

        The loop already retries on a *different* endpoint; letting the SDK also
        retry the *same* endpoint stacks multiplicatively and adds backoff
        sleeps that hold each request (and its buffered body) alive longer,
        amplifying connection-pool pressure during an upstream incident.
        """
        with patch(
            "switchyard.lib.backends"
            ".latency_service_llm_backend.OpenAILLMClient"
        ) as mock_cls:
            mock_cls.side_effect = lambda **kw: MagicMock()
            with patch.object(HealthPoller, "start"):
                LatencyServiceLLMBackend(_config("model-A", "model-B"))

        assert mock_cls.call_args_list, "expected one OpenAILLMClient per endpoint"
        for call in mock_cls.call_args_list:
            assert call.kwargs.get("max_retries") == 0


# ---------------------------------------------------------------------------
# Tiered endpoint selection
# ---------------------------------------------------------------------------


class TestEndpointSelection:
    def test_healthy_preferred_over_unknown(self):
        backend = _make_backend(_config("model-A", "model-B"))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.UNKNOWN,
        })
        picks = {backend._select_endpoint() for _ in range(50)}
        assert picks == {"model-A"}

    def test_unknown_preferred_over_degraded(self):
        backend = _make_backend(_config("model-A", "model-B"))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.UNKNOWN,
            "model-B": EndpointHealthStatus.DEGRADED,
        })
        picks = {backend._select_endpoint() for _ in range(50)}
        assert picks == {"model-A"}

    def test_all_healthy_random_distribution(self):
        backend = _make_backend(_config("model-A", "model-B", "model-C"))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.HEALTHY,
            "model-C": EndpointHealthStatus.HEALTHY,
        })
        picks = {backend._select_endpoint() for _ in range(200)}
        assert len(picks) > 1

    def test_all_degraded_still_picks(self):
        backend = _make_backend(_config("model-A", "model-B"))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.DEGRADED,
            "model-B": EndpointHealthStatus.DEGRADED,
        })
        pick = backend._select_endpoint()
        assert pick in {"model-A", "model-B"}

    def test_initial_state_is_unknown(self):
        backend = _make_backend(_config("model-A", "model-B"))
        assert backend._health_cache["model-A"].status == EndpointHealthStatus.UNKNOWN
        assert backend._health_cache["model-B"].status == EndpointHealthStatus.UNKNOWN
        assert backend._health_cache["model-A"].last_latency_ms is None


# ---------------------------------------------------------------------------
# Inverse-latency weighted selection within a tier
# ---------------------------------------------------------------------------


class TestLatencyWeightedSelection:
    def test_picks_skew_toward_lower_latency(self):
        """Two HEALTHY endpoints; the faster one should attract most traffic."""
        backend = _make_backend(_config("fast", "slow"))
        _set_health(backend, {
            "fast": EndpointHealth(EndpointHealthStatus.HEALTHY, 50.0),
            "slow": EndpointHealth(EndpointHealthStatus.HEALTHY, 500.0),
        })
        random.seed(0)
        picks = [backend._select_endpoint() for _ in range(2000)]
        fast = picks.count("fast")
        slow = picks.count("slow")
        # weights are 1/50 vs 1/500 → expected 10:1; bound loosely to avoid flake.
        assert fast > slow * 5, f"fast={fast} slow={slow}"

    def test_unknown_latency_falls_back_to_uniform(self):
        """If any candidate's last_latency_ms is None, the tier picks uniformly."""
        backend = _make_backend(_config("model-A", "model-B"))
        _set_health(backend, {
            "model-A": EndpointHealth(EndpointHealthStatus.HEALTHY, 100.0),
            "model-B": EndpointHealth(EndpointHealthStatus.HEALTHY, None),
        })
        random.seed(0)
        picks = [backend._select_endpoint() for _ in range(2000)]
        a = picks.count("model-A")
        b = picks.count("model-B")
        # Uniform within ±15% of 50/50 over 2000 draws is comfortably non-flaky.
        assert 850 < a < 1150, f"model-A picked {a}/2000"
        assert 850 < b < 1150, f"model-B picked {b}/2000"

    def test_zero_latency_falls_back_to_uniform(self):
        """Non-positive samples are treated as bogus and trigger uniform fallback."""
        backend = _make_backend(_config("model-A", "model-B"))
        _set_health(backend, {
            "model-A": EndpointHealth(EndpointHealthStatus.HEALTHY, 0.0),
            "model-B": EndpointHealth(EndpointHealthStatus.HEALTHY, 100.0),
        })
        # Must not raise (no 1/0); just exercise selection a few times.
        picks = {backend._select_endpoint() for _ in range(50)}
        assert picks <= {"model-A", "model-B"}

    def test_weighting_only_within_winning_tier(self):
        """A faster DEGRADED endpoint must not beat a slower HEALTHY one."""
        backend = _make_backend(_config("slow-healthy", "fast-degraded"))
        _set_health(backend, {
            "slow-healthy": EndpointHealth(EndpointHealthStatus.HEALTHY, 500.0),
            "fast-degraded": EndpointHealth(EndpointHealthStatus.DEGRADED, 10.0),
        })
        picks = {backend._select_endpoint() for _ in range(50)}
        assert picks == {"slow-healthy"}


# ---------------------------------------------------------------------------
# Retry with dedup
# ---------------------------------------------------------------------------


class TestRetryDedup:
    async def test_retry_avoids_same_endpoint(self):
        backend = _make_backend(_config("model-A", "model-B", max_retries=1))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.HEALTHY,
        })

        fail_mock = AsyncMock(side_effect=RuntimeError("down"))
        success_mock = AsyncMock(return_value=_make_completion())
        backend._clients["model-A"].acompletion = fail_mock
        backend._clients["model-B"].acompletion = success_mock

        ctx = ProxyContext()
        result = await backend.call(ctx, _openai_request())

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        assert success_mock.called

    async def test_all_retries_exhausted_raises(self):
        backend = _make_backend(_config("model-A", "model-B", max_retries=1))

        for mid in backend._clients:
            backend._clients[mid].acompletion = AsyncMock(
                side_effect=RuntimeError("down")
            )

        ctx = ProxyContext()
        with pytest.raises(RuntimeError, match="down"):
            await backend.call(ctx, _openai_request())


# ---------------------------------------------------------------------------
# Retry policy: transient errors retry, 4xx client errors fail fast
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 415, 422])
    def test_client_errors_are_not_retryable(self, status):
        from switchyard.lib.backends.latency_service_llm_backend import (
            _is_retryable_status,
        )

        assert _is_retryable_status(status) is False

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_transient_errors_are_retryable(self, status):
        from switchyard.lib.backends.latency_service_llm_backend import (
            _is_retryable_status,
        )

        assert _is_retryable_status(status) is True

    async def test_400_fails_fast_without_failover(self):
        """A 400 on the first endpoint must not retry the second."""
        from switchyard.lib.proxy_context import CTX_UPSTREAM_HTTP_STATUS

        backend = _make_backend(_config("model-A", "model-B", max_retries=2))
        # model-A is the sole HEALTHY endpoint → tried first deterministically.
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.UNKNOWN,
        })
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(400)
        )
        backend._clients["model-B"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        ctx = ProxyContext()
        with pytest.raises(openai.APIStatusError):
            await backend.call(ctx, _openai_request())

        # Failed once on model-A, never failed over to model-B.
        assert backend._clients["model-A"].acompletion.call_count == 1
        assert backend._clients["model-B"].acompletion.call_count == 0
        # Upstream status is stashed so the endpoint passes the 400 through.
        assert ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] == 400

    async def test_5xx_retries_to_another_endpoint(self):
        """A 503 on the first endpoint fails over and recovers on the second."""
        backend = _make_backend(_config("model-A", "model-B", max_retries=2))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.UNKNOWN,
        })
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(503)
        )
        backend._clients["model-B"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        ctx = ProxyContext()
        result = await backend.call(ctx, _openai_request())

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        assert backend._clients["model-B"].acompletion.call_count == 1
        assert ctx.selected_model == "model-B"


# ---------------------------------------------------------------------------
# Session affinity (pin a conversation to one endpoint)
# ---------------------------------------------------------------------------


def _conv_request(text: str) -> ChatRequest:
    """A request whose first user message anchors a distinct session."""
    return _openai_request(messages=[{"role": "user", "content": text}])


class TestSessionAffinity:
    async def test_disabled_by_default_pins_nothing(self):
        """Without session_affinity, the affinity map stays empty."""
        backend = _make_backend(_config("model-A", "model-B"))
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.HEALTHY,
        })
        for mid in backend._clients:
            backend._clients[mid].acompletion = AsyncMock(
                return_value=_make_completion()
            )

        await backend.call(ProxyContext(), _conv_request("task"))
        assert len(backend._affinity) == 0

    async def test_first_turn_latency_aware_then_sticks(self):
        """Turn 1 uses the latency-aware pick; later turns reuse that endpoint
        even after latencies flip to favour the other one."""
        backend = _make_backend(_config("fast", "slow", session_affinity=True))
        for mid in backend._clients:
            backend._clients[mid].acompletion = AsyncMock(
                return_value=_make_completion()
            )
        # Turn 1: only "fast" is selectable (slow is DEGRADED) → deterministic pin.
        _set_health(backend, {
            "fast": EndpointHealthStatus.HEALTHY,
            "slow": EndpointHealthStatus.DEGRADED,
        })
        req = _conv_request("the task")
        ctx1 = ProxyContext()
        await backend.call(ctx1, req)
        assert ctx1.selected_model == "fast"

        # Flip: "slow" is now healthy AND the latency winner. Affinity must
        # still keep the conversation on "fast".
        _set_health(backend, {
            "fast": EndpointHealth(EndpointHealthStatus.HEALTHY, 1000.0),
            "slow": EndpointHealth(EndpointHealthStatus.HEALTHY, 1.0),
        })
        for _ in range(20):
            ctx = ProxyContext()
            await backend.call(ctx, req)
            assert ctx.selected_model == "fast"

    async def test_distinct_conversations_pin_independently(self):
        """Two conversations resolve to independent pins."""
        backend = _make_backend(_config("model-A", "model-B", session_affinity=True))
        for mid in backend._clients:
            backend._clients[mid].acompletion = AsyncMock(
                return_value=_make_completion()
            )

        # Conversation 1 pins to model-A (only A selectable).
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.DEGRADED,
        })
        req_a = _conv_request("conversation one")
        ctx_a = ProxyContext()
        await backend.call(ctx_a, req_a)
        assert ctx_a.selected_model == "model-A"

        # Conversation 2 pins to model-B (only B selectable).
        _set_health(backend, {
            "model-A": EndpointHealthStatus.DEGRADED,
            "model-B": EndpointHealthStatus.HEALTHY,
        })
        req_b = _conv_request("conversation two")
        ctx_b = ProxyContext()
        await backend.call(ctx_b, req_b)
        assert ctx_b.selected_model == "model-B"

        # Independent pins: each conversation resolves to its own endpoint.
        assert len(backend._affinity) == 2
        assert backend._affinity.pinned(ProxyContext(), req_a) == "model-A"
        assert backend._affinity.pinned(ProxyContext(), req_b) == "model-B"

    async def test_degraded_pin_reroutes_and_repins(self):
        """When the pinned endpoint degrades, the next turn re-routes to a
        healthy endpoint and the pin follows."""
        backend = _make_backend(_config("model-A", "model-B", session_affinity=True))
        for mid in backend._clients:
            backend._clients[mid].acompletion = AsyncMock(
                return_value=_make_completion()
            )
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.DEGRADED,
        })
        req = _conv_request("task")
        await backend.call(ProxyContext(), req)
        assert backend._affinity.pinned(ProxyContext(), req) == "model-A"

        # model-A degrades; model-B becomes the only healthy endpoint.
        _set_health(backend, {
            "model-A": EndpointHealthStatus.DEGRADED,
            "model-B": EndpointHealthStatus.HEALTHY,
        })
        ctx2 = ProxyContext()
        await backend.call(ctx2, req)
        assert ctx2.selected_model == "model-B"
        assert backend._affinity.pinned(ProxyContext(), req) == "model-B"

    async def test_pin_follows_recovery_after_call_failure(self):
        """If the first-turn endpoint fails the call, the pin records the
        endpoint that actually served the request."""
        backend = _make_backend(
            _config("model-A", "model-B", max_retries=1, session_affinity=True)
        )
        _set_health(backend, {
            "model-A": EndpointHealthStatus.HEALTHY,
            "model-B": EndpointHealthStatus.HEALTHY,
        })
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=RuntimeError("down")
        )
        backend._clients["model-B"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        req = _conv_request("task")
        ctx = ProxyContext()
        await backend.call(ctx, req)
        # Regardless of which endpoint was tried first, only model-B succeeds.
        assert ctx.selected_model == "model-B"
        assert backend._affinity.pinned(ProxyContext(), req) == "model-B"

    async def test_lru_eviction_bounds_map(self):
        """The affinity map never exceeds affinity_max_sessions."""
        backend = _make_backend(
            _config("model-A", session_affinity=True, affinity_max_sessions=2)
        )
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )
        _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})

        for i in range(3):
            await backend.call(ProxyContext(), _conv_request(f"conversation {i}"))

        assert len(backend._affinity) == 2

    def test_affinity_config_reaches_backend(self):
        """A config dict's affinity settings reach the backend's pin map."""
        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[{"model": "model-A", "base_url": "http://a.test", "api_key": "k"}],
            session_affinity=True,
            affinity_max_sessions=5,
        )
        assert config.session_affinity is True
        assert config.affinity_max_sessions == 5

        backend = _make_backend(config)
        assert backend._affinity.max_sessions == 5

    async def test_affinity_counters_rendered_on_metrics(self):
        """hits/misses are surfaced in the OTel snapshot when affinity is enabled."""
        backend = _make_backend(_config("model-A", session_affinity=True))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )
        _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})

        req = _conv_request("the task")
        await backend.call(ProxyContext(), req)   # first turn → miss (no pin yet)
        await backend.call(ProxyContext(), req)   # reuse → hit
        await backend.call(ProxyContext(), req)   # reuse → hit

        snap = backend._metrics_snapshot()
        assert snap["affinity_hits"] == 2
        assert snap["affinity_misses"] == 1

    async def test_affinity_counters_absent_when_disabled(self):
        """The warm-reuse counters stay off the snapshot by default."""
        backend = _make_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )
        _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})

        await backend.call(ProxyContext(), _conv_request("task"))

        snap = backend._metrics_snapshot()
        assert "affinity_hits" not in snap
        assert "affinity_misses" not in snap

    def test_negative_affinity_max_rejected_at_construction(self):
        """A negative cap is rejected when the config is built — otherwise the
        LRU eviction loop in SessionCache.put would pop past an empty map."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LatencyServiceBackendConfig(
                latency_service_url=LATENCY_SERVICE_URL,
                endpoints=[_ep("model-A")],
                session_affinity=True,
                affinity_max_sessions=-1,
            )

    def test_zero_affinity_max_rejected_when_enabled(self):
        """A zero cap with affinity on is rejected — it would retain nothing."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LatencyServiceBackendConfig(
                latency_service_url=LATENCY_SERVICE_URL,
                endpoints=[_ep("model-A")],
                session_affinity=True,
                affinity_max_sessions=0,
            )


# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------


class TestCall:
    async def test_non_streaming_returns_completion_chat_response(self):
        backend = _make_backend(_config("model-A"))
        completion = _make_completion(content="world")
        backend._clients["model-A"].acompletion = AsyncMock(return_value=completion)

        ctx = ProxyContext()
        result = await backend.call(ctx, _openai_request())

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        assert result.body["choices"][0]["message"]["content"] == "world"
        # ``selected_model`` is the cross-language ctx field the Rust
        # ``StatsResponseProcessor`` reads to attribute tokens and
        # end-to-end latency per endpoint on /metrics.
        assert ctx.selected_model == "model-A"
        # ``backend_call_latency_ms`` is the second cross-language hook —
        # the response processor uses it to compute ``routing_overhead_ms``.
        # Mocked upstream returns instantly so latency is small but non-None.
        assert ctx.backend_call_latency_ms is not None
        assert ctx.backend_call_latency_ms >= 0.0

    async def test_responses_endpoint_dispatches_responses_natively(self):
        backend = _make_backend(_config("model-A", request_type="openai_responses"))
        backend._clients["model-A"].aresponses = AsyncMock(
            return_value={
                "id": "resp-test",
                "object": "response",
                "model": "model-A",
                "output": [],
            }
        )
        backend._clients["model-A"].acompletion = AsyncMock()

        ctx = ProxyContext()
        result = await backend.call(
            ctx,
            ChatRequest.openai_responses({"model": "incoming-model", "input": "hi"}),
        )

        assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_COMPLETION)
        backend._clients["model-A"].aresponses.assert_awaited_once()
        backend._clients["model-A"].acompletion.assert_not_called()
        call_kwargs = backend._clients["model-A"].aresponses.call_args.kwargs
        assert call_kwargs["model"] == "model-A"
        assert call_kwargs["input"] == "hi"
        assert "messages" not in call_kwargs
        assert ctx.selected_model == "model-A"

    async def test_chat_endpoint_translates_responses_to_chat_fallback(self):
        backend = _make_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )
        backend._clients["model-A"].aresponses = AsyncMock()

        result = await backend.call(
            ProxyContext(),
            ChatRequest.openai_responses({"model": "incoming-model", "input": "hi"}),
        )

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        backend._clients["model-A"].acompletion.assert_awaited_once()
        backend._clients["model-A"].aresponses.assert_not_called()
        call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
        assert call_kwargs["model"] == "model-A"
        assert call_kwargs["messages"][0]["content"] == "hi"

    async def test_streaming_wraps_into_streaming_chat_response(self):
        backend = _make_backend(_config("model-A"))
        stream_mock = MagicMock(spec=openai.AsyncStream)
        backend._clients["model-A"].acompletion = AsyncMock(return_value=stream_mock)

        result = await backend.call(ProxyContext(), _openai_request(stream=True))
        assert response_type_matches(result, ChatResponseType.OPENAI_STREAM)

    async def test_responses_streaming_wraps_into_responses_stream(self):
        backend = _make_backend(_config("model-A", request_type="openai_responses"))
        stream_mock = MagicMock(spec=openai.AsyncStream)
        backend._clients["model-A"].aresponses = AsyncMock(return_value=stream_mock)

        result = await backend.call(
            ProxyContext(),
            ChatRequest.openai_responses({
                "model": "incoming-model",
                "input": "hi",
                "stream": True,
            }),
        )

        assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_STREAM)

    async def test_model_id_overrides_incoming_model(self):
        """The selected endpoint model ID should replace the request body's model."""
        backend = _make_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        await backend.call(ProxyContext(), _openai_request(model="incoming-model"))

        call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
        assert call_kwargs["model"] == "model-A"

    async def test_accepts_anthropic_request_via_translation(self):
        """Non-OpenAI inbound formats go through TranslationEngine."""
        backend = _make_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        anthropic_req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        })
        result = await backend.call(ProxyContext(), anthropic_req)

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
        assert call_kwargs["model"] == "model-A"
        assert "messages" in call_kwargs

    async def test_upstream_model_override_used_in_body(self):
        """When upstream_model is set, it's the value sent in body['model']."""
        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[
                LatencyServiceEndpoint(
                    model="openai/gpt-5.5",
                    upstream_model="openai/openai/gpt-5.5",
                    base_url="https://inference-api.nvidia.com/v1",
                    api_key="nvapi-test",
                ),
            ],
        )
        backend = _make_backend(config)
        backend._clients["openai/gpt-5.5"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        ctx = ProxyContext()
        await backend.call(ctx, _openai_request(model="incoming-model"))

        call_kwargs = backend._clients["openai/gpt-5.5"].acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/openai/gpt-5.5"
        assert ctx.selected_model == "openai/gpt-5.5"

    async def test_upstream_model_default_falls_back_to_model(self):
        """Without upstream_model, body['model'] should be the lookup key."""
        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[
                LatencyServiceEndpoint(
                    model="openai/gpt-5.5",
                    base_url="http://llm.test",
                    api_key="k",
                ),
            ],
        )
        backend = _make_backend(config)
        backend._clients["openai/gpt-5.5"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        await backend.call(ProxyContext(), _openai_request())

        call_kwargs = backend._clients["openai/gpt-5.5"].acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-5.5"


# ---------------------------------------------------------------------------
# Credential policy
# ---------------------------------------------------------------------------


class TestCallerApiKey:
    """Per-request caller-key forwarding is opt-in for multi-tenant deployments.

    When the HTTP endpoint extracts an ``Authorization: Bearer ...`` header
    and writes the value into ``ctx.metadata[CTX_CALLER_API_KEY]``, the
    backend ignores that key by default. ``credential_policy="caller_override"``
    makes the caller key a per-call SDK override.
    """

    async def test_default_policy_ignores_caller_key_for_chat_sdk(self):
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        backend = _make_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "nvapi-caller-supplied"
        await backend.call(ctx, _openai_request())

        call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
        assert call_kwargs["api_key"] is None

    async def test_caller_override_policy_passes_caller_key_to_chat_sdk(self):
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        backend = _make_backend(_config("model-A", credential_policy="caller_override"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "nvapi-caller-supplied"
        await backend.call(ctx, _openai_request())

        call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
        assert call_kwargs["api_key"] == "nvapi-caller-supplied"

    async def test_default_policy_ignores_caller_key_for_responses_sdk(self):
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        backend = _make_backend(_config("model-A", request_type="openai_responses"))
        backend._clients["model-A"].aresponses = AsyncMock(
            return_value={"id": "resp-test", "output": []}
        )

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "nvapi-caller-supplied"
        await backend.call(
            ctx,
            ChatRequest.openai_responses({"model": "incoming-model", "input": "hi"}),
        )

        call_kwargs = backend._clients["model-A"].aresponses.call_args.kwargs
        assert call_kwargs["api_key"] is None

    async def test_caller_override_policy_passes_caller_key_to_responses_sdk(self):
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        backend = _make_backend(_config(
            "model-A",
            request_type="openai_responses",
            credential_policy="caller_override",
        ))
        backend._clients["model-A"].aresponses = AsyncMock(
            return_value={"id": "resp-test", "output": []}
        )

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "nvapi-caller-supplied"
        await backend.call(
            ctx,
            ChatRequest.openai_responses({"model": "incoming-model", "input": "hi"}),
        )

        call_kwargs = backend._clients["model-A"].aresponses.call_args.kwargs
        assert call_kwargs["api_key"] == "nvapi-caller-supplied"

    async def test_missing_caller_key_falls_back_to_config(self):
        """ctx without a caller key sends api_key=None to OpenAILLMClient.

        ``OpenAILLMClient.acompletion`` then uses the construction-time
        config key (no ``with_options`` override). The test only asserts
        the contract at the seam — that we pass ``api_key=None`` so the
        client falls back, rather than passing a stale or empty string.
        """
        backend = _make_backend(_config("model-A"))
        backend._clients["model-A"].acompletion = AsyncMock(
            return_value=_make_completion()
        )

        await backend.call(ProxyContext(), _openai_request())

        call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
        assert call_kwargs["api_key"] is None

    async def test_missing_caller_key_falls_back_to_config_for_responses(self):
        backend = _make_backend(_config("model-A", request_type="openai_responses"))
        backend._clients["model-A"].aresponses = AsyncMock(
            return_value={"id": "resp-test", "output": []}
        )

        await backend.call(
            ProxyContext(),
            ChatRequest.openai_responses({"model": "incoming-model", "input": "hi"}),
        )

        call_kwargs = backend._clients["model-A"].aresponses.call_args.kwargs
        assert call_kwargs["api_key"] is None


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/messages", {"model": "incoming-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "incoming-model", "input": "hi"}),
    ],
)
async def test_non_chat_http_endpoints_default_to_endpoint_credentials(
    path: str,
    body: dict[str, object],
) -> None:
    """HTTP ingress may attach caller keys, but the default backend policy ignores them."""
    backend = _make_backend(_config("model-A"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._clients["model-A"].acompletion = AsyncMock(
        return_value=_make_completion(model="model-A")
    )
    app = build_switchyard_app(Switchyard(backend=backend, translator=TranslationEngine()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            json=body,
            headers={"Authorization": "Bearer CALLER-BYO-KEY"},
        )

    assert response.status_code == 200, response.text
    call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
    assert call_kwargs["api_key"] is None


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/messages", {"model": "incoming-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "incoming-model", "input": "hi"}),
    ],
)
async def test_non_chat_http_endpoints_forward_caller_key_in_byo_mode(
    path: str,
    body: dict[str, object],
) -> None:
    """Opt-in BYO mode preserves caller keys across Messages/Responses ingress."""
    backend = _make_backend(_config("model-A", credential_policy="caller_override"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._clients["model-A"].acompletion = AsyncMock(
        return_value=_make_completion(model="model-A")
    )
    app = build_switchyard_app(Switchyard(backend=backend, translator=TranslationEngine()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            json=body,
            headers={"Authorization": "Bearer CALLER-BYO-KEY"},
        )

    assert response.status_code == 200, response.text
    call_kwargs = backend._clients["model-A"].acompletion.call_args.kwargs
    assert call_kwargs["api_key"] == "CALLER-BYO-KEY"


async def test_responses_http_endpoint_can_use_native_latency_responses_mode() -> None:
    backend = _make_backend(_config("model-A", request_type="openai_responses"))
    _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
    backend._clients["model-A"].aresponses = AsyncMock(
        return_value={
            "id": "resp-test",
            "object": "response",
            "model": "model-A",
            "output": [],
        }
    )
    backend._clients["model-A"].acompletion = AsyncMock()
    app = build_switchyard_app(Switchyard(backend=backend, translator=TranslationEngine()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "incoming-model", "input": "hi"},
            headers={"Authorization": "Bearer CALLER-BYO-KEY"},
        )

    assert response.status_code == 200, response.text
    backend._clients["model-A"].aresponses.assert_awaited_once()
    backend._clients["model-A"].acompletion.assert_not_called()
    call_kwargs = backend._clients["model-A"].aresponses.call_args.kwargs
    assert call_kwargs["api_key"] is None
    assert call_kwargs["input"] == "hi"


class TestEndpointApiKeyReachesUpstream:
    """End-to-end auth: which credential the upstream HTTP call carries.

    The seam test above only asserts the ``api_key`` value handed to
    ``acompletion``; it cannot catch a regression where a blank caller key
    clobbers the configured endpoint key and the upstream call goes out
    unauthenticated. These tests run the real
    ``OpenAILLMClient`` against a mocked HTTP transport and inspect the
    ``Authorization`` header that actually reaches the wire.
    """

    @staticmethod
    def _backend_with_captured_auth(
        captured: dict[str, str | None],
        credential_policy: str = "configured_endpoint",
    ) -> LatencyServiceLLMBackend:
        """Real-client backend whose upstream auth header is captured."""

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "model-A",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }],
                },
            )

        config = LatencyServiceBackendConfig(
            latency_service_url=LATENCY_SERVICE_URL,
            endpoints=[LatencyServiceEndpoint(
                model="model-A",
                base_url="http://llm.test/v1",
                api_key="ENDPOINT-CONFIGURED-KEY",
            )],
            credential_policy=credential_policy,
        )
        with patch.object(HealthPoller, "start"):
            backend = LatencyServiceLLMBackend(config)
        backend._clients["model-A"].async_client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        _set_health(backend, {"model-A": EndpointHealthStatus.HEALTHY})
        return backend

    async def test_no_caller_key_uses_configured_endpoint_key(self):
        """No caller key → the endpoint's configured key authenticates upstream."""
        captured: dict[str, str | None] = {}
        backend = self._backend_with_captured_auth(captured)

        await backend.call(ProxyContext(), _openai_request())

        assert captured["authorization"] == "Bearer ENDPOINT-CONFIGURED-KEY"

    async def test_default_policy_ignores_caller_key(self):
        """Default policy keeps using the endpoint key even when a caller key exists."""
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        captured: dict[str, str | None] = {}
        backend = self._backend_with_captured_auth(captured)

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "CALLER-BYO-KEY"
        await backend.call(ctx, _openai_request())

        assert captured["authorization"] == "Bearer ENDPOINT-CONFIGURED-KEY"

    async def test_caller_override_policy_uses_caller_key(self):
        """BYO mode lets a caller-supplied key override the endpoint key."""
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        captured: dict[str, str | None] = {}
        backend = self._backend_with_captured_auth(
            captured,
            credential_policy="caller_override",
        )

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "CALLER-BYO-KEY"
        await backend.call(ctx, _openai_request())

        assert captured["authorization"] == "Bearer CALLER-BYO-KEY"

    async def test_blank_caller_key_does_not_clobber_endpoint_key(self):
        """A blank caller key in BYO mode still falls back to the configured key."""
        from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

        captured: dict[str, str | None] = {}
        backend = self._backend_with_captured_auth(
            captured,
            credential_policy="caller_override",
        )

        ctx = ProxyContext()
        ctx.metadata[CTX_CALLER_API_KEY] = "   "
        await backend.call(ctx, _openai_request())

        assert captured["authorization"] == "Bearer ENDPOINT-CONFIGURED-KEY"


# ---------------------------------------------------------------------------
# Readiness and shutdown
# ---------------------------------------------------------------------------


class TestReadinessAndShutdown:
    def test_not_ready_before_first_poll(self):
        backend = _make_backend(_config("model-A"))
        assert backend.is_ready() is False

    def test_ready_after_first_poll(self):
        backend = _make_backend(_config("model-A"))
        backend._poller._poll_count = 1
        assert backend.is_ready() is True

    def test_shutdown_stops_poller(self):
        backend = _make_backend(_config("model-A"))
        backend.shutdown()
        assert backend._poller._stop_event.is_set()


# ---------------------------------------------------------------------------
# Health poller
# ---------------------------------------------------------------------------


class TestHealthPoller:
    def test_poll_updates_cache(self):
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.UNKNOWN),
            "model-B": EndpointHealth(EndpointHealthStatus.UNKNOWN),
        }
        poller = _make_poller(["model-A", "model-B"], health_cache)

        response_data = {
            "endpoint_health": {
                "model-A": {"status": "healthy", "last_latency_ms": 100.0},
                "model-B": {"status": "degraded", "last_latency_ms": 1500.0},
            }
        }
        _mock_health_response(poller, _health_response(200, json=response_data))
        _run_one_poll(poller)

        assert health_cache["model-A"].status == EndpointHealthStatus.HEALTHY
        assert health_cache["model-A"].last_latency_ms == 100.0
        assert health_cache["model-B"].status == EndpointHealthStatus.DEGRADED
        assert health_cache["model-B"].last_latency_ms == 1500.0
        assert poller.has_polled

    def test_poll_captures_null_latency(self):
        """``last_latency_ms`` is nullable in the spec; absence becomes None."""
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.UNKNOWN),
        }
        poller = _make_poller(["model-A"], health_cache)

        response_data = {
            "endpoint_health": {
                "model-A": {"status": "healthy", "last_latency_ms": None},
            }
        }
        _mock_health_response(poller, _health_response(200, json=response_data))
        _run_one_poll(poller)

        assert health_cache["model-A"].status == EndpointHealthStatus.HEALTHY
        assert health_cache["model-A"].last_latency_ms is None

    def test_poll_failure_resets_to_unknown(self):
        """If the Latency Service is unreachable, all endpoints reset to UNKNOWN
        so the backend falls back to random routing rather than stale data."""
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.HEALTHY, 100.0),
            "model-B": EndpointHealth(EndpointHealthStatus.DEGRADED, 800.0),
        }
        poller = _make_poller(["model-A", "model-B"], health_cache)
        _mock_health_response(poller, _health_response(500))
        _run_one_poll(poller)

        assert health_cache["model-A"].status == EndpointHealthStatus.UNKNOWN
        assert health_cache["model-A"].last_latency_ms is None
        assert health_cache["model-B"].status == EndpointHealthStatus.UNKNOWN
        assert health_cache["model-B"].last_latency_ms is None
        assert not poller.has_polled

    def test_poll_ignores_unknown_model_ids(self):
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.UNKNOWN),
        }
        poller = _make_poller(["model-A"], health_cache)

        response_data = {
            "endpoint_health": {
                "model-A": {"status": "healthy", "last_latency_ms": 100.0},
                "model-UNKNOWN": {"status": "degraded", "last_latency_ms": 999.0},
            }
        }
        _mock_health_response(poller, _health_response(200, json=response_data))
        _run_one_poll(poller)

        assert health_cache["model-A"].status == EndpointHealthStatus.HEALTHY
        assert "model-UNKNOWN" not in health_cache

    def test_success_increments_polls_and_records_timestamp(self):
        """``/metrics`` reads these fields to expose poll-loop health."""
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.UNKNOWN),
        }
        poller = _make_poller(["model-A"], health_cache)
        _mock_health_response(
            poller,
            _health_response(200, json={"endpoint_health": {
                "model-A": {"status": "healthy", "last_latency_ms": 50.0},
            }}),
        )
        _run_one_poll(poller)

        assert poller.poll_successes == 1
        assert poller.poll_failures == 0
        assert poller.last_poll_ok is True
        assert poller.seconds_since_last_success is not None
        assert poller.seconds_since_last_success >= 0

    def test_failure_increments_failures_and_leaves_age_unset(self):
        """Pre-success failure leaves ``seconds_since_last_success`` None so
        scrapers can distinguish "never polled" from "polled but stale"."""
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.UNKNOWN),
        }
        poller = _make_poller(["model-A"], health_cache)
        _mock_health_response(poller, _health_response(500))
        _run_one_poll(poller)

        assert poller.poll_successes == 0
        assert poller.poll_failures == 1
        assert poller.last_poll_ok is False
        assert poller.seconds_since_last_success is None

    def test_stop_event_terminates_poller(self):
        health_cache: dict[str, EndpointHealth] = {
            "model-A": EndpointHealth(EndpointHealthStatus.UNKNOWN),
        }
        poller = _make_poller(["model-A"], health_cache, poll_interval_s=0.05)
        _mock_health_response(
            poller,
            _health_response(200, json={"endpoint_health": {
                "model-A": {"status": "healthy", "last_latency_ms": 50.0},
            }}),
        )

        poller.start()
        time.sleep(0.15)
        poller.stop()
        poller.join(timeout=2.0)

        assert not poller.is_alive()
        assert poller.has_polled
