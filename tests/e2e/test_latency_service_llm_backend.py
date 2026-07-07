# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for :class:`LatencyServiceLLMBackend`.

These tests make real LLM calls through the latency-service backend
to validate end-to-end routing against an external health source.

A lightweight mock Latency Service runs in-process to supply health
verdicts. The LLM calls go to a real backend (OpenRouter by default,
with NVIDIA Inference Hub as a fallback).

Prerequisites:
    - OPENROUTER_API_KEY or NVIDIA_API_KEY environment variable

Run with:
    OPENROUTER_API_KEY=sk-or-... pytest tests/e2e/test_latency_service_llm_backend.py -v
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from switchyard.lib.backends.health_poller import (
    EndpointHealthStatus,
)
from switchyard.lib.backends.latency_service_llm_backend import (
    LatencyServiceLLMBackend,
)
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import (
    ChatRequest,
    ChatResponseType,
    response_type_matches,
)

from .conftest import find_free_port, get_nvidia_config

pytestmark = pytest.mark.integration

_nvidia = get_nvidia_config()
_skip_reason = "OPENROUTER_API_KEY or NVIDIA_API_KEY not set"

BACKEND_BASE_URL = _nvidia["base_url"]

if _nvidia["provider"] == "openrouter":
    MODEL_A = _nvidia["model"]
    MODEL_B = os.environ.get("OPENROUTER_MODEL_B") or "anthropic/claude-opus-4.7"
else:
    MODEL_A = "openai/openai/gpt-5.2"
    MODEL_B = "azure/openai/gpt-5.2"

SIMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! Can you help me?"},
]


# ---------------------------------------------------------------------------
# Mock Latency Service
# ---------------------------------------------------------------------------


class _HealthState:
    """Thread-safe mutable health state for the mock Latency Service."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._health: dict[str, str] = {}
        self.request_count = 0

    def set_health(self, model_id: str, status: str) -> None:
        with self._lock:
            self._health[model_id] = status

    def get_response(self, requested_ids: list[str]) -> dict:
        with self._lock:
            self.request_count += 1
            return {
                "endpoint_health": {
                    mid: {
                        "status": self._health.get(mid, "unknown"),
                        "last_latency_ms": None,
                    }
                    for mid in requested_ids
                }
            }


def _make_handler(state: _HealthState):
    """Create an HTTP request handler class bound to the given health state."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/v1/endpoints/health":
                qs = parse_qs(parsed.query)
                endpoint_ids = qs.get("endpoint_ids", [])
                body = json.dumps(state.get_response(endpoint_ids))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass  # Suppress request logging in test output

    return _Handler


class MockLatencyService:
    """Manages a mock Latency Service HTTP server for testing."""

    def __init__(self) -> None:
        self.state = _HealthState()
        self.port = find_free_port()
        handler = _make_handler(self.state)
        self._server = HTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="mock-latency-service",
        )
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def set_health(self, model_id: str, status: str) -> None:
        self.state.set_health(model_id, status)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5.0)


@pytest.fixture()
def mock_latency_service():
    """Provide a mock Latency Service for the duration of a test."""
    svc = MockLatencyService()
    yield svc
    svc.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _endpoint(model: str) -> LatencyServiceEndpoint:
    return LatencyServiceEndpoint(
        model=model,
        api_key=_nvidia["api_key"],
        base_url=BACKEND_BASE_URL,
    )


def _make_backend(
    latency_service_url: str,
    models: list[str],
    poll_interval_s: float = 1.0,
    **kwargs,
) -> LatencyServiceLLMBackend:
    config = LatencyServiceBackendConfig(
        latency_service_url=latency_service_url,
        endpoints=[_endpoint(m) for m in models],
        poll_interval_s=poll_interval_s,
        **kwargs,
    )
    return LatencyServiceLLMBackend(config)


def _chat_request(stream: bool = False) -> ChatRequest:
    # ``max_tokens`` is deliberately generous so reasoning-capable
    # backends (e.g. the GPT-5 / Qwen 3.5 families) have room to spend
    # the first several hundred tokens on internal reasoning before
    # emitting user-visible ``content``.  Matches the value used by
    # the sibling passthrough e2e suites.
    body: dict = {
        "messages": SIMPLE_MESSAGES,
        "max_tokens": 2048,
        "stream": stream,
    }
    return ChatRequest.openai_chat(body)  # type: ignore[arg-type]


def _wait_for_first_poll(
    backend: LatencyServiceLLMBackend,
    timeout: float = 10.0,
) -> None:
    """Block until the background poller completes at least one poll."""
    deadline = time.monotonic() + timeout
    while not backend.is_ready():
        if time.monotonic() > deadline:
            raise TimeoutError("Poller did not complete first poll within timeout")
        time.sleep(0.05)


# ================================================================== #
# Non-streaming tests
# ================================================================== #


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestLatencyServiceBackendNonStreaming:
    """Non-streaming requests through the latency-service backend."""

    async def test_healthy_endpoint_serves_request(self, mock_latency_service):
        """A HEALTHY endpoint should successfully serve a non-streaming request."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        backend = _make_backend(mock_latency_service.url, [MODEL_A])
        try:
            _wait_for_first_poll(backend)

            ctx = ProxyContext()
            response = await backend.call(ctx, _chat_request())

            assert response_type_matches(response, ChatResponseType.OPENAI_COMPLETION)
            assert response.body["choices"][0]["message"]["content"]
            assert response.body["choices"][0]["message"]["role"] == "assistant"
            assert ctx.selected_model == MODEL_A
        finally:
            await backend.shutdown()

    async def test_response_has_usage(self, mock_latency_service):
        """Response should include token usage statistics."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        backend = _make_backend(mock_latency_service.url, [MODEL_A])
        try:
            _wait_for_first_poll(backend)

            ctx = ProxyContext()
            response = await backend.call(ctx, _chat_request())

            assert response_type_matches(response, ChatResponseType.OPENAI_COMPLETION)
            assert response.body["usage"] is not None
            assert response.body["usage"]["prompt_tokens"] > 0
            assert response.body["usage"]["completion_tokens"] > 0
        finally:
            await backend.shutdown()


# ================================================================== #
# Streaming tests
# ================================================================== #


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestLatencyServiceBackendStreaming:
    """Streaming requests through the latency-service backend."""

    async def test_streaming_yields_chunks(self, mock_latency_service):
        """Streaming request should yield content chunks."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        backend = _make_backend(mock_latency_service.url, [MODEL_A])
        try:
            _wait_for_first_poll(backend)

            ctx = ProxyContext()
            response = await backend.call(ctx, _chat_request(stream=True))
            assert response_type_matches(response, ChatResponseType.OPENAI_STREAM)

            chunks = [chunk async for chunk in response.stream]
            assert len(chunks) >= 2, "Expected at least one content chunk + stop chunk"

            text_parts = [
                c.choices[0].delta.content
                for c in chunks
                if c.choices and c.choices[0].delta and c.choices[0].delta.content
            ]
            full_text = "".join(text_parts)
            assert len(full_text) > 0
        finally:
            await backend.shutdown()


# ================================================================== #
# Health-based routing
# ================================================================== #


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestLatencyServiceBackendHealth:
    """Verify health-based endpoint selection with real LLM calls."""

    async def test_healthy_preferred_over_degraded(self, mock_latency_service):
        """When one endpoint is HEALTHY and another DEGRADED, all traffic
        should go to the HEALTHY one."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        mock_latency_service.set_health(MODEL_B, "degraded")
        backend = _make_backend(mock_latency_service.url, [MODEL_A, MODEL_B])
        try:
            _wait_for_first_poll(backend)

            selected_models = set()
            for _ in range(3):
                ctx = ProxyContext()
                await backend.call(ctx, _chat_request())
                selected_models.add(ctx.selected_model)

            assert selected_models == {MODEL_A}, (
                f"Expected all traffic to HEALTHY model, got: {selected_models}"
            )
        finally:
            await backend.shutdown()

    async def test_unknown_falls_back_to_random(self, mock_latency_service):
        """When all endpoints are UNKNOWN, the backend should distribute
        traffic randomly across all endpoints."""
        mock_latency_service.set_health(MODEL_A, "unknown")
        mock_latency_service.set_health(MODEL_B, "unknown")
        backend = _make_backend(mock_latency_service.url, [MODEL_A, MODEL_B])
        try:
            _wait_for_first_poll(backend)

            selected_models = set()
            for _ in range(6):
                ctx = ProxyContext()
                await backend.call(ctx, _chat_request())
                selected_models.add(ctx.selected_model)

            assert len(selected_models) > 1, (
                f"Expected random distribution across models, got: {selected_models}"
            )
        finally:
            await backend.shutdown()

    async def test_poller_receives_health_updates(self, mock_latency_service):
        """The background poller should pick up health changes from the
        Latency Service and reflect them in the health cache."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        backend = _make_backend(
            mock_latency_service.url, [MODEL_A], poll_interval_s=0.5,
        )
        try:
            _wait_for_first_poll(backend)
            assert backend._health_cache[MODEL_A].status == EndpointHealthStatus.HEALTHY

            mock_latency_service.set_health(MODEL_A, "degraded")
            time.sleep(1.5)  # Wait for at least one poll cycle

            assert backend._health_cache[MODEL_A].status == EndpointHealthStatus.DEGRADED
        finally:
            await backend.shutdown()


# ================================================================== #
# Readiness
# ================================================================== #


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestLatencyServiceBackendReadiness:
    """Verify is_ready() behavior with a real poller."""

    async def test_ready_after_first_poll(self, mock_latency_service):
        """is_ready() should return True once the poller has fetched health."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        backend = _make_backend(mock_latency_service.url, [MODEL_A])
        try:
            assert not backend.is_ready()
            _wait_for_first_poll(backend)
            assert backend.is_ready()
        finally:
            await backend.shutdown()


# ================================================================== #
# Session affinity (sticky routing)
# ================================================================== #


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestLatencyServiceBackendStickiness:
    """Verify session affinity pins a multi-turn conversation to one endpoint."""

    async def test_conversation_sticks_to_one_endpoint(self, mock_latency_service):
        """With ``session_affinity`` on, every turn of one conversation routes
        to the same endpoint — even though both endpoints are HEALTHY with no
        latency sample and would otherwise be picked at random (the foil is
        ``test_unknown_falls_back_to_random``, which spreads without affinity)."""
        mock_latency_service.set_health(MODEL_A, "healthy")
        mock_latency_service.set_health(MODEL_B, "healthy")
        backend = _make_backend(
            mock_latency_service.url, [MODEL_A, MODEL_B], session_affinity=True,
        )
        try:
            _wait_for_first_poll(backend)

            # One conversation that grows each turn but keeps a stable prefix
            # (same system + first user message → same session key).
            messages = list(SIMPLE_MESSAGES)
            selected: list[str] = []
            for turn in range(6):
                ctx = ProxyContext()
                request = ChatRequest.openai_chat(
                    {"messages": messages, "max_tokens": 128},  # type: ignore[arg-type]
                )
                response = await backend.call(ctx, request)
                assert response_type_matches(
                    response, ChatResponseType.OPENAI_COMPLETION
                )
                selected.append(ctx.selected_model)
                messages = messages + [
                    {"role": "assistant", "content": "Sure."},
                    {"role": "user", "content": f"Follow-up {turn + 1}?"},
                ]

            assert len(set(selected)) == 1, (
                f"affinity should pin the conversation to one endpoint, got {selected}"
            )
            assert selected[0] in {MODEL_A, MODEL_B}
        finally:
            await backend.shutdown()
