# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end profile tests with mocked upstream model providers.

This file is the highest-leverage entry in the mocked-test suite added after
the post-PR-#12 wiring regressions. Where ``test_inference_e2e.py`` swaps in
a fake ``LLMBackend`` subclass — bypassing the entire ``OpenAiPassthroughBackend`` +
openai-SDK + httpx code path — these tests exercise that path end-to-end:

    inbound HTTP request
       → Switchyard chain (real RequestProcessors / ResponseProcessors)
       → OpenAiPassthroughBackend (Rust HTTP call)
       → local OpenAI-compatible upstream stub
       → response back through the chain
       → outbound HTTP response

A regression anywhere in that pipeline (renamed module, missing kwarg, wrong
base_url forwarding, stale processor wiring, broken stats accumulator)
fails one of these tests instead of slipping to top-of-tree.

All tests run offline: ASGITransport intercepts inbound HTTP, the upstream
is a local loopback HTTP server, and no external network is touched.
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from switchyard import PassthroughProfileConfig, ProfileSwitchyard
from switchyard.server.switchyard_app import build_switchyard_app

# ---------------------------------------------------------------------------
# Upstream payloads
# ---------------------------------------------------------------------------


def _completion_payload(*, content: str = "hello back") -> dict[str, object]:
    """An OpenAI Chat Completion JSON body (non-streaming)."""
    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _completion_chunk(*, content: str = "", finish: str | None = None) -> dict[str, object]:
    delta: dict[str, object] = {}
    if content:
        delta["content"] = content
    return {
        "id": "chatcmpl-upstream-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "upstream-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse_stream_body(chunks: list[dict[str, object]]) -> bytes:
    """Encode chunks as the upstream SSE wire format the OpenAI SDK reads."""
    out = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


class _OpenAICompatStub:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []
        self._responses: list[tuple[int, bytes, str]] = []

    def __enter__(self) -> _OpenAICompatStub:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                with owner._lock:
                    owner._requests.append({
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "body": body,
                    })
                    if owner._responses:
                        status, content, content_type = owner._responses.pop(0)
                    else:
                        status = 500
                        content = b'{"error":{"message":"no stub response queued"}}'
                        content_type = "application/json"

                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(content)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, _format: str, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    def respond_json(self, status: int, body: dict[str, object]) -> None:
        content = json.dumps(body).encode("utf-8")
        with self._lock:
            self._responses.append((status, content, "application/json"))

    def respond_sse(self, body: bytes) -> None:
        with self._lock:
            self._responses.append((200, body, "text/event-stream"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def passthrough_upstream() -> Iterator[_OpenAICompatStub]:
    with _OpenAICompatStub() as upstream:
        yield upstream


@pytest.fixture
async def passthrough_client(
    passthrough_upstream: _OpenAICompatStub,
) -> AsyncIterator[httpx.AsyncClient]:
    """A FastAPI client wired through the real passthrough profile.

    The profile builds a real ``OpenAiPassthroughBackend`` pointed at
    a local OpenAI-compatible stub to control the upstream response.
    """
    switchyard = ProfileSwitchyard(
        PassthroughProfileConfig(
            api_key="test-key-not-used",
            base_url=passthrough_upstream.base_url,
        )
        .build()
        .with_runtime_components(enable_stats=False)
    )
    app = build_switchyard_app(switchyard)
    # ``raise_app_exceptions=False`` mirrors what uvicorn does in
    # production: an unhandled exception inside a route is mapped to a
    # 500 response, not propagated up the call stack.  Without this the
    # error-path tests would see the openai SDK exception escape the
    # transport entirely.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Passthrough profile — OpenAI inbound, OpenAI upstream
# ---------------------------------------------------------------------------


class TestPassthroughProfileOpenAI:
    """Inbound OpenAI Chat Completions through the real passthrough profile."""

    async def test_non_streaming_round_trip(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
    ) -> None:
        passthrough_upstream.respond_json(200, _completion_payload(content="pong"))

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "pong"
        assert passthrough_upstream.requests, "passthrough never invoked the upstream"

    async def test_streaming_round_trip(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
    ) -> None:
        chunks = [
            _completion_chunk(content="foo"),
            _completion_chunk(content="bar"),
            _completion_chunk(finish="stop"),
        ]
        passthrough_upstream.respond_sse(_sse_stream_body(chunks))

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200, resp.text
        assert "text/event-stream" in resp.headers["content-type"]
        assert passthrough_upstream.requests

        data_lines = [
            line[6:]
            for line in resp.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert data_lines, "expected at least one outbound data frame"
        decoded = [json.loads(line) for line in data_lines]
        joined = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in decoded
            if c.get("choices")
        )
        assert joined == "foobar"


# ---------------------------------------------------------------------------
# Passthrough profile — Anthropic inbound, OpenAI upstream
# ---------------------------------------------------------------------------


class TestPassthroughProfileAnthropic:
    """Inbound Anthropic Messages, translated to OpenAI for the upstream call."""

    async def test_non_streaming_translates_both_directions(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
    ) -> None:
        passthrough_upstream.respond_json(200, _completion_payload(content="hi"))

        resp = await passthrough_client.post(
            "/v1/messages",
            json={
                "model": "any-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"] == "hi"
        assert body["stop_reason"] == "end_turn"
        assert passthrough_upstream.requests, "Anthropic inbound did not reach the OpenAI upstream"

        # Verify the request the upstream SAW was already in OpenAI Chat
        # Completions shape — i.e. the request translator ran.
        upstream_body = passthrough_upstream.requests[-1]["body"]
        assert "messages" in upstream_body
        assert upstream_body["messages"][-1]["content"] == "ping"


# ---------------------------------------------------------------------------
# Passthrough profile — Responses API inbound, OpenAI Chat upstream
# ---------------------------------------------------------------------------


class TestPassthroughProfileResponsesApi:
    """Inbound OpenAI Responses API, translated to Chat Completions upstream."""

    async def test_non_streaming_translates_both_directions(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
    ) -> None:
        passthrough_upstream.respond_json(200, _completion_payload(content="resp-ok"))

        resp = await passthrough_client.post(
            "/v1/responses",
            json={"model": "any-model", "input": "ping"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        msg = body["output"][0]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "resp-ok"
        assert passthrough_upstream.requests


# ---------------------------------------------------------------------------
# Backend errors propagate as HTTP errors, not 200 with garbage
# ---------------------------------------------------------------------------


class TestPassthroughProfileBackendErrors:
    """Upstream HTTP errors must surface as HTTP errors to the client.

    A regression where the chain swallows an upstream 4xx/5xx and still
    returns 200 with malformed JSON is exactly the kind of silent failure
    we want CI to catch.
    """

    async def test_upstream_500_does_not_return_200(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
    ) -> None:
        passthrough_upstream.respond_json(500, {"error": {"message": "boom"}})

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        # Whatever the exact mapping is, returning 200 on an upstream 500 is wrong.
        assert resp.status_code != 200, (
            f"Upstream 500 leaked through as a successful response: body={resp.text!r}"
        )
        assert resp.status_code >= 400

    async def test_upstream_401_does_not_return_200(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
    ) -> None:
        passthrough_upstream.respond_json(
            401,
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
        )

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        assert resp.status_code != 200
        assert resp.status_code >= 400


# ---------------------------------------------------------------------------
# Upstream-attempt outcome counters are wired for the Rust passthrough backend
# ---------------------------------------------------------------------------


class TestPassthroughProfileUpstreamAttemptCounters:
    """`switchyard.upstream_attempts` must populate for non-latency chains.

    The endpoint-layer fallback records one upstream attempt per request for
    backends (here the Rust ``OpenAiPassthroughBackend``) that issue exactly
    one upstream call and have no Python retry loop. The counters are emitted
    via OpenTelemetry; the fixture binds an in-memory meter to read them.
    """

    @pytest.fixture(autouse=True)
    def _meter(self) -> Iterator[Any]:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        from switchyard.lib import metrics

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        metrics.reset_for_test(provider.get_meter("switchyard"))
        try:
            yield reader
        finally:
            metrics.reset_for_test(None)

    @staticmethod
    def _attempt_count(reader: Any, *, outcome: str, code: str) -> int:
        """Sum of ``switchyard.upstream_attempts`` points matching outcome/code."""
        total = 0
        data = reader.get_metrics_data()
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name != "switchyard.upstream_attempts":
                        continue
                    for point in metric.data.data_points:
                        attrs = point.attributes
                        if attrs.get("outcome") == outcome and attrs.get("code") == code:
                            total += point.value
        return total

    @staticmethod
    def _retry_recovered(reader: Any) -> int:
        data = reader.get_metrics_data()
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "switchyard.retry_recovered":
                        return sum(p.value for p in metric.data.data_points)
        return 0

    async def test_success_records_one_200_attempt(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
        _meter: Any,
    ) -> None:
        passthrough_upstream.respond_json(200, _completion_payload(content="ok"))

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert resp.status_code == 200, resp.text

        assert self._attempt_count(_meter, outcome="success", code="200") == 1
        assert self._retry_recovered(_meter) == 0

    async def test_upstream_500_records_one_retryable_attempt(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
        _meter: Any,
    ) -> None:
        passthrough_upstream.respond_json(500, {"error": {"message": "boom"}})

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert resp.status_code >= 400

        assert self._attempt_count(_meter, outcome="retryable_error", code="500") == 1
        assert self._attempt_count(_meter, outcome="success", code="200") == 0

    async def test_upstream_401_records_one_other_error_attempt(
        self,
        passthrough_client: httpx.AsyncClient,
        passthrough_upstream: _OpenAICompatStub,
        _meter: Any,
    ) -> None:
        passthrough_upstream.respond_json(
            401,
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
        )

        resp = await passthrough_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert resp.status_code >= 400

        assert self._attempt_count(_meter, outcome="other_error", code="401") == 1
        # A 4xx client error is not retryable and never recovers.
        assert self._retry_recovered(_meter) == 0
