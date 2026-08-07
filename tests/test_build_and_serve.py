# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``switchyard.server.server_util.build_and_serve``.

``build_and_serve`` is the second app-construction code path (the first is
``build_switchyard_app`` directly). PR #28 fixed a regression where this
function lazy-imported three modules that no longer exist after the
open-source cleanup, raising ``ModuleNotFoundError`` for every CLI subcommand
that called it (e.g., ``random-routing``).

These tests exercise the function offline by patching out ``uvicorn.run``,
capturing the FastAPI app it would have served, and driving real HTTP
requests through that app via ``httpx.ASGITransport``. A regression in any
of build_and_serve's imports, app construction, or extra-endpoint wiring
fails one of these tests.
"""

import argparse
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchyard.lib.endpoints.base import Endpoint
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard.server import server_util
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse
from switchyard_rust.translation import TranslationEngine

_ALL_REQUEST_TYPES = [
    ChatRequestType.OPENAI_CHAT,
    ChatRequestType.OPENAI_RESPONSES,
    ChatRequestType.ANTHROPIC,
]

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubBackend(LLMBackend):
    def supported_request_types(self) -> list[ChatRequestType]:
        return list(_ALL_REQUEST_TYPES)

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        completion = ChatCompletion(
            id="chatcmpl-stub",
            object="chat.completion",
            created=1700000000,
            model="stub",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        return ChatResponse.openai_completion(completion)


class _CountingStubBackend(_StubBackend):
    """Variant of _StubBackend that records each call so tests can assert short-circuit."""

    def __init__(self) -> None:
        self.call_count = 0

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        return await super().call(ctx, request)


class _SentinelEndpoint(Endpoint):
    """Adds ``GET /sentinel`` so tests can verify ``extra_endpoints`` are wired."""

    def register(self, app: FastAPI) -> None:
        router = APIRouter()

        @router.get("/sentinel")
        async def _sentinel() -> dict[str, str]:
            return {"sentinel": "ok"}

        app.include_router(router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(
    host: str = "127.0.0.1",
    port: int | None = 4000,
    reload: bool = False,
    workers: int = 1,
) -> argparse.Namespace:
    """Build the argparse namespace ``build_and_serve`` expects."""
    return argparse.Namespace(host=host, port=port, reload=reload, workers=workers)


def _capture_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch ``uvicorn.run`` to capture its kwargs without starting a server."""
    captured: dict[str, Any] = {}

    def _fake_run(
        app: FastAPI,
        host: str,
        port: int,
        reload: bool,
        workers: int,
    ) -> None:
        captured["app"] = app
        captured["kwargs"] = {
            "host": host,
            "port": port,
            "reload": reload,
            "workers": workers,
        }

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    return captured


def _assert_invalid_request_response(
    response: httpx.Response,
    path: str,
    code: str,
) -> None:
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    if path == "/v1/messages":
        assert body["type"] == "error"
        assert set(body["error"]) == {"type", "message"}
    else:
        assert body["error"]["code"] == code


def _switchyard() -> Switchyard:
    return Switchyard(backend=_StubBackend(), translator=TranslationEngine())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_and_serve_imports_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct guard for PR #28: ``build_and_serve`` must not raise on import.

    The previous regression was a lazy import of deleted modules
    (``endpoint_sets``, ``server``, ``server_config``); the function ran
    cleanly until called, then crashed at the lazy-import line. Calling
    it with a stub uvicorn proves all of its imports resolve.
    """
    captured = _capture_uvicorn(monkeypatch)
    server_util.build_and_serve(_ns(), _switchyard())
    assert "app" in captured, "build_and_serve never reached uvicorn.run"


def test_build_and_serve_passes_host_and_port_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_uvicorn(monkeypatch)
    server_util.build_and_serve(_ns(host="0.0.0.0", port=5555), _switchyard())
    assert captured["kwargs"]["host"] == "0.0.0.0"
    assert captured["kwargs"]["port"] == 5555


def test_build_and_serve_defaults_port_to_4000_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented default in --port help; regression-guard the fallback path."""
    captured = _capture_uvicorn(monkeypatch)
    server_util.build_and_serve(_ns(port=None), _switchyard())
    assert captured["kwargs"]["port"] == 4000


def test_build_and_serve_workers_default_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Namespace without ``workers`` must not raise — ``getattr`` fallback applies."""
    captured = _capture_uvicorn(monkeypatch)
    args = argparse.Namespace(host="127.0.0.1", port=4000, reload=False)
    server_util.build_and_serve(args, _switchyard())
    assert captured["kwargs"]["workers"] == 1


def test_build_and_serve_registers_extra_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extra_endpoints`` must be registered onto the app before serving."""
    captured = _capture_uvicorn(monkeypatch)
    server_util.build_and_serve(
        _ns(),
        _switchyard(),
        extra_endpoints=[_SentinelEndpoint()],
    )
    routes = {getattr(r, "path", None) for r in captured["app"].routes}
    assert "/sentinel" in routes


@pytest.fixture
async def served_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    """Drive the app ``build_and_serve`` would have served through ASGI.

    The full app boot path runs (build_switchyard_app → endpoint
    registration → app.state wiring) but uvicorn never starts.
    """
    captured = _capture_uvicorn(monkeypatch)
    server_util.build_and_serve(_ns(), _switchyard(), extra_endpoints=[_SentinelEndpoint()])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=captured["app"]),
        base_url="http://test",
    ) as client:
        yield client


class TestBuiltAppRoundTrips:
    """Round-trip through the app build_and_serve assembled.

    Catches the kind of failure PR #28 fixed: server starts cleanly, but
    every request hits ``AttributeError`` because of an internal wiring
    mismatch. A unit test on ``build_and_serve`` alone wouldn't have
    caught the ``app.state.switchyard`` vs ``app.state.switchyard``
    bug — only a real request through the assembled app does.
    """

    async def test_health(self, served_client: httpx.AsyncClient) -> None:
        resp = await served_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_openai_chat_completions(self, served_client: httpx.AsyncClient) -> None:
        resp = await served_client.post(
            "/v1/chat/completions",
            json={
                "model": "any",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "ok"

    async def test_anthropic_messages(self, served_client: httpx.AsyncClient) -> None:
        resp = await served_client.post(
            "/v1/messages",
            json={
                "model": "any",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["content"][0]["text"] == "ok"

    async def test_extra_endpoint_reachable(self, served_client: httpx.AsyncClient) -> None:
        resp = await served_client.get("/sentinel")
        assert resp.status_code == 200
        assert resp.json() == {"sentinel": "ok"}


class TestInvalidRequestBody:
    """Malformed or non-object JSON bodies must return 400 with a structured error envelope.

    REQ-AG3: agents must receive structured errors with retry semantics so they can
    distinguish client-side parse bugs (no retry) from transient server failures (retry).
    These tests confirm the fix for the bare-500 regression reported in bug 6267258.
    """

    _ENDPOINTS = [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
    ]

    @pytest.mark.parametrize("path", _ENDPOINTS)
    async def test_malformed_json_returns_400(
        self, served_client: httpx.AsyncClient, path: str
    ) -> None:
        resp = await served_client.post(
            path,
            content="{invalid json,,",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        _assert_invalid_request_response(resp, path, "invalid_body")

    @pytest.mark.parametrize("path", _ENDPOINTS)
    async def test_json_array_body_returns_400(
        self, served_client: httpx.AsyncClient, path: str
    ) -> None:
        resp = await served_client.post(
            path,
            content='["not", "an", "object"]',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        _assert_invalid_request_response(resp, path, "invalid_body")

    async def test_server_stays_healthy_after_bad_request(
        self, served_client: httpx.AsyncClient
    ) -> None:
        await served_client.post(
            "/v1/chat/completions",
            content="{bad json",
            headers={"Content-Type": "application/json"},
        )
        resp = await served_client.get("/health")
        assert resp.status_code == 200


@pytest.fixture
async def counting_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, _CountingStubBackend]]:
    """Like served_client but exposes a call-counting backend for short-circuit checks."""
    captured: dict[str, Any] = {}

    def _fake_run(
        app: FastAPI,
        host: str,
        port: int,
        reload: bool,
        workers: int,
    ) -> None:
        captured["app"] = app
        captured["kwargs"] = {
            "host": host,
            "port": port,
            "reload": reload,
            "workers": workers,
        }

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    backend = _CountingStubBackend()
    sw = Switchyard(backend=backend, translator=TranslationEngine())
    server_util.build_and_serve(_ns(), sw)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=captured["app"]),
        base_url="http://test",
    ) as client:
        yield client, backend


class TestEmptyMessages:
    """Empty messages array must short-circuit with a structured 400 before reaching the backend.

    REQ-AG3: agents must distinguish client-side errors (no retry) from server failures
    (retry). An empty messages array is a client bug — reporting it as 500 breaks
    agent retry logic. This class pins the fix and confirms dispatch is skipped.
    """

    async def test_openai_chat_empty_messages_returns_400(
        self,
        counting_client: tuple[httpx.AsyncClient, _CountingStubBackend],
    ) -> None:
        client, backend = counting_client
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "any", "messages": []},
        )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "empty_messages"
        assert "messages" in body["error"]["message"]
        assert backend.call_count == 0, "backend must not be invoked for empty messages"

    async def test_anthropic_messages_empty_messages_returns_400(
        self,
        counting_client: tuple[httpx.AsyncClient, _CountingStubBackend],
    ) -> None:
        client, backend = counting_client
        resp = await client.post(
            "/v1/messages",
            json={"model": "any", "max_tokens": 16, "messages": []},
        )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        _assert_invalid_request_response(resp, "/v1/messages", "empty_messages")
        assert backend.call_count == 0, "backend must not be invoked for empty messages"

    async def test_non_empty_messages_still_succeed(
        self,
        counting_client: tuple[httpx.AsyncClient, _CountingStubBackend],
    ) -> None:
        client, backend = counting_client
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "any", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200, resp.text
        assert backend.call_count == 1

    async def test_server_stays_healthy_after_empty_messages(
        self,
        counting_client: tuple[httpx.AsyncClient, _CountingStubBackend],
    ) -> None:
        client, _ = counting_client
        await client.post("/v1/chat/completions", json={"model": "any", "messages": []})
        resp = await client.get("/health")
        assert resp.status_code == 200
