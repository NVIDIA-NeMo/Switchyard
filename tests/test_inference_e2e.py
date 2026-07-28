# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end inference tests using a mock LLM backend.

Exercises the full HTTP stack — FastAPI endpoints → Switchyard chain →
mock backend → response translation — without touching any live LLM provider.

All tests run offline: no API keys, no network access, no running model.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse
from switchyard_rust.translation import TranslationEngine

# ---------------------------------------------------------------------------
# Mock LLM backends
# ---------------------------------------------------------------------------

_REPLY = "hello back"
_ALL_REQUEST_TYPES = [
    ChatRequestType.OPENAI_CHAT,
    ChatRequestType.OPENAI_RESPONSES,
    ChatRequestType.ANTHROPIC,
]


class _MockLLMBackend(LLMBackend):
    """Returns a canned OpenAI completion response for every call."""

    def __init__(self, completion: ChatCompletion) -> None:
        self._completion = completion

    def supported_request_types(self) -> list[ChatRequestType]:
        return list(_ALL_REQUEST_TYPES)

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        return ChatResponse.openai_completion(self._completion)


class _StreamingMockLLMBackend(LLMBackend):
    """Returns a fixed OpenAI stream response for every call."""

    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = list(chunks)

    def supported_request_types(self) -> list[ChatRequestType]:
        return list(_ALL_REQUEST_TYPES)

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        chunks = self._chunks

        async def _iter() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in chunks:
                yield chunk

        return ChatResponse.openai_stream(ResponseStream(_iter()))


class _RaisingMockLLMBackend(LLMBackend):
    """Raises ``exc`` for every call — exercises the error path through the chain."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def supported_request_types(self) -> list[ChatRequestType]:
        return list(_ALL_REQUEST_TYPES)

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        raise self._exc


class _TranslatingMockLLMBackend(LLMBackend):
    """Chat-only backend that translates the inbound request to Chat at the top of
    ``call`` — mirroring any real chat-completions backend (native / passthrough).

    An unsupported inbound field (e.g. a bad message role) is rejected by the
    translation engine here, before any upstream call, exactly as it would be for
    a production chat backend.
    """

    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        request = TranslationEngine().request_to_any_of(request, [ChatRequestType.OPENAI_CHAT])
        return ChatResponse.openai_completion(_make_completion())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(*, model: str = "mock-model", content: str = _REPLY) -> ChatCompletion:
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
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _make_tool_call_completion(
    *,
    model: str = "mock-model",
    tool_name: str = "get_weather",
    tool_args: str = '{"city": "Paris"}',
    tool_call_id: str = "call_test_123",
) -> ChatCompletion:
    """Completion whose assistant message contains a single tool call."""
    return ChatCompletion(
        id="chatcmpl-test-tool",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id=tool_call_id,
                            type="function",
                            function=Function(name=tool_name, arguments=tool_args),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=CompletionUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


def _make_chunks(*, model: str = "mock-model", content: str = _REPLY) -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-stream",
            object="chat.completion.chunk",
            created=1700000000,
            model=model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=content),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-stream",
            object="chat.completion.chunk",
            created=1700000000,
            model=model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    switchyard = Switchyard(
        backend=_MockLLMBackend(_make_completion()),
        translator=TranslationEngine(),
    )
    return build_switchyard_app(switchyard)


@pytest.fixture
def streaming_app() -> FastAPI:
    switchyard = Switchyard(
        backend=_StreamingMockLLMBackend(_make_chunks()),
        translator=TranslationEngine(),
    )
    return build_switchyard_app(switchyard)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
async def streaming_client(streaming_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=streaming_app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
def tool_call_app() -> FastAPI:
    """App backed by a mock that always returns a tool-calling completion."""
    switchyard = Switchyard(
        backend=_MockLLMBackend(_make_tool_call_completion()),
        translator=TranslationEngine(),
    )
    return build_switchyard_app(switchyard)


@pytest.fixture
async def tool_call_client(tool_call_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=tool_call_app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
def raising_app() -> FastAPI:
    """App backed by a mock that always raises a generic backend error."""
    switchyard = Switchyard(
        backend=_RaisingMockLLMBackend(RuntimeError("backend boom")),
        translator=TranslationEngine(),
    )
    return build_switchyard_app(switchyard)


@pytest.fixture
async def raising_client(raising_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # ``raise_app_exceptions=False`` mirrors uvicorn's behavior in production:
    # an unhandled exception inside a route is mapped to a 500 response, not
    # propagated up the call stack. Required for the error-path tests below.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=raising_app, raise_app_exceptions=False),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
def translating_app() -> FastAPI:
    """App whose chat-only backend translates the inbound request to Chat, so an
    invalid inbound role is rejected during translation before any upstream call."""
    switchyard = Switchyard(
        backend=_TranslatingMockLLMBackend(),
        translator=TranslationEngine(),
    )
    return build_switchyard_app(switchyard)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferenceE2E:

    async def test_health_liveness(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_openai_chat_completions(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == _REPLY
        assert choice["finish_reason"] == "stop"

    async def test_openai_chat_completions_streaming(
        self, streaming_client: httpx.AsyncClient
    ) -> None:
        resp = await streaming_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        data_lines = [
            line[6:]  # strip leading "data: "
            for line in resp.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert data_lines, "expected at least one data frame before [DONE]"

        chunks = [json.loads(line) for line in data_lines]
        content = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c.get("choices")
        )
        assert content == _REPLY

    async def test_anthropic_messages(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "any-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == _REPLY
        assert data["stop_reason"] == "end_turn"

    async def test_openai_responses_api(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "any-model",
                "input": "hello",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        msg = data["output"][0]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == _REPLY

    async def test_anthropic_messages_streaming(
        self, streaming_client: httpx.AsyncClient
    ) -> None:
        """Streaming through the Anthropic inbound — chunks must be Anthropic SSE shape."""
        resp = await streaming_client.post(
            "/v1/messages",
            json={
                "model": "any-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200, resp.text
        assert "text/event-stream" in resp.headers["content-type"]

        events = [
            json.loads(line[6:])
            for line in resp.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert events, "expected at least one Anthropic SSE event"

        # Anthropic streaming uses an event-typed envelope (message_start /
        # content_block_delta / message_stop). Pin the contract: at least
        # one event carries the canonical Anthropic event 'type' field.
        anthropic_event_types = {"message_start", "content_block_delta", "message_stop"}
        seen_types = {e.get("type") for e in events}
        assert seen_types & anthropic_event_types, (
            f"no Anthropic-shaped event types in stream: {seen_types}"
        )

        # The reply text should be reconstructible from text deltas.
        text_deltas = [
            e["delta"]["text"]
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        if text_deltas:
            assert "".join(text_deltas) == _REPLY

    async def test_openai_responses_streaming(
        self, streaming_client: httpx.AsyncClient
    ) -> None:
        """Streaming through the Responses API inbound — chunks must be Responses SSE shape."""
        resp = await streaming_client.post(
            "/v1/responses",
            json={
                "model": "any-model",
                "input": "hello",
                "stream": True,
            },
        )
        assert resp.status_code == 200, resp.text
        assert "text/event-stream" in resp.headers["content-type"]

        events = [
            json.loads(line[6:])
            for line in resp.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert events, "expected at least one Responses SSE event"

        # Responses streaming events all carry a 'type' starting with
        # 'response.' — a regression that emits Chat Completions chunks
        # on this endpoint would fail this assertion.
        types = {e.get("type", "") for e in events}
        assert any(t.startswith("response.") for t in types), (
            f"no Responses-shaped event types in stream: {types}"
        )

    async def test_backend_exception_returns_500(
        self, raising_client: httpx.AsyncClient
    ) -> None:
        """A backend that raises must surface as an HTTP error, not 200 with garbage."""
        resp = await raising_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["code"] == "internal_chain_error"
        assert "backend boom" in body["error"]["message"]


class TestToolCallRoundTrip:
    """Backend returns OpenAI tool_calls; verify each inbound format renders correctly.

    Tool-call translation is a frequent break point in refactors because
    the format-specific shapes diverge sharply (OpenAI: ``tool_calls`` on
    the assistant message; Anthropic: a ``tool_use`` block in ``content``;
    Responses: a top-level ``function_call`` output item).
    """

    async def test_openai_tool_calls_passthrough(
        self, tool_call_client: httpx.AsyncClient
    ) -> None:
        resp = await tool_call_client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "what's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tool_calls = choice["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"city": "Paris"}'

    async def test_anthropic_tool_use_translation(
        self, tool_call_client: httpx.AsyncClient
    ) -> None:
        """OpenAI ``tool_calls`` from the backend must be translated to Anthropic ``tool_use``."""
        resp = await tool_call_client.post(
            "/v1/messages",
            json={
                "model": "any-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "what's the weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "input_schema": {"type": "object"},
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == "message"
        assert body["stop_reason"] == "tool_use"

        tool_use_blocks = [b for b in body["content"] if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1, f"expected one tool_use block, got: {body['content']}"
        block = tool_use_blocks[0]
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "Paris"}


class TestEndpointErrorContract:
    """Endpoint error-mapping contracts previously covered only through the
    removed latency-service backend, re-vehicled onto in-process backends."""

    def test_post_dispatch_exception_returns_json_500(self, app: FastAPI) -> None:
        """An exception raised AFTER dispatch (e.g. during result serialization)
        must surface as a JSON 500 envelope, not FastAPI's plain-text 500."""

        class _Unserializable:
            def model_dump(self) -> None:
                raise RuntimeError("serialization exploded")

        with patch(
            "switchyard.lib.endpoints.openai_chat_endpoint.dispatch_chat_request",
            new_callable=AsyncMock,
            return_value=_Unserializable(),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "any-model", "messages": [{"role": "user", "content": "hi"}]},
                )

        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["code"] == "internal_chain_error"
        assert "serialization exploded" in body["error"]["message"]

    @pytest.mark.parametrize(
        ("path", "payload"),
        [
            (
                "/v1/responses",
                {"model": "m", "input": [{"type": "message", "role": "api", "content": "ping"}]},
            ),
            (
                "/v1/messages",
                {"model": "m", "max_tokens": 16, "messages": [{"role": "api", "content": "ping"}]},
            ),
        ],
    )
    def test_invalid_inbound_role_surfaces_as_internal_error(
        self, translating_app: FastAPI, path: str, payload: dict[str, object]
    ) -> None:
        """An unsupported inbound message role is rejected by the translation engine
        when a chat-only backend translates the request, before any upstream call.

        NOTE: this currently surfaces as a 500 ``internal_error``. The removed
        latency-service backend special-cased translation ``invalid_value`` errors
        into a provider-compatible 400; no surviving backend reproduces that, so
        the standalone-server contract is now a 500. Update this test (and the
        endpoint/translation layer) if a 400 is restored.
        """
        with TestClient(translating_app, raise_server_exceptions=False) as client:
            response = client.post(path, json=payload)

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["type"] == "internal_error"
        assert '"api"' in body["error"]["message"]
        assert "role" in body["error"]["message"]
