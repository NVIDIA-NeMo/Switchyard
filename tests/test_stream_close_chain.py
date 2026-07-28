# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Close ownership survives the Python -> Rust-core -> Python round trip.

The production path is: a Python backend returns
``ChatResponse.openai_stream(ResponseStream(sdk_stream))``; the backend adapter
converts it to a Rust-core stream via ``take_core``; response processors run in
core; the terminal translator rebuilds a Python ``ChatResponseStream`` via
``from_core``; the endpoint feeds that into an SSE helper. The rebuilt stream no
longer references the original SDK ``AsyncStream``, so unless close ownership is
preserved across the conversion, closing it on client disconnect never releases
the upstream connection — the OOM connection-pool leak.

These tests exercise the *real* chain (``Switchyard`` + ``TranslationEngine``),
not just the SSE helper in isolation, and assert the fake SDK stream's close
hook runs after the response stream is closed mid-flight — for both same-format
(OpenAI in / OpenAI backend) and translated (Anthropic in / OpenAI backend)
streaming.
"""

from __future__ import annotations

import asyncio
from typing import Any

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.endpoints.sse_helpers import (
    iter_anthropic_sse,
    iter_chat_completion_sse,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse
from switchyard_rust.translation import TranslationEngine


class _FakeSdkStream:
    """Mimics the OpenAI SDK ``AsyncStream``: async-iterable with async ``close``.

    Records whether ``close()`` ran so a test can assert the upstream response
    (and its pooled connection) would have been released.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self.closed = False

    def __aiter__(self) -> _FakeSdkStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def close(self) -> None:
        self.closed = True


def _chunk(content: str | None, *, role: str | None = None, finish: str | None = None) -> dict:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": "chatcmpl-close",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "close-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _chunks() -> list[dict]:
    return [
        _chunk("", role="assistant"),
        _chunk("hello"),
        _chunk(" world"),
        _chunk(None, finish="stop"),
    ]


class _StreamingSdkBackend(LLMBackend):
    """Backend returning a streaming response wrapping a fake SDK stream.

    Holds the fake so a test can assert it was closed. Supports both OpenAI Chat
    and Anthropic inbound so the same backend serves the same-format and the
    translated paths (the response is always an OpenAI chat stream).
    """

    def __init__(self, fake: _FakeSdkStream) -> None:
        self._fake = fake

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT, ChatRequestType.ANTHROPIC]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        _ = ctx, request
        return ChatResponse.openai_stream(ResponseStream(self._fake))


async def _wait_closed(fake: _FakeSdkStream, timeout: float = 2.0) -> bool:
    """Wait for the best-effort, runtime-scheduled source close to land.

    ``ChatResponseStream``'s drop-time close runs as a fire-and-forget task on
    the Rust tokio runtime (a separate thread), so it may not have completed the
    instant ``aclose`` returns to the asyncio loop. Poll briefly for it.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not fake.closed and loop.time() < deadline:
        await asyncio.sleep(0.005)
    return fake.closed


def _openai_request() -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": "close-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })


def _anthropic_request() -> ChatRequest:
    return ChatRequest.anthropic({
        "model": "close-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })


async def test_same_format_stream_close_releases_sdk_source_through_chain() -> None:
    """OpenAI in / OpenAI backend: closing the SSE stream after one frame closes
    the SDK stream the backend wrapped — across take_core/from_core."""
    fake = _FakeSdkStream(_chunks())
    chain = Switchyard(backend=_StreamingSdkBackend(fake), translator=TranslationEngine())

    result = await chain.call(_openai_request())
    sse = iter_chat_completion_sse(result)
    first = await sse.__anext__()
    assert first.startswith("data:")

    await sse.aclose()  # client disconnect mid-stream

    assert await _wait_closed(fake), "backend SDK stream was not closed after teardown"


async def test_translated_stream_close_releases_sdk_source_through_chain() -> None:
    """Anthropic in / OpenAI backend: the response is translated through
    ``translate_stream``; closing the SSE stream still closes the SDK source."""
    fake = _FakeSdkStream(_chunks())
    chain = Switchyard(backend=_StreamingSdkBackend(fake), translator=TranslationEngine())

    result = await chain.call(_anthropic_request())
    sse = iter_anthropic_sse(result)
    first = await sse.__anext__()
    assert first.startswith("event:")

    await sse.aclose()  # client disconnect mid-stream

    assert await _wait_closed(fake), "backend SDK stream was not closed after translated teardown"


class _FakeTranslatableInput:
    """Async-iterable openai chunk source with an async ``aclose`` it records."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self.aclosed = False

    def __aiter__(self) -> _FakeTranslatableInput:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self) -> None:
        self.aclosed = True


async def test_translate_stream_closes_input_on_disconnect() -> None:
    """``translate_stream``'s ``finally`` closes its input when the consuming
    SSE generator is ``aclose``-ed mid-stream (the client-disconnect path)."""
    fake = _FakeTranslatableInput(_chunks())
    gen = TranslationEngine().translate_stream("openai_chat", "anthropic_messages", fake)

    await gen.__anext__()
    await gen.aclose()

    assert fake.aclosed, "translate_stream did not close its input stream"


async def test_translate_stream_closes_input_on_normal_completion() -> None:
    fake = _FakeTranslatableInput(_chunks())
    gen = TranslationEngine().translate_stream("openai_chat", "anthropic_messages", fake)

    _ = [frame async for frame in gen]

    assert fake.aclosed, "translate_stream did not close its input on completion"
