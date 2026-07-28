# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empirical repro: unclosed streaming responses exhaust the httpx pool, and our
SSE-helper close path releases the connection.

Uses the REAL OpenAI ``AsyncStream`` against a loopback SSE server, with the
httpx pool capped at a single connection so exhaustion is deterministic. The
stream is driven through the real ``iter_chat_completion_sse`` helper, so the
``finally: await _aclose_stream(...)`` we added is the thing under test.

Marked ``integration`` (real sockets + a pool timeout) so the default unit gate
can deselect it with ``-m "not integration"``; it is the canonical regression
for the OOM connection-pool leak.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest
from openai import AsyncOpenAI

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.endpoints.sse_helpers import iter_chat_completion_sse
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse
from switchyard_rust.translation import TranslationEngine

pytestmark = pytest.mark.integration


def _chunk(i: int) -> dict:
    return {
        "id": "chatcmpl-leak",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "leak-model",
        "choices": [{"index": 0, "delta": {"content": str(i)}, "finish_reason": None}],
    }


class _SlowSSEStub:
    """Loopback OpenAI-compatible server that streams SSE chunks slowly.

    Streams many chunks with a delay so a client that reads one frame and
    abandons the response leaves the connection checked out — the production
    client-disconnect scenario.
    """

    def __init__(self, n_chunks: int = 500, delay_s: float = 0.02) -> None:
        self._n = n_chunks
        self._delay = delay_s
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _SlowSSEStub:
        n, delay = self._n, self._delay

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                if length:
                    self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(data: bytes) -> None:
                    self.wfile.write(f"{len(data):X}\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

                try:
                    for i in range(n):
                        send(f"data: {json.dumps(_chunk(i))}\n\n".encode())
                        time.sleep(delay)
                    send(b"data: [DONE]\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return  # client disconnected — expected in these tests

            def log_message(self, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"


def _client(base_url: str) -> AsyncOpenAI:
    # max_connections=1 makes exhaustion deterministic: one leaked stream takes
    # the only slot. pool=1.5s fails the next acquire fast; max_retries=0 keeps
    # the SDK from retrying the pool timeout (which would only slow the test).
    return AsyncOpenAI(
        base_url=base_url,
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            timeout=httpx.Timeout(5.0, pool=1.5),
        ),
    )


def _pool_conns(client: AsyncOpenAI) -> int:
    """Best-effort live connection count for evidence logging (version-dependent)."""
    try:
        return len(client._client._transport._pool.connections)  # type: ignore[attr-defined]
    except Exception:
        return -1


async def _open_and_read_one(client: AsyncOpenAI, *, close: bool):
    """Open a streaming completion, drive it through the real SSE helper for one
    frame, then either close it (our fix) or abandon it (the leak)."""
    stream = await client.chat.completions.create(
        model="leak-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    sse = iter_chat_completion_sse(stream)
    first = await sse.__anext__()
    assert first.startswith("data:"), first
    assert "error" not in first, f"stream errored on first frame: {first}"
    if close:
        await sse.aclose()  # our fix: finally -> _aclose_stream -> AsyncStream.close()
        return None
    return sse, stream  # hold refs so the connection stays checked out


async def test_unclosed_stream_exhausts_pool() -> None:
    """An abandoned stream pins its pooled connection; the next request starves."""
    with _SlowSSEStub() as stub:
        client = _client(stub.base_url)
        try:
            held = await _open_and_read_one(client, close=False)  # leak: never closed
            assert held is not None
            assert _pool_conns(client) == 1
            # The only pool slot is occupied by the abandoned stream; a new
            # streaming request cannot acquire a connection and must fail.
            with pytest.raises(
                (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    httpx.PoolTimeout,
                    httpx.TimeoutException,
                )
            ):
                await _open_and_read_one(client, close=False)
        finally:
            await client.close()


async def test_our_close_releases_pool_connection() -> None:
    """Closing via the SSE helper frees the slot, so a sequence of streams runs."""
    with _SlowSSEStub() as stub:
        client = _client(stub.base_url)
        try:
            # Our fix closes each stream, freeing the single pool slot, so a
            # whole sequence of streaming requests succeeds — no exhaustion.
            for _ in range(5):
                await _open_and_read_one(client, close=True)
            assert _pool_conns(client) <= 1
        finally:
            await client.close()


class _RealSdkStreamingBackend(LLMBackend):
    """Backend that issues a real streaming SDK call and wraps the AsyncStream.

    Mirrors the production shape: ``call`` returns
    ``ChatResponse.openai_stream(ResponseStream(sdk_stream))`` where
    ``sdk_stream`` is a genuine OpenAI ``AsyncStream`` holding a pooled httpx
    connection. Running this through ``Switchyard`` exercises the real
    ``take_core`` -> Rust-core -> ``from_core`` round trip that drops the
    source — the exact path the in-process fakes cannot reach.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        _ = ctx, request
        sdk_stream = await self._client.chat.completions.create(
            model="leak-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        return ChatResponse.openai_stream(ResponseStream(sdk_stream))


def _chain_request() -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": "leak-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })


async def test_chain_round_trip_releases_pool_connection() -> None:
    """End-to-end through the production chain: a real SDK stream survives
    ``take_core``/``from_core`` and is released on client disconnect.

    With a single pool slot, each iteration acquires the only connection, reads
    one frame, and abandons the stream. The next iteration's upstream call must
    re-acquire that slot, which only succeeds if the prior stream's connection
    was actually returned to the pool. If close ownership were lost across the
    Rust-core conversion, the second iteration would starve and raise
    ``PoolTimeout``; the loop completing proves the fix holds through the real
    chain (backend -> take_core -> core -> from_core -> translator -> SSE).
    """
    with _SlowSSEStub() as stub:
        client = _client(stub.base_url)
        chain = Switchyard(
            backend=_RealSdkStreamingBackend(client),
            translator=TranslationEngine(),
        )
        try:
            for _ in range(5):
                result = await chain.call(_chain_request())
                sse = iter_chat_completion_sse(result)
                first = await sse.__anext__()
                assert first.startswith("data:"), first
                assert "error" not in first, f"stream errored on first frame: {first}"
                # Disconnect mid-stream. The next iteration's acquire blocks on
                # the pool until the runtime-scheduled close returns this slot.
                await sse.aclose()
            assert _pool_conns(client) <= 1
        finally:
            await client.close()
