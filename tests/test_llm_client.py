# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`OpenAILLMClient`."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
import respx
from openai import AsyncOpenAI

from switchyard.lib.llm_client import OpenAILLMClient, RawSSEFrameStream


def test_constructs_only_the_async_client() -> None:
    """Only the async client is built.

    Backends call ``acompletion`` exclusively, so a sync ``OpenAI`` client
    would only allocate a second, never-used httpx connection pool (1000
    connections by default) per instance.
    """
    client = OpenAILLMClient(api_key="test-key")
    assert isinstance(client.async_client, AsyncOpenAI)
    assert not hasattr(client, "client")


def test_max_retries_reaches_the_sdk_client() -> None:
    """``max_retries`` is forwarded to the underlying SDK client."""
    client = OpenAILLMClient(api_key="test-key", max_retries=0)
    assert client.async_client.max_retries == 0


class TestAcompletionApiKeyOverride:
    """``acompletion`` overrides the construction-time key only for a real key.

    A blank or absent per-call ``api_key`` must fall back to the
    construction-time key (the configured endpoint key) instead of overriding
    it with nothing, which would unauthenticate the upstream call.
    """

    @staticmethod
    def _client_with_spied_options() -> tuple[OpenAILLMClient, MagicMock]:
        client = OpenAILLMClient(api_key="endpoint-key")
        client.async_client = MagicMock()
        client.async_client.chat.completions.create = AsyncMock(return_value="base")
        client.async_client.responses.create = AsyncMock(return_value="responses-base")
        overridden = MagicMock()
        overridden.chat.completions.create = AsyncMock(return_value="overridden")
        overridden.responses.create = AsyncMock(return_value="responses-overridden")
        client.async_client.with_options.return_value = overridden
        # Non-streaming ``aresponses`` fetches the raw HTTP response and
        # returns its exact JSON body; wire raw spies per client.
        for target, label in ((client.async_client, "base"), (overridden, "overridden")):
            raw = MagicMock()
            raw.http_response.json.return_value = {"src": f"responses-{label}"}
            target.responses.with_raw_response.create = AsyncMock(return_value=raw)
        return client, client.async_client

    async def test_real_caller_key_uses_with_options(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.acompletion(api_key="caller-key", model="m")
        async_client.with_options.assert_called_once_with(api_key="caller-key")
        assert result == "overridden"

    async def test_none_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.acompletion(api_key=None, model="m")
        async_client.with_options.assert_not_called()
        assert result == "base"

    async def test_blank_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.acompletion(api_key="   ", model="m")
        async_client.with_options.assert_not_called()
        assert result == "base"

    async def test_responses_real_caller_key_uses_with_options(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.aresponses(api_key="caller-key", model="m", input="hi")
        async_client.with_options.assert_called_once_with(api_key="caller-key")
        assert result == {"src": "responses-overridden"}

    async def test_responses_missing_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.aresponses(api_key=None, model="m", input="hi")
        async_client.with_options.assert_not_called()
        assert result == {"src": "responses-base"}

    async def test_responses_blank_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.aresponses(api_key="   ", model="m", input="hi")
        async_client.with_options.assert_not_called()
        assert result == {"src": "responses-base"}

    async def test_responses_streaming_uses_raw_streaming_path(self) -> None:
        """Streaming fetches the raw SSE response (verbatim frames) and enters
        the SDK context manager eagerly so error statuses raise at call time."""
        client, async_client = self._client_with_spied_options()

        async def _lines():
            yield "data: {}"
            yield ""

        api_response = MagicMock()
        api_response.http_response.aiter_lines = _lines
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=api_response)
        cm.__aexit__ = AsyncMock(return_value=False)
        async_client.responses.with_streaming_response.create = MagicMock(return_value=cm)

        result = await client.aresponses(api_key=None, model="m", input="hi", stream=True)

        assert isinstance(result, RawSSEFrameStream)
        cm.__aenter__.assert_awaited_once()
        async_client.responses.create.assert_not_called()
        async_client.responses.with_raw_response.create.assert_not_called()
        await result.aclose()
        cm.__aexit__.assert_awaited_once()


@respx.mock
async def test_aresponses_returns_exact_upstream_json() -> None:
    """Non-streaming Responses calls return the upstream body as-is.

    Provider-specific extras and explicit-null fields must survive — the SDK
    typed-model round-trip would normalize the former and ``exclude_none``
    serialization would drop the latter.
    """
    upstream = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1719890000,
        "model": "gpt-5",
        "status": "completed",
        "output": [],
        "store": False,
        "temperature": 1.0,
        "top_p": 0.9,
        "previous_response_id": None,
        "provider_meta": {"azure_region": "eastus"},
    }
    respx.post("http://upstream.test/v1/responses").mock(
        return_value=httpx.Response(200, json=upstream)
    )

    client = OpenAILLMClient(api_key="k", base_url="http://upstream.test/v1")
    result = await client.aresponses(model="gpt-5", input="hi")

    assert result == upstream


_SSE_FRAMES = [
    (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","store":false,'
        '"temperature":1.0,"reasoning":{"effort":null},"provider_meta":{"az":"eastus"}}}\n\n'
    ),
    ": keep-alive\n\n",
    (
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"hi","vendor_extra":123}\n\n'
    ),
    (
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","usage":'
        '{"input_tokens":3,"output_tokens":2}}}\n\n'
    ),
]


@respx.mock
async def test_aresponses_streaming_yields_verbatim_sse_frames() -> None:
    """Streaming Responses calls yield the upstream SSE frames byte-for-byte
   : unknown provider fields, explicit nulls, comment keep-alives,
    and event names all survive because no typed-model parse happens.
    """
    respx.post("http://upstream.test/v1/responses").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(_SSE_FRAMES).encode(),
        )
    )

    client = OpenAILLMClient(api_key="k", base_url="http://upstream.test/v1")
    stream = await client.aresponses(model="gpt-5", input="hi", stream=True)

    frames = [frame async for frame in stream]
    assert frames == _SSE_FRAMES


@respx.mock
async def test_aresponses_streaming_error_status_raises_at_call_time() -> None:
    """A non-2xx on a streaming Responses call raises ``APIStatusError`` from
    ``aresponses`` itself — before any frame is consumed — so the backend's
    retry/failover contract is unchanged by the raw-frame path."""
    respx.post("http://upstream.test/v1/responses").mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom"}})
    )

    client = OpenAILLMClient(api_key="k", base_url="http://upstream.test/v1")
    with pytest.raises(openai.APIStatusError):
        await client.aresponses(model="gpt-5", input="hi", stream=True)
