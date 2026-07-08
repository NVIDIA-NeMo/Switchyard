# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal LLM client wrapper for backends.

This provides a thin wrapper around the OpenAI SDK to support
OpenAI-compatible backends in the chain.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from switchyard.telemetry import get_telemetry_headers


class OpenAILLMClient:
    """Client that wraps the official OpenAI Python SDK.

    Works with any OpenAI-compatible API (OpenAI, NVIDIA NIM, Azure,
    vLLM, etc.) by accepting a custom ``base_url``.

    Used by :class:`~switchyard.lib.backends.openai_llm_backend.OpenAiNativeBackend`
    and other OpenAI-compatible backends.
    """

    async_client: AsyncOpenAI

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the async OpenAI client.

        Args:
            api_key: API key for authentication. When omitted or empty,
                an inert placeholder is used so the SDK can construct;
                callers must then supply a real key per-request via
                ``acompletion(api_key=...)`` (BYO-key mode). The
                placeholder never reaches a real upstream — without a
                caller key the SDK call fails with the upstream's 401.
            base_url: Custom base URL for OpenAI-compatible APIs (e.g.,
                Azure, vLLM, NVIDIA NIM). Defaults to OpenAI's standard URL.
            timeout: Request timeout in seconds. None means no timeout.
            max_retries: Override the OpenAI SDK's default 2-retry budget.
                ``None`` (default) keeps the SDK default. Set ``0`` for the
                classifier path so a slow-upstream ``ReadTimeout`` falls
                through to our own ``fail_open`` fallback at the configured
                timeout rather than compounding via SDK exponential backoff.
        """
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            # BYO-key deployments construct the client with no
            # server-side credential and supply the caller's key per
            # request via ``acompletion(api_key=...)``. The SDK refuses
            # to construct without *some* key, so we inject an inert
            # placeholder that is overridden on every real call. If a
            # caller forgets to send a key, the upstream sees this
            # placeholder and returns 401 — no real secret can leak.
            client_kwargs["api_key"] = "switchyard-byo-key-required"
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        client_kwargs["default_headers"] = get_telemetry_headers()

        # Only the async client is ever used (backends call ``acompletion``).
        # A sync ``OpenAI`` client would allocate a second, never-used httpx
        # connection pool (1000 connections by default) per instance, so it is
        # intentionally not constructed.
        self.async_client = AsyncOpenAI(**client_kwargs)

    def _client_for_api_key(self, api_key: str | None) -> AsyncOpenAI:
        if api_key and api_key.strip():
            return self.async_client.with_options(api_key=api_key)
        return self.async_client

    async def acompletion(
        self,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async wrapper for chat completions.

        When a non-blank ``api_key`` is supplied, the call uses that credential
        via the SDK's ``with_options`` override instead of the client's
        construction-time key. Used by backends that forward a per-request
        caller credential (BYO-key multi-tenant deployments).

        When ``api_key`` is ``None`` or blank (no caller key, or a
        whitespace-only header), the override is skipped so the call falls
        back to the construction-time key — the per-endpoint ``api_key`` an
        operator configured. A blank value must not override a real configured
        key with nothing, which would unauthenticate the upstream call (a 401
        even though a valid key was configured).
        """
        return await self._client_for_api_key(api_key).chat.completions.create(**kwargs)

    async def aresponses(
        self,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async wrapper for the OpenAI Responses API.

        When a non-blank ``api_key`` is supplied, the call uses that
        per-request credential via the SDK's ``with_options`` override. When
        ``api_key`` is ``None`` or blank, no override is applied and the
        construction-time key configured on the client is used. SDK validation
        and upstream errors are intentionally propagated unchanged.

        Non-streaming calls return the upstream's **exact JSON body** (a
        ``dict``) rather than the SDK's typed ``Response`` model: round-tripping
        through the typed model re-normalizes the payload and its
        ``exclude_none`` serialization drops explicit-null fields, eroding
        schema fidelity for Responses passthrough.

        Streaming calls return a :class:`RawSSEFrameStream` yielding the
        upstream's SSE frames as **verbatim strings** (modulo CRLF → LF line
        normalization) for the same reason — the SDK's typed event stream
        drops provider extras and explicit nulls per event. The HTTP request
        is sent (and error statuses raise) *before* this method returns, so
        the caller's retry/failover contract is unchanged.
        """
        client = self._client_for_api_key(api_key)
        if kwargs.get("stream"):
            cm = client.responses.with_streaming_response.create(**kwargs)
            # Enter eagerly: the request goes out and non-2xx statuses raise
            # ``APIStatusError`` here, not at first iteration — after first
            # iteration the endpoint has already committed an HTTP 200.
            response = await cm.__aenter__()
            return RawSSEFrameStream(cm, response.http_response)
        raw = await client.responses.with_raw_response.create(**kwargs)
        return raw.http_response.json()


class RawSSEFrameStream:
    """Async iterator over an SSE response's frames as verbatim strings.

    Each item is one complete frame (all lines up to and including the blank
    separator, e.g. ``"event: x\\ndata: {...}\\n\\n"``), byte-equivalent to the
    upstream modulo CRLF → LF normalization. Comment/keep-alive frames pass
    through unchanged. ``aclose`` releases the underlying HTTP response and is
    safe to call at any point, including before the first ``__anext__``.
    """

    def __init__(self, cm: Any, http_response: Any) -> None:
        self._cm = cm
        self._lines = http_response.aiter_lines()
        self._closed = False

    def __aiter__(self) -> RawSSEFrameStream:
        return self

    async def __anext__(self) -> str:
        buffer: list[str] = []
        try:
            async for line in self._lines:
                if line == "":
                    if buffer:
                        return "\n".join(buffer) + "\n\n"
                    continue
                buffer.append(line)
        except BaseException:
            await self.aclose()
            raise
        await self.aclose()
        if buffer:
            # Upstream closed without a trailing blank line; emit the tail as
            # a well-formed frame rather than dropping it.
            return "\n".join(buffer) + "\n\n"
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cm.__aexit__(None, None, None)
