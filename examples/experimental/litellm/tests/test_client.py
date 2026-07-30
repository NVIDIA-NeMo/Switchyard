# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import httpx
import pytest
import respx
from litellm import AuthenticationError, ModelResponse, RateLimitError
from switchyard_litellm import LiteLLMSyClient

BASE_URL = "http://gateway.test/v1"


def request_body() -> dict[str, object]:
    return {
        "model": "auto",
        "instructions": [],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hello."}],
            }
        ],
        "tools": [],
        "tool_choice": None,
        "sampling": {"temperature": 0.2, "top_p": 0.9, "top_k": None},
        "output": {"max_output_tokens": 64, "response_format": None},
        "reasoning": {"effort": "low", "raw": None},
        "stream": False,
        "extensions": {"fields": {}},
        "preservation": {"requests": {}, "responses": {}},
    }


def gateway_response(model: str = "moonshotai/kimi-k3") -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


async def test_call_uses_litellm_async_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> ModelResponse:
        captured.update(kwargs)
        return ModelResponse(**gateway_response())

    monkeypatch.setattr("switchyard_litellm.client.acompletion", fake_acompletion)
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        await client.call(request_body())
    finally:
        await client.aclose()

    assert captured["model"] == "openai/fast"
    assert captured["api_base"] == BASE_URL
    assert captured["api_key"] == "not-needed"
    assert captured["num_retries"] == 0
    assert captured["allowed_openai_params"] == ["reasoning_effort"]
    assert captured["stream"] is False


@respx.mock
async def test_call_translates_request_and_normalizes_response() -> None:
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=gateway_response())
    )
    request = request_body()
    original = copy.deepcopy(request)
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        response = await client.call(request)
    finally:
        await client.aclose()

    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "model": "fast",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hello."}],
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_completion_tokens": 64,
        "reasoning_effort": "low",
        "allowed_openai_params": ["reasoning_effort"],
    }
    assert sent["model"] == "fast"
    assert sent["model"] != "openai/fast"
    assert request == original
    assert response == {
        "id": "chatcmpl-test",
        "model": "moonshotai/kimi-k3",
        "outputs": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello."}],
                "stop_reason": "end_turn",
            }
        ],
        "usage": {
            "input_tokens": 10,
            "cached_input_tokens": 2,
            "output_tokens": 7,
            "total_tokens": 19,
            "reasoning_tokens": 3,
        },
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda body: body.update(messages="not-a-sequence"), "messages"),
        (lambda body: body.update(sampling=[]), "sampling"),
        (lambda body: body.update(output=[]), "output"),
        (lambda body: body.update(reasoning=[]), "reasoning"),
        (lambda body: body.update(extensions=[]), "extensions"),
        (lambda body: body.update(preservation=[]), "preservation"),
        (lambda body: body.update(stream=True), "stream"),
        (
            lambda body: body["instructions"].append(
                {"role": "system", "content": [{"type": "text", "text": "x"}]}
            ),
            "instructions",
        ),
        (
            lambda body: body["tools"].append(
                {"name": "lookup", "description": None, "parameters": {}}
            ),
            "tools",
        ),
        (
            lambda body: body["messages"][0]["content"].append(
                {"type": "image", "source": {"type": "url", "data": {"url": "x"}}}
            ),
            r"messages\[0\]\.content\[1\]",
        ),
        (
            lambda body: body["sampling"].update(top_k=4),
            "sampling.top_k",
        ),
        (
            lambda body: body["output"].update(response_format={"type": "json_object"}),
            "output.response_format",
        ),
        (
            lambda body: body["reasoning"].update(raw={"provider": "value"}),
            "reasoning.raw",
        ),
        (
            lambda body: body["extensions"]["fields"].update(provider="value"),
            "extensions",
        ),
        (
            lambda body: body["extensions"]["fields"].update(provider=None),
            "extensions",
        ),
        (
            lambda body: body["preservation"]["requests"].update(provider=None),
            "preservation",
        ),
        (
            lambda body: body["preservation"]["responses"].update(provider={}),
            "preservation",
        ),
    ],
)
async def test_call_rejects_unsupported_normalized_fields(
    mutate: object,
    match: str,
) -> None:
    request = request_body()
    assert callable(mutate)
    mutate(request)
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=gateway_response())
        )
        client = LiteLLMSyClient("fast", base_url=BASE_URL)
        try:
            with pytest.raises(ValueError, match=match):
                await client.call(request)
        finally:
            await client.aclose()


async def test_call_rejects_missing_messages() -> None:
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        with pytest.raises(ValueError, match="messages"):
            await client.call({})
    finally:
        await client.aclose()


@respx.mock
async def test_call_rejects_a_response_without_text() -> None:
    payload = gateway_response()
    payload["choices"][0]["message"]["content"] = None
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=payload)
    )
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        with pytest.raises(ValueError, match="no text content"):
            await client.call(request_body())
    finally:
        await client.aclose()


async def test_call_rejects_a_response_without_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = gateway_response()
    payload["choices"] = []

    async def fake_acompletion(**_: object) -> ModelResponse:
        return ModelResponse(**payload)

    monkeypatch.setattr("switchyard_litellm.client.acompletion", fake_acompletion)
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        with pytest.raises(ValueError, match="no choices"):
            await client.call(request_body())
    finally:
        await client.aclose()


@respx.mock
async def test_litellm_errors_propagate() -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )
    )
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        with pytest.raises(AuthenticationError):
            await client.call(request_body())
    finally:
        await client.aclose()


@respx.mock
async def test_retryable_litellm_error_is_not_retried() -> None:
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {
                    "message": "rate limited",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )
    )
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        with pytest.raises(RateLimitError):
            await client.call(request_body())
    finally:
        await client.aclose()

    assert route.call_count == 1


@respx.mock
async def test_cached_token_count_preserves_explicit_zero() -> None:
    payload = gateway_response()
    payload["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=payload)
    )
    client = LiteLLMSyClient("fast", base_url=BASE_URL)
    try:
        response = await client.call(request_body())
    finally:
        await client.aclose()

    assert response["usage"]["cached_input_tokens"] == 0
