# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the token-continuity injection backend."""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from switchyard.lib.backends.token_injection_backend import (
    TokenInjectionBackend,
    _slice_env_suffix,
)
from switchyard.lib.backends.vllm_parsers import (
    parse_generation,
    tool_choice_json_schema,
)
from switchyard.lib.processors.token_capture_request_processor import CTX_TOKEN_CAPTURE_SESSION
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequest, ChatResponse, ProxyContext

_BASE_URL = "http://vllm.test/v1"
_MODEL = "Qwen/Qwen3-0.6B"
_EOT = 9

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


class _FakeInner(LLMBackend):
    """Chat backend double returning a preset vLLM-shaped completion."""

    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.calls = 0
        self._body = body if body is not None else _first_turn_body()

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        return ChatResponse.openai_completion(self._body)


def _first_turn_body() -> dict[str, Any]:
    """vLLM chat body for the seed call: p1=[1,2,3], g1=[4,5,EOT], natural stop."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": _MODEL,
        "prompt_token_ids": [1, 2, 3],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "step one"},
                "finish_reason": "stop",
                "token_ids": [4, 5, _EOT],
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}, {"logprob": -0.3}]},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


def _backend(inner: _FakeInner | None = None, **kwargs: Any) -> TokenInjectionBackend:
    return TokenInjectionBackend(
        inner or _FakeInner(),
        base_url=_BASE_URL,
        model=_MODEL,
        **kwargs,
    )


def _ctx(session: str | None = "sess-1") -> ProxyContext:
    ctx = ProxyContext()
    if session is not None:
        ctx.metadata[CTX_TOKEN_CAPTURE_SESSION] = session
    return ctx


def _chat_request(turns: int = 1) -> ChatRequest:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
    if turns > 1:
        messages += [
            {"role": "assistant", "content": "step one"},
            {"role": "user", "content": "continue"},
        ]
    return ChatRequest.openai_chat({"model": _MODEL, "messages": messages})


def _mock_tokenize(respx_mock: respx.MockRouter, token_lists: list[list[int]]) -> None:
    """Queue /tokenize responses in call order."""
    respx_mock.post("http://vllm.test/tokenize").mock(
        side_effect=[httpx.Response(200, json={"tokens": tokens}) for tokens in token_lists]
    )


def _completion_response(prompt: list[int], generation: list[int], text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-2",
            "object": "text_completion",
            "created": 1700000100,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                    "prompt_token_ids": prompt,
                    "token_ids": generation,
                    "logprobs": {
                        "tokens": ["x"] * len(generation),
                        "token_logprobs": [-0.5] * len(generation),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(generation),
                "total_tokens": len(prompt) + len(generation),
            },
        },
    )


# ---------------------------------------------------------------------------
# Injection path
# ---------------------------------------------------------------------------


@respx.mock
async def test_second_call_injects_contiguous_prompt(respx_mock: respx.MockRouter) -> None:
    """The injected prompt is exactly accumulated history + env suffix."""
    inner = _FakeInner()
    backend = _backend(inner)
    ctx = _ctx()

    # Seed: prefix (no gen prompt) = [1, 2]; p1 = [1, 2, 3] starts with it.
    _mock_tokenize(
        respx_mock,
        [
            [1, 2],  # seed prefix
            [1, 2, _EOT, 20, 21, 30],  # turn-2 rendered (gen prompt on)
            [1, 2, _EOT, 20, 21],  # turn-2 prefix (gen prompt off)
        ],
    )
    injected = [1, 2, 3, 4, 5, _EOT, 20, 21, 30]
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response(injected, [40, 41], "The answer")
    )

    await backend.call(ctx, _chat_request())
    response = await backend.call(ctx, _chat_request(turns=2))

    assert inner.calls == 1  # only the seed call hit the chat path
    sent = json.loads(completions.calls[0].request.content)
    assert sent["prompt"] == injected
    assert sent["return_token_ids"] is True
    assert sent["logprobs"] == 0

    body = dict(response.body)
    assert body["prompt_token_ids"] == injected
    choice = body["choices"][0]
    assert choice["token_ids"] == [40, 41]
    assert choice["message"]["content"] == "The answer"
    assert choice["logprobs"]["content"] == [
        {"token_id": 40, "logprob": -0.5},
        {"token_id": 41, "logprob": -0.5},
    ]
    # Contiguity by construction: the injected prompt extends p1 + g1.
    history = [1, 2, 3, 4, 5, _EOT]
    assert body["prompt_token_ids"][: len(history)] == history


@respx.mock
async def test_third_call_extends_advanced_state(respx_mock: respx.MockRouter) -> None:
    """State advances across injected calls: call 3 extends call 2's tokens."""
    backend = _backend()
    ctx = _ctx()
    injected_2 = [1, 2, 3, 4, 5, _EOT, 20, 21, 30]
    injected_3 = injected_2 + [40, 41, 50, 51]
    _mock_tokenize(
        respx_mock,
        [
            [1, 2],
            [1, 2, _EOT, 20, 21, 30],
            [1, 2, _EOT, 20, 21],
            [1, 2, _EOT, 20, 21, 41, 50, 51],  # turn-3 rendered; EOT 41 ends prior turn
            [1, 2, _EOT, 20, 21, 41, 50],
        ],
    )
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        side_effect=[
            _completion_response(injected_2, [40, 41], "step two"),
            _completion_response(injected_3, [60], "done"),
        ]
    )

    await backend.call(ctx, _chat_request())
    await backend.call(ctx, _chat_request(turns=2))
    await backend.call(ctx, _chat_request(turns=2))

    sent = json.loads(completions.calls[1].request.content)
    assert sent["prompt"] == injected_3


@respx.mock
async def test_max_tokens_takes_smaller_of_both_keys(respx_mock: respx.MockRouter) -> None:
    """Harnesses may send both max_tokens and max_completion_tokens; honor the smaller."""
    backend = _backend()
    ctx = _ctx()
    _mock_tokenize(respx_mock, [[1, 2], [1, 2, _EOT, 20], [1, 2, _EOT]])
    injected = [1, 2, 3, 4, 5, _EOT, 20]
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response(injected, [40], "ok")
    )

    await backend.call(ctx, _chat_request())
    request = ChatRequest.openai_chat({
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32000,
        "max_completion_tokens": 512,
    })
    await backend.call(ctx, request)
    assert json.loads(completions.calls[0].request.content)["max_tokens"] == 512


@respx.mock
async def test_max_tokens_clamped_to_model_context(respx_mock: respx.MockRouter) -> None:
    """The chat endpoint caps the budget to remaining context; the injected
    completions call must do the same or vLLM rejects it with a 400."""
    backend = _backend()
    ctx = _ctx()
    respx_mock.post("http://vllm.test/tokenize").mock(
        side_effect=[
            httpx.Response(200, json={"tokens": [1, 2], "max_model_len": 40}),
            httpx.Response(200, json={"tokens": [1, 2, _EOT, 20], "max_model_len": 40}),
            httpx.Response(200, json={"tokens": [1, 2, _EOT], "max_model_len": 40}),
        ]
    )
    injected = [1, 2, 3, 4, 5, _EOT, 20]  # 7 tokens; remaining context = 33
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response(injected, [40], "ok")
    )

    await backend.call(ctx, _chat_request())
    request = ChatRequest.openai_chat({
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32000,
    })
    await backend.call(ctx, request)
    assert json.loads(completions.calls[0].request.content)["max_tokens"] == 33


@respx.mock
async def test_sampling_params_forwarded(respx_mock: respx.MockRouter) -> None:
    backend = _backend()
    ctx = _ctx()
    _mock_tokenize(respx_mock, [[1, 2], [1, 2, _EOT, 20], [1, 2, _EOT]])
    injected = [1, 2, 3, 4, 5, _EOT, 20]
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response(injected, [40], "ok")
    )

    await backend.call(ctx, _chat_request())
    request = ChatRequest.openai_chat({
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "max_completion_tokens": 128,
        "stop": ["\n\n"],
    })
    await backend.call(ctx, request)

    sent = json.loads(completions.calls[0].request.content)
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 128
    assert sent["stop"] == ["\n\n"]


@respx.mock
async def test_required_tool_choice_uses_guided_decoding(respx_mock: respx.MockRouter) -> None:
    """tool_choice: required rides structured_outputs and parses the JSON array."""
    backend = _backend()
    ctx = _ctx()
    _mock_tokenize(respx_mock, [[1, 2], [1, 2, _EOT, 20], [1, 2, _EOT]])
    injected = [1, 2, 3, 4, 5, _EOT, 20]
    guided_text = json.dumps([{"name": "get_weather", "parameters": {"city": "Paris"}}])
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response(injected, [40, 41], guided_text)
    )

    await backend.call(ctx, _chat_request())
    request = ChatRequest.openai_chat({
        "model": _MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": _TOOLS,
        "tool_choice": "required",
    })
    response = await backend.call(ctx, request)

    sent = json.loads(completions.calls[0].request.content)
    assert sent["structured_outputs"]["json"]["minItems"] == 1

    choice = dict(response.body)["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Fallback pinning
# ---------------------------------------------------------------------------


async def test_no_session_delegates_without_state() -> None:
    inner = _FakeInner()
    backend = _backend(inner)
    await backend.call(_ctx(session=None), _chat_request())
    assert inner.calls == 1
    assert backend._sessions == {}


async def test_missing_token_fields_pins_fallback() -> None:
    inner = _FakeInner(body={"id": "x", "model": _MODEL, "choices": [{"message": {}}]})
    backend = _backend(inner)
    ctx = _ctx()
    await backend.call(ctx, _chat_request())
    await backend.call(ctx, _chat_request(turns=2))
    assert inner.calls == 2  # second call went straight to the chat path
    assert backend._sessions["sess-1"].fallen_back


@respx.mock
async def test_prefix_mismatch_falls_back_per_call_without_pinning(
    respx_mock: respx.MockRouter,
) -> None:
    """A non-extending call (e.g. opencode's title generator on the same
    session) is served via the chat path, but the chain state survives and the
    next extending call still injects."""
    inner = _FakeInner()
    backend = _backend(inner)
    ctx = _ctx()
    _mock_tokenize(
        respx_mock,
        [
            [1, 2],  # seed prefix
            [99, 98],  # title-gen call: does NOT start with [1, 2]
            [1, 2, _EOT, 20, 21, 30],  # main turn 2: extends the chain
            [1, 2, _EOT, 20, 21],
        ],
    )
    injected = [1, 2, 3, 4, 5, _EOT, 20, 21, 30]
    completions = respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response(injected, [40, 41], "back on chain")
    )

    await backend.call(ctx, _chat_request())  # seed (chat path)
    await backend.call(ctx, _chat_request(turns=2))  # title-gen: chat path
    await backend.call(ctx, _chat_request(turns=2))  # main turn 2: injected

    assert inner.calls == 2  # seed + title-gen only
    assert not backend._sessions["sess-1"].fallen_back
    assert json.loads(completions.calls[0].request.content)["prompt"] == injected


@respx.mock
async def test_length_stop_leaves_no_eot_and_falls_back_per_call(
    respx_mock: respx.MockRouter,
) -> None:
    body = _first_turn_body()
    body["choices"][0]["finish_reason"] = "length"
    inner = _FakeInner(body=body)
    backend = _backend(inner)
    ctx = _ctx()
    _mock_tokenize(respx_mock, [[1, 2]])

    await backend.call(ctx, _chat_request())
    await backend.call(ctx, _chat_request(turns=2))
    assert inner.calls == 2
    assert not backend._sessions["sess-1"].fallen_back


@respx.mock
async def test_multi_choice_falls_back(respx_mock: respx.MockRouter) -> None:
    inner = _FakeInner()
    backend = _backend(inner)
    ctx = _ctx()
    _mock_tokenize(respx_mock, [[1, 2]])

    await backend.call(ctx, _chat_request())
    request = ChatRequest.openai_chat(
        {"model": _MODEL, "messages": [{"role": "user", "content": "hi"}], "n": 2}
    )
    await backend.call(ctx, request)
    assert inner.calls == 2
    assert backend._sessions["sess-1"].fallen_back


@respx.mock
async def test_prompt_echo_mismatch_falls_back(respx_mock: respx.MockRouter) -> None:
    """A completions response that does not echo the injected prompt is rejected."""
    inner = _FakeInner()
    backend = _backend(inner)
    ctx = _ctx()
    _mock_tokenize(respx_mock, [[1, 2], [1, 2, _EOT, 20]])
    respx_mock.post(f"{_BASE_URL}/completions").mock(
        return_value=_completion_response([7, 7, 7], [40], "bad echo")
    )

    await backend.call(ctx, _chat_request())
    await backend.call(ctx, _chat_request(turns=2))
    assert inner.calls == 2
    assert backend._sessions["sess-1"].fallen_back


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_env_suffix_skips_eot_after_natural_stop() -> None:
    assert _slice_env_suffix([_EOT, 20, 21], [4, 5, _EOT], _EOT) == [20, 21]


def test_env_suffix_keeps_eot_after_truncation() -> None:
    assert _slice_env_suffix([_EOT, 20, 21], [4, 5], _EOT) == [_EOT, 20, 21]


def test_parse_generation_without_parsers_passes_through() -> None:
    parsed = parse_generation(
        "raw text", model=_MODEL, tools=None, tool_parser=None, reasoning_parser=None
    )
    assert parsed.content == "raw text"
    assert parsed.reasoning_content is None
    assert parsed.tool_calls == []


# ---------------------------------------------------------------------------
# Route bundle wiring
# ---------------------------------------------------------------------------


def _bundle(route: dict[str, Any]) -> dict[str, Any]:
    return {"routes": {"policy-model": route}}


def _injection_route(**overrides: Any) -> dict[str, Any]:
    route: dict[str, Any] = {
        "type": "model",
        "target": {"model": _MODEL, "base_url": _BASE_URL, "api_key": "k"},
        "token_capture_engine": "vllm",
        "token_injection": True,
        "tool_parser": "hermes",
        "reasoning_parser": "qwen3",
    }
    route.update(overrides)
    return route


def _route_backends(table: Any, model_id: str = "policy-model") -> list[Any]:
    return list(table.lookup_switchyard(model_id).iter_components())


def test_bundle_builds_injection_backend() -> None:
    from switchyard.cli.route_bundle import build_route_bundle_table

    table = build_route_bundle_table(_bundle(_injection_route()), token_capture_enabled=True)
    components = _route_backends(table)
    injection = [c for c in components if isinstance(c, TokenInjectionBackend)]
    assert len(injection) == 1
    backend = injection[0]
    assert backend._model == _MODEL
    assert backend._tool_parser == "hermes"
    assert backend._reasoning_parser == "qwen3"


def test_bundle_injection_requires_capture_engine() -> None:
    import pytest

    from switchyard.cli.route_bundle import RouteBundleConfigError, build_route_bundle_table

    route = _injection_route()
    del route["token_capture_engine"]
    with pytest.raises(RouteBundleConfigError, match="token_capture_engine"):
        build_route_bundle_table(_bundle(route), token_capture_enabled=True)


def test_bundle_injection_stripped_when_capture_disabled() -> None:
    from switchyard.cli.route_bundle import build_route_bundle_table

    table = build_route_bundle_table(_bundle(_injection_route()), token_capture_enabled=False)
    components = _route_backends(table)
    assert not any(isinstance(c, TokenInjectionBackend) for c in components)


def test_bundle_without_injection_key_is_unwrapped() -> None:
    from switchyard.cli.route_bundle import build_route_bundle_table

    route = _injection_route()
    del route["token_injection"]
    table = build_route_bundle_table(_bundle(route), token_capture_enabled=True)
    components = _route_backends(table)
    assert not any(isinstance(c, TokenInjectionBackend) for c in components)


def test_tool_choice_schema_shapes() -> None:
    assert tool_choice_json_schema(_TOOLS, "auto") is None
    assert tool_choice_json_schema(_TOOLS, None) is None
    assert tool_choice_json_schema([], "required") is None

    required = tool_choice_json_schema(_TOOLS, "required")
    assert required is not None
    assert required["items"]["anyOf"][0]["properties"]["name"]["enum"] == ["get_weather"]

    named = tool_choice_json_schema(
        _TOOLS, {"type": "function", "function": {"name": "get_weather"}}
    )
    assert named == _TOOLS[0]["function"]["parameters"]
