# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python host adapter over ``switchyard-llm-client``."""

from __future__ import annotations

import pytest

from switchyard.lib.backends.backend_format_resolver import (
    BackendFormatResolution,
    BackendFormatResolver,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.llm_client_builder import (
    build_llm_client,
    build_target_llm_client,
    prepare_llm_target,
    resolve_llm_target,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.components import StatsAccumulator
from switchyard_rust.core import ChatRequest
from switchyard_rust.llm_client import LlmClient
from tests._chain_test_helpers import (
    _backend_payload,
    _last_body,
    _OpenAICompatStub,
    _sse_body,
    _stream_chunk,
)


def _target(
    target_id: str,
    model: str,
    *,
    format: object = BackendFormat.OPENAI,
    api_key: str | None = "sk-test",
    base_url: str | None = "https://example.invalid/v1",
    extra_body: dict[str, object] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> LlmTarget:
    return LlmTarget(
        id=target_id,
        model=model,
        format=format,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )


def _request(model: str = "client-model", *, stream: bool = False) -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    })


def test_builder_constructs_one_client_for_all_targets() -> None:
    client = build_llm_client(
        {
            "strong": _target("strong", "strong-model"),
            "weak": _target("weak", "weak-model"),
        },
        default_target_id="strong",
    )

    assert isinstance(client, LlmClient)
    assert client.target_ids() == ["strong", "weak"]
    assert client.default_target_id == "strong"
    assert [value.value for value in client.supported_request_types] == [
        "openai_chat",
        "openai_responses",
        "anthropic",
    ]


def test_mapping_keys_fill_default_target_ids() -> None:
    client = build_llm_client({
        "strong": LlmTarget(
            model="strong-model",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        "weak": LlmTarget(
            model="weak-model",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
    })

    assert client.target_ids() == ["strong", "weak"]


def test_runtime_defaults_are_applied_before_client_construction() -> None:
    defaulted = prepare_llm_target(
        _target("weak", "nvidia/nvidia/nemotron-3-super-v3")
    )
    explicit = prepare_llm_target(
        LlmTarget(
            id="weak",
            model="nvidia/nvidia/nemotron-3-super-v3",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
    )

    assert defaulted.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert explicit.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_auto_target_is_resolved_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target("strong", "anthropic/claude-test", format=BackendFormat.AUTO)

    monkeypatch.setattr(
        BackendFormatResolver,
        "resolve",
        lambda value: BackendFormatResolution(
            format=BackendFormat.ANTHROPIC,
            reason=f"resolved {value.model}",
        ),
    )

    assert resolve_llm_target(target).format == BackendFormat.ANTHROPIC
    assert isinstance(build_target_llm_client(target), LlmClient)


@pytest.mark.parametrize("missing", ["base_url", "api_key"])
def test_auto_resolution_requires_probe_inputs(missing: str) -> None:
    target = _target(
        "strong",
        "anthropic/claude-test",
        format=BackendFormat.AUTO,
        base_url=None if missing == "base_url" else "https://example.invalid/v1",
        api_key=None if missing == "api_key" else "sk-test",
    )

    with pytest.raises(ValueError, match=f"requires {missing}"):
        build_target_llm_client(target)


def test_client_rejects_duplicate_and_unknown_default_targets() -> None:
    target = _target("duplicate", "model")
    with pytest.raises(RuntimeError, match="duplicate LLM target id"):
        build_llm_client([target, target])
    with pytest.raises(RuntimeError, match="default target missing is not configured"):
        build_llm_client([target], default_target_id="missing")


async def test_client_rejects_missing_or_unknown_selection_before_network() -> None:
    client = build_llm_client([
        _target("strong", "strong-model"),
        _target("weak", "weak-model"),
    ])

    with pytest.raises(RuntimeError, match="multiple targets but no selected target"):
        await client.call(ProxyContext(), _request())

    ctx = ProxyContext()
    ctx.selected_target = "missing"
    with pytest.raises(RuntimeError, match="selected target missing is not configured"):
        await client.call(ctx, _request())


async def test_client_routes_call_and_records_stats() -> None:
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            _backend_payload(content="served", model="strong-model")
        )
        client = build_llm_client([
            _target(
                "strong",
                "strong-model",
                base_url=upstream.base_url,
                extra_body={"provider": {"allow_fallbacks": False}},
                extra_headers={"x-provider-option": "enabled"},
            ),
            _target("weak", "weak-model", base_url=upstream.base_url),
        ])
        stats = StatsAccumulator()
        client.attach_stats(stats)
        ctx = ProxyContext()
        ctx.selected_target = "strong"

        response = await client.call(ctx, _request())

    assert response.body["choices"][0]["message"]["content"] == "served"
    assert response.body["model"] == "client-model"
    assert _last_body(upstream)["model"] == "strong-model"
    assert _last_body(upstream)["provider"] == {"allow_fallbacks": False}
    assert upstream.requests[0]["headers"]["authorization"] == "Bearer sk-test"
    assert upstream.requests[0]["headers"]["x-provider-option"] == "enabled"
    assert upstream.requests[0]["headers"]["x-switchyard-version"]
    assert ctx.selected_model == "strong-model"
    snapshot = stats.snapshot_sync()
    assert snapshot["total_requests"] == 1
    assert snapshot["models"]["strong-model"]["calls"] == 1


async def test_client_translates_openai_request_to_anthropic() -> None:
    with _OpenAICompatStub() as upstream:
        upstream.respond_json({
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "translated"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })
        client = build_target_llm_client(
            _target(
                "anthropic",
                "claude-test",
                format=BackendFormat.ANTHROPIC,
                base_url=upstream.base_url,
            )
        )

        response = await client.call(ProxyContext(), _request())

    assert upstream.requests[0]["path"] == "/v1/messages"
    assert upstream.requests[0]["headers"]["x-api-key"] == "sk-test"
    assert response.body["choices"][0]["message"]["content"] == "translated"


async def test_streaming_client_requests_usage_and_returns_events() -> None:
    with _OpenAICompatStub() as upstream:
        upstream.respond_sse(
            _sse_body([
                _stream_chunk(content="hello"),
                {
                    **_stream_chunk(finish="stop"),
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            ])
        )
        client = build_target_llm_client(
            _target("target", "upstream-model", base_url=upstream.base_url)
        )

        response = await client.call(
            ProxyContext(), _request("client-model", stream=True)
        )
        events = [event async for event in response.stream]

    assert events
    assert _last_body(upstream)["stream_options"] == {"include_usage": True}


async def test_client_respects_telemetry_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_TELEMETRY_OPT_OUT", "true")
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(_backend_payload(content="ok", model="target-model"))
        client = build_target_llm_client(
            _target("target", "target-model", base_url=upstream.base_url)
        )

        await client.call(ProxyContext(), _request())

    assert "x-switchyard-version" not in upstream.requests[0]["headers"]
