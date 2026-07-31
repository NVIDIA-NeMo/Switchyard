# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for direct Rust component bindings."""

from __future__ import annotations

import math

import pytest

from switchyard_rust import (
    AnthropicNativeBackend,
    BackendFormat,
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    EndpointConfig,
    LLMBackend,
    LlmTarget,
    LlmTargetBackend,
    MultiLlmBackend,
    OpenAiNativeBackend,
    ProxyContext,
    RandomRoutingProcessorConfig,
    StatsAccumulator,
    StatsLlmBackend,
    StatsRequestProcessor,
    StatsResponseProcessor,
)


def _target(
    target_id: str,
    model: str,
    *,
    format: object = BackendFormat.OPENAI,
) -> LlmTarget:
    return LlmTarget(
        target_id,
        model,
        format=format,
        endpoint=EndpointConfig(base_url="https://example.test/v1", api_key="test-key"),
    )


def test_component_exports_are_callable_processors_and_native_backends() -> None:
    openai_target = _target("openai", "gpt-test", format=BackendFormat.OPENAI)
    anthropic_target = _target("anthropic", "claude-test", format=BackendFormat.ANTHROPIC)

    assert callable(StatsRequestProcessor().process)
    assert callable(StatsResponseProcessor(StatsAccumulator()).process)
    assert isinstance(OpenAiNativeBackend(openai_target), LLMBackend)
    assert isinstance(AnthropicNativeBackend(anthropic_target), LLMBackend)


def test_config_bindings_validate_and_own_values() -> None:
    endpoint = EndpointConfig(base_url="https://example.test/v1", api_key="secret", timeout_secs=3.5)
    target = LlmTarget(
        "target-a",
        "model-a",
        format="openai",
        endpoint=endpoint,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        extra_headers={"X-Inference-Priority": "batch"},
    )

    assert BackendFormat("openai") == BackendFormat.OPENAI
    assert BackendFormat.OPENAI == "openai"
    assert BackendFormat("responses") == BackendFormat.RESPONSES
    assert BackendFormat.RESPONSES == "responses"
    assert target.id == "target-a"
    assert target.model == "model-a"
    assert target.format == BackendFormat.OPENAI
    assert target.endpoint.to_dict() == {
        "api_key": "secret",
        "base_url": "https://example.test/v1",
        "timeout_secs": 3.5,
    }
    assert target.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert target.extra_headers == {"X-Inference-Priority": "batch"}
    with pytest.raises(ValueError, match="Unknown backend format"):
        BackendFormat("bedrock")
    with pytest.raises(ValueError, match="must not be empty"):
        LlmTarget(" ", "model-a")
    with pytest.raises(ValueError, match="requires a model string"):
        LlmTarget(id="target-a")
    with pytest.raises(RuntimeError, match="strong_probability"):
        RandomRoutingProcessorConfig(
            _target("strong", "strong-model"),
            _target("weak", "weak-model"),
            strong_probability=math.nan,
        )


async def test_stats_processors_share_rust_accumulator() -> None:
    stats = StatsAccumulator()
    request_processor = StatsRequestProcessor()
    response_processor = StatsResponseProcessor(stats)
    ctx = ProxyContext()
    request = ChatRequest.openai_chat({"model": "client-model", "messages": []})

    processed = await request_processor.process(ctx, request)
    ctx.selected_model = "served-model"
    response = await response_processor.process(
        ctx,
        ChatResponse.openai_completion({
            "model": "served-model",
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "prompt_tokens_details": {
                    "cached_tokens": 3,
                    "cache_creation_tokens": 2,
                },
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }),
    )

    assert processed.model == "client-model"
    assert response.body["model"] == "served-model"
    snapshot = stats.snapshot_sync()
    model_stats = snapshot["models"]["served-model"]
    assert model_stats["prompt_tokens"] == 11
    assert model_stats["completion_tokens"] == 7
    assert model_stats["cached_tokens"] == 3
    assert model_stats["cache_creation_tokens"] == 2
    assert model_stats["reasoning_tokens"] == 5
    assert model_stats["total_latency"]["count"] == 1


async def test_stream_callbacks_survive_handoff_to_rust_response_processor() -> None:
    async def source():
        yield {"choices": [{"delta": {"content": "hi"}}]}
        yield {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    tapped: list[dict[str, object]] = []
    mapped: list[dict[str, object]] = []
    completed = False

    async def tap(event: dict[str, object]) -> None:
        tapped.append(dict(event))

    async def map_event(event: dict[str, object]) -> dict[str, object]:
        mapped.append(dict(event))
        return {**event, "mapped": True}

    async def on_complete() -> None:
        nonlocal completed
        completed = True

    stats = StatsAccumulator()
    ctx = ProxyContext()
    ctx.selected_model = "served-model"
    response = ChatResponse.openai_stream(source())
    response.stream.tap(tap).map(map_event).on_complete(on_complete)

    processed = await StatsResponseProcessor(stats).process(ctx, response)
    events = [event async for event in processed.stream]

    assert [event["mapped"] for event in events] == [True, True]
    assert len(tapped) == 2
    assert len(mapped) == 2
    assert completed is True
    assert stats.snapshot_sync()["models"]["served-model"]["prompt_tokens"] == 1


def test_backend_bindings_construct_without_provider_sdks_or_network() -> None:
    openai_target = _target("openai", "gpt-test", format=BackendFormat.OPENAI)
    responses_target = _target("responses", "gpt-responses", format=BackendFormat.RESPONSES)
    anthropic_target = _target("anthropic", "claude-test", format=BackendFormat.ANTHROPIC)
    openai = OpenAiNativeBackend(openai_target)
    responses = OpenAiNativeBackend(responses_target)
    anthropic = AnthropicNativeBackend(anthropic_target)
    stats = StatsAccumulator()

    multi = MultiLlmBackend([
        LlmTargetBackend(openai_target, openai),
        (anthropic_target, anthropic),
    ], default_target_id="openai")
    stats_backend = StatsLlmBackend(openai, stats)

    assert [request_type.value for request_type in openai.supported_request_types] == [
        "openai_chat"
    ]
    assert [request_type.value for request_type in responses.supported_request_types] == [
        "openai_responses"
    ]
    assert [request_type.value for request_type in anthropic.supported_request_types] == [
        "anthropic"
    ]
    assert set(multi.target_ids()) == {"openai", "anthropic"}
    assert multi.default_target_id() == "openai"
    assert stats_backend.supported_request_types == openai.supported_request_types


def test_backend_bindings_reject_invalid_native_composition() -> None:
    openai_target = _target("openai", "gpt-test", format=BackendFormat.OPENAI)
    openai = OpenAiNativeBackend(openai_target)

    with pytest.raises(RuntimeError, match="requires a target with resolved OpenAI format"):
        OpenAiNativeBackend(_target("bad", "claude-test", format=BackendFormat.ANTHROPIC))
    with pytest.raises(RuntimeError, match="requires a target with resolved Anthropic format"):
        AnthropicNativeBackend(_target("bad", "gpt-test", format=BackendFormat.OPENAI))
    with pytest.raises(RuntimeError, match="duplicate LLM target id"):
        MultiLlmBackend([
            LlmTargetBackend(openai_target, openai),
            LlmTargetBackend(openai_target, openai),
        ])
    with pytest.raises(RuntimeError, match="default target missing is not configured"):
        MultiLlmBackend(
            [LlmTargetBackend(openai_target, openai)],
            default_target_id="missing",
        )
    with pytest.raises(RuntimeError, match="at least one request type"):
        MultiLlmBackend([LlmTargetBackend(openai_target, openai)], supported_request_types=[])


def test_wrappers_require_rust_native_backend_instances() -> None:
    class PythonOnlyBackend(LLMBackend):
        @property
        def supported_request_types(self) -> list[ChatRequestType]:
            return [ChatRequestType.OPENAI_CHAT]

        async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
            return ChatResponse.openai_completion({"model": request.model})

    with pytest.raises(TypeError):
        StatsLlmBackend(PythonOnlyBackend(), StatsAccumulator())
