# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Rust-owned switchyard_rust core bindings."""

from __future__ import annotations

import pytest

from switchyard.lib.chat_response import (
    AnthropicResponseStream,
    ResponsesApiStream,
    ResponseStream,
)
from switchyard_rust import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseStream,
    ChatResponseType,
    LLMBackend,
    ProxyContext,
    ProxyMetadata,
)


def test_openai_chat_request_owns_body_and_exposes_model() -> None:
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }

    request = ChatRequest.openai_chat(body)

    assert request.request_type == ChatRequestType.OPENAI_CHAT
    assert request.request_type is ChatRequestType.OPENAI_CHAT
    assert hash(request.request_type) == hash(ChatRequestType.OPENAI_CHAT)
    assert request.request_type.value == "openai_chat"
    assert request.model == "gpt-4o"
    assert request.body == body
    assert request.body is not body

    body["model"] = "mutated-after-construction"
    assert request.model == "gpt-4o"


def test_request_constructors_preserve_wire_format() -> None:
    responses = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hello"})
    anthropic = ChatRequest.anthropic({"model": "claude-sonnet-4.5", "messages": []})

    assert responses.request_type == ChatRequestType.OPENAI_RESPONSES
    assert responses.request_type.value == "openai_responses"
    assert responses.model == "gpt-4o"
    assert anthropic.request_type == ChatRequestType.ANTHROPIC
    assert anthropic.request_type.value == "anthropic"
    assert anthropic.model == "claude-sonnet-4.5"


def test_set_model_mutates_rust_owned_body() -> None:
    request = ChatRequest.openai_chat({"model": "old", "messages": []})

    request.set_model("new")

    assert request.model == "new"
    assert request.to_body() == {"model": "new", "messages": []}


def test_set_model_recovers_malformed_non_object_body() -> None:
    request = ChatRequest.anthropic(["not", "an", "object"])

    request.set_model("claude-sonnet-4.5")

    assert request.model == "claude-sonnet-4.5"
    assert request.body == {"model": "claude-sonnet-4.5"}


def test_replace_body_preserves_request_type() -> None:
    request = ChatRequest.openai_responses({"model": "old", "input": "hello"})

    request.replace_body({"model": "new", "input": "replacement"})

    assert request.request_type == ChatRequestType.OPENAI_RESPONSES
    assert request.request_type.value == "openai_responses"
    assert request.model == "new"
    assert request.body == {"model": "new", "input": "replacement"}


def test_non_json_body_is_rejected() -> None:
    class NonJsonable:
        pass

    with pytest.raises(ValueError):
        ChatRequest.openai_chat(NonJsonable())


def test_openai_completion_response_owns_body() -> None:
    body = {
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    }

    response = ChatResponse.openai_completion(body)

    assert response.response_type == ChatResponseType.OPENAI_COMPLETION
    assert response.response_type is ChatResponseType.OPENAI_COMPLETION
    assert hash(response.response_type) == hash(ChatResponseType.OPENAI_COMPLETION)
    assert response.response_type.value == "openai_completion"
    assert response.body == body
    assert response.body is not body

    body["model"] = "mutated-after-construction"
    assert response.body["model"] == "gpt-4o"


def test_response_constructors_preserve_wire_shape() -> None:
    responses = ChatResponse.openai_responses_completion({
        "id": "resp-test",
        "model": "gpt-4o",
        "output": [],
    })
    anthropic = ChatResponse.anthropic_completion({
        "id": "msg-test",
        "model": "claude-sonnet-4.5",
        "content": [],
    })

    assert responses.response_type == ChatResponseType.OPENAI_RESPONSES_COMPLETION
    assert responses.response_type.value == "openai_responses_completion"
    assert anthropic.response_type == ChatResponseType.ANTHROPIC_COMPLETION
    assert anthropic.response_type.value == "anthropic_completion"


async def test_stream_response_uses_rust_owned_async_stream_and_rejects_body_access() -> None:
    async def source():
        yield {"delta": "hello"}

    response = ChatResponse.openai_stream(source())

    assert response.response_type == ChatResponseType.OPENAI_STREAM
    assert [event async for event in response.stream] == [{"delta": "hello"}]
    with pytest.raises(AttributeError):
        _ = response.body
    with pytest.raises(ValueError):
        response.replace_body({"not": "allowed"})


def test_stream_replace_body_rejects_before_serializing_body() -> None:
    class ExplodingBody:
        def model_dump(self, **kwargs: object) -> object:
            raise RuntimeError("should not serialize streaming replacement")

    response = ChatResponse.openai_stream(object())

    with pytest.raises(ValueError, match="streaming ChatResponse"):
        response.replace_body(ExplodingBody())


async def test_stream_response_supports_taps_maps_and_completion_callbacks() -> None:
    async def source():
        yield {"index": 0}
        yield {"index": 1}

    tapped: list[dict[str, int]] = []
    completed = False

    async def tap(event: dict[str, int]) -> None:
        tapped.append(dict(event))

    async def map_event(event: dict[str, int]) -> dict[str, int]:
        return {"index": event["index"] + 10}

    async def on_complete() -> None:
        nonlocal completed
        completed = True

    response = ChatResponse.openai_stream(source())
    stream = response.stream.tap(tap).map(map_event).on_complete(on_complete)

    assert [event async for event in stream] == [{"index": 10}, {"index": 11}]
    assert tapped == [{"index": 0}, {"index": 1}]
    assert completed is True

    with pytest.raises(RuntimeError, match="already been consumed"):
        _ = [event async for event in stream]


def test_proxy_context_is_rust_owned_with_rust_metadata_mapping() -> None:
    metadata = {"request_id": "client-visible", "nested": {"value": 1}}

    ctx = ProxyContext(metadata=metadata, request_id="rust-request")

    assert ctx.request_id == "rust-request"
    assert isinstance(ctx.metadata, ProxyMetadata)
    assert ctx.metadata is ctx.metadata
    assert ctx.metadata == metadata
    assert ctx.metadata is not metadata

    ctx.metadata["new"] = "value"
    ctx.metadata.setdefault("order", []).append("first")
    ctx.metadata.update({"updated": True})
    ctx.selected_model = "model-a"
    ctx.selected_target = "target-a"
    ctx.inbound_format = ChatRequestType.OPENAI_CHAT
    ctx.backend_call_latency_ms = 42.5

    assert ctx.metadata["new"] == "value"
    assert ctx.metadata["order"] == ["first"]
    assert ctx.metadata.get("missing", "fallback") == "fallback"
    assert ctx.metadata.copy()["updated"] is True
    assert sorted(ctx.metadata.keys()) == ["nested", "new", "order", "request_id", "updated"]
    del ctx.metadata["updated"]
    assert "updated" not in ctx.metadata
    assert ctx.selected_model == "model-a"
    assert ctx.selected_target == "target-a"
    assert ctx.inbound_format == ChatRequestType.OPENAI_CHAT
    assert ctx.backend_call_latency_ms == 42.5

    ctx.backend_call_latency_ms = None
    assert ctx.backend_call_latency_ms is None


def test_proxy_context_records_opaque_routing_trace_events() -> None:
    ctx = ProxyContext(request_id="request-1")

    evaluation = ctx.record_routing_event(
        "llm_classifier.output", {"recommended_tier": "complex"}
    )
    decision = ctx.record_routing_event("llm_classifier.decision", {"tier": "strong"})

    assert evaluation["sequence"] == 0
    assert decision["sequence"] == 1
    assert ctx.routing_trace == {
        "request_id": "request-1",
        "events": [evaluation, decision],
    }


def test_proxy_context_evicted_targets_are_mutable_from_python() -> None:
    ctx = ProxyContext()

    assert ctx.evicted_targets is None
    ctx.evicted_targets = ["weak", "strong"]

    assert ctx.evicted_targets == ["strong", "weak"]

    ctx.evicted_targets = None
    assert ctx.evicted_targets is None
    with pytest.raises(ValueError, match="invalid evicted target"):
        ctx.evicted_targets = [" "]


def test_provider_stream_adapters_are_rust_chat_response_stream_aliases() -> None:
    assert ResponseStream is ChatResponseStream
    assert ResponsesApiStream is ChatResponseStream
    assert AnthropicResponseStream is ChatResponseStream


def test_backend_role_class_is_rust_owned_public_export() -> None:
    from switchyard.lib.roles import LLMBackend as PublicLLMBackend

    assert PublicLLMBackend is LLMBackend


async def test_request_response_components_are_method_based() -> None:
    class Passthrough:
        async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
            return request

    processor = Passthrough()
    request = ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})

    assert await processor.process(ProxyContext(), request) is request


def test_replace_body_preserves_response_type() -> None:
    response = ChatResponse.anthropic_completion({"model": "old"})

    response.replace_body({"model": "new", "content": []})

    assert response.response_type == ChatResponseType.ANTHROPIC_COMPLETION
    assert response.body == {"model": "new", "content": []}
