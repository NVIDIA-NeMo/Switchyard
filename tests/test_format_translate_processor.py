# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.chat_response.openai_responses import ResponsesApiStream
from switchyard.lib.processors.format_translate import (
    FormatTranslateResponseProcessor,
    ModelFormatLookupProcessor,
    StampOriginalFormatProcessor,
    TranslateConfig,
)
from switchyard.lib.proxy_context import (
    CTX_ORIGINAL_FORMAT,
    CTX_ORIGINAL_REQUEST,
    CTX_PROXY_ACTUAL_MODEL,
    CTX_TARGET_FORMAT,
    ProxyContext,
)
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    request_type_value,
    response_type_matches,
)


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _chat_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "backend-model",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _chat_tool_chunk(
    *,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    function: dict[str, str] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "backend-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_search",
                                "type": "function",
                                "function": function,
                            }
                        ]
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _anthropic_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-upstream",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]


def _anthropic_tool_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_tool",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-upstream",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_weather",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":"SF"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]


def _responses_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "response.created",
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1700000000,
                "status": "in_progress",
                "model": "responses-upstream",
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "responses-upstream",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
        },
    ]


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


def _frame_data(frame: str) -> dict[str, Any]:
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


async def test_model_format_lookup_prefers_rust_selected_model() -> None:
    processor = ModelFormatLookupProcessor(
        TranslateConfig(models=[
            {"model": "rust-selected", "backend_format": "anthropic"},
            {"model": "legacy-selected", "backend_format": "openai"},
            {"model": "responses-selected", "backend_format": "responses"},
        ])
    )
    ctx = ProxyContext(metadata={CTX_PROXY_ACTUAL_MODEL: "legacy-selected"})
    ctx.selected_model = "rust-selected"

    await processor.process(
        ctx,
        ChatRequest.openai_chat({"model": "client-model", "messages": []}),
    )

    assert ctx.metadata[CTX_TARGET_FORMAT] == ChatRequestType.ANTHROPIC


async def test_model_format_lookup_preserves_responses_target() -> None:
    processor = ModelFormatLookupProcessor(
        TranslateConfig(models=[{"model": "responses-selected", "backend_format": "responses"}])
    )
    ctx = ProxyContext()
    ctx.selected_model = "responses-selected"

    await processor.process(
        ctx,
        ChatRequest.openai_responses({"model": "client-model", "input": "hi"}),
    )

    assert ctx.metadata[CTX_TARGET_FORMAT] == ChatRequestType.OPENAI_RESPONSES


async def test_model_format_lookup_falls_back_to_legacy_metadata() -> None:
    processor = ModelFormatLookupProcessor(
        TranslateConfig(models=[{"model": "legacy-selected", "backend_format": "openai"}])
    )
    ctx = ProxyContext(metadata={CTX_PROXY_ACTUAL_MODEL: "legacy-selected"})

    await processor.process(
        ctx,
        ChatRequest.anthropic({"model": "client-model", "messages": []}),
    )

    assert ctx.metadata[CTX_TARGET_FORMAT] == ChatRequestType.OPENAI_CHAT


@pytest.mark.asyncio
async def test_stamp_original_format_preserves_original_body_snapshot() -> None:
    body = {"model": "gpt-client", "messages": [{"role": "user", "content": "hi"}]}
    ctx = ProxyContext()

    await StampOriginalFormatProcessor().process(ctx, ChatRequest.openai_chat(body))
    body["messages"][0]["content"] = "mutated"

    assert request_type_value(ctx.metadata[CTX_ORIGINAL_FORMAT]) == "openai_chat"
    assert ctx.metadata[CTX_ORIGINAL_REQUEST] == {
        "model": "gpt-client",
        "messages": [{"role": "user", "content": "hi"}],
    }


@pytest.mark.asyncio
async def test_streaming_openai_response_translates_back_to_anthropic() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.ANTHROPIC,
            CTX_ORIGINAL_REQUEST: {"model": "claude-client"},
        }
    )
    ctx.selected_model = "served/model"
    response = ChatResponse.openai_stream(
        ResponseStream(
            _aiter(
                [
                    _chat_chunk(content="hel"),
                    _chat_chunk(content="lo", finish_reason="stop"),
                ]
            )
        )
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.ANTHROPIC_STREAM)
    events = await _collect(result.stream)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
    )
    assert events[0]["message"]["model"] == "served/model"
    assert text == "hello"
    assert [event["type"] for event in events].count("content_block_start") == 1


@pytest.mark.asyncio
async def test_streaming_openai_length_finish_translates_to_anthropic_max_tokens() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.ANTHROPIC,
            CTX_ORIGINAL_REQUEST: {"model": "claude-client"},
        }
    )
    response = ChatResponse.openai_stream(
        ResponseStream(_aiter([_chat_chunk(content="truncated", finish_reason="length")]))
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.ANTHROPIC_STREAM)
    events = await _collect(result.stream)
    message_delta = next(event for event in events if event["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_streaming_openai_response_translates_back_to_responses() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_RESPONSES,
            CTX_ORIGINAL_REQUEST: {"model": "responses-client", "input": "hi"},
        }
    )
    ctx.selected_model = "served/model"
    response = ChatResponse.openai_stream(
        ResponseStream(
            _aiter(
                [
                    _chat_chunk(content="hel"),
                    _chat_chunk(content="lo", finish_reason="stop"),
                ]
            )
        )
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_STREAM)
    frames = await _collect(result.stream)
    payloads = [_frame_data(frame) for frame in frames]
    assert payloads[0]["response"]["model"] == "served/model"
    assert (
        "".join(
            payload["delta"]
            for payload in payloads
            if payload["type"] == "response.output_text.delta"
        )
        == "hello"
    )
    assert payloads[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_streaming_openai_to_responses_keeps_unique_output_indexes() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_RESPONSES,
            CTX_ORIGINAL_REQUEST: {"model": "responses-client", "input": "search"},
        }
    )
    response = ChatResponse.openai_stream(
        ResponseStream(
            _aiter(
                [
                    _chat_tool_chunk(name="search", arguments='{"q":"x"}'),
                    _chat_chunk(content="Checking"),
                    _chat_chunk(finish_reason="tool_calls"),
                ]
            )
        )
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_STREAM)
    payloads = [_frame_data(frame) for frame in await _collect(result.stream)]
    added = [
        (payload["output_index"], payload["item"]["type"])
        for payload in payloads
        if payload["type"] == "response.output_item.added"
    ]
    assert added == [(0, "function_call"), (1, "message")]

    function_index = added[0][0]
    args_done = next(
        payload
        for payload in payloads
        if payload["type"] == "response.function_call_arguments.done"
    )
    assert args_done["output_index"] == function_index

    completed = payloads[-1]["response"]
    assert [item["type"] for item in completed["output"]] == [
        "function_call",
        "message",
    ]


@pytest.mark.asyncio
async def test_streaming_anthropic_response_translates_back_to_responses() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_RESPONSES,
            CTX_ORIGINAL_REQUEST: {"model": "responses-client", "input": "hi"},
        }
    )
    response = ChatResponse.anthropic_stream(
        AnthropicResponseStream(_aiter(_anthropic_events())),
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_STREAM)
    payloads = [_frame_data(frame) for frame in await _collect(result.stream)]
    assert (
        "".join(
            payload["delta"]
            for payload in payloads
            if payload["type"] == "response.output_text.delta"
        )
        == "hello"
    )
    assert payloads[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_streaming_anthropic_response_translates_back_to_openai() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_CHAT,
            CTX_ORIGINAL_REQUEST: {"model": "gpt-client"},
        }
    )
    response = ChatResponse.anthropic_stream(
        AnthropicResponseStream(_aiter(_anthropic_events())),
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_STREAM)
    chunks = [chunk.model_dump(exclude_none=True) for chunk in await _collect(result.stream)]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["model"] == "claude-upstream"
    assert chunks[1]["choices"][0]["delta"]["content"] == "hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens"] == 3
    assert chunks[-1]["usage"]["completion_tokens"] == 1


@pytest.mark.asyncio
async def test_streaming_anthropic_cache_usage_translates_back_to_openai() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_CHAT,
            CTX_ORIGINAL_REQUEST: {"model": "gpt-client"},
        }
    )
    events = _anthropic_events()
    events[0]["message"]["usage"] = {
        "input_tokens": 3,
        "cache_creation_input_tokens": 4,
        "cache_read_input_tokens": 2,
        "output_tokens": 0,
    }
    response = ChatResponse.anthropic_stream(AnthropicResponseStream(_aiter(events)))

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_STREAM)
    chunks = [chunk.model_dump(exclude_none=True) for chunk in await _collect(result.stream)]
    usage = chunks[-1]["usage"]
    assert usage["prompt_tokens"] == 9
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == 10
    assert usage["prompt_tokens_details"]["cached_tokens"] == 2
    assert usage["prompt_tokens_details"]["cache_creation_tokens"] == 4


@pytest.mark.asyncio
async def test_streaming_responses_response_translates_back_to_openai() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_CHAT,
            CTX_ORIGINAL_REQUEST: {"model": "gpt-client"},
        }
    )
    response = ChatResponse.openai_responses_stream(
        ResponsesApiStream(_aiter(_responses_events())),
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_STREAM)
    chunks = [chunk.model_dump(exclude_none=True) for chunk in await _collect(result.stream)]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"]["content"] == "hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens"] == 5
    assert chunks[-1]["usage"]["completion_tokens"] == 2


@pytest.mark.asyncio
async def test_streaming_responses_response_translates_back_to_anthropic() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.ANTHROPIC,
            CTX_ORIGINAL_REQUEST: {"model": "claude-client"},
        }
    )
    ctx.selected_model = "served/model"
    response = ChatResponse.openai_responses_stream(
        ResponsesApiStream(_aiter(_responses_events())),
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.ANTHROPIC_STREAM)
    events = await _collect(result.stream)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
    )
    assert events[0]["message"]["model"] == "served/model"
    assert text == "hello"


@pytest.mark.asyncio
async def test_streaming_anthropic_tool_use_translates_back_to_openai_tool_call() -> None:
    ctx = ProxyContext(
        metadata={
            CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_CHAT,
            CTX_ORIGINAL_REQUEST: {"model": "gpt-client"},
        }
    )
    response = ChatResponse.anthropic_stream(
        AnthropicResponseStream(_aiter(_anthropic_tool_events())),
    )

    result = await FormatTranslateResponseProcessor().process(ctx, response)

    assert response_type_matches(result, ChatResponseType.OPENAI_STREAM)
    chunks = [chunk.model_dump(exclude_none=True) for chunk in await _collect(result.stream)]
    first_tool_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    args_delta = chunks[2]["choices"][0]["delta"]["tool_calls"][0]
    assert first_tool_delta["id"] == "toolu_weather"
    assert first_tool_delta["function"]["name"] == "get_weather"
    assert args_delta["function"]["arguments"] == '{"city":"SF"}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
