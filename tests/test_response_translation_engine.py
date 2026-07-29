# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified TranslationEngine response wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import Response as OpenAIResponse

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard_rust.core import (
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)
from switchyard_rust.translation import TranslationEngine

E = TranslationEngine()


def _make_completion(content: str = "Hello!", model: str = "gpt-4o") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_anthropic_message(content: str = "Hi there", model: str = "claude-sonnet-4-20250514"):
    from anthropic.types import Message, TextBlock, Usage

    return Message(
        id="msg_test123",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text=content)],
        model=model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _make_responses_api_response(
    *, content: str = "Hello from Responses", model: str = "gpt-4o",
) -> OpenAIResponse:
    return OpenAIResponse(
        id="resp_test",
        created_at=1700000000,
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_test",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ],
        tool_choice="auto",
        tools=[],
        status="completed",
        parallel_tool_calls=True,
        text={"format": {"type": "text"}},
    )


# =========================================================================
# to_openai_chat
# =========================================================================


class TestToOpenAIChat:
    def test_completion_passthrough(self):
        """OpenAI completion responses pass through unchanged."""
        resp = ChatResponse.openai_completion(_make_completion())
        result = E.response_to(ChatRequestType.OPENAI_CHAT, resp)
        assert result is resp

    def test_streaming_passthrough(self):
        """OpenAI streaming responses pass through unchanged."""
        stream = MagicMock(spec=ResponseStream)
        resp = ChatResponse.openai_stream(stream)
        result = E.response_to(ChatRequestType.OPENAI_CHAT, resp)
        assert result is resp

    def test_anthropic_to_openai(self):
        """Anthropic responses are converted to OpenAI completion responses."""
        resp = ChatResponse.anthropic_completion(_make_anthropic_message(content="Hello world"))
        result = E.response_to(ChatRequestType.OPENAI_CHAT, resp)

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        assert result.body["choices"][0]["message"]["content"] == "Hello world"
        assert result.body["choices"][0]["finish_reason"] == "stop"

    def test_responses_api_to_openai(self):
        """Responses API responses are converted to OpenAI completion responses."""
        resp = ChatResponse.openai_responses_completion(
            _make_responses_api_response(content="Hello from Responses"),
        )
        result = E.response_to(ChatRequestType.OPENAI_CHAT, resp)

        assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)
        assert result.body["model"] == "gpt-4o"
        assert result.body["choices"][0]["message"]["content"] == "Hello from Responses"
        assert result.body["choices"][0]["finish_reason"] == "stop"


# =========================================================================
# to_anthropic
# =========================================================================


class TestToAnthropic:
    def test_anthropic_passthrough(self):
        """Anthropic completion responses pass through unchanged."""
        resp = ChatResponse.anthropic_completion(_make_anthropic_message())
        result = E.response_to(ChatRequestType.ANTHROPIC, resp)
        assert result is resp

    def test_anthropic_streaming_passthrough(self):
        """Anthropic streaming responses pass through unchanged."""
        from switchyard.lib.chat_response.anthropic import AnthropicResponseStream

        stream = MagicMock(spec=AnthropicResponseStream)
        resp = ChatResponse.anthropic_stream(stream)
        result = E.response_to(ChatRequestType.ANTHROPIC, resp)
        assert result is resp

    def test_openai_to_anthropic(self):
        """OpenAI completion responses are converted to Anthropic responses."""
        resp = ChatResponse.openai_completion(_make_completion(content="Hello world"))
        result = E.response_to(ChatRequestType.ANTHROPIC, resp)

        assert response_type_matches(result, ChatResponseType.ANTHROPIC_COMPLETION)
        body = result.body
        assert body["role"] == "assistant"
        assert body["stop_reason"] == "end_turn"
        assert len(body["content"]) == 1
        assert body["content"][0]["text"] == "Hello world"

    def test_responses_api_to_anthropic(self):
        """Responses API responses are converted to Anthropic responses."""
        resp = ChatResponse.openai_responses_completion(_make_responses_api_response(content="Hello world"))
        result = E.response_to(ChatRequestType.ANTHROPIC, resp)

        assert response_type_matches(result, ChatResponseType.ANTHROPIC_COMPLETION)
        assert result.body["content"][0]["text"] == "Hello world"


# =========================================================================
# to_responses
# =========================================================================


class TestToResponses:
    def test_responses_passthrough(self):
        """Responses API completion responses pass through unchanged."""
        resp = ChatResponse.openai_responses_completion(_make_responses_api_response())
        result = E.response_to(
            ChatRequestType.OPENAI_RESPONSES,
            resp,
            original_body={"model": "gpt-4o"},
        )
        assert result is resp

    def test_responses_streaming_passthrough(self):
        """Responses API streaming responses pass through unchanged."""
        from switchyard.lib.chat_response.openai_responses import ResponsesApiStream

        resp = ChatResponse.openai_responses_stream(ResponsesApiStream(_async_iter([])))
        result = E.response_to(
            ChatRequestType.OPENAI_RESPONSES,
            resp,
            original_body={"model": "gpt-4o"},
        )
        assert result is resp

    def test_openai_to_responses(self):
        """OpenAI completion responses are converted to Responses API responses."""
        resp = ChatResponse.openai_completion(_make_completion(content="Hello!"))
        result = E.response_to(
            ChatRequestType.OPENAI_RESPONSES,
            resp,
            original_body={"model": "gpt-4o", "input": "Hi"},
        )

        assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_COMPLETION)
        body = result.body
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert len(body["output"]) == 1
        assert body["output"][0]["type"] == "message"

    def test_anthropic_to_responses(self):
        """Anthropic responses are converted to Responses API responses."""
        resp = ChatResponse.anthropic_completion(_make_anthropic_message())
        result = E.response_to(
            ChatRequestType.OPENAI_RESPONSES,
            resp,
            original_body={"model": "gpt-4o"},
        )

        assert response_type_matches(result, ChatResponseType.OPENAI_RESPONSES_COMPLETION)
        assert result.body["output"][0]["content"][0]["text"] == "Hi there"


# =========================================================================
# stream_openai_to_anthropic — usage propagation
# =========================================================================


def _chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Build a minimal OpenAI-shaped streaming chunk."""
    delta = SimpleNamespace(
        content=content,
        reasoning=None,
        reasoning_content=reasoning_content,
        tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    chunk = SimpleNamespace(choices=[choice])
    if usage is not None:
        chunk.usage = usage
    return chunk


async def _async_iter(items: list) -> AsyncIterator:
    for item in items:
        yield item


async def _collect_events(chunks: list) -> list[dict]:
    events: list[dict] = []
    async for ev in E.translate_stream(
        ChatRequestType.OPENAI_CHAT,
        ChatRequestType.ANTHROPIC,
        _async_iter(chunks),
        model="test-model",
    ):
        events.append(ev)
    return events


def _find_event(events: list[dict], event_type: str) -> dict:
    return next(ev for ev in events if ev["type"] == event_type)


class TestStreamOpenAIToAnthropicUsage:
    @pytest.mark.asyncio
    async def test_backend_usage_preferred_over_heuristic(self):
        """Real usage from the backend replaces the per-delta +1 counter."""
        usage = SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=42,
            total_tokens=542,
            prompt_tokens_details=None,
        )
        chunks = [
            _chunk(content="Hello"),
            _chunk(content=" world", usage=usage),
            _chunk(finish_reason="stop"),
        ]
        events = await _collect_events(chunks)
        msg_delta = _find_event(events, "message_delta")

        assert msg_delta["usage"]["input_tokens"] == 500
        assert msg_delta["usage"]["output_tokens"] == 42

    @pytest.mark.asyncio
    async def test_cached_tokens_propagated(self):
        """Anthropic usage splits uncached input from cache-read tokens."""
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=10,
            total_tokens=1010,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        )
        chunks = [_chunk(content="Hi", usage=usage)]
        events = await _collect_events(chunks)
        msg_delta = _find_event(events, "message_delta")

        assert msg_delta["usage"]["cache_read_input_tokens"] == 800
        assert msg_delta["usage"]["input_tokens"] == 200

    @pytest.mark.asyncio
    async def test_heuristic_fallback_when_no_backend_usage(self):
        """Without backend usage, output_tokens falls back to delta counting."""
        chunks = [
            _chunk(content="a"),
            _chunk(content="b"),
            _chunk(content="c"),
            _chunk(finish_reason="stop"),
        ]
        events = await _collect_events(chunks)
        msg_delta = _find_event(events, "message_delta")

        assert msg_delta["usage"]["output_tokens"] == 3
        assert "input_tokens" not in msg_delta["usage"]
        assert "cache_read_input_tokens" not in msg_delta["usage"]

    @pytest.mark.asyncio
    async def test_reasoning_stream_events_validate_against_anthropic_sdk(self):
        from anthropic.types import (  # noqa: PLC0415
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
        )

        events = await _collect_events([
            _chunk(reasoning_content="private"),
            _chunk(content="visible"),
            _chunk(finish_reason="stop"),
        ])

        thinking_start = next(
            event for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "thinking"
        )
        signature_delta = next(
            event for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "signature_delta"
        )
        assert thinking_start["content_block"]["signature"] == ""
        assert signature_delta["delta"]["signature"] == ""
        RawContentBlockStartEvent.model_validate(thinking_start)
        RawContentBlockDeltaEvent.model_validate(signature_delta)


# ---------------------------------------------------------------------------
# Raw SSE frame parsing for cross-format stream translation
# ---------------------------------------------------------------------------


class TestSseFramePayloads:
    """``sse_frame_payloads`` feeds verbatim-passthrough frame strings back
    into the translator; the SSE grammar corners must parse correctly."""

    def test_single_frame_with_event_name(self):
        from switchyard_rust.translation import sse_frame_payloads

        payloads = sse_frame_payloads(
            'event: response.created\ndata: {"type":"response.created"}\n\n'
        )
        assert payloads == [{"type": "response.created"}]

    def test_multi_data_lines_join_with_newline(self):
        from switchyard_rust.translation import sse_frame_payloads

        payloads = sse_frame_payloads('data: {"a":\ndata:  1}\n\n')
        assert payloads == [{"a": 1}]

    def test_comment_done_and_non_json_frames_yield_nothing(self):
        from switchyard_rust.translation import sse_frame_payloads

        assert sse_frame_payloads(": keep-alive\n\n") == []
        assert sse_frame_payloads("data: [DONE]\n\n") == []
        assert sse_frame_payloads("data: not-json\n\n") == []

    def test_multiple_frames_in_one_string_parse_in_order(self):
        from switchyard_rust.translation import sse_frame_payloads

        payloads = sse_frame_payloads('data: {"a":1}\n\ndata: {"b":2}\n\n')
        assert payloads == [{"a": 1}, {"b": 2}]


async def test_translate_stream_accepts_raw_sse_frame_strings():
    """Cross-format translation parses raw Responses frames into chat chunks."""

    async def _frames() -> AsyncIterator[str]:
        yield (
            'event: response.created\n'
            'data: {"type":"response.created","response":{"id":"r1","model":"m"}}\n\n'
        )
        yield (
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
        )
        yield ": keep-alive\n\n"
        yield (
            'event: response.completed\n'
            'data: {"type":"response.completed","response":{"id":"r1","usage":'
            '{"input_tokens":1,"output_tokens":1}}}\n\n'
        )

    chunks = [
        chunk
        async for chunk in E.translate_stream(
            "openai_responses", "openai_chat", _frames(), model="m",
        )
    ]

    deltas = [
        choice.get("delta", {}).get("content")
        for chunk in chunks
        if isinstance(chunk, dict)
        for choice in chunk.get("choices", [])
    ]
    assert "hello" in deltas
