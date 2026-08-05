# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden translation tests for the token-capture flow.

Fixed request/response fixtures, asserted structurally against their validated
translated forms — covering the two translation boundaries token capture
relies on:

* inbound harness request (Anthropic / Responses) → the OpenAI Chat body the
  vLLM backend receives, and
* the synthesized OpenAI chunk stream (built from a captured buffered vLLM
  completion) → the client's native stream events.

The fixtures are plain dicts at module top so more validated pairs can be
appended without touching the test logic. Assertions are structural (event
types, block shapes, exact field values) rather than substring checks, so a
translation that emits the right strings in the wrong shape still fails.
"""

from __future__ import annotations

import json

from switchyard.lib.processors.rl_logging_request_processor import RlLoggingRequestProcessor
from switchyard.lib.processors.token_capture_request_processor import (
    TokenCaptureRequestProcessor,
)
from switchyard.lib.processors.token_capture_response_processor import (
    TokenCaptureResponseProcessor,
)
from switchyard.lib.request_metadata import CTX_REQUEST_METADATA
from switchyard_rust.components import RequestMetadata
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse, ProxyContext
from switchyard_rust.translation import TranslationEngine

_SESSION = "claude-1700000000000-abc12345"

# --- Fixed inbound requests (what harnesses send) ----------------------------

ANTHROPIC_REQUEST = {
    "model": "tito-model",
    "max_tokens": 64,
    "stream": True,
    "system": "be brief",
    "messages": [{"role": "user", "content": "say hello"}],
}

# Turn 2 of a tool-using conversation: the shape capture sees on every call
# after the first (assistant tool_use + tool_result history).
ANTHROPIC_TOOL_HISTORY_REQUEST = {
    "model": "tito-model",
    "max_tokens": 64,
    "messages": [
        {"role": "user", "content": "create hello.py"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Creating the file now."},
                {
                    "type": "tool_use",
                    "id": "call_abc123",
                    "name": "write_file",
                    "input": {"path": "hello.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_abc123",
                    "content": "file written",
                }
            ],
        },
    ],
}

RESPONSES_REQUEST = {
    "model": "tito-model",
    "stream": True,
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "say hello"}],
        }
    ],
}

RESPONSES_TOOL_HISTORY_REQUEST = {
    "model": "tito-model",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "create hello.py"}],
        },
        {
            "type": "function_call",
            "call_id": "call_abc123",
            "name": "write_file",
            "arguments": '{"path": "hello.py"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_abc123",
            "output": "file written",
        },
    ],
}

# --- Fixed captured vLLM completion (what the backend returns) ---------------

VLLM_COMPLETION_WITH_TOOL_CALL = {
    "id": "chatcmpl-golden",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "Qwen/Qwen3-0.6B",
    "prompt_token_ids": [1, 2, 3],
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Creating the file now.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "hello.py"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "token_ids": [7, 8, 9],
            "logprobs": {
                "content": [
                    {"token": "a", "logprob": -0.1},
                    {"token": "b", "logprob": -0.2},
                    {"token": "c", "logprob": -0.3},
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
}


def _capture_ctx() -> ProxyContext:
    ctx = ProxyContext()
    ctx.metadata[CTX_REQUEST_METADATA] = RequestMetadata.from_headers(
        {"proxy_x_session_id": _SESSION}
    )
    return ctx


async def _synthesized_stream(tmp_path, inbound: ChatRequest) -> ChatResponse:
    """Run the capture pair the way the chain does and return the synthetic stream."""
    ctx = _capture_ctx()
    await RlLoggingRequestProcessor().process(ctx, inbound)
    await TokenCaptureRequestProcessor().process(ctx, inbound)
    return await TokenCaptureResponseProcessor(tmp_path).process(
        ctx, ChatResponse.openai_completion(dict(VLLM_COMPLETION_WITH_TOOL_CALL))
    )


# --- Request direction: harness format → upstream OpenAI Chat body -----------


def test_anthropic_request_translates_to_openai_chat() -> None:
    engine = TranslationEngine()
    translated = engine.request_to(
        ChatRequestType.OPENAI_CHAT, ChatRequest.anthropic(dict(ANTHROPIC_REQUEST))
    )
    body = dict(translated.body)

    assert body["model"] == "tito-model"
    assert body["max_completion_tokens"] == 64
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == "be brief"
    assert body["messages"][1]["content"] == "say hello"


def test_anthropic_tool_history_translates_to_openai_chat() -> None:
    engine = TranslationEngine()
    translated = engine.request_to(
        ChatRequestType.OPENAI_CHAT,
        ChatRequest.anthropic(dict(ANTHROPIC_TOOL_HISTORY_REQUEST)),
    )
    messages = dict(translated.body)["messages"]

    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool"]

    assistant = messages[1]
    calls = assistant["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_abc123"
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "hello.py"}

    tool = messages[2]
    # The tool result must round-trip keyed by the SAME call id.
    assert tool["tool_call_id"] == "call_abc123"
    assert "file written" in json.dumps(tool["content"])


def test_responses_request_translates_to_openai_chat() -> None:
    engine = TranslationEngine()
    translated = engine.request_to(
        ChatRequestType.OPENAI_CHAT, ChatRequest.openai_responses(dict(RESPONSES_REQUEST))
    )
    body = dict(translated.body)

    assert body["model"] == "tito-model"
    user_messages = [m for m in body["messages"] if m["role"] == "user"]
    assert len(user_messages) == 1
    content = user_messages[0]["content"]
    # Typed input_text parts must arrive as clean text, not serialized JSON.
    text = content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content
    )
    assert text == "say hello"
    assert "{" not in text


def test_responses_tool_history_translates_to_openai_chat() -> None:
    engine = TranslationEngine()
    translated = engine.request_to(
        ChatRequestType.OPENAI_CHAT,
        ChatRequest.openai_responses(dict(RESPONSES_TOOL_HISTORY_REQUEST)),
    )
    messages = dict(translated.body)["messages"]

    assistant = next(m for m in messages if m["role"] == "assistant")
    calls = assistant["tool_calls"]
    assert calls[0]["id"] == "call_abc123"
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "hello.py"}

    tool = next(m for m in messages if m["role"] == "tool")
    assert tool["tool_call_id"] == "call_abc123"
    assert "file written" in json.dumps(tool["content"])


# --- Stream direction: synthetic OpenAI chunks → client-native events --------


async def test_synthetic_stream_translates_to_anthropic_events(tmp_path) -> None:
    inbound = ChatRequest.anthropic(dict(ANTHROPIC_REQUEST))
    out = await _synthesized_stream(tmp_path, inbound)

    events = [event async for event in TranslationEngine().stream_for_request(inbound, out)]
    types = [e["type"] for e in events]

    # Anthropic stream grammar: proper frame, in order.
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"

    # Text arrives as a text block with the exact delta — and ONLY the text:
    # a tool call leaking into a text delta (the JSON-blob failure class)
    # must fail here.
    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
    ]
    assert text_deltas == ["Creating the file now."]

    # The tool call arrives as a structured tool_use block with the same id.
    tool_starts = [
        e["content_block"]
        for e in events
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["id"] == "call_abc123"
    assert tool_starts[0]["name"] == "write_file"
    json_deltas = [
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
    ]
    assert json.loads("".join(json_deltas)) == {"path": "hello.py"}

    # Terminal frame: tool_use stop reason and usage from the captured body.
    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"]["output_tokens"] == 6 - 3


def _parse_sse(events: list[str]) -> list[dict]:
    parsed = []
    for raw in events:
        for line in raw.splitlines():
            if line.startswith("data: "):
                parsed.append(json.loads(line[len("data: "):]))
    return parsed


async def test_synthetic_stream_translates_to_responses_events(tmp_path) -> None:
    inbound = ChatRequest.openai_responses(dict(RESPONSES_REQUEST))
    out = await _synthesized_stream(tmp_path, inbound)

    raw = [event async for event in TranslationEngine().stream_for_request(inbound, out)]
    events = _parse_sse([str(e) for e in raw])
    types = [e["type"] for e in events]

    assert types[0] == "response.created"
    assert types[-1] == "response.completed"

    # Text arrives only through output_text deltas, with the exact payload.
    text_deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    assert text_deltas == ["Creating the file now."]

    # The tool call arrives as a function_call output item with the same
    # call id, plus its arguments through the dedicated delta channel.
    fc_items = [
        e["item"]
        for e in events
        if e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"
    ]
    assert len(fc_items) == 1
    assert fc_items[0]["call_id"] == "call_abc123"
    assert fc_items[0]["name"] == "write_file"
    args_done = next(
        e for e in events if e["type"] == "response.function_call_arguments.done"
    )
    assert json.loads(args_done["arguments"]) == {"path": "hello.py"}
