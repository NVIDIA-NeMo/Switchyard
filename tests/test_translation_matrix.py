# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-format translation matrix: every source format to every target.

Each scenario builds semantically equivalent request/response/stream payloads
per wire format, translates them across every (source, target) pair, and
asserts marker survival plus target-specific structural invariants. Markers
are unique strings (and a distinctive base64 payload for attachments), so the
assertions hold regardless of how a target codec arranges the content.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from switchyard_rust.translation import TranslationEngine

ENGINE = TranslationEngine()

OPENAI = "openai_chat"
ANTHROPIC = "anthropic_messages"
GEMINI = "gemini"
FORMATS = [OPENAI, ANTHROPIC, GEMINI]

SYSTEM_MARKER = "SYSTEM_MARKER_be_terse"
TURN1_MARKER = "TURN1_MARKER_first_question"
REPLY_MARKER = "REPLY_MARKER_assistant_answer"
TURN2_MARKER = "TURN2_MARKER_follow_up"
RESULT_MARKER = "RESULT_MARKER_22C_sunny"
# base64 of "image_bytes_marker" — a distinctive payload to trace attachments.
IMAGE_B64 = "aW1hZ2VfYnl0ZXNfbWFya2Vy"
PDF_B64 = "cGRmX2J5dGVzX21hcmtlcg=="
TOOL_NAME = "get_weather"
TOOL_ARG_VALUE = "Paris"


def translate(source: str, target: str, body: dict[str, Any]) -> dict[str, Any]:
    return ENGINE.translate_request(source, target, body)


def dumps(body: dict[str, Any]) -> str:
    return json.dumps(body)


def assert_structurally_valid(target: str, body: dict[str, Any]) -> None:
    """Spot-check the encoded body satisfies the target API's hard rules."""
    if target == OPENAI:
        roles = {m["role"] for m in body["messages"]}
        assert roles <= {"system", "developer", "user", "assistant", "tool"}
        for message in body["messages"]:
            if message["role"] == "tool":
                assert message.get("tool_call_id"), "tool message must correlate to a call"
    elif target == ANTHROPIC:
        assert "max_tokens" in body
        for message in body["messages"]:
            assert message["role"] in {"user", "assistant"}
    elif target == GEMINI:
        for content in body["contents"]:
            assert content["role"] in {"user", "model"}
            assert content["parts"], "Gemini rejects empty parts arrays"


# ---------------------------------------------------------------------------
# Request scenarios
# ---------------------------------------------------------------------------


def _openai_image_part(data: str = IMAGE_B64) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _anthropic_image_block(data: str = IMAGE_B64) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _gemini_image_part(data: str = IMAGE_B64) -> dict[str, Any]:
    return {"inlineData": {"mimeType": "image/png", "data": data}}


def text_system_bodies() -> dict[str, dict[str, Any]]:
    return {
        OPENAI: {
            "model": "m",
            "messages": [
                {"role": "system", "content": SYSTEM_MARKER},
                {"role": "user", "content": TURN1_MARKER},
                {"role": "assistant", "content": REPLY_MARKER},
                {"role": "user", "content": TURN2_MARKER},
            ],
        },
        ANTHROPIC: {
            "model": "m",
            "max_tokens": 256,
            "system": SYSTEM_MARKER,
            "messages": [
                {"role": "user", "content": TURN1_MARKER},
                {"role": "assistant", "content": REPLY_MARKER},
                {"role": "user", "content": TURN2_MARKER},
            ],
        },
        GEMINI: {
            "model": "m",
            "systemInstruction": {"parts": [{"text": SYSTEM_MARKER}]},
            "contents": [
                {"role": "user", "parts": [{"text": TURN1_MARKER}]},
                {"role": "model", "parts": [{"text": REPLY_MARKER}]},
                {"role": "user", "parts": [{"text": TURN2_MARKER}]},
            ],
        },
    }


def image_start_bodies() -> dict[str, dict[str, Any]]:
    return {
        OPENAI: {
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    _openai_image_part(),
                    {"type": "text", "text": TURN1_MARKER},
                ],
            }],
        },
        ANTHROPIC: {
            "model": "m",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    _anthropic_image_block(),
                    {"type": "text", "text": TURN1_MARKER},
                ],
            }],
        },
        GEMINI: {
            "model": "m",
            "contents": [{
                "role": "user",
                "parts": [_gemini_image_part(), {"text": TURN1_MARKER}],
            }],
        },
    }


def image_mid_conversation_bodies() -> dict[str, dict[str, Any]]:
    return {
        OPENAI: {
            "model": "m",
            "messages": [
                {"role": "user", "content": TURN1_MARKER},
                {"role": "assistant", "content": REPLY_MARKER},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TURN2_MARKER},
                        _openai_image_part(),
                    ],
                },
            ],
        },
        ANTHROPIC: {
            "model": "m",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": TURN1_MARKER},
                {"role": "assistant", "content": REPLY_MARKER},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TURN2_MARKER},
                        _anthropic_image_block(),
                    ],
                },
            ],
        },
        GEMINI: {
            "model": "m",
            "contents": [
                {"role": "user", "parts": [{"text": TURN1_MARKER}]},
                {"role": "model", "parts": [{"text": REPLY_MARKER}]},
                {
                    "role": "user",
                    "parts": [{"text": TURN2_MARKER}, _gemini_image_part()],
                },
            ],
        },
    }


def _openai_tools() -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def tools_round_trip_bodies() -> dict[str, dict[str, Any]]:
    return {
        OPENAI: {
            "model": "m",
            "messages": [
                {"role": "user", "content": TURN1_MARKER},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": json.dumps({"city": TOOL_ARG_VALUE}),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": RESULT_MARKER},
                {"role": "user", "content": TURN2_MARKER},
            ],
            "tools": _openai_tools(),
        },
        ANTHROPIC: {
            "model": "m",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": TURN1_MARKER},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": TOOL_NAME,
                        "input": {"city": TOOL_ARG_VALUE},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": RESULT_MARKER,
                    }],
                },
                {"role": "user", "content": TURN2_MARKER},
            ],
            "tools": [{
                "name": TOOL_NAME,
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
        },
        GEMINI: {
            "model": "m",
            "contents": [
                {"role": "user", "parts": [{"text": TURN1_MARKER}]},
                {
                    "role": "model",
                    "parts": [{
                        "functionCall": {"name": TOOL_NAME, "args": {"city": TOOL_ARG_VALUE}},
                    }],
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": TOOL_NAME,
                            "response": {"parts": [{"text": RESULT_MARKER}]},
                        },
                    }],
                },
                {"role": "user", "parts": [{"text": TURN2_MARKER}]},
            ],
            "tools": [{
                "functionDeclarations": [{
                    "name": TOOL_NAME,
                    "description": "Get weather",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {"city": {"type": "STRING"}},
                        "required": ["city"],
                    },
                }],
            }],
        },
    }


def tool_result_image_bodies() -> dict[str, dict[str, Any]]:
    """Tool results carrying an image attachment (screenshot-style tools).

    OpenAI has no wire shape for this, so it only appears as a source for
    Anthropic and Gemini.
    """
    return {
        ANTHROPIC: {
            "model": "m",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": TURN1_MARKER},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "take_screenshot",
                        "input": {},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": RESULT_MARKER},
                            _anthropic_image_block(),
                        ],
                    }],
                },
            ],
        },
        GEMINI: {
            "model": "m",
            "contents": [
                {"role": "user", "parts": [{"text": TURN1_MARKER}]},
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "take_screenshot", "args": {}}}],
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "take_screenshot",
                            "response": {"parts": [
                                {"text": RESULT_MARKER},
                                _gemini_image_part(),
                            ]},
                        },
                    }],
                },
            ],
        },
    }


def pdf_attachment_bodies() -> dict[str, dict[str, Any]]:
    return {
        ANTHROPIC: {
            "model": "m",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": TURN1_MARKER},
                    {
                        "type": "document",
                        "source": {"type": "base64", "data": PDF_B64},
                    },
                ],
            }],
        },
        GEMINI: {
            "model": "m",
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": TURN1_MARKER},
                    {"inlineData": {"mimeType": "application/pdf", "data": PDF_B64}},
                ],
            }],
        },
    }


# ---------------------------------------------------------------------------
# Request matrix tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_text_and_system_survive_every_pair(source: str, target: str) -> None:
    body = translate(source, target, text_system_bodies()[source])
    encoded = dumps(body)
    for marker in (SYSTEM_MARKER, TURN1_MARKER, REPLY_MARKER, TURN2_MARKER):
        assert marker in encoded, f"{marker} lost translating {source} -> {target}"
    assert_structurally_valid(target, body)


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_image_at_conversation_start_survives_every_pair(source: str, target: str) -> None:
    body = translate(source, target, image_start_bodies()[source])
    encoded = dumps(body)
    assert TURN1_MARKER in encoded
    assert IMAGE_B64 in encoded, f"image payload lost translating {source} -> {target}"
    assert_structurally_valid(target, body)


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_image_mid_conversation_survives_every_pair(source: str, target: str) -> None:
    body = translate(source, target, image_mid_conversation_bodies()[source])
    encoded = dumps(body)
    for marker in (TURN1_MARKER, REPLY_MARKER, TURN2_MARKER):
        assert marker in encoded
    assert IMAGE_B64 in encoded, f"mid-conversation image lost {source} -> {target}"
    # The image must stay attached to the final user turn, not migrate.
    if target == OPENAI:
        assert IMAGE_B64 in dumps(body["messages"][-1])
    elif target == ANTHROPIC:
        assert IMAGE_B64 in dumps(body["messages"][-1])
    elif target == GEMINI:
        assert IMAGE_B64 in dumps(body["contents"][-1])
    assert_structurally_valid(target, body)


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_tool_round_trip_survives_every_pair(source: str, target: str) -> None:
    body = translate(source, target, tools_round_trip_bodies()[source])
    encoded = dumps(body)
    assert TOOL_NAME in encoded, f"tool declaration/call lost {source} -> {target}"
    assert TOOL_ARG_VALUE in encoded, f"tool arguments lost {source} -> {target}"
    assert RESULT_MARKER in encoded, f"tool result lost {source} -> {target}"
    assert TURN2_MARKER in encoded
    assert_structurally_valid(target, body)
    # Call/result correlation must hold in the target's own idiom.
    if target == OPENAI:
        calls = [
            call
            for message in body["messages"]
            for call in message.get("tool_calls") or []
        ]
        results = [m for m in body["messages"] if m["role"] == "tool"]
        assert calls and results
        assert results[0]["tool_call_id"] == calls[0]["id"]
    elif target == ANTHROPIC:
        uses = [
            block
            for message in body["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "tool_use"
        ]
        results = [
            block
            for message in body["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        assert uses and results
        assert results[0]["tool_use_id"] == uses[0]["id"]
    elif target == GEMINI:
        calls = [
            part["functionCall"]
            for content in body["contents"]
            for part in content["parts"]
            if "functionCall" in part
        ]
        responses = [
            part["functionResponse"]
            for content in body["contents"]
            for part in content["parts"]
            if "functionResponse" in part
        ]
        assert calls and responses
        assert responses[0]["name"] == calls[0]["name"]


@pytest.mark.parametrize("source", [ANTHROPIC, GEMINI])
@pytest.mark.parametrize("target", FORMATS)
def test_tool_result_image_attachment_every_pair(source: str, target: str) -> None:
    body = translate(source, target, tool_result_image_bodies()[source])
    encoded = dumps(body)
    assert RESULT_MARKER in encoded, f"tool result text lost {source} -> {target}"
    assert IMAGE_B64 in encoded, f"tool-result image lost {source} -> {target}"
    if target == OPENAI:
        # OpenAI tool messages are text-only: the payload must not be
        # stringified into the tool message; it is hoisted into a following
        # user message where vision models can read it.
        tool_messages = [m for m in body["messages"] if m["role"] == "tool"]
        assert tool_messages
        assert all(IMAGE_B64 not in dumps(m) for m in tool_messages)
        tool_index = body["messages"].index(tool_messages[-1])
        hoisted = body["messages"][tool_index + 1]
        assert hoisted["role"] == "user"
        assert IMAGE_B64 in dumps(hoisted)
    if target == ANTHROPIC:
        result_blocks = [
            block
            for message in body["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        assert any(
            isinstance(block.get("content"), list)
            and any(entry.get("type") == "image" for entry in block["content"])
            for block in result_blocks
        ), "tool-result image must be a typed image block, not stringified JSON"
    if target == GEMINI and source != GEMINI:
        # Hoisted to a sibling part of the same user turn: Gemini does not
        # read media nested inside the functionResponse payload. Same-format
        # traffic is preserved verbatim, so this only applies to translations.
        result_contents = [
            content
            for content in body["contents"]
            if any("functionResponse" in part for part in content["parts"])
        ]
        assert result_contents
        assert any(
            "inlineData" in part
            for content in result_contents
            for part in content["parts"]
        ), "tool-result image must be a sibling inlineData part"
    assert_structurally_valid(target, body)


@pytest.mark.parametrize("source", [ANTHROPIC, GEMINI])
@pytest.mark.parametrize("target", FORMATS)
def test_pdf_attachment_every_pair(source: str, target: str) -> None:
    body = translate(source, target, pdf_attachment_bodies()[source])
    encoded = dumps(body)
    assert TURN1_MARKER in encoded
    assert PDF_B64 in encoded, f"document payload lost {source} -> {target}"
    assert_structurally_valid(target, body)


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("intermediate", FORMATS)
def test_two_hop_translation_preserves_content(source: str, intermediate: str) -> None:
    """source -> intermediate -> source keeps text, images, and tool traffic."""
    first = translate(source, intermediate, tools_round_trip_bodies()[source])
    back = translate(intermediate, source, first)
    encoded = dumps(back)
    for marker in (TOOL_NAME, TOOL_ARG_VALUE, RESULT_MARKER, TURN2_MARKER):
        assert marker in encoded, f"{marker} lost on {source} -> {intermediate} -> {source}"
    assert_structurally_valid(source, back)

    first = translate(source, intermediate, image_mid_conversation_bodies()[source])
    back = translate(intermediate, source, first)
    assert IMAGE_B64 in dumps(back), f"image lost on {source} -> {intermediate} -> {source}"


# ---------------------------------------------------------------------------
# Response matrix
# ---------------------------------------------------------------------------

RESPONSE_TEXT_MARKER = "RESPONSE_TEXT_MARKER_final_answer"


def text_response_bodies() -> dict[str, dict[str, Any]]:
    return {
        OPENAI: {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": RESPONSE_TEXT_MARKER},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
        ANTHROPIC: {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": RESPONSE_TEXT_MARKER}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        },
        GEMINI: {
            "candidates": [{
                "content": {"parts": [{"text": RESPONSE_TEXT_MARKER}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "totalTokenCount": 18,
            },
            "modelVersion": "m",
            "responseId": "resp-1",
        },
    }


def tool_call_response_bodies() -> dict[str, dict[str, Any]]:
    return {
        OPENAI: {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": json.dumps({"city": TOOL_ARG_VALUE}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        },
        ANTHROPIC: {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": TOOL_NAME,
                "input": {"city": TOOL_ARG_VALUE},
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
        GEMINI: {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {"name": TOOL_NAME, "args": {"city": TOOL_ARG_VALUE}},
                    }],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
        },
    }


TOOLISH_STOP = {OPENAI: "tool_calls", ANTHROPIC: "tool_use", GEMINI: "STOP"}


def response_stop_value(target: str, body: dict[str, Any]) -> str:
    if target == OPENAI:
        return str(body["choices"][0]["finish_reason"])
    if target == ANTHROPIC:
        return str(body["stop_reason"])
    return str(body["candidates"][0]["finishReason"])


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_text_response_translates_every_pair(source: str, target: str) -> None:
    body = ENGINE.translate_response(source, target, text_response_bodies()[source])
    assert RESPONSE_TEXT_MARKER in dumps(body)


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_tool_call_response_translates_every_pair(source: str, target: str) -> None:
    body = ENGINE.translate_response(source, target, tool_call_response_bodies()[source])
    encoded = dumps(body)
    assert TOOL_NAME in encoded
    assert TOOL_ARG_VALUE in encoded
    assert response_stop_value(target, body) == TOOLISH_STOP[target], (
        f"tool-call stop reason wrong for {source} -> {target}"
    )


# ---------------------------------------------------------------------------
# Streaming matrix
# ---------------------------------------------------------------------------


def text_stream_events() -> dict[str, list[dict[str, Any]]]:
    return {
        OPENAI: [
            {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
            },
            {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
            },
        ],
        ANTHROPIC: [
            {"type": "message_start", "message": {"id": "msg_1", "model": "m", "usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ],
        GEMINI: [
            {
                "candidates": [{"content": {"parts": [{"text": "Hel"}], "role": "model"}}],
                "responseId": "resp-1",
                "modelVersion": "m",
            },
            {
                "candidates": [{
                    "content": {"parts": [{"text": "lo"}], "role": "model"},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2},
            },
        ],
    }


def tool_stream_events() -> dict[str, list[dict[str, Any]]]:
    arguments = json.dumps({"city": TOOL_ARG_VALUE})
    return {
        OPENAI: [
            {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": TOOL_NAME, "arguments": arguments[:9]},
                    }]},
                }],
            },
            {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": arguments[9:]},
                    }]},
                }],
            },
            {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ],
        ANTHROPIC: [
            {"type": "message_start", "message": {"id": "msg_1", "model": "m", "usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": TOOL_NAME, "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": arguments[:9]},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": arguments[9:]},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ],
        GEMINI: [
            {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {"name": TOOL_NAME, "args": {"city": TOOL_ARG_VALUE}},
                        }],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }],
                "responseId": "resp-1",
                "modelVersion": "m",
            },
        ],
    }


def run_stream(source: str, target: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from switchyard_rust.translation import _format_name

    translator = ENGINE._inner.stream(_format_name(source), _format_name(target), None, None)
    out: list[dict[str, Any]] = []
    for event in events:
        out.extend(translator.translate_event(event))
    out.extend(translator.finish())
    return out


def stream_text(target: str, events: list[dict[str, Any]]) -> str:
    if target == OPENAI:
        return "".join(
            (event.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            for event in events
        )
    if target == ANTHROPIC:
        return "".join(
            event.get("delta", {}).get("text", "")
            for event in events
            if event.get("type") == "content_block_delta"
        )
    return "".join(
        part.get("text", "")
        for event in events
        for part in ((event.get("candidates") or [{}])[0].get("content") or {}).get("parts", [])
        if not part.get("thought")
    )


def stream_tool_calls(target: str, events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Returns (name, joined argument json) pairs reconstructed from a stream."""
    if target == OPENAI:
        name = ""
        arguments = ""
        for event in events:
            for call in (event.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []:
                name = call.get("function", {}).get("name") or name
                arguments += call.get("function", {}).get("arguments") or ""
        return [(name, arguments)] if name else []
    if target == ANTHROPIC:
        name = ""
        arguments = ""
        for event in events:
            if event.get("type") == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    name = block.get("name", "")
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "input_json_delta":
                    arguments += delta.get("partial_json", "")
        return [(name, arguments)] if name else []
    calls = []
    for event in events:
        for part in ((event.get("candidates") or [{}])[0].get("content") or {}).get("parts", []):
            if "functionCall" in part:
                call = part["functionCall"]
                calls.append((call.get("name", ""), json.dumps(call.get("args", {}))))
    return calls


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_text_stream_translates_every_pair(source: str, target: str) -> None:
    if source == target:
        pytest.skip("same-format streams pass through endpoints untranslated")
    events = run_stream(source, target, text_stream_events()[source])
    assert stream_text(target, events) == "Hello", f"stream text lost {source} -> {target}"


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
def test_tool_stream_translates_every_pair(source: str, target: str) -> None:
    if source == target:
        pytest.skip("same-format streams pass through endpoints untranslated")
    events = run_stream(source, target, tool_stream_events()[source])
    calls = stream_tool_calls(target, events)
    assert calls, f"tool call lost in stream {source} -> {target}"
    name, arguments = calls[0]
    assert name == TOOL_NAME
    assert TOOL_ARG_VALUE in arguments
