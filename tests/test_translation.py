# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public Python translation boundary against the native engine."""

import json

import pytest

from switchyard.translation import TranslationEngine, TranslationError, WireFormat


def test_translate_request() -> None:
    body = {
        "model": "example",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 64,
    }
    result = TranslationEngine().translate_request(
        WireFormat.OPENAI_CHAT, "anthropic_messages", body
    )
    assert result == {
        "body": {
            "model": "example",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64,
        },
        "diagnostics": [],
    }


def test_translate_response() -> None:
    body = {
        "id": "response-1",
        "model": "example",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ],
    }
    result = TranslationEngine().translate_response(
        "openai_responses", WireFormat.ANTHROPIC_MESSAGES, body
    )
    assert set(result) == {"body", "diagnostics"}
    assert result["diagnostics"] == []
    assert result["body"]["model"] == "example"
    assert result["body"]["content"] == [{"type": "text", "text": "Hello"}]
    assert result["body"]["stop_reason"] == "end_turn"


def test_lossy_conversion_returns_serialized_diagnostics() -> None:
    result = TranslationEngine().translate_request(
        "openai_chat", "anthropic_messages", {"model": "example", "messages": [42]}
    )
    assert result["diagnostics"] == [
        {
            "severity": "Warning",
            "code": "lossy_conversion",
            "message": "OpenAI message at index 0 is not an object",
            "source": "openai_chat",
            "target": "anthropic_messages",
            "path": None,
        }
    ]


@pytest.mark.parametrize(
    "source,target,body",
    [
        pytest.param(42, "openai_chat", {}, id="invalid-format-type"),
        pytest.param(
            "openai_chat", "openai_responses", '{"messages": []}', id="non-dictionary-body"
        ),
    ],
)
def test_invalid_argument_types(source, target, body) -> None:
    with pytest.raises(TypeError):
        TranslationEngine().translate_request(source, target, body)


@pytest.mark.parametrize(
    "method,args",
    [
        pytest.param("translate_request", ("custom", "openai_chat", {}), id="translate-request"),
        pytest.param("decode_response", ("custom", {}), id="decode-response"),
    ],
)
def test_unknown_format_raises_translation_error(method, args) -> None:
    with pytest.raises(TranslationError) as caught:
        getattr(TranslationEngine(), method)(*args)
    assert caught.value.kind == "Other"
    assert caught.value.details == {"message": "no codec registered for custom"}


def test_translation_error_has_readable_message_and_structured_details() -> None:
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_request("openai_chat", "openai_responses", {"messages": {}})

    error = caught.value
    assert error.kind == "InvalidType"
    assert error.details == {"path": "$.messages", "expected": "array"}
    assert error.args == ("expected array at $.messages",)
    assert str(error) == error.args[0]


def test_request_ir_round_trip_and_edit() -> None:
    engine = TranslationEngine()
    body = {"model": "example", "messages": [{"role": "user", "content": "Hello"}]}
    decoded = engine.decode_request(WireFormat.OPENAI_CHAT, body)
    assert set(decoded) == {"request", "diagnostics"}
    assert decoded["diagnostics"] == []
    request = json.loads(json.dumps(decoded["request"]))
    assert request["messages"][0]["content"] == [{"type": "text", "text": "Hello"}]
    request["model"] = "edited"
    assert engine.encode_request("openai_chat", request) == {"body": body, "diagnostics": []}
    request["preservation"]["requests"].clear()
    encoded = engine.encode_request(WireFormat.OPENAI_CHAT, request)
    assert encoded["body"]["model"] == "edited"
    assert encoded["body"]["messages"] == body["messages"]
    assert encoded["diagnostics"] == []


def test_response_ir_restores_request_tool_extensions() -> None:
    engine = TranslationEngine()
    request = engine.decode_request(
        "openai_responses",
        {
            "input": "Search documentation and list files",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__docs",
                    "tools": [{"type": "function", "name": "search", "parameters": {}}],
                },
                {"type": "custom", "name": "exec", "format": {"type": "text"}},
            ],
        },
    )["request"]
    decoded = engine.decode_response(
        WireFormat.OPENAI_CHAT,
        {
            "id": "response-1",
            "model": "example",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__docs__search",
                                    "arguments": '{"query":"streaming"}',
                                },
                            },
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec",
                                    "arguments": '{"input":"ls -la"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    assert set(decoded) == {"response", "diagnostics"}
    assert decoded["diagnostics"] == []
    response = json.loads(json.dumps(decoded["response"]))
    plain = engine.encode_response("openai_responses", response)
    for empty in (None, {}, {"fields": {}}):
        assert (
            engine.encode_response("openai_responses", response, request_extensions=empty) == plain
        )
    plain_calls = plain["body"]["output"]
    assert [call["type"] for call in plain_calls] == ["function_call", "function_call"]
    assert plain_calls[0]["name"] == "mcp__docs__search"
    assert "namespace" not in plain_calls[0]
    restored = engine.encode_response(
        WireFormat.OPENAI_RESPONSES,
        response,
        request_extensions=json.loads(json.dumps(request["extensions"])),
    )
    assert restored["diagnostics"] == []
    search, custom = restored["body"]["output"]
    assert search["name"] == "search"
    assert search["namespace"] == "mcp__docs"
    assert custom["type"] == "custom_tool_call"
    assert custom["name"] == "exec"
    assert custom["input"] == "ls -la"
    assert "arguments" not in custom


def test_ir_boundary_diagnostics_and_errors() -> None:
    engine = TranslationEngine()
    decoded = engine.decode_request("openai_chat", {"messages": [42]})
    assert decoded["diagnostics"][0]["code"] == "lossy_conversion"
    assert decoded["diagnostics"][0]["severity"] == "Warning"
    with pytest.raises(ValueError):
        engine.encode_request("openai_chat", {"messages": "not a list"})
    with pytest.raises(ValueError):
        engine.encode_response("openai_chat", {}, request_extensions={"fields": 42})
    with pytest.raises(TypeError):
        engine.encode_response("openai_chat", {}, request_extensions=[])
