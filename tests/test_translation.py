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


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"request_extensions": None}, id="none"),
        pytest.param({"request_extensions": {}}, id="empty"),
        pytest.param({"request_extensions": {"fields": {}}}, id="empty-fields"),
        pytest.param({"request_extensions": {"fields": {"unrelated": True}}}, id="unrelated"),
    ],
)
def test_translate_response(kwargs) -> None:
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
        "openai_responses", WireFormat.ANTHROPIC_MESSAGES, body, **kwargs
    )
    assert set(result) == {"body", "diagnostics"}
    assert result["diagnostics"] == []
    assert result["body"]["model"] == "example"
    assert result["body"]["content"] == [{"type": "text", "text": "Hello"}]
    assert result["body"]["stop_reason"] == "end_turn"


@pytest.mark.parametrize(
    "method,args,kwargs,message,source,target",
    [
        pytest.param(
            "translate_request",
            ("openai_chat", "anthropic_messages", {"model": "example", "messages": [42]}),
            {},
            "OpenAI message at index 0 is not an object",
            "openai_chat",
            "anthropic_messages",
            id="request",
        ),
        pytest.param(
            "translate_response",
            ("openai_responses", "anthropic_messages", {"output": [42]}),
            {"request_extensions": {"fields": {"unrelated": True}}},
            "Responses output item is not an object",
            "openai_responses",
            "anthropic_messages",
            id="response",
        ),
        pytest.param(
            "decode_request",
            ("openai_chat", {"messages": [42]}),
            {},
            "OpenAI message at index 0 is not an object",
            None,
            None,
            id="decoded-request",
        ),
    ],
)
def test_lossy_conversion_returns_serialized_diagnostics(
    method, args, kwargs, message, source, target
) -> None:
    result = getattr(TranslationEngine(), method)(*args, **kwargs)
    assert result["diagnostics"] == [
        {
            "severity": "Warning",
            "code": "lossy_conversion",
            "message": message,
            "source": source,
            "target": target,
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
        pytest.param(
            "translate_response",
            ("openai_chat", "custom", {"choices": []}),
            id="translate-response",
        ),
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


@pytest.mark.parametrize("method", ["encode_response", "translate_response"])
def test_response_ir_restores_request_tool_extensions(method) -> None:
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
    body = {
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
    }
    extensions = request["extensions"]
    if method == "encode_response":
        response = engine.decode_response("openai_chat", body)["response"]
        restored = engine.encode_response(
            "openai_responses", response, request_extensions=extensions
        )
    else:
        restored = engine.translate_response(
            "openai_chat", "openai_responses", body, request_extensions=extensions
        )
    assert restored["diagnostics"] == []
    search, custom = restored["body"]["output"]
    assert search["type"] == "function_call"
    assert search["call_id"] == "call-1"
    assert search["name"] == "search"
    assert search["namespace"] == "mcp__docs"
    assert json.loads(search["arguments"]) == {"query": "streaming"}
    assert custom["type"] == "custom_tool_call"
    assert custom["call_id"] == "call-2"
    assert custom["name"] == "exec"
    assert custom["input"] == "ls -la"
    assert "arguments" not in custom


@pytest.mark.parametrize(
    "method,args,kwargs,error",
    [
        pytest.param(
            "encode_request",
            ("openai_chat", {"messages": "not a list"}),
            {},
            ValueError,
            id="request",
        ),
        pytest.param(
            "translate_request",
            (
                "openai_responses",
                "openai_chat",
                {
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "a__b",
                            "tools": [{"type": "function", "name": "c", "parameters": {}}],
                        },
                        {
                            "type": "namespace",
                            "name": "a",
                            "tools": [{"type": "function", "name": "b__c", "parameters": {}}],
                        },
                    ],
                },
            ),
            {},
            TranslationError,
            id="namespace-tool-name-collision",
        ),
        pytest.param(
            "encode_response",
            ("openai_chat", {}),
            {"request_extensions": {"fields": 42}},
            ValueError,
            id="encode-invalid-fields",
        ),
        pytest.param(
            "encode_response",
            ("openai_chat", {}),
            {"request_extensions": []},
            TypeError,
            id="encode-invalid-type",
        ),
        pytest.param(
            "translate_response",
            ("openai_chat", "openai_responses", {"choices": []}),
            {"request_extensions": {"fields": 42}},
            ValueError,
            id="translate-invalid-fields",
        ),
        pytest.param(
            "translate_response",
            ("openai_chat", "openai_responses", {"choices": []}),
            {"request_extensions": []},
            TypeError,
            id="translate-invalid-type",
        ),
    ],
)
def test_ir_boundary_errors(method, args, kwargs, error) -> None:
    with pytest.raises(error):
        getattr(TranslationEngine(), method)(*args, **kwargs)
