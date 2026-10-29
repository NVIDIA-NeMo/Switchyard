# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public Python translation boundary against the native engine."""

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


def test_invalid_format_type() -> None:
    with pytest.raises(TypeError):
        TranslationEngine().translate_request(42, "openai_chat", {})


def test_unknown_format_raises_translation_error() -> None:
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_request("custom", "openai_chat", {})
    assert caught.value.kind == "Other"
    assert caught.value.details == {"message": "no codec registered for custom"}


def test_body_must_be_a_dictionary() -> None:
    with pytest.raises(TypeError):
        TranslationEngine().translate_request("openai_chat", "openai_responses", '{"messages": []}')


def test_translation_error_has_readable_message_and_structured_details() -> None:
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_request("openai_chat", "openai_responses", {"messages": {}})

    error = caught.value
    assert error.kind == "InvalidType"
    assert error.details == {"path": "$.messages", "expected": "array"}
    assert error.args == ("expected array at $.messages",)
    assert str(error) == error.args[0]
