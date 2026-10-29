# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public Python translation boundary against the native engine."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import pytest

from switchyard.translation import TranslationEngine, TranslationError, WireFormat

REQUESTS = {
    WireFormat.OPENAI_CHAT: {
        "model": "example",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    },
    WireFormat.OPENAI_RESPONSES: {
        "model": "example",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        "max_output_tokens": 64,
    },
    WireFormat.ANTHROPIC_MESSAGES: {
        "model": "example",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "max_tokens": 64,
    },
}

RESPONSES = {
    WireFormat.OPENAI_CHAT: {
        "id": "response-1",
        "model": "example",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
    },
    WireFormat.OPENAI_RESPONSES: {
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
    },
    WireFormat.ANTHROPIC_MESSAGES: {
        "id": "response-1",
        "type": "message",
        "model": "example",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
}


@pytest.mark.parametrize("source", list(WireFormat))
@pytest.mark.parametrize("target", list(WireFormat))
def test_translate_request(source: WireFormat, target: WireFormat) -> None:
    body = deepcopy(REQUESTS[source])
    original = deepcopy(body)
    result = TranslationEngine().translate_request(source, target.value, body)

    assert set(result) == {"body", "diagnostics"}
    assert result["diagnostics"] == []
    assert body == original
    translated = result["body"]
    assert translated["model"] == "example"
    if target is WireFormat.OPENAI_RESPONSES:
        content = translated["input"][0]["content"]
        assert content == "Hello" or content == [{"type": "input_text", "text": "Hello"}]
        assert translated["max_output_tokens"] == 64
    else:
        content = translated["messages"][0]["content"]
        assert content == "Hello" or content == [{"type": "text", "text": "Hello"}]
        assert translated.get("max_tokens", translated.get("max_completion_tokens")) == 64


@pytest.mark.parametrize("source", list(WireFormat))
@pytest.mark.parametrize("target", list(WireFormat))
def test_translate_response(source: WireFormat, target: WireFormat) -> None:
    body = deepcopy(RESPONSES[source])
    original = deepcopy(body)
    result = TranslationEngine().translate_response(source.value, target, body)

    assert set(result) == {"body", "diagnostics"}
    assert result["diagnostics"] == []
    assert body == original
    translated = result["body"]
    assert translated["model"] == "example"
    if target is WireFormat.OPENAI_CHAT:
        assert translated["choices"][0]["message"]["content"] == "Hello"
    elif target is WireFormat.OPENAI_RESPONSES:
        assert translated["output"][0]["content"][0]["text"] == "Hello"
    else:
        assert translated["content"] == [{"type": "text", "text": "Hello"}]


def test_same_format_preserves_json_values() -> None:
    body = {
        **REQUESTS[WireFormat.OPENAI_CHAT],
        "metadata": {
            "list": [1, 2, 3],
            "float": 4.5,
            "boolean": True,
            "null": None,
            "text": "こんにちは",
        },
    }
    result = TranslationEngine().translate_request("openai_chat", "openai_chat", body)
    assert result["body"] == body
    assert json.loads(json.dumps(result)) == result


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


@pytest.mark.parametrize("method", ["translate_request", "translate_response"])
@pytest.mark.parametrize("argument", ["source", "target"])
@pytest.mark.parametrize("value", [None, 42])
def test_invalid_format_types(method: str, argument: str, value: Any) -> None:
    arguments = {"source": "openai_chat", "target": "openai_responses", "body": {}}
    arguments[argument] = value
    with pytest.raises(TypeError):
        getattr(TranslationEngine(), method)(**arguments)


@pytest.mark.parametrize("method", ["translate_request", "translate_response"])
@pytest.mark.parametrize("argument", ["source", "target"])
def test_unknown_formats_raise_translation_error(method: str, argument: str) -> None:
    bodies = REQUESTS if method == "translate_request" else RESPONSES
    arguments: dict[str, Any] = {
        "source": "openai_chat",
        "target": "openai_responses",
        "body": bodies[WireFormat.OPENAI_CHAT],
    }
    arguments[argument] = "custom"
    with pytest.raises(TranslationError) as caught:
        getattr(TranslationEngine(), method)(**arguments)
    assert caught.value.kind == "Other"
    assert caught.value.details == {"message": "no codec registered for custom"}
    assert str(caught.value) == "no codec registered for custom"


@pytest.mark.parametrize("method", ["translate_request", "translate_response"])
@pytest.mark.parametrize("body", ['{"messages": []}', [], None, 42])
def test_body_must_be_a_dictionary(method: str, body: Any) -> None:
    with pytest.raises(TypeError):
        getattr(TranslationEngine(), method)("openai_chat", "openai_responses", body)


def test_body_must_contain_json_compatible_values() -> None:
    with pytest.raises(ValueError):
        TranslationEngine().translate_request("openai_chat", "openai_chat", {"metadata": object()})


def test_translation_error_has_readable_message_and_structured_details() -> None:
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_request("openai_chat", "openai_responses", {"messages": {}})

    error = caught.value
    assert TranslationError.__bases__ == (Exception,)
    assert error.kind == "InvalidType"
    assert error.details == {"path": "$.messages", "expected": "array"}
    assert error.args == ("expected array at $.messages",)
    assert str(error) == error.args[0]


def test_invalid_value_preserves_path_and_message() -> None:
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_request(
            "openai_chat", "openai_responses", {"messages": [{"role": "human", "content": "Hello"}]}
        )
    assert caught.value.kind == "InvalidValue"
    assert caught.value.details["path"] == "$.messages[0].role"
    assert "human" in caught.value.details["message"]


def test_upstream_failure_preserves_provider_error() -> None:
    provider_error = {"code": "server_error", "message": "generation failed", "nested": [1, None]}
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_response(
            "openai_responses",
            "openai_chat",
            {"status": "failed", "error": provider_error, "output": []},
        )
    assert caught.value.kind == "UpstreamFailure"
    assert caught.value.details == {"error": provider_error}
    assert "generation failed" in str(caught.value)


def test_unsupported_translation_preserves_formats() -> None:
    with pytest.raises(TranslationError) as caught:
        TranslationEngine().translate_response(
            "openai_responses",
            "openai_chat",
            {
                "output": [
                    {
                        "type": "shell_call",
                        "id": "sh_1",
                        "call_id": "call_1",
                        "status": "completed",
                        "action": {"commands": ["pwd"]},
                    }
                ]
            },
        )
    assert caught.value.kind == "UnsupportedTranslation"
    assert caught.value.details == {"from": "openai_responses", "to": "openai_chat"}


def test_engine_can_be_reused_across_threads() -> None:
    engine = TranslationEngine()
    body = REQUESTS[WireFormat.OPENAI_CHAT]

    def translate(_: int) -> dict[str, Any]:
        return engine.translate_request("openai_chat", "anthropic_messages", body)

    expected = translate(0)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(translate, range(32))) == [expected] * 32


def test_public_exports() -> None:
    from switchyard_rust import translation

    assert translation.TranslationEngine is TranslationEngine
    assert translation.TranslationError is TranslationError
    assert translation.WireFormat is WireFormat
    assert TranslationEngine.__module__ == "switchyard.translation"
    assert TranslationError.__module__ == "switchyard.translation"
    assert WireFormat("openai_chat") is WireFormat.OPENAI_CHAT
    assert isinstance(WireFormat.OPENAI_CHAT, str)
    with pytest.raises(AttributeError):
        _ = translation.missing
