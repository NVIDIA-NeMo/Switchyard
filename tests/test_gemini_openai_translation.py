# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gemini generateContent translation to and from OpenAI Chat."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from switchyard_rust.translation import TranslationEngine

ENGINE = TranslationEngine()


def _translate_openai_request_to_gemini(**kwargs: Any) -> dict[str, Any]:
    return ENGINE.translate_request("openai_chat", "gemini", kwargs)


def _translate_gemini_request_to_openai(**kwargs: Any) -> dict[str, Any]:
    return ENGINE.translate_request("gemini", "openai_chat", kwargs)


def test_openai_system_and_user_map_to_system_instruction_and_contents():
    result = _translate_openai_request_to_gemini(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful."
    assert result["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
    assert result["model"] == "gemini-2.5-flash"


def test_openai_sampling_and_stop_map_to_generation_config():
    result = _translate_openai_request_to_gemini(
        model="m",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=256,
        temperature=0.4,
        top_p=0.9,
        stop=["END"],
        stream=True,
    )
    generation = result["generationConfig"]
    assert generation["maxOutputTokens"] == 256
    assert generation["temperature"] == 0.4
    assert generation["topP"] == 0.9
    assert generation["stopSequences"] == ["END"]
    assert result["stream"] is True


def test_openai_tools_are_sanitized_for_gemini_schema_subset():
    result = _translate_openai_request_to_gemini(
        model="m",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string", "default": "/tmp"},
                        "limit": {"type": ["integer", "null"], "exclusiveMinimum": 0},
                    },
                    "required": ["path"],
                },
            },
        }],
        tool_choice="required",
    )
    declaration = result["tools"][0]["functionDeclarations"][0]
    parameters = declaration["parameters"]
    assert declaration["name"] == "read_file"
    assert parameters["type"] == "OBJECT"
    assert "$schema" not in parameters
    assert "additionalProperties" not in parameters
    assert parameters["properties"]["path"]["type"] == "STRING"
    assert "default" not in parameters["properties"]["path"]
    assert parameters["properties"]["limit"]["type"] == "INTEGER"
    assert parameters["properties"]["limit"]["nullable"] is True
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


def test_openai_tool_results_become_function_responses():
    result = _translate_openai_request_to_gemini(
        model="m",
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"},
        ],
    )
    model_parts = result["contents"][1]["parts"]
    assert model_parts[0]["functionCall"] == {"name": "get_weather", "args": {"city": "Paris"}}
    response_part = result["contents"][2]["parts"][0]["functionResponse"]
    assert response_part["name"] == "get_weather"
    assert response_part["response"]["parts"][0]["text"] == "22C sunny"


def test_gemini_request_maps_roles_tools_and_generation_config():
    result = _translate_gemini_request_to_openai(
        model="gemini-2.5-flash",
        stream=True,
        systemInstruction={"parts": [{"text": "Be terse."}]},
        contents=[
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
        ],
        tools=[{
            "functionDeclarations": [{
                "name": "run",
                "parameters": {"type": "OBJECT", "properties": {"cmd": {"type": "STRING"}}},
            }],
        }],
        toolConfig={"functionCallingConfig": {"mode": "AUTO"}},
        generationConfig={"maxOutputTokens": 128, "temperature": 0.2, "stopSequences": ["END"]},
    )
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert result["model"] == "gemini-2.5-flash"
    assert result["stream"] is True
    assert result["max_completion_tokens"] == 128
    assert result["temperature"] == 0.2
    assert result["stop"] == ["END"]
    assert result["tool_choice"] == "auto"
    # Gemini's uppercase schema types are restored to JSON Schema form.
    parameters = result["tools"][0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["cmd"]["type"] == "string"


def test_gemini_function_call_pairs_with_response_by_name_and_order():
    result = _translate_gemini_request_to_openai(
        contents=[
            {"role": "user", "parts": [{"text": "Weather?"}]},
            {"role": "model", "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"name": "get_weather", "response": {"result": "22C"}}},
            ]},
        ],
    )
    tool_calls = result["messages"][1]["tool_calls"]
    tool_message = result["messages"][2]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == tool_calls[0]["id"]


def test_gemini_response_translates_to_openai_completion():
    result = ENGINE.translate_response(
        "gemini",
        "openai_chat",
        {
            "candidates": [{
                "content": {"parts": [{"text": "OK"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": 6,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 20,
                "totalTokenCount": 27,
            },
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp-1",
        },
    )
    choice = result["choices"][0]
    assert choice["message"]["content"] == "OK"
    assert choice["finish_reason"] == "stop"
    # Thinking tokens fold into completion tokens.
    assert result["usage"]["prompt_tokens"] == 6
    assert result["usage"]["completion_tokens"] == 21
    assert result["model"] == "gemini-2.5-flash"


def test_gemini_function_call_response_maps_to_tool_calls_finish_reason():
    result = ENGINE.translate_response(
        "gemini",
        "openai_chat",
        {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                    }],
                    "role": "model",
                },
                # Gemini reports STOP even for function-call turns.
                "finishReason": "STOP",
            }],
        },
    )
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"


def test_openai_completion_translates_to_gemini_response():
    result = ENGINE.translate_response(
        "openai_chat",
        "gemini",
        {
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    )
    candidate = result["candidates"][0]
    assert candidate["content"]["parts"] == [{"text": "Hello"}]
    assert candidate["content"]["role"] == "model"
    assert candidate["finishReason"] == "STOP"
    assert result["usageMetadata"]["promptTokenCount"] == 4
    assert result["usageMetadata"]["candidatesTokenCount"] == 2
    assert result["responseId"] == "chatcmpl-1"


def test_gemini_max_tokens_finish_reason_maps_to_length():
    result = ENGINE.translate_response(
        "gemini",
        "openai_chat",
        {
            "candidates": [{
                "content": {"parts": [{"text": "truncat"}], "role": "model"},
                "finishReason": "MAX_TOKENS",
            }],
        },
    )
    assert result["choices"][0]["finish_reason"] == "length"


def test_gemini_thought_parts_do_not_leak_into_openai_content():
    result = ENGINE.translate_response(
        "gemini",
        "openai_chat",
        {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "private planning", "thought": True},
                        {"text": "Public answer"},
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
        },
    )
    content = result["choices"][0]["message"]["content"]
    assert "private planning" not in content
    assert "Public answer" in content


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in stream]


async def _iter_events(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        yield event


async def test_gemini_stream_translates_to_openai_chunks():
    chunks = await _collect(
        ENGINE.translate_stream(
            "gemini",
            "openai_chat",
            _iter_events([
                {
                    "candidates": [{"content": {"parts": [{"text": "Hel"}], "role": "model"}}],
                    "responseId": "r1",
                    "modelVersion": "gemini-2.5-flash",
                },
                {
                    "candidates": [{
                        "content": {"parts": [{"text": "lo"}], "role": "model"},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2},
                },
            ]),
        )
    )
    text = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks)
    assert text == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_openai_stream_translates_to_gemini_chunks():
    chunks = await _collect(
        ENGINE.translate_stream(
            "openai_chat",
            "gemini",
            _iter_events([
                {
                    "id": "chatcmpl-1",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
                },
                {
                    "id": "chatcmpl-1",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                },
            ]),
        )
    )
    parts = [
        part
        for chunk in chunks
        for part in chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    ]
    assert {"text": "Hi"} in parts
    final = chunks[-1]["candidates"][0]
    assert final["finishReason"] == "STOP"
    assert chunks[-1]["usageMetadata"]["promptTokenCount"] == 3
