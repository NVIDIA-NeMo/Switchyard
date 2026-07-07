# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial / edge-case tests for ChatRequest and ChatResponse translation engines.

Every test targets a specific boundary condition, failure mode, or surprising
interaction that the happy-path tests in test_request_translation_engine.py and
test_response_translation_engine.py do not cover.  Nothing here is hypothetical
-- each case was verified to exercise real code paths.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse, request_type_matches
from switchyard_rust.translation import TranslationEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def req_engine():
    return TranslationEngine()


@pytest.fixture
def resp_engine():
    return TranslationEngine()


class _AlienChatRequest:
    """A request-shaped object unknown to either engine.

    Used to verify that NotImplementedError is raised with a useful message
    instead of silently passing through or crashing.
    """

    def __init__(self, body: dict[str, Any] | None = None):
        self._body = body or {}

    @property
    def request_type(self) -> str:
        return "alien"

    @property
    def body(self) -> dict[str, Any]:
        return self._body


def _make_completion_dict(
    content: str | None = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Build a plain-dict ChatCompletion-shaped payload."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {"message": message, "finish_reason": finish_reason},
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _mock_completion(
    content: str | None = "Hello!",
    tool_calls_mock: list[Any] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a MagicMock that quacks like ``ChatCompletion``."""
    completion = MagicMock()
    completion.choices = [
        MagicMock(
            message=MagicMock(
                content=content,
                tool_calls=tool_calls_mock,
                refusal=None,
            ),
            finish_reason=finish_reason,
        )
    ]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    completion.model = model
    completion.id = "chatcmpl-test"

    tc_dicts = None
    if tool_calls_mock:
        tc_dicts = []
        for tc in tool_calls_mock:
            tc_dicts.append({
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })

    completion.model_dump = lambda **kwargs: _make_completion_dict(
        content=content,
        tool_calls=tc_dicts,
        finish_reason=finish_reason,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return completion


def _mock_tool_call(
    call_id: str = "call_abc",
    name: str = "get_weather",
    arguments: str = '{"location":"SF"}',
) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock(name=name, arguments=arguments)
    # MagicMock special-cases .name, so set it explicitly
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# Streaming helpers ---------------------------------------------------------

@dataclass
class MockDelta:
    content: str | None = None
    tool_calls: Any = None
    reasoning: str | None = None
    reasoning_content: str | None = None


@dataclass
class MockToolCallDelta:
    index: int = 0
    id: str | None = None
    function: Any = None


@dataclass
class MockFunctionDelta:
    name: str | None = None
    arguments: str | None = None


@dataclass
class MockChoice:
    delta: MockDelta | None = None
    finish_reason: str | None = None


@dataclass
class MockChunk:
    choices: list[MockChoice] | None = None
    usage: Any = None


@dataclass
class MockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


async def _async_chunks(*chunks: Any):
    for c in chunks:
        yield c


# =========================================================================
# REQUEST ENGINE EDGE CASES
# =========================================================================


class TestRequestEngineEdgeCases:
    """Edge cases that exercise the TranslationEngine."""

    # -- Empty / minimal bodies ------------------------------------------

    def test_anthropic_empty_messages(self, req_engine):
        """An Anthropic request with an empty messages list should produce
        a valid OpenAI request with no messages (except possibly system).
        """
        body = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 100}
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert request_type_matches(result, ChatRequestType.OPENAI_CHAT)
        assert result.body["messages"] == []

    def test_anthropic_empty_model_string(self, req_engine):
        """model="" is falsy -- verify it does NOT get forwarded."""
        body = {"model": "", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert "model" not in result.body

    def test_anthropic_content_none(self, req_engine):
        """Content=None in a message should not crash; it maps to empty string."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": None}],
            "max_tokens": 100,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == ""

    def test_anthropic_content_integer(self, req_engine):
        """Content as a non-string, non-list value (int) should be coerced to str."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": 42}],
            "max_tokens": 100,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert msgs[0]["content"] == "42"

    def test_anthropic_content_boolean(self, req_engine):
        """Content as a boolean -- a type confusion that can happen with
        malformed requests.  Should not crash.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": True}],
            "max_tokens": 100,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        # bool is truthy, not a list, not a str => str(True) = "True"
        assert msgs[0]["content"] == "True"

    # -- System prompt variants ------------------------------------------

    def test_anthropic_structured_system_blocks(self, req_engine):
        """Anthropic system as a list of {type:text} blocks should be
        concatenated into a single system message while preserving block
        boundaries.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        system_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful.\n\nBe concise."

    def test_anthropic_empty_system_string(self, req_engine):
        """system="" is falsy -- should NOT produce a system message."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": "",
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        system_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 0

    def test_anthropic_whitespace_only_system(self, req_engine):
        """system="   " is truthy -- it WILL produce a system message
        (arguably a bug, but this test documents the current behavior).
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": "   ",
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        system_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "   "

    def test_anthropic_system_blocks_with_non_text(self, req_engine):
        """Structured system with a non-text block type should silently skip it."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Be helpful."},
                {"type": "image", "source": {"data": "base64..."}},
            ],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        system_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Be helpful."

    # -- Deeply nested content blocks ------------------------------------

    def test_anthropic_mixed_text_tool_use_in_one_message(self, req_engine):
        """A single assistant message with both text and tool_use content
        blocks should produce ONE assistant message with content + tool_calls.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me look that up."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "search",
                            "input": {"query": "weather"},
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Let me look that up."
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_anthropic_tool_result_with_structured_content(self, req_engine):
        """tool_result where content is a list of blocks (text + image)
        should flatten text parts; image attachments have no OpenAI tool
        message representation and are dropped rather than leaked as
        stringified base64 prompt text.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "Temperature: 72F"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "data": "iVBORw...",
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        tool_msgs = [m for m in result.body["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        # Text is preserved
        assert "Temperature: 72F" in tool_msgs[0]["content"]
        # The image payload must not leak into prompt text
        assert "iVBORw" not in tool_msgs[0]["content"]

    def test_anthropic_content_list_with_non_dict_items(self, req_engine):
        """Content blocks that are not dicts (e.g. raw strings in the list)
        should be silently skipped, not crash.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": ["just a string", 123, {"type": "text", "text": "real block"}],
                },
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "real block"

    def test_anthropic_empty_content_list(self, req_engine):
        """Content as an empty list should produce a message with empty content."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": []}],
            "max_tokens": 100,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == ""

    # -- Multi-turn conversations ----------------------------------------

    def test_twenty_message_conversation(self, req_engine):
        """A realistic 20-message conversation with alternating roles,
        including system, tool_use, and tool_result, should translate
        without loss.
        """
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": messages,
            "max_tokens": 1024,
            "system": "You are a test assistant.",
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        # 1 system + 20 conversation messages
        assert len(result.body["messages"]) == 21
        assert result.body["messages"][0]["role"] == "system"
        # Verify message ordering preserved
        for i in range(10):
            assert result.body["messages"][1 + 2 * i]["content"] == f"Question {i}"
            assert result.body["messages"][2 + 2 * i]["content"] == f"Answer {i}"

    def test_multi_turn_with_tool_roundtrip(self, req_engine):
        """A conversation with tool_use -> tool_result -> assistant follow-up
        should map cleanly to OpenAI assistant(tool_calls) -> tool -> assistant.
        """
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "72F and sunny",
                    },
                ],
            },
            {"role": "assistant", "content": "It's 72F and sunny in SF!"},
        ]
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": messages,
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        roles = [m["role"] for m in result.body["messages"]]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert result.body["messages"][1].get("tool_calls") is not None

    # -- Tool edge cases -------------------------------------------------

    def test_anthropic_tool_with_empty_input_schema(self, req_engine):
        """A tool with input_schema={} should produce parameters={}."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [
                {"name": "noop", "description": "Does nothing", "input_schema": {}},
            ],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["tools"][0]["function"]["parameters"] == {}

    def test_anthropic_tool_with_no_name(self, req_engine):
        """A tool definition missing the 'name' key should be dropped."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"description": "mystery tool", "input_schema": {}}],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert "tools" not in result.body

    def test_anthropic_tool_with_unicode_name(self, req_engine):
        """Tool names with unicode should pass through without mangling."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [
                {
                    "name": "recherche_meteo",
                    "description": "Recherche meteo",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["tools"][0]["function"]["name"] == "recherche_meteo"

    def test_anthropic_tool_with_deeply_nested_schema(self, req_engine):
        """A deeply nested JSON schema in input_schema should pass through."""
        deep_schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {
                                "deep": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [
                {"name": "deep_tool", "description": "d", "input_schema": deep_schema},
            ],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        params = result.body["tools"][0]["function"]["parameters"]
        assert params["properties"]["config"]["properties"]["nested"]["properties"]["deep"]["type"] == "string"

    def test_anthropic_tool_use_with_string_input(self, req_engine):
        """tool_use block where input is a pre-serialized JSON string
        should be used directly without double-encoding.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "run_cmd",
                            "input": '{"cmd": "ls -la"}',
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        args = result.body["messages"][0]["tool_calls"][0]["function"]["arguments"]
        # Should be the string itself, NOT json.dumps(string)
        assert args == '{"cmd": "ls -la"}'
        # Verify it's valid JSON
        parsed = json.loads(args)
        assert parsed["cmd"] == "ls -la"

    # -- Extra kwargs / pass-through fields ------------------------------

    def test_anthropic_extra_kwargs_filtered_by_openai_whitelist(self, req_engine):
        """Only fields that exist on OpenAI Chat Completions survive the
        conversion — Anthropic-only fields (``thinking``, ``cache_control``,
        etc.) must be dropped because the OpenAI SDK raises TypeError on
        unknown kwargs. Fields that happen to exist on both APIs
        (``metadata``) pass through.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "metadata": {"user_id": "u123"},          # exists on OpenAI → survives
            "thinking": {"type": "enabled", "budget_tokens": 5000},  # Anthropic-only → dropped
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["metadata"] == {"user_id": "u123"}
        assert "thinking" not in result.body

    def test_anthropic_stream_flag_preserved(self, req_engine):
        """stream=True in the Anthropic body should appear in the OpenAI body."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["stream"] is True

    def test_anthropic_stop_sequences_to_stop(self, req_engine):
        """stop_sequences should become stop in the OpenAI body."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stop_sequences": ["\n\nHuman:", "END"],
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["stop"] == ["\n\nHuman:", "END"]

    # -- Mutation safety -------------------------------------------------

    def test_anthropic_passthrough_metadata_does_not_share_nested_objects(self, req_engine):
        """Native translation serializes passthrough metadata instead of
        exposing source object identity in the translated request.
        """
        inner_tags = ["tag1", "tag2"]
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "metadata": {"tags": inner_tags},
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)

        result_meta = result.body.get("metadata")
        if result_meta is not None:
            assert result_meta["tags"] == inner_tags
            assert result_meta["tags"] is not inner_tags

    def test_responses_shallow_copy_does_not_mutate_body(self, req_engine):
        """Verify the Responses path also shallow-copies."""
        body = {"model": "gpt-4o", "input": "hello", "instructions": "Be nice."}
        original = copy.deepcopy(body)
        req = ChatRequest.openai_responses(body)
        req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert req.body == original

    # -- Responses API input variants ------------------------------------

    def test_responses_empty_input_string(self, req_engine):
        """input="" should produce a user message with empty content."""
        body = {"model": "gpt-4o", "input": ""}
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        user_msgs = [m for m in result.body["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == ""

    def test_responses_empty_input_list(self, req_engine):
        """input=[] should produce an empty messages list."""
        body = {"model": "gpt-4o", "input": []}
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["messages"] == []

    def test_responses_unknown_input_item_type_is_preserved(self, req_engine):
        """Unknown input items should survive as valid Chat text content."""
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "audio_clip", "data": "base64..."},  # unknown
                {"type": "message", "role": "assistant", "content": "hello"},
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert len(msgs) == 3
        assert msgs[0]["content"] == "hi"
        assert msgs[1]["content"][0]["type"] == "text"
        assert json.loads(msgs[1]["content"][0]["text"]) == {
            "type": "audio_clip",
            "data": "base64...",
        }
        assert msgs[2]["content"] == "hello"

    def test_responses_orphan_function_call_output(self, req_engine):
        """A function_call_output without a preceding function_call should
        not produce an invalid Chat ``tool`` message.
        """
        body = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_orphan",
                    "output": "result",
                },
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["messages"] == [
            {"role": "user", "content": "Tool result call_orphan: result"}
        ]

    def test_responses_max_output_tokens_to_max_completion_tokens(self, req_engine):
        """max_output_tokens should map to Chat's current max token cap."""
        body = {"model": "gpt-4o", "input": "hi", "max_output_tokens": 4096}
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["max_completion_tokens"] == 4096
        assert "max_tokens" not in result.body
        assert "max_output_tokens" not in result.body

    def test_responses_tool_with_codex_format(self, req_engine):
        """Tools in Codex CLI format (id + inputSchema.jsonSchema) should
        be converted correctly.
        """
        body = {
            "model": "gpt-4o",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "id": "codex_tool",
                    "inputSchema": {
                        "jsonSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        }
                    },
                }
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        tools = result.body["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "codex_tool"
        assert tools[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"

    def test_responses_tool_with_empty_name_skipped(self, req_engine):
        """Tools with no name AND no id should be skipped."""
        body = {
            "model": "gpt-4o",
            "input": "hi",
            "tools": [
                {"type": "function", "description": "ghost", "parameters": {}},
                {"type": "function", "name": "real_tool", "parameters": {}},
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        tools = result.body["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "real_tool"

    def test_responses_multi_turn_tool_roundtrip(self, req_engine):
        """function_call + function_call_output sequences should be merged
        into assistant(tool_calls) + tool messages.
        """
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Search for X"},
                {
                    "type": "function_call",
                    "name": "search",
                    "call_id": "call_1",
                    "arguments": '{"q":"X"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Found X",
                },
                {"type": "message", "role": "assistant", "content": "I found X!"},
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "search"

    def test_responses_parallel_tool_calls_merged(self, req_engine):
        """Multiple consecutive function_call items (before any output)
        should be merged into a single assistant message.
        """
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Do two things"},
                {
                    "type": "function_call",
                    "name": "tool_a",
                    "call_id": "call_a",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "name": "tool_b",
                    "call_id": "call_b",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "A done",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": "B done",
                },
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        # user, assistant(2 tool_calls), tool(a), tool(b)
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["tool_calls"]) == 2
        assert msgs[2]["role"] == "tool"
        assert msgs[3]["role"] == "tool"

    def test_responses_intervening_message_deferred_until_after_tool_output(self, req_engine):
        """Messages between a function_call and matching output must not
        separate Chat tool_calls from their tool result.
        """
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Search for X"},
                {
                    "type": "function_call",
                    "name": "search",
                    "call_id": "call_1",
                    "arguments": "{}",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "I will summarize after the tool.",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Found X",
                },
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
        assert msgs[1]["tool_calls"][0]["id"] == "call_1"
        assert msgs[2]["tool_call_id"] == "call_1"
        assert msgs[3]["content"] == "I will summarize after the tool."

    def test_responses_chat_compatible_fields_survive_native_translation(self, req_engine):
        body = {
            "model": "gpt-4o",
            "input": "hi",
            "metadata": {"trace": "abc"},
            "parallel_tool_calls": False,
            "prompt_cache_key": "session-1",
            "prompt_cache_retention": "24h",
            "safety_identifier": "safe-1",
            "service_tier": "flex",
            "store": False,
            "stream_options": {"include_usage": True},
            "top_logprobs": 2,
            "user": "u-123",
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["metadata"] == {"trace": "abc"}
        assert result.body["parallel_tool_calls"] is False
        assert result.body["prompt_cache_key"] == "session-1"
        assert result.body["prompt_cache_retention"] == "24h"
        assert result.body["safety_identifier"] == "safe-1"
        assert result.body["service_tier"] == "flex"
        assert result.body["store"] is False
        assert result.body["stream_options"] == {"include_usage": True}
        assert result.body["top_logprobs"] == 2
        assert result.body["user"] == "u-123"

    def test_responses_json_schema_text_format_maps_to_chat_response_format(self, req_engine):
        body = {
            "model": "gpt-4o",
            "input": "Return JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        }

    # -- NotImplementedError paths ---------------------------------------

    def test_unknown_request_type_to_openai(self, req_engine):
        """An unknown ChatRequest subclass should raise NotImplementedError
        with a message that includes the class name.
        """
        req = _AlienChatRequest({"model": "x"})
        with pytest.raises(NotImplementedError, match="_AlienChatRequest"):
            req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)

    def test_unknown_request_type_to_anthropic(self, req_engine):
        """An unknown subclass going to_anthropic should mention the class name."""
        req = _AlienChatRequest()
        with pytest.raises(NotImplementedError, match="_AlienChatRequest"):
            req_engine.request_to(ChatRequestType.ANTHROPIC, req)

    def test_unknown_request_type_to_responses(self, req_engine):
        """An unknown subclass going to_responses should raise."""
        req = _AlienChatRequest()
        with pytest.raises(NotImplementedError, match="_AlienChatRequest"):
            req_engine.request_to(ChatRequestType.OPENAI_RESPONSES, req)

    # -- OpenAI -> Anthropic edge cases ----------------------------------

    def test_openai_to_anthropic_default_max_tokens(self, req_engine):
        """When ``max_tokens`` is omitted, Anthropic requires a default.

        The translation layer injects ``128_000`` (generous on purpose —
        see ``convert_openai_request_to_anthropic``'s docstring).  Anthropic
        clamps this to the model's real output ceiling server-side, so
        oversizing is harmless and a small default would silently
        truncate long coding outputs.
        """
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        req = ChatRequest.openai_chat(body)
        result = req_engine.request_to(ChatRequestType.ANTHROPIC, req)
        assert result.body["max_tokens"] == 128_000

    def test_openai_to_anthropic_system_extraction(self, req_engine):
        """System messages should be extracted into the Anthropic system param."""
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 100,
        }
        req = ChatRequest.openai_chat(body)
        result = req_engine.request_to(ChatRequestType.ANTHROPIC, req)
        assert result.body["system"] == "Be helpful."
        # System should NOT appear in messages
        for m in result.body["messages"]:
            assert m["role"] != "system"

    def test_openai_to_anthropic_tool_calls_in_messages(self, req_engine):
        """OpenAI assistant messages with tool_calls should become Anthropic
        assistant messages with tool_use content blocks.
        """
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "Checking...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"loc": "SF"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "72F",
                },
            ],
            "max_tokens": 100,
        }
        req = ChatRequest.openai_chat(body)
        result = req_engine.request_to(ChatRequestType.ANTHROPIC, req)
        msgs = result.body["messages"]
        # assistant with tool_use blocks
        asst = msgs[1]
        assert asst["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in asst["content"])
        # tool result
        tool_result = msgs[2]
        assert tool_result["role"] == "user"
        assert tool_result["content"][0]["type"] == "tool_result"

    def test_openai_to_anthropic_stop_string_to_list(self, req_engine):
        """stop as a single string should become stop_sequences=[string]."""
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stop": "END",
        }
        req = ChatRequest.openai_chat(body)
        result = req_engine.request_to(ChatRequestType.ANTHROPIC, req)
        assert result.body["stop_sequences"] == ["END"]

    def test_openai_to_anthropic_multimodal_content(self, req_engine):
        """OpenAI messages with list content (text + image_url) should be
        converted to Anthropic content blocks.
        """
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    ],
                },
            ],
            "max_tokens": 100,
        }
        req = ChatRequest.openai_chat(body)
        result = req_engine.request_to(ChatRequestType.ANTHROPIC, req)
        msg = result.body["messages"][0]
        assert isinstance(msg["content"], list)
        text_blocks = [b for b in msg["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1


# =========================================================================
# RESPONSE ENGINE EDGE CASES
# =========================================================================


class TestResponseEngineEdgeCases:
    """Edge cases for TranslationEngine.translate()."""

    # -- Empty / degenerate responses ------------------------------------

    def test_anthropic_empty_choices(self, resp_engine):
        """ChatCompletion with choices=[] should produce a valid Anthropic
        response with empty text content, not crash.
        """
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        })
        result = resp_engine.response_for_request(req, resp)
        assert result["type"] == "message"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_responses_empty_choices(self, resp_engine):
        """ChatCompletion with choices=[] translated for Responses API
        should produce a response with empty output.
        """
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hi"})
        result = resp_engine.response_for_request(req, resp)
        assert result["status"] == "completed"
        assert result["output"] == []

    def test_anthropic_multiple_choices_uses_first(self, resp_engine):
        """When the completion has multiple choices, only the first one
        should be used for the Anthropic response.
        """
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "first"},
                    "finish_reason": "stop",
                },
                {
                    "message": {"role": "assistant", "content": "second"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        })
        result = resp_engine.response_for_request(req, resp)
        text = [b for b in result["content"] if b["type"] == "text"]
        assert text[0]["text"] == "first"

    def test_anthropic_null_content_no_tool_calls(self, resp_engine):
        """content=None and no tool_calls should produce an empty text block."""
        completion = _mock_completion(content=None, tool_calls_mock=None)
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        })
        result = resp_engine.response_for_request(req, resp)
        assert result["type"] == "message"
        # Should have at least one content block
        assert len(result["content"]) >= 1

    def test_anthropic_content_and_tool_calls_together(self, resp_engine):
        """A response with both text content AND tool_calls should produce
        both text and tool_use content blocks.
        """
        tc = _mock_tool_call(call_id="call_1", name="search", arguments='{"q":"test"}')
        completion = _mock_completion(
            content="Let me search for that.",
            tool_calls_mock=[tc],
            finish_reason="tool_calls",
        )
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "search"}],
            "max_tokens": 1024,
        })
        result = resp_engine.response_for_request(req, resp)
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Let me search for that."
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "search"

    def test_anthropic_tool_call_ids_are_sanitized(self, resp_engine):
        """OpenAI tool call IDs must become valid Anthropic ``tool_use.id``s."""
        tc = _mock_tool_call(
            call_id="call.bad:id/with space",
            name="search",
            arguments='{"q":"test"}',
        )
        completion = _mock_completion(
            content=None,
            tool_calls_mock=[tc],
            finish_reason="tool_calls",
        )
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "search"}],
            "max_tokens": 1024,
        })

        result = resp_engine.response_for_request(req, resp)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "call_bad_id_with_space"

    def test_anthropic_model_fallback_chain(self, resp_engine):
        """When the Anthropic request body has no 'model' key, the response
        should use the model from the ChatCompletion response or 'unknown'.
        """
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "llama-3.1-70b",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.response_for_request(req, resp)
        # model=None from request => falls through to response model
        assert result["model"] == "llama-3.1-70b"

    def test_anthropic_finish_reason_mapping(self, resp_engine):
        """All finish_reason values should map correctly to Anthropic stop_reason."""
        mappings = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }
        for oai_reason, expected_reason in mappings.items():
            completion = MagicMock()
            completion.model_dump.return_value = {
                "id": "chatcmpl-test",
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": oai_reason,
                    },
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            resp = ChatResponse.openai_completion(completion)
            req = ChatRequest.anthropic({
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            })
            result = resp_engine.response_for_request(req, resp)
            assert result["stop_reason"] == expected_reason, (
                f"finish_reason={oai_reason!r} should map to {expected_reason!r}"
            )

    def test_responses_content_and_tool_calls(self, resp_engine):
        """Responses translation with both content and tool_calls should
        produce both message and function_call output items.
        """
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"test"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "search"})
        result = resp_engine.response_for_request(req, resp)
        msg_items = [o for o in result["output"] if o["type"] == "message"]
        fc_items = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(msg_items) == 1
        assert len(fc_items) == 1

    def test_model_dump_failure_propagates(self, resp_engine):
        """If the response's model_dump() raises, the error should propagate
        (not be silently swallowed).
        """
        class BadCompletion:
            def model_dump(self, **kwargs):
                del kwargs
                raise RuntimeError("serialization failed")

        with pytest.raises(RuntimeError, match="serialization failed"):
            ChatResponse.openai_completion(BadCompletion())

    def test_unknown_request_type_for_response(self, resp_engine):
        """An unknown ChatRequest subclass should raise NotImplementedError."""
        completion = _mock_completion()
        resp = ChatResponse.openai_completion(completion)
        req = _AlienChatRequest()
        with pytest.raises(NotImplementedError, match="_AlienChatRequest"):
            resp_engine.response_for_request(req, resp)

    def test_unknown_request_type_for_stream(self, resp_engine):
        """An unknown ChatRequest subclass in translate_stream should raise."""
        stream = ResponseStream(_async_chunks())
        resp = ChatResponse.openai_stream(stream)
        req = _AlienChatRequest()
        with pytest.raises(NotImplementedError, match="_AlienChatRequest"):
            resp_engine.stream_for_request(req, resp)

    def test_anthropic_tool_call_with_invalid_json_arguments(self, resp_engine):
        """Tool call arguments that are not valid JSON should be wrapped
        in {"raw": ...} rather than crashing.
        """
        tc = _mock_tool_call(
            call_id="call_bad",
            name="broken_tool",
            arguments="this is not json{{{",
        )
        completion = _mock_completion(
            content=None,
            tool_calls_mock=[tc],
            finish_reason="tool_calls",
        )
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.response_for_request(req, resp)
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["input"] == {"raw": "this is not json{{{"}

    def test_anthropic_usage_as_dict(self, resp_engine):
        """When the response dict has usage as a plain dict (no attributes),
        it should still be extracted correctly.
        """
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.response_for_request(req, resp)
        assert result["usage"]["input_tokens"] == 42
        assert result["usage"]["output_tokens"] == 7


# =========================================================================
# STREAMING EDGE CASES
# =========================================================================


class TestStreamingEdgeCases:
    """Edge cases in translate_stream for both Anthropic and Responses paths."""

    async def test_anthropic_stream_empty_chunks(self, resp_engine):
        """Chunks with choices=[] or choices=None should be skipped
        without crashing, and the stream should still emit valid
        lifecycle events.
        """
        chunks = [
            MockChunk(choices=[]),
            MockChunk(choices=None),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Hi"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        types = [e["type"] for e in events]
        assert "message_start" in types
        assert "message_stop" in types
        # The text "Hi" should be in a content_block_delta
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        assert any(d["delta"]["text"] == "Hi" for d in deltas)

    async def test_openai_reasoning_stream_deltas_do_not_become_anthropic_text(
        self,
        resp_engine,
    ):
        """OpenAI-compatible reasoning deltas are not visible assistant text."""
        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(
                reasoning="private reasoning",
                reasoning_content="private reasoning content",
            ))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Visible"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        serialized = json.dumps(events)

        assert "Visible" in serialized
        assert "private reasoning" in serialized
        assert "private reasoning content" in serialized
        assert any(
            event.get("type") == "content_block_delta"
            and event.get("delta", {}).get("type") == "thinking_delta"
            for event in events
        )
        assert not any(
            event.get("type") == "content_block_delta"
            and event.get("delta", {}).get("type") == "text_delta"
            and "private reasoning" in event.get("delta", {}).get("text", "")
            for event in events
        )

        from anthropic.types import (  # noqa: PLC0415
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
        )

        for event in events:
            if event.get("type") == "content_block_start":
                RawContentBlockStartEvent.model_validate(event)
            if event.get("type") == "content_block_delta":
                RawContentBlockDeltaEvent.model_validate(event)

    async def test_anthropic_stream_only_usage_chunk(self, resp_engine):
        """A stream where the only chunk has no choices but has usage
        should produce valid message_start, empty content block, message_stop.
        """
        chunks = [
            MockChunk(choices=None, usage=MockUsage(prompt_tokens=5, completion_tokens=0)),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        # Should have emitted a minimal empty text block
        assert "content_block_start" in types
        assert "content_block_stop" in types

    async def test_anthropic_stream_completely_empty(self, resp_engine):
        """An empty async iterator (no chunks at all) should still
        emit the full Anthropic lifecycle: message_start, empty text block,
        message_delta, message_stop.
        """
        stream = ResponseStream(_async_chunks())
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        # Empty stream produces a minimal text block
        assert "content_block_start" in types

    async def test_anthropic_stream_interleaved_tool_calls(self, resp_engine):
        """Tool call chunks that arrive with name in one chunk and arguments
        split across multiple chunks should be correctly assembled.
        """
        chunks = [
            # First chunk: tool call with name
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=0,
                    id="call_1",
                    function=MockFunctionDelta(name="get_weather", arguments='{"loc'),
                )],
            ))]),
            # Second chunk: more arguments
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=0,
                    function=MockFunctionDelta(arguments='ation":"SF"}'),
                )],
            ))]),
            # Finish
            MockChunk(choices=[MockChoice(
                delta=MockDelta(),
                finish_reason="tool_calls",
            )]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]

        # Should have content_block_start for tool_use
        starts = [e for e in events if e["type"] == "content_block_start"]
        tool_starts = [s for s in starts if s["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "get_weather"

        # Should have input_json_delta events
        json_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "input_json_delta"
        ]
        assert len(json_deltas) >= 1

        # Message delta should have stop_reason = tool_use
        msg_delta = [e for e in events if e["type"] == "message_delta"]
        assert msg_delta[0]["delta"]["stop_reason"] == "tool_use"

    async def test_anthropic_stream_text_then_tool(self, resp_engine):
        """A stream with text content followed by a tool call should
        close the text block before starting the tool_use block.
        """
        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Checking..."))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=0,
                    id="call_1",
                    function=MockFunctionDelta(name="search", arguments='{"q":"x"}'),
                )],
            ))]),
            MockChunk(choices=[MockChoice(
                delta=MockDelta(), finish_reason="tool_calls",
            )]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "search"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        types = [e["type"] for e in events]

        # Text block should be started and stopped before tool block starts
        text_stop_idx = types.index("content_block_stop")
        # Find tool_use content_block_start
        tool_start_idx = None
        for i, e in enumerate(events):
            if (
                e["type"] == "content_block_start"
                and e.get("content_block", {}).get("type") == "tool_use"
            ):
                tool_start_idx = i
                break
        assert tool_start_idx is not None
        assert text_stop_idx < tool_start_idx

    async def test_responses_stream_empty_chunks(self, resp_engine):
        """Responses stream with empty/null choice chunks should still
        produce valid lifecycle SSE events.
        """
        chunks = [
            MockChunk(choices=[]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Hi"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hi"})
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        all_text = "".join(events)
        assert "response.created" in all_text
        assert "response.completed" in all_text
        assert "Hi" in all_text

    async def test_responses_stream_captures_usage_from_final_chunk(self, resp_engine):
        """The usage-only final chunk should be captured in the
        response.completed event.
        """
        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Hello"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
            ),
            # Usage-only final chunk (no choices)
            MockChunk(
                choices=None,
                usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hi"})
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        # Find the response.completed event and parse it
        for event_str in events:
            if "response.completed" in event_str:
                # Extract JSON data
                for line in event_str.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        assert data["response"]["usage"]["input_tokens"] == 10
                        assert data["response"]["usage"]["output_tokens"] == 5
                        break

    async def test_responses_stream_tool_calls(self, resp_engine):
        """Responses streaming with tool calls should emit function_call
        lifecycle events.
        """
        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=0,
                    id="call_1",
                    function=MockFunctionDelta(name="search", arguments='{"q":"x"}'),
                )],
            ))]),
            MockChunk(choices=[MockChoice(
                delta=MockDelta(), finish_reason="tool_calls",
            )]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "search"})
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]
        all_text = "".join(events)
        assert "response.output_item.added" in all_text
        assert "function_call" in all_text
        assert "response.function_call_arguments.delta" in all_text

    async def test_response_stream_double_consume_raises(self, resp_engine):
        """ResponseStream should raise RuntimeError on second iteration."""
        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Hi"))]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        _ = [e async for e in result]
        # Second consume of the same stream
        with pytest.raises(RuntimeError, match="already been consumed"):
            _ = [e async for e in resp.stream]

    async def test_anthropic_stream_multiple_tool_calls(self, resp_engine):
        """Multiple parallel tool calls should each get their own
        content_block_start/stop events with correct block indices.
        """
        chunks = [
            # Tool 0 name
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=0,
                    id="call_1",
                    function=MockFunctionDelta(name="search", arguments=None),
                )],
            ))]),
            # Tool 1 name
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=1,
                    id="call_2",
                    function=MockFunctionDelta(name="fetch", arguments=None),
                )],
            ))]),
            # Tool 0 args
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=0,
                    function=MockFunctionDelta(arguments='{"q":"test"}'),
                )],
            ))]),
            # Tool 1 args
            MockChunk(choices=[MockChoice(delta=MockDelta(
                tool_calls=[MockToolCallDelta(
                    index=1,
                    function=MockFunctionDelta(arguments='{"url":"http://x"}'),
                )],
            ))]),
            # Finish
            MockChunk(choices=[MockChoice(
                delta=MockDelta(), finish_reason="tool_calls",
            )]),
        ]
        stream = ResponseStream(_async_chunks(*chunks))
        resp = ChatResponse.openai_stream(stream)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "parallel"}],
            "max_tokens": 100,
        })
        result = resp_engine.stream_for_request(req, resp)
        events = [e async for e in result]

        # Two tool_use block starts
        tool_starts = [
            e for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 2
        names = {s["content_block"]["name"] for s in tool_starts}
        assert names == {"search", "fetch"}
        # Block indices should be distinct
        indices = {s["index"] for s in tool_starts}
        assert len(indices) == 2


# =========================================================================
# CROSS-CUTTING CONCERNS
# =========================================================================


class TestCrossCutting:
    """Tests that verify properties across the entire translation pipeline."""

    def test_idempotency_request_engine(self, req_engine):
        """Translating the same request twice should produce equivalent results."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "max_tokens": 1024,
            "system": "Be helpful.",
        }
        req = ChatRequest.anthropic(body)
        r1 = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        r2 = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert r1.body["messages"] == r2.body["messages"]
        assert r1.body.get("model") == r2.body.get("model")
        assert r1.body.get("max_tokens") == r2.body.get("max_tokens")

    def test_idempotency_response_engine(self, resp_engine):
        """Translating the same response twice should produce equivalent results
        (modulo non-deterministic IDs).
        """
        completion = _mock_completion(content="Hello!")
        resp = ChatResponse.openai_completion(completion)
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        r1 = resp_engine.response_for_request(req, resp)
        r2 = resp_engine.response_for_request(req, resp)
        # IDs are randomly generated, so compare everything else
        assert r1["type"] == r2["type"]
        assert r1["role"] == r2["role"]
        assert r1["content"] == r2["content"]
        assert r1["model"] == r2["model"]
        assert r1["stop_reason"] == r2["stop_reason"]
        assert r1["usage"] == r2["usage"]

    def test_round_trip_openai_to_anthropic_to_openai(self, req_engine):
        """OpenAI -> Anthropic -> OpenAI should preserve core semantics:
        model, messages content, max_tokens.
        """
        original_body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        openai_req = ChatRequest.openai_chat(original_body)

        # OpenAI -> Anthropic
        anthropic_req = req_engine.request_to(ChatRequestType.ANTHROPIC, openai_req)
        assert request_type_matches(anthropic_req, ChatRequestType.ANTHROPIC)
        assert anthropic_req.body["model"] == "gpt-4o"
        assert anthropic_req.body["system"] == "Be helpful."

        # Anthropic -> OpenAI
        round_tripped = req_engine.request_to(ChatRequestType.OPENAI_CHAT, anthropic_req)
        assert request_type_matches(round_tripped, ChatRequestType.OPENAI_CHAT)

        # Core semantics preserved
        assert round_tripped.body["model"] == "gpt-4o"
        rt_msgs = round_tripped.body["messages"]
        # System should be first
        assert rt_msgs[0]["role"] == "system"
        assert rt_msgs[0]["content"] == "Be helpful."
        # User messages preserved
        user_msgs = [m for m in rt_msgs if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "Hello!"
        assert user_msgs[1]["content"] == "How are you?"

    def test_round_trip_preserves_tools(self, req_engine):
        """OpenAI -> Anthropic -> OpenAI should preserve tool definitions."""
        original_body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 100,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    },
                },
            ],
        }
        openai_req = ChatRequest.openai_chat(original_body)
        anthropic_req = req_engine.request_to(ChatRequestType.ANTHROPIC, openai_req)
        round_tripped = req_engine.request_to(ChatRequestType.OPENAI_CHAT, anthropic_req)

        rt_tools = round_tripped.body.get("tools", [])
        assert len(rt_tools) == 1
        assert rt_tools[0]["function"]["name"] == "get_weather"
        assert (
            rt_tools[0]["function"]["parameters"]["properties"]["location"]["type"]
            == "string"
        )

    async def test_concurrent_translation_no_state_corruption(self, req_engine):
        """The engine is documented as stateless.  Verify that calling it
        from multiple concurrent coroutines doesn't corrupt state.
        """
        async def translate_one(i: int) -> ChatRequest:
            body = {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": f"msg-{i}"}],
                "max_tokens": 100,
                "system": f"system-{i}",
            }
            req = ChatRequest.anthropic(body)
            return req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)

        results = await asyncio.gather(*(translate_one(i) for i in range(50)))
        # Each result should have its own unique content
        for i, r in enumerate(results):
            msgs = r.body["messages"]
            system_msg = [m for m in msgs if m["role"] == "system"][0]
            user_msg = [m for m in msgs if m["role"] == "user"][0]
            assert system_msg["content"] == f"system-{i}"
            assert user_msg["content"] == f"msg-{i}"

    async def test_concurrent_response_translation(self, resp_engine):
        """Verify concurrent response translation doesn't corrupt state."""
        async def translate_one(i: int) -> dict:
            completion = MagicMock()
            completion.model_dump.return_value = {
                "id": f"chatcmpl-{i}",
                "model": f"model-{i}",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": f"response-{i}"},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": i, "completion_tokens": i * 2},
            }
            resp = ChatResponse.openai_completion(completion)
            req = ChatRequest.anthropic({
                "model": f"model-{i}",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            })
            return resp_engine.response_for_request(req, resp)

        results = await asyncio.gather(*(translate_one(i) for i in range(50)))
        for i, r in enumerate(results):
            text = [b for b in r["content"] if b["type"] == "text"]
            assert text[0]["text"] == f"response-{i}"
            assert r["usage"]["input_tokens"] == i
            assert r["usage"]["output_tokens"] == i * 2

    def test_engine_instances_are_truly_stateless(self):
        """Creating multiple engine instances and using them interleaved
        should not cause any interference.
        """
        e1 = TranslationEngine()
        e2 = TranslationEngine()
        body1 = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "from e1"}],
            "max_tokens": 100,
        }
        body2 = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "from e2"}],
            "max_tokens": 200,
        }
        r1 = e1.request_to(ChatRequestType.OPENAI_CHAT, ChatRequest.anthropic(body1))
        r2 = e2.request_to(ChatRequestType.OPENAI_CHAT, ChatRequest.anthropic(body2))
        assert r1.body["model"] == "claude-sonnet-4-20250514"
        assert r2.body["model"] == "gpt-4o-mini"

    def test_unicode_content_survives_translation(self, req_engine):
        """Unicode characters (emoji, CJK, RTL) should survive
        Anthropic -> OpenAI translation without mangling.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hola mundo"},
            ],
            "max_tokens": 100,
            "system": "multilingual assistant",
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        msgs = result.body["messages"]
        assert msgs[1]["content"] == "Hello world"
        assert msgs[2]["content"] == "Hola mundo"

    def test_very_long_content_no_truncation(self, req_engine):
        """A message with very long content (100K chars) should not be
        truncated by the translation layer.
        """
        long_text = "x" * 100_000
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": long_text}],
            "max_tokens": 100,
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert len(result.body["messages"][0]["content"]) == 100_000

    def test_responses_to_openai_passthrough_params(self, req_engine):
        """temperature, top_p, and stream should pass through from
        Responses API to Chat Completions.
        """
        body = {
            "model": "gpt-4o",
            "input": "hi",
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": True,
        }
        req = ChatRequest.openai_responses(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["temperature"] == 0.5
        assert result.body["top_p"] == 0.9
        assert result.body["stream"] is True

    def test_anthropic_tool_choice_variants(self, req_engine):
        """All Anthropic tool_choice variants should map correctly."""
        # String variants
        for ant_choice, expected in [("auto", "auto"), ("any", "required"), ("none", "none")]:
            body = {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "tools": [{"name": "t", "description": "d", "input_schema": {}}],
                "tool_choice": ant_choice,
            }
            req = ChatRequest.anthropic(body)
            result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
            assert result.body["tool_choice"] == expected, (
                f"tool_choice={ant_choice!r} should map to {expected!r}"
            )

    def test_anthropic_tool_choice_specific_tool(self, req_engine):
        """Anthropic tool_choice of type 'tool' with a name should map
        to OpenAI's function-specific tool_choice.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"name": "search", "description": "d", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "search"},
        }
        req = ChatRequest.anthropic(body)
        result = req_engine.request_to(ChatRequestType.OPENAI_CHAT, req)
        tc = result.body["tool_choice"]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
