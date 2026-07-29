# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI Responses <-> OpenAI Chat translation."""

from unittest.mock import MagicMock

import pytest

from switchyard_rust.translation import TranslationEngine

ENGINE = TranslationEngine()


def _responses_request_to_chat(body: object) -> dict:
    return ENGINE.translate_request("openai_responses", "openai_chat", body)


def _chat_response_to_responses(response: object) -> dict:
    return ENGINE.translate_response("openai_chat", "openai_responses", response)

# ---------------------------------------------------------------------------
# Request conversion tests
# ---------------------------------------------------------------------------


class TestConvertResponsesRequestToChatCompletions:
    """Tests for convert_responses_request_to_chat_completions."""

    def test_simple_string_input(self):
        body = {
            "model": "gpt-4",
            "input": "Hello, world!",
        }
        result = _responses_request_to_chat(body)

        assert result["model"] == "gpt-4"
        assert len(result["messages"]) == 1
        assert result["messages"][0] == {"role": "user", "content": "Hello, world!"}

    def test_instructions_becomes_system_message(self):
        body = {
            "model": "gpt-4",
            "input": "What is 2+2?",
            "instructions": "You are a math tutor.",
        }
        result = _responses_request_to_chat(body)

        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "system", "content": "You are a math tutor."}
        assert result["messages"][1] == {"role": "user", "content": "What is 2+2?"}

    def test_max_output_tokens_becomes_max_completion_tokens(self):
        body = {
            "model": "gpt-4",
            "input": "Hi",
            "max_output_tokens": 1024,
        }
        result = _responses_request_to_chat(body)

        assert result["max_completion_tokens"] == 1024
        assert "max_tokens" not in result
        assert "max_output_tokens" not in result

    def test_tools_conversion(self):
        body = {
            "model": "gpt-4",
            "input": "Get weather",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
        }
        result = _responses_request_to_chat(body)

        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get the weather"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_tools_preserve_strict_and_nested_chat_shape(self):
        body = {
            "model": "gpt-4",
            "input": "Get weather",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup data",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
        }
        result = _responses_request_to_chat(body)

        assert result["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ]

    def test_codex_tools_format(self):
        """Codex CLI sends tools with id/inputSchema instead of name/parameters."""
        body = {
            "model": "gpt-4",
            "input": "List files",
            "tools": [
                {
                    "id": "exec_command",
                    "description": "Runs a command in a PTY.",
                    "inputSchema": {
                        "jsonSchema": {
                            "type": "object",
                            "properties": {
                                "cmd": {"type": "string", "description": "Shell command"},
                            },
                            "required": ["cmd"],
                            "additionalProperties": False,
                        }
                    },
                },
                {
                    "id": "view_image",
                    "description": "View a local image.",
                    "inputSchema": {
                        "jsonSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                            },
                            "required": ["path"],
                        }
                    },
                },
            ],
        }
        result = _responses_request_to_chat(body)

        assert len(result["tools"]) == 2
        tool0 = result["tools"][0]
        assert tool0["type"] == "function"
        assert tool0["function"]["name"] == "exec_command"
        assert tool0["function"]["description"] == "Runs a command in a PTY."
        assert tool0["function"]["parameters"]["type"] == "object"
        assert "cmd" in tool0["function"]["parameters"]["properties"]

        tool1 = result["tools"][1]
        assert tool1["function"]["name"] == "view_image"
        assert tool1["function"]["parameters"]["required"] == ["path"]

    def test_empty_tools_filtered(self):
        """Ghost tools with empty id/name should be filtered out."""
        body = {
            "model": "gpt-4",
            "input": "Hi",
            "tools": [
                {
                    "id": "exec_command",
                    "description": "Run a command",
                    "inputSchema": {"jsonSchema": {"type": "object", "properties": {}}},
                },
                {
                    "id": "",
                    "description": "",
                    "inputSchema": {"jsonSchema": {}},
                },
            ],
        }
        result = _responses_request_to_chat(body)

        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "exec_command"

    def test_tool_choice_dropped_when_no_tools(self):
        # tool_choice without tools is invalid for OpenAI — must not be forwarded
        body = {
            "model": "gpt-4",
            "input": "Hi",
            "tool_choice": "required",
        }
        result = _responses_request_to_chat(body)
        assert "tool_choice" not in result

    def test_function_tool_choice_maps_to_chat_shape(self):
        body = {
            "model": "gpt-4",
            "input": "Hi",
            "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
            "tool_choice": {"type": "function", "name": "lookup"},
        }
        result = _responses_request_to_chat(body)
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "lookup"},
        }

    def test_unsupported_hosted_tools_and_tool_choice_are_dropped(self):
        body = {
            "model": "gpt-4",
            "input": "Search the web",
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": {"type": "web_search_preview"},
        }
        result = _responses_request_to_chat(body)
        assert "tools" not in result
        assert "tool_choice" not in result

    def test_passthrough_params(self):
        body = {
            "model": "gpt-4",
            "input": "Hi",
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "parallel_tool_calls": False,
            "metadata": {"trace": "abc"},
            "store": False,
            "stream_options": {"include_usage": True},
            "prompt_cache_key": "session-1",
            "service_tier": "flex",
            "user": "u-123",
        }
        result = _responses_request_to_chat(body)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is True
        assert result["parallel_tool_calls"] is False
        assert result["metadata"] == {"trace": "abc"}
        assert result["store"] is False
        assert result["stream_options"] == {"include_usage": True}
        assert result["prompt_cache_key"] == "session-1"
        assert result["service_tier"] == "flex"
        assert result["user"] == "u-123"

    def test_reasoning_and_text_format_map_to_chat_fields(self):
        body = {
            "model": "gpt-5",
            "input": "Return JSON",
            "reasoning": {"effort": "high"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }
        result = _responses_request_to_chat(body)
        assert result["reasoning_effort"] == "high"
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        }

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("previous_response_id", "resp_123"),
            ("conversation", "conv_123"),
            ("conversation", {"id": "conv_123"}),
        ],
    )
    def test_stateful_responses_fields_are_forgotten(self, field, value):
        body = {
            "model": "gpt-4",
            "input": "Continue",
            field: value,
        }
        result = _responses_request_to_chat(body)

        assert result["messages"] == [{"role": "user", "content": "Continue"}]
        assert result["model"] == "gpt-4"
        assert field not in result

    def test_multi_turn_input_with_messages(self):
        body = {
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi there!"}],
                },
                {"type": "message", "role": "user", "content": "How are you?"},
            ],
        }
        result = _responses_request_to_chat(body)

        assert len(result["messages"]) == 3
        assert result["messages"][0] == {"role": "user", "content": "Hello"}
        assert result["messages"][1] == {"role": "assistant", "content": "Hi there!"}
        assert result["messages"][2] == {"role": "user", "content": "How are you?"}

    def test_multi_turn_with_function_calls_and_outputs(self):
        body = {
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "Get weather in SF and NY"},
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "call_id": "call_abc",
                    "arguments": '{"location": "SF"}',
                },
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "call_id": "call_def",
                    "arguments": '{"location": "NY"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "72F sunny",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_def",
                    "output": "65F cloudy",
                },
            ],
        }
        result = _responses_request_to_chat(body)

        msgs = result["messages"]
        assert msgs[0] == {"role": "user", "content": "Get weather in SF and NY"}

        # Consecutive function_calls should be merged into one assistant message
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] is None
        assert len(msgs[1]["tool_calls"]) == 2
        assert msgs[1]["tool_calls"][0]["id"] == "call_abc"
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[1]["tool_calls"][1]["id"] == "call_def"

        # function_call_output -> tool messages
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "call_abc"
        assert msgs[2]["content"] == "72F sunny"

        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "call_def"
        assert msgs[3]["content"] == "65F cloudy"

    def test_interleaved_function_calls_produce_separate_turns(self):
        """Sequential function_call/output pairs represent separate turns.

        Codex sends interleaved pairs (call A, output A, call B, output B)
        when each call was a separate LLM turn.  The transition from
        function_call_output -> function_call marks a turn boundary.
        Each turn should produce its own assistant message.
        """
        body = {
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "Explore the repo"},
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_1",
                    "arguments": '{"cmd": "ls -la"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file1.py file2.py",
                },
                {
                    "type": "function_call",
                    "name": "str_replace_editor",
                    "call_id": "call_2",
                    "arguments": '{"filename": "AGENTS.md", "command": "view"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "# Agents\n...",
                },
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_3",
                    "arguments": '{"cmd": "cat README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_3",
                    "output": "# README",
                },
            ],
        }
        result = _responses_request_to_chat(body)
        msgs = result["messages"]

        # user message
        assert msgs[0] == {"role": "user", "content": "Explore the repo"}

        # Turn 1: call_1
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] is None
        assert len(msgs[1]["tool_calls"]) == 1
        assert msgs[1]["tool_calls"][0]["id"] == "call_1"
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "exec_command"
        assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "content": "file1.py file2.py"}

        # Turn 2: call_2
        assert msgs[3]["role"] == "assistant"
        assert len(msgs[3]["tool_calls"]) == 1
        assert msgs[3]["tool_calls"][0]["id"] == "call_2"
        assert msgs[3]["tool_calls"][0]["function"]["name"] == "str_replace_editor"
        assert msgs[4] == {"role": "tool", "tool_call_id": "call_2", "content": "# Agents\n..."}

        # Turn 3: call_3
        assert msgs[5]["role"] == "assistant"
        assert len(msgs[5]["tool_calls"]) == 1
        assert msgs[5]["tool_calls"][0]["id"] == "call_3"
        assert msgs[5]["tool_calls"][0]["function"]["name"] == "exec_command"
        assert msgs[6] == {"role": "tool", "tool_call_id": "call_3", "content": "# README"}

        # Total: 1 user + 3 * (assistant + tool) = 7 messages
        assert len(msgs) == 7

    def test_tool_blocks_separated_by_message(self):
        """Tool blocks separated by a message should produce separate assistant messages."""
        body = {
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "Do task"},
                {
                    "type": "function_call",
                    "name": "tool_a",
                    "call_id": "call_a",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "result_a",
                },
                # Assistant text response separates the two blocks
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I found something."}],
                },
                {
                    "type": "function_call",
                    "name": "tool_b",
                    "call_id": "call_b",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": "result_b",
                },
            ],
        }
        result = _responses_request_to_chat(body)
        msgs = result["messages"]

        # user, assistant(tool_a), tool_a, assistant_text, assistant(tool_b), tool_b
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["tool_calls"]) == 1
        assert msgs[1]["tool_calls"][0]["id"] == "call_a"
        assert msgs[2]["role"] == "tool"
        assert msgs[3] == {"role": "assistant", "content": "I found something."}
        assert msgs[4]["role"] == "assistant"
        assert len(msgs[4]["tool_calls"]) == 1
        assert msgs[4]["tool_calls"][0]["id"] == "call_b"
        assert msgs[5]["role"] == "tool"
        assert len(msgs) == 6

    def test_message_between_function_call_and_output_is_deferred(self):
        """Codex can inject warnings between a function_call and its output.

        Chat/Anthropic-compatible histories need the tool result immediately
        after the assistant tool call, so preserve the warning as later context.
        """
        body = {
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "Do task"},
                {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "tooluse_1",
                    "arguments": '{"command":["apply_patch","..."]}',
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Warning: apply_patch was requested via shell. "
                                "Use the apply_patch tool instead."
                            ),
                        }
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "tooluse_1",
                    "output": "Success",
                },
            ],
        }
        result = _responses_request_to_chat(body)

        assert result["messages"] == [
            {"role": "user", "content": "Do task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tooluse_1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command":["apply_patch","..."]}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tooluse_1", "content": "Success"},
            {
                "role": "user",
                "content": (
                    "Warning: apply_patch was requested via shell. "
                    "Use the apply_patch tool instead."
                ),
            },
        ]

    def test_content_blocks_flattened(self):
        """Content blocks with input_text/output_text types should be flattened."""
        body = {
            "model": "gpt-4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello "},
                        {"type": "input_text", "text": "world"},
                    ],
                },
            ],
        }
        result = _responses_request_to_chat(body)
        assert result["messages"][0]["content"] == "Hello \nworld"

    def test_multimodal_content_blocks_preserved_for_chat(self):
        body = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.test/cat.png",
                            "detail": "low",
                        },
                    ],
                },
            ],
        }
        result = _responses_request_to_chat(body)

        assert result["messages"][0]["content"] == [
            {"type": "text", "text": "Describe this"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.test/cat.png",
                    "detail": "low",
                },
            },
        ]

    def test_orphan_function_call_output_becomes_user_context(self):
        body = {
            "model": "gpt-4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_orphan",
                    "output": "result",
                },
            ],
        }
        result = _responses_request_to_chat(body)

        assert result["messages"] == [
            {"role": "user", "content": "Tool result call_orphan: result"}
        ]

    def test_non_json_tool_payloads_fall_back_to_text(self):
        recursive: list[object] = []
        recursive.append(recursive)
        body = {
            "model": "gpt-4",
            "input": [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "call_id": "call_recursive",
                    "arguments": recursive,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_recursive",
                    "output": recursive,
                },
            ],
        }
        result = _responses_request_to_chat(body)

        messages = result["messages"]
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == "[[...]]"
        assert messages[1]["content"] == "[[...]]"

    def test_codex_reasoning_items_do_not_become_empty_assistant_messages(self):
        """Codex replays each model turn as message + reasoning + function_call items.

        The reasoning item must ride with the turn's tool-call message instead of
        surfacing as a fabricated ``{"role": "assistant", "content": ""}`` chat
        message; those empty turns accumulate every round-trip and degrade the
        upstream model until it stops emitting tool calls.
        """
        body = {
            "model": "big-reasoner",
            "instructions": "You are a coding agent.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix the broken pip install."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "\n\n"}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": "Let me check the python setup."}
                    ],
                    "encrypted_content": None,
                },
                {
                    "type": "function_call",
                    "name": "shell_command",
                    "arguments": '{"command":"pip3 --version","workdir":"/app"}',
                    "call_id": "call-1",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "ModuleNotFoundError: No module named 'pip'",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "\n\n"}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "pip is missing; ensurepip."}],
                    "encrypted_content": None,
                },
                {
                    "type": "function_call",
                    "name": "shell_command",
                    "arguments": '{"command":"python3 -m ensurepip","workdir":"/app"}',
                    "call_id": "call-2",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-2",
                    "output": "Successfully installed pip",
                },
            ],
        }
        result = _responses_request_to_chat(body)

        assert result["messages"] == [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the broken pip install."},
            {"role": "assistant", "content": "\n\n"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": '{"command":"pip3 --version","workdir":"/app"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "ModuleNotFoundError: No module named 'pip'",
            },
            {"role": "assistant", "content": "\n\n"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": '{"command":"python3 -m ensurepip","workdir":"/app"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": "Successfully installed pip",
            },
        ]

    def test_reasoning_item_merges_into_following_assistant_message(self):
        """Native item ordering puts reasoning before the assistant message it belongs to."""
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "Check the file"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Reading it now."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
                {
                    "type": "function_call",
                    "name": "read_file",
                    "call_id": "call-r",
                    "arguments": '{"path":"a.py"}',
                },
                {"type": "function_call_output", "call_id": "call-r", "output": "print(1)"},
            ],
        }
        result = _responses_request_to_chat(body)

        assert result["messages"] == [
            {"role": "user", "content": "Check the file"},
            {"role": "assistant", "content": "Let me check."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-r",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-r", "content": "print(1)"},
        ]

    def test_reasoning_only_turn_does_not_add_empty_assistant_message(self):
        """A reasoning item directly before a tool call must not split the turn."""
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "List files"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Simple ls."}],
                },
                {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call-ls",
                    "arguments": '{"command":"ls"}',
                },
                {"type": "function_call_output", "call_id": "call-ls", "output": "a.py"},
            ],
        }
        result = _responses_request_to_chat(body)

        empty_assistants = [
            message
            for message in result["messages"]
            if message.get("role") == "assistant" and message.get("content") == ""
        ]
        assert empty_assistants == []
        assert result["messages"][1]["tool_calls"][0]["id"] == "call-ls"


# ---------------------------------------------------------------------------
# Response conversion tests
# ---------------------------------------------------------------------------


class TestConvertChatResponseToResponses:
    """Tests for convert_chat_response_to_responses."""

    def test_text_response(self):
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = _chat_response_to_responses(response)

        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "gpt-4"

        assert len(result["output"]) == 1
        out = result["output"][0]
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["content"][0]["type"] == "output_text"
        assert out["content"][0]["text"] == "Hello!"

    def test_tool_calls_response(self):
        response = {
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "SF"}',
                                },
                            },
                            {
                                "id": "call_def",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NY"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50,
            },
        }
        result = _chat_response_to_responses(response)

        assert result["status"] == "completed"
        # No text message, just function calls
        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "function_call"
        assert result["output"][0]["call_id"] == "call_abc"
        assert result["output"][0]["name"] == "get_weather"
        assert result["output"][0]["arguments"] == '{"location": "SF"}'

        assert result["output"][1]["type"] == "function_call"
        assert result["output"][1]["call_id"] == "call_def"

    def test_usage_mapping(self):
        response = {
            "id": "chatcmpl-789",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        }
        result = _chat_response_to_responses(response)

        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150
        assert result["usage"]["output_tokens_details"] == {"reasoning_tokens": 12}

    def test_response_object_with_model_dump(self):
        """Test with a response object that has model_dump (like litellm responses)."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-obj",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "From object"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        result = _chat_response_to_responses(mock_response)
        assert result["output"][0]["content"][0]["text"] == "From object"

    def test_mixed_text_and_tool_calls(self):
        response = {
            "id": "chatcmpl-mixed",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = _chat_response_to_responses(response)

        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["text"] == "Let me check that."
        assert result["output"][1]["type"] == "function_call"
        assert result["output"][1]["name"] == "lookup"

    def test_reasoning_content_with_tool_calls_has_no_empty_message_item(self):
        """Reasoning + tool calls with empty text must not fabricate an empty message item."""
        response = {
            "id": "chatcmpl-reason",
            "model": "big-reasoner",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Inspect the repository first.",
                        "tool_calls": [
                            {
                                "id": "call-9",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"ls"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = _chat_response_to_responses(response)

        assert [item["type"] for item in result["output"]] == ["reasoning", "function_call"]
        reasoning = result["output"][0]
        assert reasoning["content"] == [
            {"type": "reasoning_text", "text": "Inspect the repository first."}
        ]
        function_call = result["output"][1]
        assert function_call["call_id"] == "call-9"
        assert function_call["name"] == "shell"
        assert function_call["arguments"] == '{"command": "ls"}'


# ---------------------------------------------------------------------------
# Streaming conversion tests (chat chunks -> Responses SSE events)
# ---------------------------------------------------------------------------


def _chat_chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "model": "big-reasoner",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def _translate_chat_stream_to_responses(chunks: list[dict]) -> list[dict]:
    async def _source():
        for chunk in chunks:
            yield chunk

    return [
        event
        async for event in ENGINE.translate_stream("openai_chat", "openai_responses", _source())
    ]


class TestChatStreamToResponsesSse:
    """Pins the SSE contract codex consumes for streamed tool calls."""

    async def test_tool_call_stream_produces_well_formed_function_call_events(self):
        events = await _translate_chat_stream_to_responses(
            [
                _chat_chunk({"role": "assistant", "content": ""}),
                _chat_chunk({"reasoning_content": "Need to inspect the repo."}),
                _chat_chunk({"content": "\n\n"}),
                _chat_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-9",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"command":'},
                            }
                        ]
                    }
                ),
                _chat_chunk(
                    {"tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]}
                ),
                _chat_chunk({}, finish="tool_calls"),
            ]
        )

        added = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "function_call"
        ]
        assert len(added) == 1
        assert added[0]["call_id"] == "call-9"
        assert added[0]["name"] == "shell"

        argument_deltas = [
            event["delta"]
            for event in events
            if event["type"] == "response.function_call_arguments.delta"
        ]
        assert "".join(argument_deltas) == '{"command":"ls"}'

        done = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        ]
        assert len(done) == 1
        assert done[0]["call_id"] == "call-9"
        assert done[0]["name"] == "shell"
        assert done[0]["arguments"] == '{"command":"ls"}'
        assert done[0]["status"] == "completed"

        completed = [event for event in events if event["type"] == "response.completed"]
        assert len(completed) == 1
        output_types = [item["type"] for item in completed[0]["response"]["output"]]
        assert output_types == ["reasoning", "message", "function_call"]
        function_item = completed[0]["response"]["output"][2]
        assert function_item["call_id"] == "call-9"
        assert function_item["arguments"] == '{"command":"ls"}'
