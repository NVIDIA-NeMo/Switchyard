# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared message-condensing helpers."""

from __future__ import annotations

from switchyard.lib.processors.message_condensing import (
    content_text,
    message_text,
    strip_markdown_fence,
    truncate_middle,
)


def test_content_text_renders_anthropic_tool_use() -> None:
    """tool_use blocks have neither text nor nested content; render name+input."""
    blocks = [
        {"type": "text", "text": "let me check"},
        {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "pytest -x"}},
    ]

    text = content_text(blocks)

    assert "let me check" in text
    assert 'tool_call bash({"command": "pytest -x"})' in text


def test_content_text_renders_tool_result_content() -> None:
    blocks = [
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": [{"type": "text", "text": "1 failed"}],
        },
    ]
    assert content_text(blocks) == "1 failed"


def test_message_text_covers_all_three_tool_call_shapes() -> None:
    chat = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
    }
    assert message_text(chat) == 'tool_call read_file({"path": "a.py"})'

    responses_item = {"type": "function_call", "name": "bash", "arguments": '{"command": "ls"}'}
    assert message_text(responses_item) == 'tool_call bash({"command": "ls"})'

    anthropic = {
        "role": "assistant",
        "content": [{"type": "tool_use", "name": "bash", "input": {"command": "ls"}}],
    }
    assert message_text(anthropic) == 'tool_call bash({"command": "ls"})'


def test_truncate_middle_keeps_head_and_tail() -> None:
    text = "H" * 100 + "T" * 100
    out = truncate_middle(text, 60)
    assert len(out) <= 60
    assert out.startswith("H")
    assert out.endswith("T")
    assert "...[trimmed]" in out


def test_strip_markdown_fence_unwraps_json() -> None:
    fenced = '```json\n{"escalate": true}\n```'
    assert strip_markdown_fence(fenced) == '{"escalate": true}'
    assert strip_markdown_fence('{"escalate": true}') == '{"escalate": true}'
