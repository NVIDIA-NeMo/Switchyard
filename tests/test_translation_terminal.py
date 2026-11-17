# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from switchyard.translation import TranslationEngine


@pytest.mark.parametrize(
    ("protocol", "terminal"),
    [
        ("anthropic_messages", {"type": "message_stop"}),
        ("openai_responses", {"type": "response.completed", "response": {}}),
        ("openai_responses", {"type": "response.incomplete", "response": {}}),
        ("openai_responses", {"type": "response.failed", "response": {}}),
    ],
)
async def test_decoder_stops_at_terminal_event(protocol, terminal):
    """The decoder yields the terminal event without polling upstream again."""

    async def upstream():
        """The iterator raises if the decoder reads beyond the terminal event."""
        yield f"data: {json.dumps(terminal)}\n\n".encode()
        raise AssertionError("The decoder read after the terminal event.")

    async with TranslationEngine().decode_stream(protocol, upstream()) as decoded:
        assert [event["preservation"]["raw"] async for event in decoded] == [terminal]


async def test_chat_decoder_yields_usage_after_finish_reason():
    """The decoder keeps Chat Completions usage that follows finish_reason."""
    usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}

    async def upstream():
        """The iterator sends usage after finish_reason and raises after [DONE]."""
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()
        yield b"data: [DONE]\n\n"
        raise AssertionError("The decoder read after [DONE].")

    async with TranslationEngine().decode_stream("openai_chat", upstream()) as decoded:
        events = [event async for event in decoded]
    assert events[-1]["preservation"]["raw"]["usage"] == usage
    decoded_usage = next(chunk["Usage"] for chunk in events[-1]["normalized"] if "Usage" in chunk)
    assert decoded_usage["input_tokens"] == 2
    assert decoded_usage["output_tokens"] == 3
    assert decoded_usage["total_tokens"] == 5
