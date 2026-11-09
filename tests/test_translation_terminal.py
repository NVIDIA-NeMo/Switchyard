# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder stops at protocol completion without waiting for EOF."""

import asyncio
import json

import pytest

from switchyard.translation import TranslationEngine

FORMATS = ["openai_chat", "anthropic_messages", "openai_responses"]


def source_events(source):
    """Chat usage follows finish_reason; Responses text comes from the final snapshot."""
    if source == "openai_chat":
        return [
            {
                "id": "chat-1",
                "model": "backend",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}}],
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        ]
    if source == "anthropic_messages":
        return [
            {
                "type": "message_start",
                "message": {
                    "id": "msg-1",
                    "model": "backend",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
    return [
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "backend",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
        }
    ]


def sse(events, source):
    payload = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events)
    return payload + (b"data: [DONE]\n\n" if source == "openai_chat" else b"")


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", FORMATS)
@pytest.mark.parametrize("split_tail", [False, True])
async def test_translation_ignores_data_after_terminal(source, target, split_tail):
    payload = sse(source_events(source), source)
    tail_event = {
        "openai_chat": {"choices": [{"index": 0, "delta": {"content": "AFTER_TERMINAL"}}]},
        "anthropic_messages": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "AFTER_TERMINAL"},
        },
        "openai_responses": {
            "type": "response.output_text.delta",
            "item_id": "msg-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "AFTER_TERMINAL",
        },
    }[source]
    # The invalid frame must be ignored even when it is already buffered.
    tail = sse([tail_event], source) + b"data: invalid-json\n\n"
    reads = 0

    async def chunks():
        """The second chunk exposes any read after protocol completion."""
        nonlocal reads
        reads += 1
        yield payload if split_tail else payload + tail
        reads += 1
        yield tail

    engine = TranslationEngine()
    async with engine.translate_stream(source, target, chunks()) as translated:
        output = [event async for event in translated]
    assert reads == 1
    assert "AFTER_TERMINAL" not in json.dumps(output)
    if target == "openai_chat":
        assert any(
            event.get("choices") and event["choices"][0]["delta"].get("content") == "hello"
            for event in output
        )
        usage = next(event["usage"] for event in output if "usage" in event)
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 2
    elif target == "anthropic_messages":
        assert any(event.get("delta", {}).get("text") == "hello" for event in output)
        assert output[-1]["type"] == "message_stop"
        assert (
            next(
                event["usage"]["output_tokens"]
                for event in output
                if event["type"] == "message_delta"
            )
            == 2
        )
    else:
        assert output[-1]["type"] == "response.completed"
        response = output[-1]["response"]
        assert response["output"][0]["content"][0]["text"] == "hello"
        assert response["usage"]["input_tokens"] == 5
        assert response["usage"]["output_tokens"] == 2


@pytest.mark.parametrize(
    "source,terminal",
    [
        ("openai_chat", None),
        ("anthropic_messages", {"type": "message_stop"}),
        *[
            (
                "openai_responses",
                {
                    key: kind,
                    "response": {
                        "id": "resp-1",
                        "status": kind.removeprefix("response."),
                        "output": [],
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                        **(
                            {"error": {"code": "server_error", "message": "provider failed"}}
                            if kind == "response.failed"
                            else {}
                        ),
                    },
                },
            )
            for key in ("type", "event")
            for kind in ("response.completed", "response.incomplete", "response.failed")
        ],
    ],
)
async def test_decoder_completes_without_reading_after_terminal(source, terminal):
    """The terminal payload survives decoding even when the input never reaches EOF."""
    events = source_events(source) if terminal is None else [terminal]
    read_after_terminal = False

    async def held_open():
        nonlocal read_after_terminal
        # Byte-sized chunks exercise partial frames without reading beyond completion.
        for byte in sse(events, source):
            yield bytes([byte])
        read_after_terminal = True
        await asyncio.Future()

    async def collect():
        async with TranslationEngine().decode_stream(source, held_open()) as decoded:
            return [event async for event in decoded]

    output = await asyncio.wait_for(collect(), 2)
    assert not read_after_terminal
    assert [event["preservation"]["raw"] for event in output] == events
    if terminal is not None and terminal.get("type", terminal.get("event")) == "response.failed":
        assert {"StreamError": {"message": "provider failed"}} in output[-1]["normalized"]
