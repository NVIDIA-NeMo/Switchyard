# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming translation, error propagation, and caller-owned input lifetimes."""

import asyncio
import json
import weakref

import pytest

from switchyard.libsy import (
    CustomClassifierConfig,
    LlmClassifierConfig,
    LlmResponse,
    Step,
    algorithms,
)
from switchyard.translation import TranslationEngine, TranslationError, WireFormat


async def byte_stream(events, *, terminal=True):
    payload = b"".join(
        ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode() for event in events
    )
    if terminal:
        payload += b"data: [DONE]\n\n"
    # Split SSE boundaries and multi-byte UTF-8 characters across reads.
    for byte in payload:
        yield bytes([byte])


async def items(values):
    for value in values:
        yield value


async def test_stream_translation_preserves_ir_and_restores_extensions():
    engine = TranslationEngine()
    request = engine.decode_request(
        WireFormat.OPENAI_RESPONSES,
        {
            "input": "Search",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__docs",
                    "tools": [{"type": "function", "name": "search", "parameters": {}}],
                }
            ],
        },
    )
    events = [
        {
            "id": "response-1",
            "model": "backend",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Héllo"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__docs__search",
                                    "arguments": '{"query":"streaming"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    async with engine.decode_stream(WireFormat.OPENAI_CHAT, byte_stream(events)) as decoded:
        ir = [event async for event in decoded]
    assert [event["preservation"]["raw"] for event in ir] == events
    assert {"TextDelta": {"index": 0, "text": "Héllo"}} in ir[0]["normalized"]

    # Direct iteration works, including replay and empty extension equivalence.
    for extensions in (None, {}, {"fields": {}}):
        replay = engine.encode_stream("openai_chat", items(ir), request_extensions=extensions)
        assert [event async for event in replay] == events

    encoded = engine.encode_stream(
        WireFormat.OPENAI_RESPONSES,
        items(ir),
        served_model="selected",
        request_extensions=request["request"]["extensions"],
    )
    del engine
    async with encoded:
        output = [event async for event in encoded]
    assert output[-1]["type"] == "response.completed"
    assert output[-1]["response"]["model"] == "selected"
    call = next(
        item for item in output[-1]["response"]["output"] if item["type"] == "function_call"
    )
    assert call["name"] == "search"
    assert call["namespace"] == "mcp__docs"

    engine = TranslationEngine()
    async with engine.translate_stream(
        "openai_chat",
        "openai_responses",
        byte_stream(events),
        served_model="selected",
        request_extensions=request["request"]["extensions"],
    ) as translated:
        translated_output = [event async for event in translated]
    assert translated_output[-1]["response"]["model"] == "selected"
    assert translated_output[-1]["response"]["output"] == output[-1]["response"]["output"]

    async with engine.decode_stream("openai_chat", byte_stream(events)) as decoded:
        async with engine.encode_stream("anthropic_messages", decoded) as encoded:
            output = [event async for event in encoded]
    assert output[-1]["type"] == "message_stop"
    assert any(event.get("delta", {}).get("text") == "Héllo" for event in output)
    translated = engine.translate_stream("openai_chat", "anthropic_messages", byte_stream(events))
    assert [event async for event in translated] == output

    # A decoded stream can also be consumed by the existing libsy response adapter.
    classifier = algorithms.llm_classifier(
        LlmClassifierConfig.custom(
            default_target="capable",
            config=CustomClassifierConfig(
                "Choose a target.",
                {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                "/target",
            ),
        )
    )
    classification = [
        {"choices": [{"delta": {"content": '{"target":"efficient"}'}, "finish_reason": "stop"}]}
    ]
    async with engine.decode_stream("openai_chat", byte_stream(classification)) as decoded:

        async def response():
            return LlmResponse.Stream(decoded)

        outcome = None
        async for step in classifier.run_stream(
            request["request"],
            {
                "judge": ["judge"],
                "capable": ["large"],
                "efficient": ["small"],
                "any": ["large", "small"],
            },
        ):
            if isinstance(step, Step.CallModel):
                await step.call.respond(response())
            elif isinstance(step, Step.Done):
                outcome = step.outcome
        assert outcome is not None
        assert outcome.selected_model_ids[0] == "small"


@pytest.mark.parametrize(
    "method,kind",
    [
        ("decode_stream", "ResponseTranslation"),
        ("encode_stream", "Configuration"),
        ("translate_stream", "ResponseTranslation"),
    ],
)
async def test_stream_errors_preserve_exceptions(method, kind):
    engine = TranslationEngine()
    make_stream = getattr(engine, method)
    formats = ("openai_chat", "anthropic_messages") if method == "translate_stream" else ("openai_chat",)
    with pytest.raises(TranslationError, match="no stream codec registered for fake") as caught:
        make_stream("fake", *formats[1:], items([]))
    assert caught.value.kind == kind
    assert caught.value.details == {"message": "no stream codec registered for fake"}

    if method == "translate_stream":
        with pytest.raises(TranslationError, match="no stream codec registered for fake") as caught:
            make_stream("openai_chat", "fake", items([]))
        assert caught.value.kind == "Configuration"

    if method != "decode_stream":
        with pytest.raises(ValueError):
            make_stream(*formats, items([]), request_extensions={"fields": []})

    failure = OSError("input failed")

    async def failing_input():
        raise failure
        yield  # pragma: no cover

    async with make_stream(*formats, failing_input()) as stream:
        with pytest.raises(OSError) as caught:
            await anext(stream)
        assert caught.value is failure

    invalid = [] if method == "encode_stream" else "not bytes"
    async with make_stream(*formats, items([invalid])) as stream:
        with pytest.raises(TypeError):
            await anext(stream)

    if method == "encode_stream":
        async with engine.encode_stream(
            "openai_chat", items([{"normalized": "invalid"}])
        ) as stream:
            with pytest.raises(ValueError):
                await anext(stream)
        return

    for payload in (b"data: {\n\n", b""):
        async with make_stream(*formats, items([payload])) as stream:
            with pytest.raises(TranslationError) as caught:
                await anext(stream)
            assert caught.value.kind == "ResponseTranslation"
            assert caught.value.details["message"]


@pytest.mark.parametrize("method", ["decode_encode", "translate_stream"])
async def test_stream_error_events_become_upstream_failures(method):
    engine = TranslationEngine()
    for raw, variant, message in [
        ([], "DecodeError", "OpenAI stream event is not an object"),
        ({"error": {"message": "provider failed"}}, "StreamError", "provider failed"),
    ]:
        if method == "decode_encode":
            async with engine.decode_stream(
                "openai_chat", byte_stream([raw], terminal=False)
            ) as decoded:
                ir = [event async for event in decoded]
            assert ir == [
                {
                    "preservation": {"source": "openai_chat", "raw": raw},
                    "normalized": [{variant: {"message": message}}],
                }
            ]
            stream = engine.encode_stream("anthropic_messages", items(ir))
        else:
            stream = engine.translate_stream(
                "openai_chat", "anthropic_messages", byte_stream([raw], terminal=False)
            )
        async with stream:
            with pytest.raises(TranslationError) as caught:
                async for _ in stream:
                    pass
        assert caught.value.kind == "UpstreamFailure"
        assert caught.value.details["error"]["type"] == "error"
        assert caught.value.details["error"]["error"]["message"] == message


@pytest.mark.parametrize("method", ["decode_stream", "encode_stream", "translate_stream"])
@pytest.mark.parametrize("exit_mode", ["break", "cancel", "close", "cleanup-error"])
async def test_stream_cleanup_owns_only_translation_resources(method, exit_mode):
    started = asyncio.Event()
    cleaned = asyncio.Event()
    cleanup_error = RuntimeError("input read cleanup failed")

    class Input:
        def __init__(self):
            self.reads = 0
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.reads += 1
            if self.reads == 1:
                if method != "encode_stream":
                    return b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                return {
                    "preservation": None,
                    "normalized": [{"TextDelta": {"index": 0, "text": "hello"}}],
                }
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()
                if exit_mode == "cleanup-error":
                    raise cleanup_error

        async def aclose(self):
            self.close_calls += 1

    source = Input()
    source_ref = weakref.ref(source)
    engine = TranslationEngine()
    if method == "translate_stream":
        stream = engine.translate_stream("openai_chat", "openai_chat", source)
    else:
        stream = getattr(engine, method)("openai_chat", source)
    del engine
    async with stream:
        async for _ in stream:
            break
        assert source.reads == 1
        if exit_mode != "break":
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(started.wait(), 2)
            if exit_mode == "close":
                await asyncio.gather(stream.aclose(), stream.aclose())
            else:
                pending.cancel()
            expected = RuntimeError if exit_mode == "cleanup-error" else asyncio.CancelledError
            with pytest.raises(expected) as caught:
                await asyncio.wait_for(pending, 2)
            if exit_mode == "cleanup-error":
                assert caught.value is cleanup_error
                assert isinstance(caught.value.__context__, asyncio.CancelledError)
            assert cleaned.is_set()
    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert source.close_calls == 0
    if exit_mode == "break":
        del source
        assert source_ref() is None
