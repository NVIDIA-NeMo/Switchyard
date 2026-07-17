# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import switchyard
from switchyard import libsy


def test_libsy_uses_an_explicit_namespace() -> None:
    assert switchyard.libsy is libsy
    assert libsy.protocol.__name__ == "switchyard.libsy.protocol"
    assert not hasattr(libsy, "RunStream")


def test_protocol_request_and_response_fields_stay_typed() -> None:
    image: libsy.protocol.ImageSourceValue = libsy.protocol.ImageSource.Url(
        "https://example.test/image.png", "high"
    )
    tool_call = libsy.protocol.ToolCall("call-1", "lookup", {"query": "switchyard"})
    request = libsy.protocol.LlmRequest(
        model="requested-model",
        instructions=[
            libsy.protocol.InstructionBlock(
                libsy.protocol.Role.DEVELOPER,
                [libsy.protocol.ContentBlock.Text("route carefully")],
            )
        ],
        messages=[
            libsy.protocol.Message(
                libsy.protocol.Role.USER,
                [libsy.protocol.ContentBlock.Image(image)],
            ),
            libsy.protocol.Message(
                libsy.protocol.Role.ASSISTANT,
                [libsy.protocol.ContentBlock.ToolCallBlock(tool_call)],
            ),
        ],
        tools=[
            libsy.protocol.ToolDefinition(
                "lookup",
                {"type": "object"},
                description="Look up a value",
                strict=True,
            )
        ],
        tool_choice=libsy.protocol.ToolChoice.tool("lookup"),
        sampling=libsy.protocol.SamplingParams(temperature=0.2, top_p=0.9, top_k=20),
        output=libsy.protocol.OutputParams(
            max_output_tokens=128,
            response_format={"type": "json_object"},
        ),
        reasoning=libsy.protocol.ReasoningParams(effort="medium", raw={"budget": 64}),
        stream=True,
        extensions=libsy.protocol.ProviderExtensions({"provider_flag": True}),
        preservation=libsy.protocol.PreservationMetadata(
            requests={"openai_chat": {"model": "requested-model"}}
        ),
    )

    assert request.instructions[0].role == libsy.protocol.Role.DEVELOPER
    assert request.tools[0].parameters == {"type": "object"}
    assert request.tool_choice is not None
    assert request.tool_choice.name == "lookup"
    assert request.sampling.temperature == 0.2
    assert request.output.max_output_tokens == 128
    assert request.reasoning.effort == "medium"
    assert request.extensions.fields == {"provider_flag": True}
    assert request.preservation.requests == {"openai_chat": {"model": "requested-model"}}
    match request.messages[0].content[0]:
        case libsy.protocol.ContentBlock.Image(
            source=libsy.protocol.ImageSource.Url(
                url="https://example.test/image.png", detail="high"
            )
        ):
            pass
        case block:
            pytest.fail(f"unexpected image block: {block!r}")
    match request.messages[1].content[0]:
        case libsy.protocol.ContentBlock.ToolCallBlock(tool_call=round_tripped):
            assert round_tripped.arguments == {"query": "switchyard"}
        case block:
            pytest.fail(f"unexpected tool call block: {block!r}")

    response = libsy.protocol.AggLlmResponse(
        id="response-1",
        model="selected-model",
        outputs=[
            libsy.protocol.ResponseOutput(
                libsy.protocol.Role.ASSISTANT,
                [libsy.protocol.ContentBlock.Text("done")],
                stop_reason=libsy.protocol.StopReason.END_TURN,
            )
        ],
        usage=libsy.protocol.Usage(input_tokens=4, output_tokens=1, total_tokens=5),
    )

    assert response.outputs[0].role == libsy.protocol.Role.ASSISTANT
    assert response.outputs[0].stop_reason == libsy.protocol.StopReason.END_TURN
    assert response.usage.total_tokens == 5


def test_request_and_metadata_round_trip_as_python_values() -> None:
    context = libsy.protocol.Context(values={"tenant": "test"})
    metadata = libsy.protocol.Metadata(
        session_id="session-1",
        correlation_id="correlation-1",
        extra_metadata={"tenant": "test"},
        wire_format=libsy.protocol.WireFormat.OPENAI_CHAT,
    )
    llm_request = libsy.protocol.LlmRequest(
        model="inbound-model",
        messages=[
            libsy.protocol.Message(
                libsy.protocol.Role.USER,
                [libsy.protocol.ContentBlock.Text("hello")],
            )
        ],
    )
    request = libsy.protocol.Request(
        llm_request,
        raw_request={"provider_field": True},
        metadata=metadata,
    )

    assert context.values == {"tenant": "test"}
    assert request.requested_model == "inbound-model"
    assert request.llm_request.model == "inbound-model"
    assert request.llm_request.messages[0].role == libsy.protocol.Role.USER
    match request.llm_request.messages[0].content[0]:
        case libsy.protocol.ContentBlock.Text(text="hello"):
            pass
        case block:
            pytest.fail(f"unexpected content block: {block!r}")
    assert request.raw_request == {"provider_field": True}
    assert request.metadata is not None
    assert request.metadata.to_dict() == {
        "agent_id": None,
        "correlation_id": "correlation-1",
        "extra_metadata": {"tenant": "test"},
        "http_headers": None,
        "session_id": "session-1",
        "task_id": None,
        "wire_format": "openai_chat",
    }


def test_response_chunks_are_typed_variants() -> None:
    text = libsy.protocol.LlmResponseChunk.TextDelta(0, "hello")
    usage = libsy.protocol.LlmResponseChunk.UsageUpdate(
        libsy.protocol.Usage(input_tokens=3, output_tokens=2, total_tokens=5)
    )

    match text:
        case libsy.protocol.LlmResponseChunk.TextDelta(index=0, text="hello"):
            pass
        case chunk:
            pytest.fail(f"unexpected response chunk: {chunk!r}")
    assert usage.usage.input_tokens == 3
    assert usage.usage.output_tokens == 2
    assert usage.usage.total_tokens == 5


def test_aggregate_response_exposes_neutral_values() -> None:
    response = libsy.protocol.Response(libsy.protocol.AggLlmResponse(model="selected-model"))

    assert response.is_streaming is False
    assert response.selected_model == "selected-model"
    assert response.aggregate.model == "selected-model"
    with pytest.raises(AttributeError, match="aggregate Response has no stream"):
        _ = response.stream


class _ChunkSource:
    def __init__(self) -> None:
        self._chunks: AsyncIterator[libsy.protocol.LlmResponseChunkValue] = self._iterate()
        self.closed = False

    async def _iterate(self) -> AsyncIterator[libsy.protocol.LlmResponseChunkValue]:
        yield libsy.protocol.LlmResponseChunk.MessageStart("response-1", "selected-model")
        yield libsy.protocol.LlmResponseChunk.TextDelta(0, "hello")
        yield libsy.protocol.LlmResponseChunk.MessageStop("end_turn")

    def __aiter__(self) -> _ChunkSource:
        return self

    async def __anext__(self) -> libsy.protocol.LlmResponseChunkValue:
        return await anext(self._chunks)

    async def aclose(self) -> None:
        self.closed = True
        await self._chunks.aclose()


async def test_python_response_stream_round_trips_and_closes_source() -> None:
    source = _ChunkSource()
    response = libsy.protocol.Response.from_stream(source)

    assert response.is_streaming is True
    stream = response.stream
    chunks = [chunk async for chunk in stream]
    await asyncio.sleep(0)

    match chunks:
        case [
            libsy.protocol.LlmResponseChunk.MessageStart(id="response-1", model="selected-model"),
            libsy.protocol.LlmResponseChunk.TextDelta(index=0, text="hello"),
            libsy.protocol.LlmResponseChunk.MessageStop(reason="end_turn"),
        ]:
            pass
        case _:
            pytest.fail(f"unexpected response chunks: {chunks!r}")
    assert source.closed is True
    with pytest.raises(RuntimeError, match="LlmResponseStream has already been consumed"):
        async for _ in stream:
            pass


async def test_response_stream_aclose_releases_source() -> None:
    source = _ChunkSource()
    stream = libsy.protocol.Response.from_stream(source).stream

    await stream.aclose()
    await asyncio.sleep(0)

    assert source.closed is True


async def test_streaming_response_can_only_release_its_stream_once() -> None:
    response = libsy.protocol.Response.from_stream(_ChunkSource())

    _ = response.stream
    with pytest.raises(RuntimeError, match="Response has already been consumed"):
        _ = response.stream
