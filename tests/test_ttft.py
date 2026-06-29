# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TTFT + streaming token usage recorded via the otel_usage stream tap."""

from __future__ import annotations

import importlib
import time
from collections.abc import AsyncIterator

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from switchyard.lib import otel_usage
from switchyard.lib.chat_response import ResponseStream
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatResponse


def _reader():
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("switchyard")
    metrics = importlib.import_module("switchyard.lib.metrics")
    metrics.reset_for_test(meter)
    spans = importlib.import_module("switchyard.lib.spans")
    spans.reset_for_test(None)  # no tracer needed for this test
    return reader


def _names(reader) -> set[str]:
    names: set[str] = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                names.add(m.name)
    return names


def _chunk(*, content: str = "hi", usage: CompletionUsage | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=1700000000,
        model="gpt-test",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=content), finish_reason=None)],
        usage=usage,
    )


async def _fake_stream(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


async def test_streaming_records_ttft_and_tokens():
    reader = _reader()
    chunks = [
        _chunk(content="hel"),
        _chunk(content="lo", usage=CompletionUsage(prompt_tokens=7, completion_tokens=2, total_tokens=9)),
    ]
    resp = ChatResponse.openai_stream(ResponseStream(_fake_stream(chunks)))
    ctx = ProxyContext()
    ctx.metadata[otel_usage.CTX_REQUEST_START] = time.monotonic() - 0.005

    otel_usage.record_response_usage(ctx, resp, "gpt-test", "strong")

    drained = [c async for c in resp.stream]
    assert len(drained) == 2  # tap is transparent — does not consume chunks

    names = _names(reader)
    assert "switchyard.ttft_ms" in names
    assert "switchyard.prompt_tokens" in names
    assert "switchyard.completion_tokens" in names
