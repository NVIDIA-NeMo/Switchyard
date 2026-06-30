# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTel spans + latency metrics emitted by the ComponentChainProfile executor."""

from __future__ import annotations

import importlib

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard_rust.core import ChatRequest
from switchyard_rust.profiles import ProfileInput


class _FakeBackend:
    async def call(self, ctx, request):
        ctx.selected_model = "gpt-test"
        return _build_response()


def _build_response():
    # Minimal OpenAI chat completion response the chain accepts.
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from switchyard_rust.core import ChatResponse

    completion = ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )
    return ChatResponse.openai_completion(completion)


def _request() -> ChatRequest:
    return ChatRequest.openai_chat(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
    )


def _metric_names(reader) -> set[str]:
    names: set[str] = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.add(metric.name)
    return names


async def test_executor_emits_spans_and_latency_metrics():
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[metric_reader]).get_meter("switchyard")

    spans = importlib.import_module("switchyard.lib.spans")
    metrics = importlib.import_module("switchyard.lib.metrics")
    spans.reset_for_test(tracer_provider.get_tracer("switchyard"))
    metrics.reset_for_test(meter)

    profile = ComponentChainProfile(backend=_FakeBackend())
    await profile.run_with_context(ProfileInput(_request()), _make_ctx())

    span_names = {s.name for s in span_exporter.get_finished_spans()}
    assert "switchyard.request_processors" in span_names
    assert "switchyard.backend_call" in span_names
    assert "switchyard.response_processors" in span_names

    names = _metric_names(metric_reader)
    assert "switchyard.total_latency_ms" in names
    assert "switchyard.model_call_latency_ms" in names
    assert "switchyard.routing_overhead_ms" in names
    assert "switchyard.requests" in names
    # Token usage recorded from the non-streaming response body.
    assert "switchyard.prompt_tokens" in names
    assert "switchyard.completion_tokens" in names


def _make_ctx():
    from switchyard.lib.proxy_context import ProxyContext

    return ProxyContext()
