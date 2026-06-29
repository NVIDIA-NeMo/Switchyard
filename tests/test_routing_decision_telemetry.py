# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing processors emit route-decision spans and the routing_decisions counter."""

from __future__ import annotations

import importlib

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.processors.random_routing_request_processor import (
    RandomRoutingRequestProcessor,
)
from switchyard.lib.proxy_context import CTX_ROUTER_NAME, ProxyContext
from switchyard_rust.components import RandomRoutingProcessorConfig
from switchyard_rust.core import ChatRequest


def _otel():
    span_exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(span_exporter))
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("switchyard")
    spans = importlib.import_module("switchyard.lib.spans")
    metrics = importlib.import_module("switchyard.lib.metrics")
    spans.reset_for_test(tp.get_tracer("switchyard"))
    metrics.reset_for_test(meter)
    return span_exporter, reader


def _metric_names(reader) -> set[str]:
    names: set[str] = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.add(metric.name)
    return names


async def test_random_routing_emits_decision_span_and_counter():
    span_exporter, reader = _otel()
    config = RandomRoutingProcessorConfig(
        strong=LlmTarget(id="strong", model="strong-model", base_url="http://s", api_key="k"),
        weak=LlmTarget(id="weak", model="weak-model", base_url="http://w", api_key="k"),
        strong_probability=1.0,  # deterministic: always strong
        rng_seed=1,
    )
    proc = RandomRoutingRequestProcessor(config)
    ctx = ProxyContext()
    req = ChatRequest.openai_chat({"model": "x", "messages": [{"role": "user", "content": "hi"}]})

    await proc.process(ctx, req)

    assert ctx.metadata[CTX_ROUTER_NAME] == "random"
    assert ctx.metadata["_random_routing_tier"] == "strong"

    decision_spans = [s for s in span_exporter.get_finished_spans() if s.name == "switchyard.route_decision"]
    assert len(decision_spans) == 1
    attrs = decision_spans[0].attributes
    assert attrs["router"] == "random"
    assert attrs["tier"] == "strong"
    assert attrs["selected_model"] == "strong-model"
    assert attrs["strong_probability"] == 1.0

    assert "switchyard.routing_decisions" in _metric_names(reader)
