# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OTel span helpers (switchyard.lib.spans)."""

from __future__ import annotations

import importlib

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _spans_with_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("switchyard")
    spans = importlib.import_module("switchyard.lib.spans")
    spans.reset_for_test(tracer)
    return spans, exporter


def test_nested_spans_and_attributes():
    spans, exporter = _spans_with_exporter()

    with spans.request_span("switchyard.request", {"inbound_format": "openai", "stream": True}):
        with spans.stage_span("switchyard.request_processors"):
            with spans.route_decision_span(
                router="random",
                tier="strong",
                selected_model="m",
                selected_target="t",
                draw=0.1,
                missing=None,  # None attrs are dropped
            ):
                pass

    finished = {s.name: s for s in exporter.get_finished_spans()}
    assert "switchyard.request" in finished
    assert "switchyard.request_processors" in finished
    assert "switchyard.route_decision" in finished

    root = finished["switchyard.request"]
    assert root.attributes["inbound_format"] == "openai"
    assert root.attributes["stream"] is True

    decision = finished["switchyard.route_decision"]
    assert decision.attributes["router"] == "random"
    assert decision.attributes["tier"] == "strong"
    assert decision.attributes["selected_model"] == "m"
    assert decision.attributes["draw"] == 0.1
    assert "missing" not in decision.attributes


def test_current_traceparent_round_trips():
    spans, _ = _spans_with_exporter()
    with spans.request_span("switchyard.request", {}):
        tp = spans.current_traceparent()
    assert tp is not None
    assert tp.startswith("00-")  # W3C traceparent version-00


def test_noop_without_tracer():
    spans = importlib.import_module("switchyard.lib.spans")
    spans.reset_for_test(None)
    with spans.request_span("switchyard.request", {"k": "v"}):
        with spans.stage_span("switchyard.backend_call"):
            assert spans.current_traceparent() is None
