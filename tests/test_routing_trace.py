# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for minimal routing trace export."""

from __future__ import annotations

import json
import logging

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.routing_trace import (
    ROUTING_TRACE_CAPTURE_CONTENT_ENV,
    ROUTING_TRACE_JSONL_ENV,
    capture_routing_text,
    record_routing_event,
)


def test_record_is_noop_without_output_path(monkeypatch) -> None:
    monkeypatch.delenv(ROUTING_TRACE_JSONL_ENV, raising=False)
    ctx = ProxyContext(request_id="request-1")

    assert record_routing_event(ctx, "router.decision", {"tier": "strong"}) is None
    assert ctx.routing_trace is None


def test_record_writes_minimal_flat_event(monkeypatch, tmp_path) -> None:
    output = tmp_path / "events.jsonl"
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(output))
    ctx = ProxyContext(request_id="request-1")

    record_routing_event(ctx, "router.input", {"prompt": "hello"})
    record_routing_event(ctx, "router.decision", {"tier": "strong"})

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {
            "request_id": "request-1",
            "sequence": 0,
            "timestamp_ms": rows[0]["timestamp_ms"],
            "name": "router.input",
            "payload": {"prompt": "hello"},
        },
        {
            "request_id": "request-1",
            "sequence": 1,
            "timestamp_ms": rows[1]["timestamp_ms"],
            "name": "router.decision",
            "payload": {"tier": "strong"},
        },
    ]


def test_capture_content_requires_explicit_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "events.jsonl"))
    monkeypatch.delenv(ROUTING_TRACE_CAPTURE_CONTENT_ENV, raising=False)
    assert "content" not in capture_routing_text("secret")

    monkeypatch.setenv(ROUTING_TRACE_CAPTURE_CONTENT_ENV, "true")
    assert capture_routing_text("secret")["content"] == "secret"


def test_export_failure_does_not_change_routing(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path))
    ctx = ProxyContext(request_id="request-1")

    with caplog.at_level(logging.ERROR):
        recorded = record_routing_event(ctx, "router.decision", {"tier": "strong"})

    assert recorded is not None
    assert ctx.routing_trace is not None
    assert "Failed to append routing trace event" in caplog.text
