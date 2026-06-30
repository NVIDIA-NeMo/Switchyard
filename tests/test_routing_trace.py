# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in routing trace recording and JSONL export."""

from __future__ import annotations

import json
import logging
import stat

import pytest

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.routing_trace import (
    ROUTING_TRACE_CAPTURE_CONTENT_ENV,
    ROUTING_TRACE_JSONL_ENV,
    capture_routing_text,
    record_routing_event,
    routing_trace_content_enabled,
    routing_trace_enabled,
)


def _decision_event() -> dict[str, object]:
    return {
        "kind": "decision",
        "producer": "test_router",
        "name": "tier_selection",
        "selection": {"tier": "strong"},
        "source": "policy",
    }


def test_record_routing_event_is_noop_when_capture_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv(ROUTING_TRACE_JSONL_ENV, raising=False)
    ctx = ProxyContext(request_id="request-1")

    assert record_routing_event(ctx, _decision_event()) is None
    assert ctx.routing_trace is None
    assert not routing_trace_enabled()


def test_record_routing_event_appends_validated_jsonl(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "nested" / "routing-trace.jsonl"
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(trace_path))
    ctx = ProxyContext(request_id="request-1")

    recorded = record_routing_event(ctx, _decision_event())

    assert recorded is not None
    assert recorded["sequence"] == 0
    assert ctx.routing_trace is not None
    assert json.loads(trace_path.read_text()) == {
        "schema_version": 1,
        "request_id": "request-1",
        "event": recorded,
    }
    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600


def test_record_routing_event_preserves_prior_jsonl_rows(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "routing-trace.jsonl"
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(trace_path))
    ctx = ProxyContext(request_id="request-1")

    record_routing_event(ctx, _decision_event())
    record_routing_event(ctx, _decision_event())

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [row["event"]["sequence"] for row in rows] == [0, 1]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_content_capture_requires_trace_and_explicit_opt_in(
    monkeypatch,
    tmp_path,
    value: str,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(ROUTING_TRACE_CAPTURE_CONTENT_ENV, value)

    assert routing_trace_content_enabled()

    monkeypatch.delenv(ROUTING_TRACE_JSONL_ENV)
    assert not routing_trace_content_enabled()


def test_text_capture_is_hash_only_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "trace.jsonl"))
    monkeypatch.delenv(ROUTING_TRACE_CAPTURE_CONTENT_ENV, raising=False)

    captured = capture_routing_text("héllo")

    assert captured == {
        "sha256": "3c48591d8d098a4538f5e013dfcf406e948eac4d3277b10bf614e295d6068179",
        "bytes": 6,
        "chars": 5,
    }


def test_text_capture_includes_content_with_explicit_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(ROUTING_TRACE_CAPTURE_CONTENT_ENV, "true")

    assert capture_routing_text("classifier output")["content"] == "classifier output"


def test_trace_capture_is_fail_open_for_unpaired_unicode_surrogate(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(ROUTING_TRACE_CAPTURE_CONTENT_ENV, "true")
    ctx = ProxyContext(request_id="request-1")
    captured = capture_routing_text("\ud800")

    recorded = record_routing_event(
        ctx,
        {
            "kind": "evaluation",
            "producer": "test_router",
            "name": "unicode_input",
            "schema": "test_router.unicode_input/v1",
            "input": captured,
        },
    )

    assert captured["content"] == r"\ud800"
    assert recorded is not None
    assert recorded["sequence"] == 0


def test_export_failure_does_not_change_routing_behavior(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path))
    ctx = ProxyContext(request_id="request-1")

    with caplog.at_level(logging.ERROR):
        recorded = record_routing_event(ctx, _decision_event())

    assert recorded is not None
    assert ctx.routing_trace is not None
    assert "Failed to append routing trace event" in caplog.text


def test_invalid_expanduser_path_does_not_change_routing_behavior(monkeypatch, caplog) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, "~__switchyard_missing_user__/trace.jsonl")
    ctx = ProxyContext(request_id="request-1")

    with caplog.at_level(logging.ERROR):
        recorded = record_routing_event(ctx, _decision_event())

    assert recorded is not None
    assert ctx.routing_trace is not None
    assert "Failed to append routing trace event" in caplog.text


def test_invalid_event_does_not_change_routing_behavior(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "trace.jsonl"))
    ctx = ProxyContext(request_id="request-1")

    with caplog.at_level(logging.ERROR):
        recorded = record_routing_event(ctx, {"kind": "decision"})

    assert recorded is None
    assert ctx.routing_trace is None
    assert "Rejected invalid routing trace event" in caplog.text
