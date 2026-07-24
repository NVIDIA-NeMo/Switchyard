# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool-result signal extraction now built into DimensionCollector."""

from __future__ import annotations

import pytest

from switchyard_rust.components import DimensionCollector, get_tool_result_signal
from switchyard_rust.core import ChatRequest, ProxyContext

# ─── unit: classify_text (via Rust) ──────────────────────────────────────────
# DimensionCollector is Rust; test the logic through the full processor path.


async def _run_collector(body: dict, fmt: str = "openai_chat") -> ProxyContext:
    """Run DimensionCollector.process() and return the populated context."""
    collector = DimensionCollector()
    if fmt == "anthropic":
        request = ChatRequest.anthropic(body)
    elif fmt == "openai_responses":
        request = ChatRequest.openai_responses(body)
    else:
        request = ChatRequest.openai_chat(body)
    ctx = ProxyContext()
    await collector.process(ctx, request)
    return ctx


# ─── severity tests via DimensionCollector ────────────────────────────────────


async def test_traceback_stamps_hard_severity():
    ctx = await _run_collector({
        "messages": [
            {"role": "user", "content": "do something"},
            {"role": "tool", "tool_call_id": "1",
             "content": "Traceback (most recent call last):\n  ValueError"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.severity == pytest.approx(0.7)


async def test_oom_stamps_critical_severity():
    ctx = await _run_collector({
        "messages": [
            {"role": "tool", "tool_call_id": "1", "content": "Out of memory: kill process"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.severity == pytest.approx(1.0)


async def test_clean_result_has_zero_severity():
    ctx = await _run_collector({
        "messages": [
            {"role": "tool", "tool_call_id": "1", "content": "file written successfully"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.severity == pytest.approx(0.0)


async def test_no_tool_results_has_zero_severity():
    ctx = await _run_collector({
        "messages": [{"role": "user", "content": "hello"}]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.severity == pytest.approx(0.0)


# ─── conversation metrics ─────────────────────────────────────────────────────


async def test_edit_and_write_counts():
    ctx = await _run_collector({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "Edit", "arguments": "{}"}},
                {"function": {"name": "Edit", "arguments": "{}"}},
                {"function": {"name": "Write", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.edit_count == 2
    assert signal.write_count == 1


async def test_no_error_streak_all_clean():
    ctx = await _run_collector({
        "messages": [
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "tool", "tool_call_id": "2", "content": "also ok"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.no_error_streak == 2


async def test_no_error_streak_stops_at_error():
    ctx = await _run_collector({
        "messages": [
            {"role": "tool", "tool_call_id": "1",
             "content": "Traceback (most recent call last):\n  ValueError"},
            {"role": "tool", "tool_call_id": "2", "content": "ok"},
            {"role": "tool", "tool_call_id": "3", "content": "ok"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.no_error_streak == 2


async def test_tests_passed_detection():
    ctx = await _run_collector({
        "messages": [
            {"role": "tool", "tool_call_id": "1",
             "content": "====== 5 passed in 0.3s ======"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.tests_passed is True


async def test_tests_passed_false_when_failures_present():
    ctx = await _run_collector({
        "messages": [
            {"role": "tool", "tool_call_id": "1",
             "content": "2 failed, 3 passed in 0.5s"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.tests_passed is False


async def test_turn_depth_matches_message_count():
    ctx = await _run_collector({
        "messages": [
            {"role": "user", "content": "step 1"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "1", "content": "done"},
        ]
    })
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.turn_depth == 3


# ─── Anthropic format ─────────────────────────────────────────────────────────


async def test_anthropic_tool_result_extracted():
    ctx = await _run_collector({
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1",
                 "content": "Traceback (most recent call last):\n  ImportError:"}
            ]}
        ]
    }, fmt="anthropic")
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.severity == pytest.approx(0.7)


# ─── OpenAI Responses format ──────────────────────────────────────────────────


async def test_responses_api_tool_output_extracted():
    ctx = await _run_collector({
        "input": [
            {"type": "function_call", "name": "Write"},
            {"type": "function_call_output", "call_id": "1",
             "output": "file created"},
        ]
    }, fmt="openai_responses")
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    assert signal.severity == pytest.approx(0.0)
    assert signal.write_count == 1
