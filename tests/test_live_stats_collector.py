# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LiveStatsCollector and StatsResponseProcessor."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from switchyard.lib.live_stats_collector import (
    LiveStatsCollector,
    RequestStats,
)
from switchyard.lib.processors.stats_response_processor_live_collector import (
    StatsResponseProcessor,
)
from switchyard_rust.core import ChatResponse

# ---------------------------------------------------------------------------
# LiveStatsCollector unit tests
# ---------------------------------------------------------------------------


def test_initial_snapshot_is_zero():
    c = LiveStatsCollector()
    s = c.snapshot()
    assert s == RequestStats()


def test_record_increments_all_fields():
    c = LiveStatsCollector()
    c.record(
        prompt_tokens=100,
        completion_tokens=50,
        cache_read_tokens=20,
        cache_creation_tokens=5,
    )
    s = c.snapshot()
    assert s.request_count == 1
    assert s.prompt_tokens == 100
    assert s.completion_tokens == 50
    assert s.cache_read_tokens == 20
    assert s.cache_creation_tokens == 5


def test_record_accumulates_across_calls():
    c = LiveStatsCollector()
    c.record(prompt_tokens=10, completion_tokens=5)
    c.record(prompt_tokens=20, completion_tokens=10)
    s = c.snapshot()
    assert s.request_count == 2
    assert s.prompt_tokens == 30
    assert s.completion_tokens == 15
    assert c.to_dict()["models"]["unknown"]["max_observed_context_tokens"] == 30


def test_snapshot_returns_copy():
    c = LiveStatsCollector()
    c.record(prompt_tokens=10)
    snap1 = c.snapshot()
    c.record(prompt_tokens=10)
    snap2 = c.snapshot()
    assert snap1.request_count == 1
    assert snap2.request_count == 2


def test_record_defaults_to_zero_tokens():
    c = LiveStatsCollector()
    c.record()
    s = c.snapshot()
    assert s.request_count == 1
    assert s.prompt_tokens == 0
    assert s.completion_tokens == 0


def test_thread_safety():
    c = LiveStatsCollector()
    n_threads = 50
    calls_per_thread = 100

    def _worker():
        for _ in range(calls_per_thread):
            c.record(prompt_tokens=1, completion_tokens=1)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = c.snapshot()
    assert s.request_count == n_threads * calls_per_thread
    assert s.prompt_tokens == n_threads * calls_per_thread


# ---------------------------------------------------------------------------
# tier_breakdown — ordering + isolation
# ---------------------------------------------------------------------------


def test_tier_breakdown_empty_when_no_records():
    c = LiveStatsCollector()
    assert c.tier_breakdown() == []


def test_tier_breakdown_orders_strong_before_weak():
    """Insertion order must NOT determine UI order — the random-routing
    footer relies on a stable ``strong → weak`` layout regardless of
    which tier rolled first.
    """
    c = LiveStatsCollector()
    # Record weak first to confirm sort, not insertion order, drives output.
    c.record(model="nemotron", tier="weak", prompt_tokens=10, completion_tokens=5)
    c.record(model="kimi-k2.5", tier="strong", prompt_tokens=20, completion_tokens=8)

    rows = c.tier_breakdown()
    assert [name for name, _ in rows] == ["kimi-k2.5", "nemotron"]
    assert [b.tier for _, b in rows] == ["strong", "weak"]


def test_tier_breakdown_unknown_tier_sorts_last():
    """Models recorded without a tier (or with an unrecognised label)
    sink to the bottom — the strong/weak rows are always on top.
    """
    c = LiveStatsCollector()
    c.record(model="other-model", tier="", prompt_tokens=1)
    c.record(model="kimi", tier="strong", prompt_tokens=1)
    c.record(model="nemotron", tier="weak", prompt_tokens=1)

    rows = c.tier_breakdown()
    assert [name for name, _ in rows] == ["kimi", "nemotron", "other-model"]


def test_tier_breakdown_returns_copies():
    """Mutating the returned ``ModelStats`` must not affect the
    collector's internal state — the footer paints from this list and
    rendering work shouldn't be able to corrupt request accounting.
    """
    c = LiveStatsCollector()
    c.record(model="kimi", tier="strong", prompt_tokens=10, completion_tokens=5)

    rows = c.tier_breakdown()
    rows[0][1].calls = 999

    # Re-fetch; collector's truth is unchanged.
    again = c.tier_breakdown()
    assert again[0][1].calls == 1


# ---------------------------------------------------------------------------
# StatsResponseProcessor — non-streaming Anthropic
# ---------------------------------------------------------------------------


def _make_ctx():
    from switchyard.lib.proxy_context import ProxyContext
    return ProxyContext()


def _make_anthropic_usage(input_tokens=10, output_tokens=5, cache_read=0, cache_creation=0):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


async def test_records_anthropic_completion():
    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)

    body = {"usage": _make_anthropic_usage(input_tokens=100, output_tokens=40, cache_read=30)}
    response = ChatResponse.anthropic_completion(body)

    result = await proc.process(_make_ctx(), response)
    assert result is response

    s = collector.snapshot()
    assert s.request_count == 1
    # prompt_tokens sums input + cache_read + cache_creation for Anthropic
    assert s.prompt_tokens == 130
    assert s.completion_tokens == 40
    assert s.cache_read_tokens == 30


# ---------------------------------------------------------------------------
# StatsResponseProcessor — non-streaming OpenAI
# ---------------------------------------------------------------------------


async def test_records_openai_completion():
    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)

    usage = {
        "prompt_tokens": 80,
        "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 15},
    }

    body = {"usage": usage}
    response = ChatResponse.openai_completion(body)

    await proc.process(_make_ctx(), response)

    s = collector.snapshot()
    assert s.request_count == 1
    assert s.prompt_tokens == 80
    assert s.completion_tokens == 30
    assert s.cache_read_tokens == 15


async def test_records_rust_context_selected_model_and_target():
    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)
    ctx = _make_ctx()
    ctx.selected_model = "served-model"
    ctx.selected_target = "weak"

    await proc.process(
        ctx,
        ChatResponse.openai_completion({
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 4,
            },
        }),
    )

    rows = collector.tier_breakdown()
    assert [(model, stats.tier, stats.prompt_tokens) for model, stats in rows] == [
        ("served-model", "weak", 9),
    ]


async def test_handles_missing_openai_usage():
    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)

    body = {"usage": None}
    response = ChatResponse.openai_completion(body)

    await proc.process(_make_ctx(), response)
    assert collector.snapshot().request_count == 0


# ---------------------------------------------------------------------------
# StatsResponseProcessor — streaming Anthropic tap
# ---------------------------------------------------------------------------


async def test_attaches_anthropic_stream_tap():
    from switchyard.lib.chat_response.anthropic import AnthropicResponseStream

    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)

    async def _events():
        start = MagicMock()
        start.type = "message_start"
        start.message.usage.input_tokens = 50
        start.message.usage.cache_read_input_tokens = 10
        start.message.usage.cache_creation_input_tokens = 0
        yield start

        delta = MagicMock()
        delta.type = "message_delta"
        delta.usage.output_tokens = 20
        yield delta

        stop = MagicMock()
        stop.type = "message_stop"
        yield stop

    stream = AnthropicResponseStream(_events())
    response = ChatResponse.anthropic_stream(stream)

    await proc.process(_make_ctx(), response)

    events = [e async for e in response.stream]
    assert len(events) == 3

    s = collector.snapshot()
    assert s.request_count == 1
    # prompt_tokens sums input(50) + cache_read(10) + cache_creation(0)
    assert s.prompt_tokens == 60
    assert s.completion_tokens == 20
    assert s.cache_read_tokens == 10


async def test_anthropic_tap_delta_input_tokens_fallback():
    """Backends that report input_tokens in message_delta instead of message_start.

    Some Anthropic-compatible APIs (e.g. NVIDIA Inference Hub) set
    input_tokens=0 in message_start and report the real count in
    message_delta.usage.input_tokens (Optional[int]).  The tap must use
    the delta value as a fallback so the ``in`` counter is non-zero.
    """
    from switchyard.lib.chat_response.anthropic import AnthropicResponseStream

    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)

    async def _events():
        start = MagicMock()
        start.type = "message_start"
        start.message.usage.input_tokens = 0  # backend defers counting
        start.message.usage.cache_read_input_tokens = 0
        start.message.usage.cache_creation_input_tokens = 0
        yield start

        delta = MagicMock()
        delta.type = "message_delta"
        delta.usage.output_tokens = 30
        # Backend reports actual input tokens here
        delta.usage.input_tokens = 75
        delta.usage.cache_read_input_tokens = None
        delta.usage.cache_creation_input_tokens = None
        yield delta

        stop = MagicMock()
        stop.type = "message_stop"
        yield stop

    stream = AnthropicResponseStream(_events())
    response = ChatResponse.anthropic_stream(stream)
    await proc.process(_make_ctx(), response)
    [_ async for _ in response.stream]

    s = collector.snapshot()
    assert s.request_count == 1
    assert s.prompt_tokens == 75
    assert s.completion_tokens == 30


# ---------------------------------------------------------------------------
# StatsResponseProcessor — streaming OpenAI Responses API tap
# ---------------------------------------------------------------------------


async def test_attaches_openai_responses_stream_tap():
    from openai.types.responses import Response as OpenAIResponse
    from openai.types.responses import ResponseCompletedEvent
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
        ResponseUsage,
    )

    from switchyard.lib.chat_response.openai_responses import ResponsesApiStream

    collector = LiveStatsCollector()
    proc = StatsResponseProcessor(collector)

    response_obj = OpenAIResponse.model_construct(
        id="resp-test", object="response", created_at=1_700_000_000,
        model="gpt-4o", output=[], status="completed",
        usage=ResponseUsage(
            input_tokens=300, output_tokens=120, total_tokens=420,
            input_tokens_details=InputTokensDetails(cached_tokens=60),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=40),
        ),
    )

    async def _events():
        yield ResponseCompletedEvent(
            type="response.completed", response=response_obj, sequence_number=1,
        )

    response = ChatResponse.openai_responses_stream(ResponsesApiStream(_events()))
    await proc.process(_make_ctx(), response)
    [_ async for _ in response.stream]

    s = collector.snapshot()
    assert s.request_count == 1
    assert s.prompt_tokens == 300
    assert s.completion_tokens == 120
    assert s.cache_read_tokens == 60


# ---------------------------------------------------------------------------
# Cost estimation — model name normalization
# ---------------------------------------------------------------------------


def test_cost_estimate_strips_date_suffix():
    """Versioned model names like 'claude-sonnet-4-6-20251022' should match
    the base name in the price table."""
    collector = LiveStatsCollector()
    collector.record("claude-sonnet-4-6-20251022", prompt_tokens=1_000_000, completion_tokens=0)
    m = collector.to_dict()["models"]["claude-sonnet-4-6-20251022"]
    # Should resolve to claude-sonnet-4-6 price: $3.00 / 1M input
    assert m["estimated_cost_usd"] == pytest.approx(3.00)


def test_cost_estimate_none_for_unknown_model():
    collector = LiveStatsCollector()
    collector.record("some-unknown-model", prompt_tokens=100, completion_tokens=50)
    m = collector.to_dict()["models"]["some-unknown-model"]
    assert m["estimated_cost_usd"] is None
