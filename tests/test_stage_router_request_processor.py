# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`StageRouterRequestProcessor`.

The processor is a thin async dispatcher: it runs an async picker against the
:class:`ToolResultSignal` stamped upstream by :class:`DimensionCollector` and
stamps ``ctx.selected_target`` + ``ctx.selected_model`` for the downstream
``MultiLlmBackend`` to dispatch on.
"""

from __future__ import annotations

import pytest

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.stage_router import pick_capable_first, pick_efficient_first
from switchyard.lib.processors.stage_router_request_processor import StageRouterRequestProcessor
from switchyard_rust.components import DimensionCollector
from switchyard_rust.core import ChatRequest, ProxyContext


def _target(label: str, model: str) -> LlmTarget:
    return LlmTarget(
        id=label,
        model=model,
        api_key="sk-test",
        base_url="https://test.invalid/v1",
        format=BackendFormat.OPENAI,
    )


EFFICIENT = _target("efficient", "vendor/efficient-model")
CAPABLE = _target("capable", "vendor/capable-model")


async def _populated_ctx(messages: list[dict]) -> tuple[ProxyContext, ChatRequest]:
    collector = DimensionCollector()
    request = ChatRequest.openai_chat({"messages": messages})
    ctx = ProxyContext()
    await collector.process(ctx, request)
    return ctx, request


async def _capable_pick(ctx: ProxyContext) -> int:
    return await pick_capable_first(ctx, confidence_threshold=0.7)


async def _efficient_pick(ctx: ProxyContext) -> int:
    return await pick_efficient_first(ctx, confidence_threshold=0.7)


def test_requires_exactly_two_targets():
    with pytest.raises(ValueError, match="exactly 2 targets"):
        StageRouterRequestProcessor(targets=(EFFICIENT,), picker=_capable_pick)
    with pytest.raises(ValueError, match="exactly 2 targets"):
        StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE, CAPABLE), picker=_capable_pick)


@pytest.mark.asyncio
async def test_capable_first_stamps_capable_on_first_turn_no_signal():
    """First turn: no ToolResultSignal yet → no_signal path → default tier."""
    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=_capable_pick)
    ctx, request = await _populated_ctx([{"role": "user", "content": "hi"}])
    await processor.process(ctx, request)
    assert ctx.selected_target == "capable"
    assert ctx.selected_model == "vendor/capable-model"


@pytest.mark.asyncio
async def test_efficient_first_stamps_efficient_on_first_turn_no_signal():
    """First turn: no ToolResultSignal yet → no_signal path → default tier."""
    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=_efficient_pick)
    ctx, request = await _populated_ctx([{"role": "user", "content": "hi"}])
    await processor.process(ctx, request)
    assert ctx.selected_target == "efficient"


@pytest.mark.asyncio
async def test_capable_first_falls_open_to_capable_on_low_confidence():
    """Signal present but scorer below threshold + no classifier → fall_open to default."""
    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=_capable_pick)
    # One Read + one clean tool_result + a follow-up user message. This produces
    # a non-None ToolResultSignal (so the no_signal short-circuit is bypassed)
    # but the scorer only sees a small no_error_streak penalty — confidence
    # well under 0.7, classifier not configured → fall_open returns default.
    ctx, request = await _populated_ctx([
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
        {"role": "user", "content": "next"},
    ])
    await processor.process(ctx, request)
    assert ctx.selected_target == "capable"


@pytest.mark.asyncio
async def test_efficient_first_falls_open_to_efficient_on_low_confidence():
    """Sibling check on the efficient-first picker, same low-confidence shape."""
    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=_efficient_pick)
    ctx, request = await _populated_ctx([
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
        {"role": "user", "content": "next"},
    ])
    await processor.process(ctx, request)
    assert ctx.selected_target == "efficient"


@pytest.mark.asyncio
async def test_critical_severity_escalates_both_pickers():
    fatal = [
        {"role": "tool", "tool_call_id": "1",
         "content": "Out of memory: cannot allocate memory"},
        {"role": "user", "content": "try again"},
    ]
    for picker in (_capable_pick, _efficient_pick):
        processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=picker)
        ctx, request = await _populated_ctx(fatal)
        await processor.process(ctx, request)
        assert ctx.selected_target == "capable"


@pytest.mark.asyncio
async def test_request_is_not_mutated():
    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=_capable_pick)
    ctx, request = await _populated_ctx([{"role": "user", "content": "hi"}])
    returned = await processor.process(ctx, request)
    assert returned is request


@pytest.mark.asyncio
async def test_buggy_picker_falls_back_to_efficient():
    async def bad_picker(_ctx: ProxyContext) -> int:
        raise RuntimeError("boom")

    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=bad_picker)
    ctx, request = await _populated_ctx([{"role": "user", "content": "hi"}])
    await processor.process(ctx, request)
    assert ctx.selected_target == "efficient"


@pytest.mark.asyncio
async def test_picker_index_is_clamped():
    async def overshooting_picker(_ctx: ProxyContext) -> int:
        return 99

    processor = StageRouterRequestProcessor(targets=(EFFICIENT, CAPABLE), picker=overshooting_picker)
    ctx, request = await _populated_ctx([{"role": "user", "content": "hi"}])
    await processor.process(ctx, request)
    assert ctx.selected_target == "capable"
