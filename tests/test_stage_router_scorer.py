# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the stage-router dimensions + scorer."""

from __future__ import annotations

import pytest

from switchyard.lib.processors.stage_router.dimensions import (
    CodingAgentDimensions,
    from_signal,
)
import math

from switchyard.lib.processors.stage_router.scorer import DEFAULT_STEEPNESS, DEFAULT_WEIGHTS, score
from switchyard_rust.components import DimensionCollector
from switchyard_rust.core import ChatRequest, ProxyContext


def _zero_dimensions() -> CodingAgentDimensions:
    return CodingAgentDimensions(
        severity=0.0,
        no_error_streak_intensity=0.0,
        write_intensity=0.0,
        edit_intensity=0.0,
        recent_write_intensity=0.0,
        planning_active=0.0,
        pure_bash_intensity=0.0,
        stuck_exploring=0.0,
        no_progress=0.0,
        tests_passed=0.0,
    )


def test_zero_signal_scores_to_zero():
    result = score(_zero_dimensions())
    assert result.score == 0.0
    assert result.confidence == 0.0


def test_critical_severity_pushes_toward_capable():
    dims = _zero_dimensions()
    dims = CodingAgentDimensions(**{**dims.__dict__, "severity": 1.0})
    result = score(dims)
    assert result.score > 0
    assert result.confidence == abs(result.score)


def test_tests_passed_pushes_toward_efficient():
    dims = _zero_dimensions()
    dims = CodingAgentDimensions(**{**dims.__dict__, "tests_passed": 1.0})
    result = score(dims)
    assert result.score < 0


def test_score_bounded_by_unit_interval():
    """tanh keeps score strictly in (-1, 1); extreme raw sums saturate near ±1."""
    dims = CodingAgentDimensions(**{**_zero_dimensions().__dict__, "severity": 1.0})
    high = score(dims, weights={"severity": 5.0})
    assert -1.0 < high.score <= 1.0
    assert high.score > 0.99  # tanh(10) is ~1.0 to 5 decimal places
    assert high.confidence == abs(high.score)
    low = score(dims, weights={"severity": -5.0})
    assert -1.0 <= low.score < 1.0
    assert low.score < -0.99
    assert low.confidence == abs(low.score)


def test_custom_weights_can_invert_decision():
    """Researchers override weights via the call site, not YAML."""
    dims = CodingAgentDimensions(**{**_zero_dimensions().__dict__, "severity": 1.0})
    default = score(dims, weights=DEFAULT_WEIGHTS)
    inverted = score(dims, weights={"severity": -0.5})
    assert default.score > 0
    assert inverted.score < 0


def test_contributions_are_pre_sigmoid_raw_products():
    """contributions are the raw weight×dim products; score = tanh(k * sum(contributions))."""
    dims = CodingAgentDimensions(**{**_zero_dimensions().__dict__, "tests_passed": 1.0})
    result = score(dims)
    raw_sum = sum(result.contributions.values())
    expected_score = math.tanh(DEFAULT_STEEPNESS * raw_sum)
    assert abs(result.score - expected_score) < 1e-9


@pytest.mark.asyncio
async def test_from_signal_normalises_real_extracted_signal():
    """End-to-end: DimensionCollector → ToolResultSignal → CodingAgentDimensions."""
    collector = DimensionCollector()
    ctx = ProxyContext()
    request = ChatRequest.openai_chat({
        "messages": [
            {"role": "assistant",
             "tool_calls": [{"function": {"name": "Write", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
    })
    await collector.process(ctx, request)
    from switchyard_rust.components import get_tool_result_signal
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    dims = from_signal(signal)
    assert 0.0 <= dims.severity <= 1.0
    assert 0.0 <= dims.write_intensity <= 1.0
    assert dims.write_intensity > 0  # we issued one Write call
