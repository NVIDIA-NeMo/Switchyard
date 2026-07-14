# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the stage-router dimensions + scorer."""

from __future__ import annotations

import math

import pytest

from switchyard.lib.processors.stage_router.dimensions import (
    CodingAgentDimensions,
    from_signal,
)
from switchyard.lib.processors.stage_router.scorer import (
    _SCORE_GAIN,
    DEFAULT_WEIGHTS,
    score,
)
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


def test_tests_passed_is_not_scored():
    """tests_passed is routed to the picker's default tier, not scored here."""
    dims = _zero_dimensions()
    dims = CodingAgentDimensions(**{**dims.__dict__, "tests_passed": 1.0})
    result = score(dims)
    assert result.score == 0.0


def test_progress_signal_points_efficient():
    """A progress signal scores negative (→EFFICIENT), picker-independent."""
    dims = CodingAgentDimensions(
        **{**_zero_dimensions().__dict__, "recent_write_intensity": 1.0}
    )
    assert score(dims).score < 0


def test_wrong_signal_points_capable():
    """A wrong signal scores positive (→CAPABLE), picker-independent."""
    dims = CodingAgentDimensions(
        **{**_zero_dimensions().__dict__, "stuck_exploring": 1.0}
    )
    assert score(dims).score > 0


def test_tanh_spread_and_monotonic_confidence():
    """Raw sum is tanh-squashed: one signal spreads into the range, more agreeing
    signals raise confidence monotonically, and realistic turns never saturate.
    """
    one = score(
        CodingAgentDimensions(**{**_zero_dimensions().__dict__, "stuck_exploring": 1.0})
    )
    assert one.score == pytest.approx(math.tanh(_SCORE_GAIN * 0.10))  # tanh(gain*unit)
    assert one.score > 0                          # wrong signal → CAPABLE

    two = score(
        CodingAgentDimensions(**{
            **_zero_dimensions().__dict__,
            "recent_write_intensity": 1.0,
            "pure_bash_intensity": 1.0,
        })
    )
    assert two.confidence > one.confidence        # more signals → more confident
    assert two.score < 0                          # progress → EFFICIENT
    assert abs(two.score) < 1.0                   # realistic turns never saturate


def test_extreme_weights_saturate_via_tanh():
    """A large raw sum saturates smoothly to ±1 through tanh (no hard clip)."""
    dims = CodingAgentDimensions(**{**_zero_dimensions().__dict__, "severity": 1.0})
    high = score(dims, weights={"severity": 5.0})   # raw 5.0 → tanh(25) ≈ 1.0
    assert high.score == pytest.approx(1.0)
    assert high.confidence == pytest.approx(1.0)
    low = score(dims, weights={"severity": -5.0})
    assert low.score == pytest.approx(-1.0)
    assert low.confidence == pytest.approx(1.0)


def test_custom_weights_can_invert_decision():
    """Researchers override weights via the call site, not YAML."""
    dims = CodingAgentDimensions(**{**_zero_dimensions().__dict__, "severity": 1.0})
    default = score(dims, weights=DEFAULT_WEIGHTS)
    inverted = score(dims, weights={"severity": -0.5})
    assert default.score > 0
    assert inverted.score < 0


def test_contributions_are_raw_presquash():
    """``contributions`` are pre-squash: they sum to raw, score = tanh(gain*raw)."""
    dims = CodingAgentDimensions(
        **{**_zero_dimensions().__dict__, "recent_write_intensity": 0.5}
    )
    result = score(dims)
    raw = sum(result.contributions.values())
    assert result.score == pytest.approx(math.tanh(_SCORE_GAIN * raw))


def test_contributions_exceed_squashed_score():
    """A large raw sum stays unsquashed in contributions while score saturates."""
    dims = CodingAgentDimensions(**{**_zero_dimensions().__dict__, "severity": 1.0})
    result = score(dims, weights={"severity": 5.0})
    raw = sum(result.contributions.values())
    assert raw == 5.0
    assert result.score == pytest.approx(1.0)
    assert raw > result.score


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
