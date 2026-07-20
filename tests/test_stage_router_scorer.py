# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the stage-router dimensions + two-axis scorer."""

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


def _zero() -> CodingAgentDimensions:
    return CodingAgentDimensions(
        severity=0.0,
        spinning=0.0,
        exploring=0.0,
        recent_production_intensity=0.0,
        repeated_cmd_ratio=0.0,
    )


def _with(**kw: float) -> CodingAgentDimensions:
    return CodingAgentDimensions(**{**_zero().__dict__, **kw})


def test_zero_signal_scores_to_zero():
    result = score(_zero())
    assert result.score == 0.0
    assert result.confidence == 0.0


def test_wrong_signals_point_capable():
    assert score(_with(severity=0.7)).score > 0
    assert score(_with(spinning=1.0)).score > 0
    assert score(_with(exploring=1.0)).score > 0


def test_progress_signal_points_efficient():
    assert score(_with(recent_production_intensity=1.0)).score < 0


def test_hard_severity_and_spinning_contribute_one_unit():
    # severity is normalised by its HARD cap (0.7), so a HARD error lands at the
    # same unit as a maxed boolean signal like spinning.
    hard = score(_with(severity=0.7))
    spin = score(_with(spinning=1.0))
    assert hard.score == pytest.approx(math.tanh(_SCORE_GAIN * 0.10))
    assert spin.score == pytest.approx(math.tanh(_SCORE_GAIN * 0.10))


def test_exploring_is_a_full_escalation_signal():
    """exploring is a full-weight WRONG signal → clears a default threshold on its own
    (it's the persistent 'reading-without-producing' latch), same weight as spinning."""
    explore = score(_with(exploring=1.0))
    spin = score(_with(spinning=1.0))
    assert explore.score > 0
    assert explore.score == pytest.approx(math.tanh(_SCORE_GAIN * 0.10))  # ~0.462
    assert explore.confidence == pytest.approx(spin.confidence)
    assert explore.confidence > 0.30  # escalates alone at a typical threshold


def test_repeated_cmd_ratio_is_a_wrong_signal():
    """A weak model looping one command (repeated_cmd_ratio→1) is a full WRONG signal
    → CAPABLE, same weight as spinning. Catches churn that severity/spinning miss."""
    churn = score(_with(repeated_cmd_ratio=1.0))
    assert churn.score > 0
    assert churn.score == pytest.approx(math.tanh(_SCORE_GAIN * 0.10))


def test_corroboration_raises_confidence():
    """Two agreeing wrong signals corroborate to higher confidence than one."""
    one = score(_with(severity=0.7))
    two = score(_with(severity=0.7, spinning=1.0))
    assert two.confidence > one.confidence
    assert two.score == pytest.approx(math.tanh(_SCORE_GAIN * 0.20))  # ~0.762


def test_axes_can_cancel_to_neutral():
    """A turn that both errored and produced nets toward zero confidence."""
    dims = _with(severity=0.7, recent_production_intensity=1.0)
    result = score(dims)
    assert result.confidence < 0.30  # roughly cancels → defers to classifier/default


def test_tanh_saturates_smoothly_via_weight_override():
    dims = _with(severity=1.0)
    high = score(dims, weights={"severity": 5.0})  # raw 5.0 → tanh(25) ≈ 1.0
    assert high.score == pytest.approx(1.0)
    low = score(dims, weights={"severity": -5.0})
    assert low.score == pytest.approx(-1.0)


def test_custom_weights_can_invert_decision():
    dims = _with(severity=1.0)
    assert score(dims, weights=DEFAULT_WEIGHTS).score > 0
    assert score(dims, weights={"severity": -0.5}).score < 0


def test_contributions_are_raw_presquash():
    dims = _with(recent_production_intensity=0.5)
    result = score(dims)
    raw = sum(result.contributions.values())
    assert result.score == pytest.approx(math.tanh(_SCORE_GAIN * raw))


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
    assert dims.recent_production_intensity > 0  # we issued one Write call
    # too shallow (turn_depth < 8) for a stall signal to fire
    assert dims.spinning == 0.0
    assert dims.exploring == 0.0
