# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted linear scorer: signed score in ``[-1, +1]``, confidence = ``abs(score)``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from switchyard.lib.processors.stage_router.dimensions import CodingAgentDimensions

#: Default linear weights. Positive ⇒ STRONG; negative ⇒ WEAK. Calibrated so
#: a single high-impact axis lands past the 0.5 default confidence threshold.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "severity":                    0.80,
    "stuck_exploring":             0.70,
    "no_progress":                 0.60,
    "tests_passed":               -0.80,
    "planning_active":            -0.70,
    "write_intensity":            -0.40,
    "edit_intensity":             -0.30,
    "recent_write_intensity":     -0.30,
    "pure_bash_intensity":        -0.30,
    "no_error_streak_intensity":  -0.20,
}


@dataclass(frozen=True)
class ScoreResult:
    """Output of :func:`score`. ``confidence == abs(score)`` by construction."""

    score: float
    confidence: float
    contributions: Mapping[str, float] = field(default_factory=dict)


def score(
    dimensions: CodingAgentDimensions,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> ScoreResult:
    """Score ``dimensions`` against ``weights``; raw sum is clipped to ``[-1, +1]``."""
    contributions: dict[str, float] = {}
    raw = 0.0
    for field_name, weight in weights.items():
        value = getattr(dimensions, field_name, 0.0)
        contribution = value * weight
        contributions[field_name] = contribution
        raw += contribution
    clipped = max(-1.0, min(1.0, raw))
    return ScoreResult(score=clipped, confidence=abs(clipped), contributions=contributions)


__all__ = ["DEFAULT_WEIGHTS", "ScoreResult", "score"]
