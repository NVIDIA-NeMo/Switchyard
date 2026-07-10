# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted scorer with sigmoid (tanh) shaping: signed score in (-1, +1), confidence = abs(score).

The raw weighted sum is passed through tanh(k * raw) before thresholding. This amplifies
moderate signals toward ±1, ensuring that efficient turns with several weak negative
dimensions cross the confidence threshold rather than sitting in the ambiguous zone.

Score zones relative to threshold t:
  [-1, -t)  → strong efficient signal  → route EFFICIENT
  [-t, t]   → ambiguous               → fall through to classifier / default
  (t,  1]   → strong capable signal   → route CAPABLE
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from switchyard.lib.processors.stage_router.dimensions import CodingAgentDimensions

#: Default linear weights. Positive ⇒ CAPABLE; negative ⇒ EFFICIENT.
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

#: Sigmoid steepness. tanh(k * raw): k=2 means raw=±0.5 → score≈±0.76,
#: pushing moderate efficient/capable signals past the default t=0.5 threshold.
DEFAULT_STEEPNESS: float = 2.0


@dataclass(frozen=True)
class ScoreResult:
    """Output of :func:`score`. ``confidence == abs(score)`` by construction.

    ``contributions`` are the raw per-dimension products (before sigmoid);
    their sum is the pre-sigmoid input, not necessarily equal to ``score``.
    """

    score: float
    confidence: float
    contributions: Mapping[str, float] = field(default_factory=dict)


def score(
    dimensions: CodingAgentDimensions,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    steepness: float = DEFAULT_STEEPNESS,
) -> ScoreResult:
    """Score ``dimensions`` against ``weights`` with sigmoid shaping.

    Raw weighted sum is passed through ``tanh(steepness * raw)`` to produce a
    score in (-1, +1). Confidence is ``abs(score)``.
    """
    contributions: dict[str, float] = {}
    raw = 0.0
    for field_name, weight in weights.items():
        value = getattr(dimensions, field_name, 0.0)
        contribution = value * weight
        contributions[field_name] = contribution
        raw += contribution
    shaped = math.tanh(steepness * raw)
    return ScoreResult(score=shaped, confidence=abs(shaped), contributions=contributions)


__all__ = ["DEFAULT_STEEPNESS", "DEFAULT_WEIGHTS", "ScoreResult", "score"]
