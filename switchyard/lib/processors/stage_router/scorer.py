# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted linear scorer, tanh-squashed to ``(-1, +1)``; confidence = ``abs(score)``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from switchyard.lib.processors.stage_router.dimensions import CodingAgentDimensions

#: Gain applied before the tanh squash. The raw weighted sum is small (typical
#: multi-signal turns land near ±0.35), so ``tanh(gain * raw)`` spreads it across
#: most of ``(-1, +1)``. That turns the symmetric ``confidence_threshold`` into a
#: full-range dial: users pick a positive ``t`` in ``(0, 1)`` and the confident
#: fraction sweeps smoothly (e.g. t=0.1 → most turns, t=0.7 → few). Monotonic, so
#: routing sign is unchanged — only the confidence scale is spread out.
_SCORE_GAIN: float = 5.0

#: Max severity that reaches the scorer: `1.0` (critical) is caught by the picker
#: override, so `0.7` (hard) is the strongest error the linear scorer ever sees.
_HARD_SEVERITY: float = 0.7
#: Fixed weight one maxed signal contributes. Weights are constant (independent of
#: threshold), so the ``confidence_threshold`` is the sole corroboration dial:
#: ``signals-to-clear = threshold / _SIGNAL_UNIT``. With unit 0.10 the threshold
#: maps to whole signal counts — 0.10 → one maxed signal clears, 0.20 → two must
#: agree, etc. (unit and threshold are redundant for routing; only their ratio
#: matters, so tuning lives entirely in the threshold). Kept well under 1 so the
#: summed score never saturates to ±1 — no single axis can peg the decision.
_SIGNAL_UNIT: float = 0.10

#: "Something is wrong" signals — always push toward CAPABLE (strong), both pickers.
#: severity is per-turn; the other two are windowed gates — all self-clear.
_WRONG_SIGNALS: tuple[str, ...] = ("severity", "stuck_exploring", "no_progress")
#: "Progress / settled" signals — push toward EFFICIENT (weak), both pickers.
#: capable_first / efficient_first differ only in the fall_open default. All windowed.
_PROGRESS_SIGNALS: tuple[str, ...] = (
    "recent_write_intensity",
    "planning_active",
    "pure_bash_intensity",
    "no_error_streak_intensity",
)
#: Max value each signal reaches at the scorer; used to normalise so a maxed
#: signal contributes exactly ``_SIGNAL_UNIT``. Defaults to 1.0 for [0, 1] gates.
_MAX_VALUE: Mapping[str, float] = {"severity": _HARD_SEVERITY}


def _build_weights(unit: float = _SIGNAL_UNIT) -> dict[str, float]:
    """Signed, fixed linear weights: wrong → CAPABLE (+), progress → EFFICIENT (-).

    A single maxed signal contributes ``unit`` (severity is normalised by its
    0.7 cap so it too lands at ``unit``). Because every weight is well under 1,
    the summed score never saturates; corroboration is set downstream by the
    ``confidence_threshold`` (signals-to-clear = threshold / unit).
    """
    weights: dict[str, float] = {}
    for name in _WRONG_SIGNALS:
        weights[name] = unit / _MAX_VALUE.get(name, 1.0)
    for name in _PROGRESS_SIGNALS:
        weights[name] = -unit / _MAX_VALUE.get(name, 1.0)
    return weights


#: Fixed scorer weights. Corroboration is dialed by the confidence_threshold.
DEFAULT_WEIGHTS: Mapping[str, float] = _build_weights()


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
    """Score ``dimensions``; raw weighted sum is tanh-squashed into ``(-1, +1)``.

    ``contributions`` are the raw per-signal weighted values (pre-squash, so they
    sum to the raw score); ``score`` is ``tanh(gain * raw)``. The squash spreads
    the small raw sum across the range so the ``confidence_threshold`` reads as a
    symmetric ``(0, 1)`` dial.
    """
    contributions: dict[str, float] = {}
    raw = 0.0
    for field_name, weight in weights.items():
        value = getattr(dimensions, field_name, 0.0)
        contribution = value * weight
        contributions[field_name] = contribution
        raw += contribution
    squashed = math.tanh(_SCORE_GAIN * raw)
    return ScoreResult(score=squashed, confidence=abs(squashed), contributions=contributions)


__all__ = ["DEFAULT_WEIGHTS", "ScoreResult", "score"]
