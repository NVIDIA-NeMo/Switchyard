# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed linear scorer over two axes (error, production), tanh-squashed to ``(-1, +1)``.

Each signal contributes a fixed weight; WRONG signals push toward CAPABLE (+),
PROGRESS signals toward EFFICIENT (−). The raw weighted sum is squashed with
``tanh(gain * raw)`` and ``confidence = abs(score)``.

**On the threshold (read this before tuning):** the raw sum is small — one maxed
signal is ``±0.10``, two corroborating signals ``±0.20``. The gain spreads that
into a usable confidence range; it does **not** make the threshold an integer
"signals-to-clear" count. Empirically the dial reads:

    conf 0.245  exploring alone (half weight)
    conf 0.462  one full signal (a HARD error, or spinning)
    conf 0.635  severity + exploring
    conf 0.762  two full signals (severity + spinning)

So a threshold near 0.3 escalates on ~one signal, ~0.5 needs ~1.5, ~0.7 needs two
to corroborate. The reachable range is ``(0, ~0.76)`` for these two axes — a
threshold above ~0.76 can't be met, so it always defers to the classifier/default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from switchyard.lib.processors.stage_router.dimensions import CodingAgentDimensions

#: Gain applied before the tanh squash — spreads the small raw sum across the
#: usable confidence range (see the module docstring for the resulting dial). Load-
#: bearing: without it, confidence would cap near ±0.20 and mid/high thresholds
#: would be unreachable.
_SCORE_GAIN: float = 5.0

#: Max severity the scorer ever sees: ``1.0`` (critical) is caught by the picker
#: override, so ``0.7`` (hard) is the strongest error that reaches the linear sum.
#: Used to normalise ``severity`` so a HARD error contributes exactly ``_SIGNAL_UNIT``.
_HARD_SEVERITY: float = 0.7
#: Weight one maxed signal contributes. Small enough that no single axis pegs the
#: decision; corroboration across the two axes is what pushes confidence up.
_SIGNAL_UNIT: float = 0.10

#: "Something is wrong" → CAPABLE (strong). ``exploring`` is neutral (half weight):
#: it never escalates alone at a sane threshold, only when corroborated.
_WRONG_SIGNALS: tuple[str, ...] = ("severity", "spinning", "exploring")
#: "Making progress" → EFFICIENT (weak). Both are the good poles of the two axes.
_PROGRESS_SIGNALS: tuple[str, ...] = ("recent_production_intensity", "no_error_streak_intensity")
#: Max value each signal reaches, used to normalise so a maxed signal contributes
#: ``_SIGNAL_UNIT``. Defaults to 1.0 for ``[0, 1]`` gates/ratios.
_MAX_VALUE: Mapping[str, float] = {"severity": _HARD_SEVERITY}
#: Signals weighted at half unit — deliberately weak, needs corroboration to matter.
_HALF_WEIGHT: frozenset[str] = frozenset({"exploring"})


def _build_weights(unit: float = _SIGNAL_UNIT) -> dict[str, float]:
    """Signed, fixed linear weights: wrong → CAPABLE (+), progress → EFFICIENT (−).

    A maxed signal contributes ``unit`` (``severity`` is normalised by its HARD cap
    so it too lands at ``unit``); ``exploring`` is halved to keep it neutral.
    """
    weights: dict[str, float] = {}
    for name in _WRONG_SIGNALS:
        w = unit / _MAX_VALUE.get(name, 1.0)
        weights[name] = w / 2.0 if name in _HALF_WEIGHT else w
    for name in _PROGRESS_SIGNALS:
        weights[name] = -unit / _MAX_VALUE.get(name, 1.0)
    return weights


#: Fixed scorer weights. Corroboration across axes is dialed by confidence_threshold.
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
    sum to the raw score); ``score`` is ``tanh(gain * raw)``; ``confidence`` its
    magnitude. Positive ``score`` → CAPABLE, negative → EFFICIENT.
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
