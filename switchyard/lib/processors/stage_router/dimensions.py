# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scorer-ready view of :class:`ToolResultSignal` — two axes, five signals.

The stage_router scorer models a coding turn on two independent axes:

* **error** — did the recent tool results error?  ``severity`` (bad pole) vs
  ``no_error_streak_intensity`` (good pole).
* **production** — is the agent producing code?  ``spinning`` / ``exploring``
  (bad poles) vs ``recent_production_intensity`` (good pole).

``spinning`` and ``exploring`` are mutually exclusive, split by whether the
agent is doing *any* investigative work (reads / plans) in the recent window:
``spinning`` = not even looking, ``exploring`` = looking but not building.

Every field here is consumed — ``severity`` by the picker override, the rest by
the scorer weights. All counts are over the recent window; see the notes on
:func:`from_signal` for the two windowing caveats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from switchyard_rust.components import ToolResultSignal

#: Turn-depth below which stall signals stay quiet — early no-write turns are
#: normal exploration, not a stall.
_STALL_MIN_TURN_DEPTH: int = 8


def _saturating(x: float, scale: float) -> float:
    """Map non-negative counts to ``[0, 1)``; ``scale`` is the half-saturation point."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-x / scale)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class CodingAgentDimensions:
    """Normalised, scorer-ready view of a single :class:`ToolResultSignal`."""

    #: Windowed max error severity in ``[0, 1]`` (used by the picker override; the
    #: scorer weights it capped at HARD since CRITICAL short-circuits upstream).
    severity: float
    #: 1.0 when the recent window has no reads, plans, writes, or edits — the agent
    #: is only cycling non-inspecting commands (a struggle signal → strong).
    spinning: float
    #: 1.0 when the recent window has reads/plans but no writes or edits — the agent
    #: is investigating without converging (a neutral-ish signal → strong, half weight).
    exploring: float
    #: Fraction of recent tool ops that produced code (writes + edits) → weak.
    recent_production_intensity: float
    #: Saturating count of consecutive clean recent results → weak.
    no_error_streak_intensity: float


def from_signal(signal: ToolResultSignal) -> CodingAgentDimensions:
    """Project a :class:`ToolResultSignal` onto the two-axis dimension space.

    Windowing notes (both inherent to routing on the normalised request):

    * ``turn_depth`` is a raw *message* count, so its scale varies by wire format
      (Anthropic batches tool results into fewer messages than OpenAI-chat). The
      ``_STALL_MIN_TURN_DEPTH`` gate is therefore approximate across origins.
    * ``severity`` is windowed over the last N tool *results* while the ``recent_*``
      counts are over the last N tool *calls* — usually the same turns, but a
      trailing call without a result yet can offset them by one.
    """
    recent_ops = (
        signal.recent_write_count
        + signal.recent_edit_count
        + signal.recent_read_count
        + signal.recent_todowrite_count
    )
    deep_enough = signal.turn_depth >= _STALL_MIN_TURN_DEPTH
    no_production = signal.recent_write_count == 0 and signal.recent_edit_count == 0
    investigating = signal.recent_read_count >= 1 or signal.recent_todowrite_count >= 1
    # spinning vs exploring partition the "not producing" case by investigative
    # activity, so at most one fires — no double-counting on the production axis.
    spinning = deep_enough and no_production and not investigating
    exploring = deep_enough and no_production and investigating
    return CodingAgentDimensions(
        severity=float(signal.severity),
        spinning=1.0 if spinning else 0.0,
        exploring=1.0 if exploring else 0.0,
        recent_production_intensity=_ratio(
            signal.recent_write_count + signal.recent_edit_count, recent_ops
        ),
        no_error_streak_intensity=_saturating(signal.no_error_streak, scale=3.0),
    )


__all__ = ["CodingAgentDimensions", "from_signal"]
