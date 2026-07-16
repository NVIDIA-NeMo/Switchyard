# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scorer-ready view of :class:`ToolResultSignal` — all fields normalised to ``[0, 1]``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from switchyard_rust.components import ToolResultSignal


_PURE_BASH_NORM: float = 8.0


def _saturating(x: float, scale: float) -> float:
    """Map non-negative counts to ``[0, 1]``; ``scale`` is the half-saturation point."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-x / scale)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class CodingAgentDimensions:
    """Normalised, scorer-ready view of a single :class:`ToolResultSignal`."""

    severity: float
    no_error_streak_intensity: float
    write_intensity: float
    edit_intensity: float
    recent_write_intensity: float
    planning_active: float
    pure_bash_intensity: float
    stuck_exploring: float
    no_progress: float
    tests_passed: float


def from_signal(signal: ToolResultSignal) -> CodingAgentDimensions:
    """Project a :class:`ToolResultSignal` onto the normalised dimension space.

    Note: `turn_depth`, `recent_read_count`, `recent_write_count`, `write_count`,
    and `edit_count` are read from `signal` for the `stuck_exploring` /
    `no_progress` boolean gates, but their normalised intensities aren't
    exposed as separate dimensions because nothing in
    :data:`DEFAULT_WEIGHTS` keys off them.
    """
    total_tool_ops = signal.write_count + signal.edit_count + signal.read_count
    recent_tool_ops = signal.recent_write_count + signal.recent_edit_count + signal.recent_read_count
    # stuck_exploring: a *recent* read-stall — spinning on reads without writing
    # in the recent window. Windowed, so it drops the moment a write lands.
    stuck = (
        signal.turn_depth >= 8
        and signal.recent_write_count == 0
        and signal.recent_read_count >= 2
    )
    # no_progress: a *windowed* dead-end — deep into the run (turn_depth > 30) with no
    # write or edit in the recent window. Windowed like every other signal, so it clears
    # once the agent produces something recent and re-fires if it stalls again. The
    # deeper turn_depth gate keeps it distinct from stuck_exploring (turn_depth >= 8 and
    # requires recent reads), so the two still corroborate rather than duplicate.
    no_progress = (
        signal.turn_depth > 30
        and signal.recent_write_count == 0
        and signal.recent_edit_count == 0
    )
    return CodingAgentDimensions(
        severity=float(signal.severity),
        no_error_streak_intensity=_saturating(signal.no_error_streak, scale=3.0),
        write_intensity=_ratio(signal.write_count, total_tool_ops),
        edit_intensity=_ratio(signal.edit_count, total_tool_ops),
        recent_write_intensity=_ratio(signal.recent_write_count, recent_tool_ops),
        planning_active=1.0 if signal.recent_todowrite_count > 0 else 0.0,
        pure_bash_intensity=_saturating(signal.pure_bash_streak, scale=_PURE_BASH_NORM),
        stuck_exploring=1.0 if stuck else 0.0,
        no_progress=1.0 if no_progress else 0.0,
        # Only treat tests_passed as a signal once the agent has made real changes;
        # early test runs against the unmodified codebase are exploratory, not confirmatory.
        tests_passed=1.0 if signal.tests_passed and signal.write_count >= 3 else 0.0,
    )


__all__ = ["CodingAgentDimensions", "from_signal"]
