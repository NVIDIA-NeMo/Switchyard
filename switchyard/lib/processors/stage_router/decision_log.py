# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe per-source decision counter for the stage_router picker."""

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

DecisionSource = Literal[
    "override",         # _apply_overrides short-circuited (severity ≥ 1.0, large prompt, tests_passed)
    "dimensions",       # scorer confidence ≥ confidence_threshold
    "llm-classifier",   # classifier consulted and returned a tier
    "fall_open",        # classifier configured but returned None, OR not configured at all
    "no_signal",        # ToolResultSignal not present (first turn)
]

#: ``ctx.metadata`` key the picker writes its decision source to.
CONTEXT_KEY: str = "stage_router_decision_source"


@dataclass
class StageRouterDecisionLog:
    """Thread-safe counter for stage_router decision sources."""

    counts: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, source: DecisionSource) -> None:
        with self._lock:
            self.counts[source] = self.counts.get(source, 0) + 1

    def snapshot(self) -> dict[str, int]:
        """Return a copy of the current counts; safe to publish."""
        with self._lock:
            return dict(self.counts)

    def total(self) -> int:
        with self._lock:
            return sum(self.counts.values())


__all__ = ["CONTEXT_KEY", "StageRouterDecisionLog", "DecisionSource"]
