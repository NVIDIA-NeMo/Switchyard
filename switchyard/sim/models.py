# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded task outcomes and the results of evaluating routing decisions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trial:
    """One recorded attempt. Messages contain only the input before agent execution.

    A target identifies a complete model/agent configuration, not just a model name.
    Unknown measurements stay ``None``. Usage preserves source counter names.
    """

    task_id: str
    trial_id: str
    target: str
    messages: tuple[dict[str, object], ...]
    reward: float | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    duration_seconds: float | None = None
    task_checksum: str | None = None
    model: str | None = None
    source: str | None = None
    usage: Mapping[str, int | None] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "trial_id", "target"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.task_checksum is not None and (
            not isinstance(self.task_checksum, str) or not self.task_checksum.strip()
        ):
            raise ValueError("task_checksum must be a non-empty string or None")
        for name in ("reward", "cost_usd", "duration_seconds"):
            value = getattr(self, name)
            try:
                finite = value is None or math.isfinite(value)
            except (TypeError, OverflowError):
                finite = False
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not finite
                or (name != "reward" and value < 0)
            ):
                raise ValueError(
                    f"{name} must be finite" + (" and non-negative" if name != "reward" else "")
                )


@dataclass(frozen=True)
class LoadIssue:
    """An input problem retained alongside successfully loaded trials."""

    source: str
    message: str
    task_id: str | None = None


@dataclass(frozen=True)
class Run:
    """Imported evidence from any producer, including every rejected input."""

    trials: tuple[Trial, ...]
    issues: tuple[LoadIssue, ...] = ()


HarborRun = Run


@dataclass(frozen=True)
class Outcome:
    """Mean recorded measurements for one task and target.

    Each mean is unavailable unless every repeat reports that measurement.
    Counts distinguish missing measurements from observed zero values.
    """

    trials: int
    reward: float | None
    cost_usd: float | None
    duration_seconds: float | None
    reward_trials: int
    cost_trials: int
    duration_trials: int

    @classmethod
    def from_trials(cls, trials: tuple[Trial, ...]) -> Outcome:
        if not trials:
            raise ValueError("an outcome requires at least one trial")

        def measurement(name: str) -> tuple[float | None, int]:
            values = [getattr(trial, name) for trial in trials if getattr(trial, name) is not None]
            if len(values) != len(trials):
                return None, len(values)
            try:
                mean = math.fsum(values) / len(trials)
            except OverflowError:
                # A finite mean can have a sum outside the floating-point range.
                mean = float(statistics.mean(values))
            return mean, len(values)

        reward, reward_count = measurement("reward")
        cost, cost_count = measurement("cost_usd")
        duration, duration_count = measurement("duration_seconds")
        return cls(len(trials), reward, cost, duration, reward_count, cost_count, duration_count)


@dataclass(frozen=True)
class Task:
    """One routing input with recorded outcomes for each candidate target."""

    task_id: str
    messages: tuple[dict[str, object], ...]
    outcomes: Mapping[str, Outcome]
    trials: Mapping[str, tuple[Trial, ...]]


@dataclass(frozen=True)
class Result:
    """One scored decision or visible evaluation error; safe to stream to a sink."""

    task_id: str
    baselines: Mapping[str, Outcome]
    target: str | None = None
    model: str | None = None
    outcome: Outcome | None = None
    decision_id: str | None = None
    algorithm: str | None = None
    evidence: object = None
    fallbacks: tuple[str, ...] = ()
    routing_seconds: float | None = None
    routing_calls: int | None = None
    routing_failed_calls: int | None = None
    routing_cost_usd: float | None = None
    routing_usage: Mapping[str, int | None] = field(default_factory=dict)
    error: str | None = None
    routing_error_kind: str | None = None
    routing_error_status: int | None = None
    routing_error_target: str | None = None
