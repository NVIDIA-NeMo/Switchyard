# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, task-weighted summaries of recorded routing evaluations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction

from .models import Result


def _as_float(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError as error:
        raise ValueError("report aggregate exceeds finite floating-point range") from error


@dataclass
class _Sum:
    """Exact sum of finite ints/floats; binary denominators are powers of two."""

    numerator: int = 0
    denominator: int = 1

    def add(self, value: float) -> None:
        numerator, denominator = value.as_integer_ratio()
        if denominator > self.denominator:
            self.numerator *= denominator // self.denominator
            self.denominator = denominator
        self.numerator += numerator * (self.denominator // denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass
class _Measurement:
    known: int = 0
    total: _Sum = field(default_factory=_Sum)

    def add(self, value: float | None) -> None:
        if value is not None:
            self.known += 1
            self.total.add(value)

    def summary(self, count: int) -> dict[str, int | float | None]:
        complete = count > 0 and self.known == count
        total = self.total.fraction
        return {
            "known_tasks": self.known,
            "observed_total": _as_float(total) if self.known else None,
            "total": _as_float(total) if complete else None,
            "mean": _as_float(total / count) if complete else None,
        }


def _validate_measurement(value: float | None, name: str, *, nonnegative: bool = True) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (nonnegative and value < 0)
    ):
        raise ValueError(f"invalid {name}: expected a finite measurement or None")


class Report:
    """Accumulate one result per task without keeping decisions or trajectories.

    Recorded measurements already average repeats within a task and target. Tasks
    have equal weight here. Unknown measurements invalidate the corresponding
    total and mean; ``observed_total`` retains the known part. Reward and cost
    comparisons each use successful decisions with that measurement known for
    the selected and every fixed target. The empirical reward oracle takes the
    best target mean for each task, after averaging repeats. It is a retrospective
    comparison, not a deployable routing policy.
    """

    def __init__(
        self,
        expected: int,
        targets: tuple[str, ...],
        *,
        run_id: str,
        concurrency: int,
        coverage: Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError("expected must be a non-negative integer")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not targets or any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("targets must contain non-empty names")
        if len(set(targets)) != len(targets):
            raise ValueError("targets must be unique")
        self.expected = expected
        self.targets = targets
        self.run_id = run_id
        self.concurrency = concurrency
        self._coverage = deepcopy(dict(coverage)) if coverage is not None else None
        self.processed = 0
        self.routed = 0
        self.scored = 0
        self.errors = 0
        self._seen: set[str] = set()
        self._selected: Counter[str] = Counter(dict.fromkeys(targets, 0))
        self._recorded = {
            name: _Measurement() for name in ("reward", "cost_usd", "duration_seconds")
        }
        self._routing_seconds = _Measurement()
        self._routing_cost = _Measurement()
        self._routing_calls = _Measurement()
        self._routing_failed_calls = _Measurement()
        self._zero_call_tasks = 0
        self._routing_usage: dict[str, _Measurement] = {}
        self._estimated_cost = _Measurement()
        self._comparison_tasks = 0
        self._comparison_routed = _Sum()
        self._comparison_targets = {target: _Sum() for target in targets}
        self._comparison_best = _Sum()
        self._cost_comparison_tasks = 0
        self._cost_comparison_routed = _Sum()
        self._cost_comparison_targets = {target: _Sum() for target in targets}
        self._cost_comparison_routing = _Measurement()

    @property
    def complete(self) -> bool:
        return (
            self.expected > 0
            and self.processed == self.expected
            and self.scored == self.expected
            and not self.errors
        )

    def add(self, result: Result) -> None:
        """Add one result, rejecting duplicate IDs and invalid measurements."""
        if result.task_id in self._seen:
            raise ValueError(f"duplicate result for task {result.task_id!r}")
        if self.processed >= self.expected:
            raise ValueError("more results than expected tasks")
        if result.error is None and result.target is not None and result.target not in self.targets:
            raise ValueError("successful result target must be one of the configured targets")
        for name in ("routing_calls", "routing_failed_calls"):
            count = getattr(result, name)
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            result.routing_failed_calls is not None
            and result.routing_calls is not None
            and result.routing_failed_calls > result.routing_calls
        ):
            raise ValueError("routing_failed_calls cannot exceed routing_calls")
        _validate_measurement(result.routing_seconds, "routing_seconds")
        _validate_measurement(result.routing_cost_usd, "routing_cost_usd")
        for name, value in result.routing_usage.items():
            _validate_measurement(value, name)
        outcomes = [*result.baselines.values()]
        if result.outcome is not None:
            outcomes.append(result.outcome)
        for recorded in outcomes:
            _validate_measurement(recorded.reward, "reward", nonnegative=False)
            _validate_measurement(recorded.cost_usd, "cost_usd")
            _validate_measurement(recorded.duration_seconds, "duration_seconds")

        self._seen.add(result.task_id)
        self.processed += 1
        self.errors += int(result.error is not None)
        if result.target is not None:
            self.routed += 1
            self._selected[result.target] += 1
        outcome = result.outcome if result.error is None and result.target is not None else None
        self.scored += int(outcome is not None and outcome.reward is not None)
        for name, measurement in self._recorded.items():
            measurement.add(getattr(outcome, name) if outcome is not None else None)
        self._routing_seconds.add(result.routing_seconds)
        self._routing_cost.add(result.routing_cost_usd)
        self._routing_calls.add(result.routing_calls)
        self._routing_failed_calls.add(result.routing_failed_calls)
        for name in self._routing_usage.keys() | result.routing_usage.keys():
            if name not in self._routing_usage:
                self._routing_usage[name] = _Measurement(known=self._zero_call_tasks)
            value = result.routing_usage.get(name)
            self._routing_usage[name].add(
                0 if value is None and result.routing_calls == 0 else value
            )
        self._zero_call_tasks += int(result.routing_calls == 0)
        if (
            outcome is not None
            and outcome.cost_usd is not None
            and result.routing_cost_usd is not None
        ):
            self._estimated_cost.add(outcome.cost_usd)
            self._estimated_cost.total.add(result.routing_cost_usd)

        if outcome is not None and outcome.reward is not None:
            rewards = [
                result.baselines[target].reward if target in result.baselines else None
                for target in self.targets
            ]
            if all(reward is not None for reward in rewards):
                known_rewards = [reward for reward in rewards if reward is not None]
                self._comparison_tasks += 1
                self._comparison_routed.add(outcome.reward)
                for target, reward in zip(self.targets, known_rewards, strict=True):
                    self._comparison_targets[target].add(reward)
                self._comparison_best.add(max(known_rewards))

        if outcome is not None and outcome.cost_usd is not None:
            costs = [
                result.baselines[target].cost_usd if target in result.baselines else None
                for target in self.targets
            ]
            if all(cost is not None for cost in costs):
                self._cost_comparison_tasks += 1
                self._cost_comparison_routed.add(outcome.cost_usd)
                for target, cost in zip(self.targets, costs, strict=True):
                    assert cost is not None
                    self._cost_comparison_targets[target].add(cost)
                self._cost_comparison_routing.add(result.routing_cost_usd)

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON-compatible snapshot with explicit coverage."""
        count = self._comparison_tasks
        cost_count = self._cost_comparison_tasks
        routed_cost = (
            _as_float(self._cost_comparison_routed.fraction / cost_count) if cost_count else None
        )
        routing_cost = self._cost_comparison_routing.summary(cost_count)["mean"]
        estimated = self._estimated_cost.summary(self.processed)
        if self.processed != self.expected:
            estimated["total"] = None
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "concurrency": self.concurrency,
            "coverage": deepcopy(self._coverage),
            "complete": self.complete,
            "counts": {
                "total": self.expected,
                "processed": self.processed,
                "pending": self.expected - self.processed,
                "routed": self.routed,
                "scored": self.scored,
                "errors": self.errors,
                "unscored": self.processed - self.scored,
            },
            "targets": dict(self._selected),
            "recorded": {
                name: value.summary(self.processed) for name, value in self._recorded.items()
            },
            "routing": {
                "seconds": self._routing_seconds.summary(self.processed),
                "calls": self._routing_calls.summary(self.processed)["total"],
                "known_call_tasks": self._routing_calls.known,
                "observed_calls": _as_float(self._routing_calls.total.fraction),
                "failed_calls": self._routing_failed_calls.summary(self.processed)["total"],
                "known_failed_call_tasks": self._routing_failed_calls.known,
                "observed_failed_calls": _as_float(self._routing_failed_calls.total.fraction),
                "cost_usd": self._routing_cost.summary(self.processed),
                "usage": {
                    name: value.summary(self.processed)
                    for name, value in sorted(self._routing_usage.items())
                },
            },
            "estimated_cost_usd": estimated,
            "comparison": {
                "tasks": count,
                "routed_mean_reward": _as_float(self._comparison_routed.fraction / count)
                if count
                else None,
                "targets": {
                    target: _as_float(total.fraction / count) if count else None
                    for target, total in self._comparison_targets.items()
                },
                "empirical_oracle_mean_reward": _as_float(self._comparison_best.fraction / count)
                if count
                else None,
            },
            "cost_comparison": {
                "tasks": cost_count,
                "routed_mean_cost_usd": routed_cost,
                "targets": {
                    target: _as_float(total.fraction / cost_count) if cost_count else None
                    for target, total in self._cost_comparison_targets.items()
                },
                "routing_mean_cost_usd": routing_cost,
                "routed_mean_cost_with_routing_usd": _as_float(
                    (
                        self._cost_comparison_routed.fraction
                        + self._cost_comparison_routing.total.fraction
                    )
                    / cost_count
                )
                if routed_cost is not None and routing_cost is not None
                else None,
            },
        }

    def format_text(self) -> str:
        """Render coverage and the currently available headline measurements."""
        reward = self._recorded["reward"].summary(self.processed)["mean"]
        cost = self._estimated_cost.summary(self.expected)["total"]
        reward_text = f"{reward:.4f}" if reward is not None else "unknown"
        cost_text = f"${cost:.4f}" if cost is not None else "unknown"
        return (
            f"{self.processed}/{self.expected} tasks; {self.scored} scored; {self.errors} errors; "
            f"mean reward {reward_text}; estimated cost {cost_text}"
        )
