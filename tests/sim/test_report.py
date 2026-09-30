# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage, weighting, and isolation of task evaluation summaries."""

import json
from dataclasses import replace
from itertools import permutations

import pytest

from switchyard.sim.models import Outcome, Result, Trial
from switchyard.sim.report import Report


def outcome(rewards, *, cost=1.0, duration=3.0):
    return Outcome.from_trials(
        tuple(
            Trial(
                "task",
                str(index),
                "target",
                (),
                reward=reward,
                cost_usd=cost,
                duration_seconds=duration,
            )
            for index, reward in enumerate(rewards)
        )
    )


def result(task_id, *, fast=None, strong=None, target="fast", **kwargs):
    baselines = {
        "fast": fast if fast is not None else outcome([0.0]),
        "strong": strong if strong is not None else outcome([1.0], cost=2.0),
    }
    return Result(
        task_id,
        baselines,
        target=target,
        outcome=baselines[target] if target is not None else None,
        routing_seconds=0.5,
        routing_calls=1,
        routing_failed_calls=0,
        routing_cost_usd=0.1,
        routing_usage={"input_tokens": 10, "output_tokens": 2},
        **kwargs,
    )


def report(expected):
    return Report(expected, ("fast", "strong"), run_id="run", concurrency=2)


def test_repeat_means_remain_finite_when_intermediate_sum_overflows():
    value = outcome([1e308, 1e308], cost=1e308, duration=1e308)
    assert value.reward == value.cost_usd == value.duration_seconds == 1e308
    assert outcome([1e308, 1e308, -1e308]).reward == pytest.approx(1e308 / 3)


@pytest.mark.parametrize("scale,residual", [(1e16, 1.0), (1e308, 1e-300), (1.0, 5e-324)])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_signed_reward_totals_preserve_residual_independent_of_completion_order(
    scale, residual, order
):
    summary = report(3)
    rewards = (scale, residual, -scale)
    for index in order:
        reward = rewards[index]
        recorded = outcome([reward])
        summary.add(result(str(index), fast=recorded, strong=recorded))
    data = summary.to_dict()
    assert data["recorded"]["reward"]["total"] == residual
    assert data["recorded"]["reward"]["mean"] == residual / 3
    assert data["comparison"] == {
        "tasks": 3,
        "routed_mean_reward": residual / 3,
        "targets": {"fast": residual / 3, "strong": residual / 3},
        "empirical_oracle_mean_reward": residual / 3,
    }
    json.dumps(data, allow_nan=False)


def test_unrepresentable_snapshot_is_explicit_and_does_not_corrupt_later_totals():
    summary = report(3)
    for index in range(2):
        recorded = outcome([1e308])
        summary.add(result(str(index), fast=recorded, strong=recorded))
    with pytest.raises(ValueError, match="aggregate exceeds finite"):
        summary.to_dict()
    recorded = outcome([-1e308])
    summary.add(result("last", fast=recorded, strong=recorded))
    data = summary.to_dict()
    assert data["recorded"]["reward"]["total"] == 1e308
    assert data["comparison"]["routed_mean_reward"] == pytest.approx(1e308 / 3)
    assert data["complete"]
    json.dumps(data, allow_nan=False)


def test_comparison_means_do_not_require_representable_baseline_totals():
    summary = report(2)
    for index in range(2):
        summary.add(result(str(index), strong=outcome([1e308], cost=1e308)))
    data = summary.to_dict()
    assert data["comparison"]["targets"]["strong"] == 1e308
    assert data["comparison"]["empirical_oracle_mean_reward"] == 1e308
    assert data["cost_comparison"]["targets"]["strong"] == 1e308
    json.dumps(data, allow_nan=False)


def test_combined_cost_overflow_is_reported_at_snapshot_without_rejecting_result():
    summary = report(1)
    summary.add(replace(result("one", fast=outcome([1], cost=1e308)), routing_cost_usd=1e308))
    assert summary.processed == 1 and summary.complete
    with pytest.raises(ValueError, match="aggregate exceeds finite"):
        summary.to_dict()


def test_repeats_are_averaged_within_task_and_comparisons_share_cohort():
    summary = report(3)
    summary.add(result("one", fast=outcome([0, 1, 1, 0])))
    summary.add(result("two", fast=outcome([1]), strong=outcome([0])))
    summary.add(result("missing-baseline", fast=outcome([1]), strong=outcome([None])))
    data = summary.to_dict()
    assert data["complete"] is True
    assert data["counts"]["scored"] == 3
    assert data["recorded"]["reward"]["mean"] == pytest.approx(2.5 / 3)
    assert data["comparison"] == {
        "tasks": 2,
        "routed_mean_reward": 0.75,
        "targets": {"fast": 0.75, "strong": 0.5},
        "empirical_oracle_mean_reward": 1.0,
    }
    assert data["estimated_cost_usd"]["total"] == pytest.approx(3.3)
    json.dumps(data, allow_nan=False)


def test_errors_and_unknowns_preserve_denominators_and_known_spend():
    summary = report(3)
    summary.add(result("known"))
    summary.add(replace(result("missing"), outcome=outcome([None], cost=None, duration=None)))
    summary.add(
        replace(
            result("failed"),
            target=None,
            outcome=None,
            error="timeout",
            routing_failed_calls=1,
            routing_cost_usd=None,
            routing_usage={"input_tokens": None},
        )
    )
    data = summary.to_dict()
    assert data["complete"] is False
    assert data["counts"] == {
        "total": 3,
        "processed": 3,
        "pending": 0,
        "routed": 2,
        "scored": 1,
        "errors": 1,
        "unscored": 2,
    }
    assert data["recorded"]["reward"] == {
        "known_tasks": 1,
        "observed_total": 0.0,
        "total": None,
        "mean": None,
    }
    assert data["routing"]["cost_usd"]["observed_total"] == pytest.approx(0.2)
    assert data["routing"]["cost_usd"]["total"] is None
    assert data["routing"]["usage"]["input_tokens"]["known_tasks"] == 2
    assert data["routing"]["usage"]["output_tokens"]["total"] is None
    assert data["routing"]["failed_calls"] == 1
    assert data["estimated_cost_usd"]["total"] is None
    assert data["comparison"]["tasks"] == 1


def test_partial_run_snapshot_and_duplicate_rejection_are_atomic():
    summary = report(2)
    summary.add(result("one"))
    snapshot = summary.to_dict()
    assert snapshot["estimated_cost_usd"]["total"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summary.add(result("one"))
    assert summary.to_dict() == snapshot
    with pytest.raises(ValueError, match="routing_seconds"):
        summary.add(replace(result("two"), routing_seconds=float("nan")))
    assert summary.to_dict() == snapshot
    summary.add(result("two", target="strong"))
    assert snapshot["counts"]["processed"] == 1
    assert snapshot["targets"] == {"fast": 1, "strong": 0}
    assert summary.to_dict()["counts"]["processed"] == 2
    with pytest.raises(ValueError, match="more results"):
        summary.add(result("three"))


def test_zero_call_usage_and_missing_call_usage_remain_distinct():
    summary = report(3)
    summary.add(replace(result("no-call"), routing_calls=0, routing_usage={}, routing_cost_usd=0.0))
    summary.add(replace(result("unreported-call"), routing_usage={}))
    summary.add(result("reported-call"))
    usage = summary.to_dict()["routing"]["usage"]
    assert usage["input_tokens"] == {
        "known_tasks": 2,
        "observed_total": 10.0,
        "total": None,
        "mean": None,
    }


def test_selected_error_is_never_scored_and_empty_comparison_is_unknown():
    summary = report(1)
    summary.add(result("unsupported", error="request was rewritten"))
    data = summary.to_dict()
    assert data["counts"]["routed"] == 1
    assert data["counts"]["scored"] == 0
    assert data["recorded"]["cost_usd"]["total"] is None
    assert data["estimated_cost_usd"]["total"] is None
    assert data["comparison"] == {
        "tasks": 0,
        "routed_mean_reward": None,
        "targets": {"fast": None, "strong": None},
        "empirical_oracle_mean_reward": None,
    }
    assert "unknown" in summary.format_text()


def test_lost_observations_are_unknown_and_input_coverage_is_preserved():
    coverage = {"input_tasks": 4, "eligible_tasks": 2, "excluded": ["three", "four"]}
    summary = Report(2, ("fast", "strong"), run_id="run", concurrency=2, coverage=coverage)
    summary.add(result("one"))
    summary.add(
        replace(
            result("two"),
            error="canceled",
            routing_calls=None,
            routing_failed_calls=None,
            routing_cost_usd=None,
            routing_usage={},
        )
    )
    data = summary.to_dict()
    assert data["routing"]["calls"] is None
    assert data["routing"]["known_call_tasks"] == 1
    assert data["routing"]["observed_calls"] == 1
    assert data["routing"]["failed_calls"] is None
    assert data["routing"]["usage"]["input_tokens"]["known_tasks"] == 1
    assert data["coverage"] == coverage
    coverage["excluded"].append("five")
    data["coverage"]["excluded"].clear()
    assert summary.to_dict()["coverage"]["excluded"] == ["three", "four"]


def test_successful_unknown_target_is_rejected_without_changing_report():
    summary = report(1)
    before = summary.to_dict()
    with pytest.raises(ValueError, match="configured targets"):
        summary.add(replace(result("invalid"), target="other"))
    assert summary.to_dict() == before


def test_cost_comparisons_use_one_matched_task_cohort_and_equal_task_weights():
    summary = report(4)
    summary.add(result("one", fast=outcome([0, 1, 1, 0], cost=2), strong=outcome([1], cost=8)))
    summary.add(
        result("two", fast=outcome([None], cost=4), strong=outcome([1], cost=6), target="strong")
    )
    summary.add(
        replace(
            result("missing-baseline", fast=outcome([1], cost=100), strong=outcome([1], cost=None)),
            routing_cost_usd=999,
        )
    )
    summary.add(
        replace(
            result("failed", fast=outcome([1], cost=500), error="routing failed"),
            routing_cost_usd=None,
        )
    )
    data = summary.to_dict()
    assert data["comparison"]["tasks"] == 2
    assert data["cost_comparison"] == {
        "tasks": 2,
        "routed_mean_cost_usd": 4.0,
        "targets": {"fast": 3.0, "strong": 7.0},
        "routing_mean_cost_usd": pytest.approx(0.1),
        "routed_mean_cost_with_routing_usd": pytest.approx(4.1),
    }


def test_cost_comparison_routing_cost_requires_all_cohort_tasks():
    summary = report(3)
    summary.add(result("known"))
    summary.add(replace(result("unknown-routing"), routing_cost_usd=None))
    summary.add(result("outside-cohort", strong=outcome([1], cost=None)))
    costs = summary.to_dict()["cost_comparison"]
    assert costs == {
        "tasks": 2,
        "routed_mean_cost_usd": 1.0,
        "targets": {"fast": 1.0, "strong": 2.0},
        "routing_mean_cost_usd": None,
        "routed_mean_cost_with_routing_usd": None,
    }


def test_empty_cost_comparison_is_unknown_and_zero_cost_remains_known():
    summary = report(2)
    summary.add(result("missing", fast=outcome([1], cost=None)))
    assert summary.to_dict()["cost_comparison"] == {
        "tasks": 0,
        "routed_mean_cost_usd": None,
        "targets": {"fast": None, "strong": None},
        "routing_mean_cost_usd": None,
        "routed_mean_cost_with_routing_usd": None,
    }
    summary.add(
        replace(
            result("free", fast=outcome([1], cost=0), strong=outcome([1], cost=0)),
            routing_cost_usd=0,
        )
    )
    costs = summary.to_dict()["cost_comparison"]
    assert costs["tasks"] == 1
    assert costs["routed_mean_cost_usd"] == costs["routing_mean_cost_usd"] == 0
    assert costs["routed_mean_cost_with_routing_usd"] == 0
