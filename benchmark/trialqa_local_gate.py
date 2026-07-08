# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast, paired promotion gates for the local TrialQA experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import benchmark.trialqa_local_batch as batch
import benchmark.trialqa_local_dataset as dataset
import benchmark.trialqa_local_demo as demo

SCHEMA_VERSION = "switchyard.trialqa_gate_report.v3"
POLICY_NAME = "ultra-efficiency-v3"
TOKEN_REDUCTION_MIN = 0.15
OPERATIONAL_CALL_REDUCTION_MIN = 0.20
QUALITY_DELTA_MIN = -0.05
QUALITY_CONFIDENCE_LEVEL = 0.95
FUTILITY_CONFIDENCE_LEVEL = 0.95
INTERIM_QUALITY_MODE = "interim_harm_screen"
CONFIRMATORY_QUALITY_MODE = "confirmatory_noninferiority"
_REPEAT_SUFFIX = re.compile(r"-r\d+\Z")
GateKind = Literal["plumbing", "operational", "promotion"]
JsonObject = dict[str, Any]


class TrialQAGateError(RuntimeError):
    """A gate input is incomplete, inconsistent, or unsafe to compare."""


@dataclass(frozen=True)
class TaskMetrics:
    task_id: str
    pair_id: str
    condition: str
    total_tokens: int
    operational_calls: int
    successful_operational_calls: int
    skill_load_calls: int
    successful_skill_load_calls: int
    model_turns: int
    physical_attempts: int = 0
    null_eof_retries: int = 0
    retry_usage_charges: int = 0
    unpriced_null_eof_retries: int = 0
    retry_token_sensitivity: int = 0
    score: float | None = None
    terminal: bool = False
    score_bound: bool = True
    timed_out: bool = False
    generation_timeout_seconds: int = batch.DEFAULT_GENERATION_TIMEOUT_SECONDS
    generation_timeout_policy: str = batch.DEFAULT_GENERATION_TIMEOUT_POLICY


def _trimmed_mean(values: Sequence[int], fraction: float = 0.10) -> float:
    if not values:
        raise TrialQAGateError("a robust mean requires at least one paired delta")
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    if not retained:
        raise TrialQAGateError("trimmed mean removed the complete cohort")
    return statistics.fmean(retained)


def _reduction_fraction(baseline: int, treatment: int) -> float | None:
    return (baseline - treatment) / baseline if baseline else None


def _optimistic_delta_lower_bound(values: Sequence[int]) -> float | None:
    """Return a small-sample one-sided lower bound for treatment-minus-control.

    A positive result rules out even a zero mean reduction at the configured
    confidence level. Fewer than four pairs are deliberately non-decisive.
    """

    if len(values) < 4:
        return None
    point = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    degrees_of_freedom = len(values) - 1
    z_value = statistics.NormalDist().inv_cdf(FUTILITY_CONFIDENCE_LEVEL)
    critical_value = z_value * math.sqrt(degrees_of_freedom / (degrees_of_freedom - 2))
    return point - critical_value * standard_error


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    return sum(
        math.comb(trials, count) * probability**count * (1.0 - probability) ** (trials - count)
        for count in range(successes + 1)
    )


def _clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    low = 0.0
    high = successes / trials
    target = 1.0 - alpha
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(successes - 1, trials, midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    if successes == trials:
        return 1.0
    low = successes / trials
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _question_key(pair_id: str) -> str:
    return _REPEAT_SUFFIX.sub("", pair_id)


def _validate_primary_evaluation_scope(
    value: Mapping[str, object],
) -> JsonObject:
    integer_fields = {"question_start", "question_count", "repeat_count", "task_count"}
    required = {*integer_fields, "question_group_keys_sha256"}
    if set(value) != required:
        raise TrialQAGateError(
            "primary_evaluation_scope must contain exactly "
            "question_start, question_count, repeat_count, task_count, and "
            "question_group_keys_sha256"
        )
    normalized: JsonObject = {}
    for field in sorted(integer_fields):
        item = value[field]
        if not isinstance(item, int) or isinstance(item, bool):
            raise TrialQAGateError(f"primary_evaluation_scope.{field} must be an integer")
        normalized[field] = item
    digest = value["question_group_keys_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise TrialQAGateError(
            "primary_evaluation_scope.question_group_keys_sha256 must be a SHA-256 digest"
        )
    normalized["question_group_keys_sha256"] = digest
    if cast(int, normalized["question_start"]) < 0:
        raise TrialQAGateError("primary_evaluation_scope.question_start cannot be negative")
    if cast(int, normalized["question_count"]) <= 0:
        raise TrialQAGateError("primary_evaluation_scope.question_count must be positive")
    if cast(int, normalized["repeat_count"]) <= 0:
        raise TrialQAGateError("primary_evaluation_scope.repeat_count must be positive")
    expected_tasks = (
        cast(int, normalized["question_count"]) * cast(int, normalized["repeat_count"]) * 2
    )
    if normalized["task_count"] != expected_tasks:
        raise TrialQAGateError(
            "primary_evaluation_scope.task_count must equal question_count * repeat_count * 2"
        )
    return normalized


def _quality_bounds(
    pair_scores: Sequence[tuple[str, float, float]],
    *,
    mode: str,
) -> JsonObject:
    baseline_only_wins = 0
    treatment_only_wins = 0
    deltas: list[float] = []
    question_deltas: dict[str, list[float]] = {}
    for pair_id, baseline_score, treatment_score in pair_scores:
        if baseline_score not in {0.0, 1.0} or treatment_score not in {0.0, 1.0}:
            raise TrialQAGateError("quality bounds require paired binary scores")
        delta = treatment_score - baseline_score
        deltas.append(delta)
        question_deltas.setdefault(_question_key(pair_id), []).append(delta)
        if delta == -1.0:
            baseline_only_wins += 1
        elif delta == 1.0:
            treatment_only_wins += 1

    pair_count = len(deltas)
    question_count = len(question_deltas)
    point_delta = statistics.fmean(deltas)
    has_repeats = question_count < pair_count
    if has_repeats and question_count >= 4:
        cluster_means = [statistics.fmean(values) for values in question_deltas.values()]
        point_delta = statistics.fmean(cluster_means)
        standard_error = statistics.stdev(cluster_means) / math.sqrt(question_count)
        degrees_of_freedom = question_count - 1
        z_value = statistics.NormalDist().inv_cdf(QUALITY_CONFIDENCE_LEVEL)
        critical_value = z_value * math.sqrt(degrees_of_freedom / (degrees_of_freedom - 2))
        radius = critical_value * standard_error
        lower_bound = max(-1.0, point_delta - radius)
        upper_bound = min(1.0, point_delta + radius)
        method = "question-clustered-normal-small-sample-corrected"
        bound_details: JsonObject = {
            "standard_error": standard_error,
            "critical_value": critical_value,
            "degrees_of_freedom": degrees_of_freedom,
        }
    elif has_repeats:
        lower_bound = -1.0
        upper_bound = 1.0
        method = "question-clustered-insufficient-clusters"
        bound_details = {
            "standard_error": None,
            "critical_value": None,
            "degrees_of_freedom": max(question_count - 1, 0),
        }
    else:
        component_alpha = (1.0 - QUALITY_CONFIDENCE_LEVEL) / 2.0
        treatment_only_lower = _clopper_pearson_lower(
            treatment_only_wins, pair_count, component_alpha
        )
        treatment_only_upper = _clopper_pearson_upper(
            treatment_only_wins, pair_count, component_alpha
        )
        baseline_only_lower = _clopper_pearson_lower(
            baseline_only_wins, pair_count, component_alpha
        )
        baseline_only_upper = _clopper_pearson_upper(
            baseline_only_wins, pair_count, component_alpha
        )
        lower_bound = max(-1.0, treatment_only_lower - baseline_only_upper)
        upper_bound = min(1.0, treatment_only_upper - baseline_only_lower)
        method = "paired-binary-bonferroni-clopper-pearson"
        bound_details = {
            "component_alpha": component_alpha,
            "treatment_only_probability": {
                "lower": treatment_only_lower,
                "upper": treatment_only_upper,
            },
            "baseline_only_probability": {
                "lower": baseline_only_lower,
                "upper": baseline_only_upper,
            },
        }

    decision_bound = lower_bound if mode == CONFIRMATORY_QUALITY_MODE else upper_bound
    return {
        "available": True,
        "mode": mode,
        "method": method,
        "confidence_level": QUALITY_CONFIDENCE_LEVEL,
        "noninferiority_margin": QUALITY_DELTA_MIN,
        "pair_count": pair_count,
        "question_cluster_count": question_count,
        "baseline_only_wins": baseline_only_wins,
        "treatment_only_wins": treatment_only_wins,
        "concordant_pairs": pair_count - baseline_only_wins - treatment_only_wins,
        "point_delta": point_delta,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "decision_bound": decision_bound,
        "decision_operator": ">=",
        "decision_threshold": QUALITY_DELTA_MIN,
        "details": bound_details,
    }


def _criterion(
    name: str,
    value: int | float | None,
    operator: str,
    threshold: int | float,
) -> JsonObject:
    if value is None:
        passed = False
    elif operator == ">=":
        passed = value >= threshold
    elif operator == ">":
        passed = value > threshold
    elif operator == "<":
        passed = value < threshold
    elif operator == "<=":
        passed = value <= threshold
    else:  # pragma: no cover - closed internal vocabulary.
        raise AssertionError(f"unsupported gate operator: {operator}")
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def build_gate_report(
    metrics: Sequence[TaskMetrics],
    *,
    gate: GateKind,
    manifest_id: str = "synthetic",
    manifest_kind: str = "full",
    manifest_task_count: int | None = None,
    candidate: Mapping[str, object] | None = None,
    source_attestation_current: bool = True,
    control_design: str = "concurrent-paired",
    baseline_manifest_id: str | None = None,
    primary_evaluation_scope: Mapping[str, object] | None = None,
    manifest_performance_eligible: bool | None = None,
    scope_attestation: Mapping[str, object] | None = None,
) -> JsonObject:
    """Build a deterministic report and apply the named promotion policy."""

    if gate not in {"plumbing", "operational", "promotion"}:
        raise TrialQAGateError(f"unknown gate: {gate!r}")
    pairs: dict[str, dict[str, TaskMetrics]] = {}
    for item in metrics:
        if item.condition not in {"baseline", "treatment"}:
            raise TrialQAGateError("gates only accept baseline/treatment tasks")
        conditions = pairs.setdefault(item.pair_id, {})
        if item.condition in conditions:
            raise TrialQAGateError(f"duplicate {item.condition} task for {item.pair_id}")
        conditions[item.condition] = item
    if not pairs:
        raise TrialQAGateError("gate scope contains no pairs")
    incomplete = [
        pair_id
        for pair_id, conditions in pairs.items()
        if set(conditions) != {"baseline", "treatment"}
    ]
    if incomplete:
        raise TrialQAGateError(f"gate scope contains an incomplete pair: {incomplete[0]}")

    if manifest_performance_eligible is not None and not isinstance(
        manifest_performance_eligible, bool
    ):
        raise TrialQAGateError("manifest performance_eligible must be boolean")
    if primary_evaluation_scope is not None:
        if manifest_performance_eligible is not True:
            raise TrialQAGateError(
                "primary_evaluation_scope requires manifest performance_eligible=true"
            )
        declared_scope = _validate_primary_evaluation_scope(primary_evaluation_scope)
        declared_task_count = cast(int, declared_scope["task_count"])
        if manifest_kind != "full" or manifest_task_count != declared_task_count:
            raise TrialQAGateError(
                "primary_evaluation_scope disagrees with manifest kind or task count"
            )
        if len(metrics) > declared_task_count:
            raise TrialQAGateError("gate scope exceeds primary_evaluation_scope.task_count")
        full_protocol = len(metrics) == declared_task_count
    else:
        declared_scope = {
            "question_start": 0,
            "question_count": dataset.SERGEI_TEST_COUNT,
            "repeat_count": demo.FULL_REPEATS,
            "task_count": dataset.SERGEI_TEST_COUNT * demo.FULL_REPEATS * 2,
        }
        full_protocol = manifest_performance_eligible is not False and (
            manifest_kind == "full"
            and manifest_task_count == len(metrics)
            and len(metrics) == cast(int, declared_scope["task_count"])
        )
    quality_mode = CONFIRMATORY_QUALITY_MODE if full_protocol else INTERIM_QUALITY_MODE
    ordered_pairs = sorted(pairs.items())
    cluster_counts: dict[str, int] = {}
    for pair_id, _conditions in ordered_pairs:
        question_key = _question_key(pair_id)
        cluster_counts[question_key] = cluster_counts.get(question_key, 0) + 1
    if primary_evaluation_scope is not None:
        expected_questions = cast(int, declared_scope["question_count"])
        expected_repeats = cast(int, declared_scope["repeat_count"])
        if len(cluster_counts) > expected_questions or any(
            count > expected_repeats for count in cluster_counts.values()
        ):
            raise TrialQAGateError(
                "gate scope falls outside primary evaluation question/repeat bounds"
            )
        if full_protocol and (
            len(cluster_counts) != expected_questions
            or set(cluster_counts.values()) != {expected_repeats}
        ):
            raise TrialQAGateError(
                "complete gate scope disagrees with primary evaluation question/repeat counts"
            )
    elif full_protocol:
        expected_questions = cast(int, declared_scope["question_count"])
        expected_repeats = cast(int, declared_scope["repeat_count"])
        if len(cluster_counts) != expected_questions or set(cluster_counts.values()) != {
            expected_repeats
        }:
            raise TrialQAGateError(
                "complete gate scope disagrees with primary evaluation question/repeat counts"
            )
    raw_token_deltas: list[int] = []
    token_deltas: list[int] = []
    call_deltas: list[int] = []
    pair_rows: list[JsonObject] = []
    token_cheaper = {"baseline": 0, "treatment": 0, "equal": 0}
    call_cheaper = {"baseline": 0, "treatment": 0, "equal": 0}
    for pair_id, conditions in ordered_pairs:
        baseline = conditions["baseline"]
        treatment = conditions["treatment"]
        raw_token_delta = treatment.total_tokens - baseline.total_tokens
        token_delta = (
            treatment.total_tokens + treatment.retry_token_sensitivity - baseline.total_tokens
        )
        call_delta = treatment.operational_calls - baseline.operational_calls
        raw_token_deltas.append(raw_token_delta)
        token_deltas.append(token_delta)
        call_deltas.append(call_delta)
        token_cheaper[
            "treatment" if token_delta < 0 else "baseline" if token_delta > 0 else "equal"
        ] += 1
        call_cheaper[
            "treatment" if call_delta < 0 else "baseline" if call_delta > 0 else "equal"
        ] += 1
        pair_rows.append(
            {
                "pair_id": pair_id,
                "baseline_task_id": baseline.task_id,
                "treatment_task_id": treatment.task_id,
                "token_delta": token_delta,
                "raw_token_delta": raw_token_delta,
                "baseline_retry_token_sensitivity": baseline.retry_token_sensitivity,
                "treatment_retry_token_sensitivity": treatment.retry_token_sensitivity,
                "operational_call_delta": call_delta,
                "score_delta": (
                    treatment.score - baseline.score
                    if treatment.score is not None and baseline.score is not None
                    else None
                ),
            }
        )

    condition_metrics: JsonObject = {}
    for condition in ("baseline", "treatment"):
        items = [conditions[condition] for _pair_id, conditions in ordered_pairs]
        condition_metrics[condition] = {
            "tasks": len(items),
            "total_tokens": sum(item.total_tokens for item in items),
            "retry_token_sensitivity": sum(item.retry_token_sensitivity for item in items),
            "conservative_efficiency_tokens": sum(
                item.total_tokens
                + (item.retry_token_sensitivity if condition == "treatment" else 0)
                for item in items
            ),
            "operational_calls": sum(item.operational_calls for item in items),
            "successful_operational_calls": sum(
                item.successful_operational_calls for item in items
            ),
            "skill_load_calls": sum(item.skill_load_calls for item in items),
            "successful_skill_load_calls": sum(item.successful_skill_load_calls for item in items),
            "model_turns": sum(item.model_turns for item in items),
            "physical_attempts": sum(item.physical_attempts for item in items),
            "null_eof_retries": sum(item.null_eof_retries for item in items),
            "retry_usage_charges": sum(item.retry_usage_charges for item in items),
            "unpriced_null_eof_retries": sum(item.unpriced_null_eof_retries for item in items),
            "mean_score": (
                statistics.fmean(cast(float, item.score) for item in items)
                if all(item.score is not None for item in items)
                else None
            ),
            "terminal_tasks": sum(item.terminal for item in items),
            "terminal_rate": sum(item.terminal for item in items) / len(items),
        }

    baseline_summary = cast(JsonObject, condition_metrics["baseline"])
    treatment_summary = cast(JsonObject, condition_metrics["treatment"])
    raw_baseline_tokens = cast(int, baseline_summary["total_tokens"])
    raw_treatment_tokens = cast(int, treatment_summary["total_tokens"])
    baseline_tokens = cast(int, baseline_summary["conservative_efficiency_tokens"])
    treatment_tokens = cast(int, treatment_summary["conservative_efficiency_tokens"])
    baseline_calls = cast(int, baseline_summary["operational_calls"])
    treatment_calls = cast(int, treatment_summary["operational_calls"])
    token_reduction = _reduction_fraction(baseline_tokens, treatment_tokens)
    raw_token_reduction = _reduction_fraction(raw_baseline_tokens, raw_treatment_tokens)
    call_reduction = _reduction_fraction(baseline_calls, treatment_calls)
    quality_complete = all(item.score is not None and item.score_bound for item in metrics)
    if quality_complete:
        quality = _quality_bounds(
            [
                (
                    pair_id,
                    cast(float, conditions["baseline"].score),
                    cast(float, conditions["treatment"].score),
                )
                for pair_id, conditions in ordered_pairs
            ],
            mode=quality_mode,
        )
        quality_delta: float | None = cast(float, quality["point_delta"])
    else:
        quality_delta = None
        quality = {
            "available": False,
            "mode": quality_mode,
            "method": None,
            "confidence_level": QUALITY_CONFIDENCE_LEVEL,
            "noninferiority_margin": QUALITY_DELTA_MIN,
            "pair_count": len(pairs),
            "question_cluster_count": len(
                {_question_key(pair_id) for pair_id, _conditions in ordered_pairs}
            ),
            "baseline_only_wins": None,
            "treatment_only_wins": None,
            "concordant_pairs": None,
            "point_delta": None,
            "lower_bound": None,
            "upper_bound": None,
            "decision_bound": None,
            "decision_operator": ">=",
            "decision_threshold": QUALITY_DELTA_MIN,
            "details": None,
        }
    terminal_rate_delta = cast(float, treatment_summary["terminal_rate"]) - cast(
        float, baseline_summary["terminal_rate"]
    )
    early_primary_checkpoint = primary_evaluation_scope is not None and len(cluster_counts) < cast(
        int, declared_scope["question_count"]
    )
    token_optimistic_lower = _optimistic_delta_lower_bound(token_deltas)
    call_optimistic_lower = _optimistic_delta_lower_bound(call_deltas)
    efficiency_futile = (
        token_optimistic_lower is not None
        and call_optimistic_lower is not None
        and token_optimistic_lower > 0
        and call_optimistic_lower > 0
    )
    futility_criteria = [
        {
            "name": "at_least_one_efficiency_signal_remains_plausible",
            "value": not efficiency_futile,
            "operator": "is",
            "threshold": True,
            "passed": not efficiency_futile,
        }
    ]
    early_aggregate_benefit_criteria = [
        _criterion(
            "early_aggregate_token_reduction_positive",
            token_reduction,
            ">",
            0.0,
        ),
        _criterion(
            "early_aggregate_operational_call_reduction_positive",
            call_reduction,
            ">",
            0.0,
        ),
    ]
    early_paired_majority_benefit_criteria = [
        _criterion(
            "early_treatment_token_cheaper_pairs_majority",
            token_cheaper["treatment"],
            ">",
            token_cheaper["baseline"],
        ),
        _criterion(
            "early_treatment_call_cheaper_pairs_majority",
            call_cheaper["treatment"],
            ">",
            call_cheaper["baseline"],
        ),
    ]
    early_visible_benefit_criteria = [
        *early_aggregate_benefit_criteria,
        *early_paired_majority_benefit_criteria,
    ]
    timed_out_task_ids = [item.task_id for item in metrics if item.timed_out]
    timeout_by_condition = {
        condition: {
            "seconds": sorted(
                {item.generation_timeout_seconds for item in metrics if item.condition == condition}
            ),
            "policies": sorted(
                {item.generation_timeout_policy for item in metrics if item.condition == condition}
            ),
        }
        for condition in ("baseline", "treatment")
    }

    treatment_items = [item for item in metrics if item.condition == "treatment"]
    treatment_without_successful_skill_load = [
        item.task_id for item in treatment_items if item.successful_skill_load_calls < 1
    ]
    tasks_without_operational_call = [
        item.task_id for item in metrics if item.operational_calls < 1
    ]
    compliance_diagnostics = [
        {
            "name": "at_least_one_successful_skill_load_per_treatment",
            "value": not treatment_without_successful_skill_load,
            "operator": "is",
            "threshold": True,
            "passed": not treatment_without_successful_skill_load,
            "gating": False,
            "scope": "treatment",
            "task_count": len(treatment_items),
            "compliant_task_count": len(treatment_items)
            - len(treatment_without_successful_skill_load),
            "compliance_rate": 1.0
            - len(treatment_without_successful_skill_load) / len(treatment_items),
            "noncompliant_task_ids": treatment_without_successful_skill_load,
        },
        {
            "name": "at_least_one_operational_call_per_task",
            "value": not tasks_without_operational_call,
            "operator": "is",
            "threshold": True,
            "passed": not tasks_without_operational_call,
            "gating": False,
            "scope": "all",
            "task_count": len(metrics),
            "compliant_task_count": len(metrics) - len(tasks_without_operational_call),
            "compliance_rate": 1.0 - len(tasks_without_operational_call) / len(metrics),
            "noncompliant_task_ids": tasks_without_operational_call,
        },
    ]

    plumbing_criteria = [
        {
            "name": "source_attestation_current",
            "value": source_attestation_current,
            "operator": "is",
            "threshold": True,
            "passed": source_attestation_current,
        },
        {
            "name": "no_generation_timeouts",
            "value": not timed_out_task_ids,
            "operator": "is",
            "threshold": True,
            "passed": not timed_out_task_ids,
        },
        {
            "name": "no_unpriced_null_eof_retries",
            "value": all(item.unpriced_null_eof_retries == 0 for item in metrics),
            "operator": "is",
            "threshold": True,
            "passed": all(item.unpriced_null_eof_retries == 0 for item in metrics),
        },
        {
            "name": "no_null_eof_retries_in_performance_capture",
            "value": all(item.null_eof_retries == 0 for item in metrics),
            "operator": "is",
            "threshold": True,
            "passed": all(item.null_eof_retries == 0 for item in metrics),
        },
    ]
    terminal_rate_criteria = [
        _criterion(
            "treatment_terminal_rate_delta",
            terminal_rate_delta,
            "<=",
            0.05,
        )
    ]
    efficiency_criteria = [
        _criterion(
            "token_reduction_fraction",
            token_reduction,
            ">=",
            TOKEN_REDUCTION_MIN,
        ),
        _criterion(
            "median_paired_token_delta",
            statistics.median(token_deltas),
            "<",
            0,
        ),
        _criterion(
            "trimmed_mean_paired_token_delta",
            _trimmed_mean(token_deltas),
            "<",
            0,
        ),
        _criterion(
            "treatment_token_cheaper_pairs",
            token_cheaper["treatment"],
            ">",
            token_cheaper["baseline"],
        ),
        _criterion(
            "operational_call_reduction_fraction",
            call_reduction,
            ">=",
            OPERATIONAL_CALL_REDUCTION_MIN,
        ),
        _criterion(
            "trimmed_mean_paired_operational_call_delta",
            _trimmed_mean(call_deltas),
            "<",
            0,
        ),
        _criterion(
            "treatment_call_cheaper_pairs",
            call_cheaper["treatment"],
            ">",
            call_cheaper["baseline"],
        ),
    ]
    operational_criteria = [*efficiency_criteria, *terminal_rate_criteria]
    quality_criterion_name = (
        "paired_quality_lower_bound"
        if quality_mode == CONFIRMATORY_QUALITY_MODE
        else "paired_quality_upper_bound"
    )
    quality_criteria = [
        _criterion(
            quality_criterion_name,
            cast(float | None, quality["decision_bound"]),
            ">=",
            QUALITY_DELTA_MIN,
        )
    ]
    plumbing_passed = all(cast(bool, item["passed"]) for item in plumbing_criteria)
    operational_passed = plumbing_passed and all(
        cast(bool, item["passed"]) for item in operational_criteria
    )
    terminal_rate_passed = all(cast(bool, item["passed"]) for item in terminal_rate_criteria)
    early_aggregate_benefit_passed = all(
        cast(bool, item["passed"]) for item in early_aggregate_benefit_criteria
    )
    early_paired_majority_benefit_passed = all(
        cast(bool, item["passed"]) for item in early_paired_majority_benefit_criteria
    )
    early_visible_benefit_passed = all(
        cast(bool, item["passed"]) for item in early_visible_benefit_criteria
    )
    checkpoint_operational_passed = (
        plumbing_passed
        and terminal_rate_passed
        and early_visible_benefit_passed
        and not efficiency_futile
        if early_primary_checkpoint
        else operational_passed
    )
    if gate == "plumbing":
        decision = "promote_to_operational" if plumbing_passed else "kill"
        evaluated_criteria = plumbing_criteria
    elif gate == "operational":
        decision = "promote_to_score" if checkpoint_operational_passed else "kill"
        evaluated_criteria = [
            *plumbing_criteria,
            *(
                [
                    *terminal_rate_criteria,
                    *early_visible_benefit_criteria,
                    *futility_criteria,
                ]
                if early_primary_checkpoint
                else operational_criteria
            ),
        ]
    elif not quality_complete:
        decision = "incomplete"
        evaluated_criteria = [
            *plumbing_criteria,
            *(
                [
                    *terminal_rate_criteria,
                    *early_visible_benefit_criteria,
                    *futility_criteria,
                ]
                if early_primary_checkpoint
                else operational_criteria
            ),
            *quality_criteria,
        ]
    else:
        promotion_passed = checkpoint_operational_passed and all(
            cast(bool, item["passed"]) for item in quality_criteria
        )
        decision = "promote_to_next_cohort" if promotion_passed else "kill"
        evaluated_criteria = [
            *plumbing_criteria,
            *(
                [
                    *terminal_rate_criteria,
                    *early_visible_benefit_criteria,
                    *futility_criteria,
                ]
                if early_primary_checkpoint
                else operational_criteria
            ),
            *quality_criteria,
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "name": POLICY_NAME,
            "token_reduction_min": TOKEN_REDUCTION_MIN,
            "operational_call_reduction_min": OPERATIONAL_CALL_REDUCTION_MIN,
            "quality_delta_min": QUALITY_DELTA_MIN,
            "quality_confidence_level": QUALITY_CONFIDENCE_LEVEL,
            "futility_confidence_level": FUTILITY_CONFIDENCE_LEVEL,
            "early_checkpoint_requires_visible_aggregate_benefit": True,
            "early_checkpoint_requires_visible_paired_majority_benefit": True,
            "interim_quality_mode": INTERIM_QUALITY_MODE,
            "confirmatory_quality_mode": CONFIRMATORY_QUALITY_MODE,
            "retry_token_policy": "treatment-raw-plus-sensitivity-vs-baseline-raw",
            "performance_retry_policy": "zero-null-eof-retries",
            "analysis_population": "intention-to-treat",
            "completed_draw_policy": "terminal-no-retry-or-replacement",
            "empty_answer_policy": "score-zero",
            "operational_call_compliance_policy": "diagnostic-only",
            "skill_load_compliance_policy": "diagnostic-only",
        },
        "manifest_id": manifest_id,
        "manifest_kind": manifest_kind,
        "candidate": dict(candidate) if candidate is not None else None,
        "gate": gate,
        "decision": decision,
        "control_design": control_design,
        "baseline_manifest_id": baseline_manifest_id,
        "performance_eligible": bool(
            full_protocol and gate == "promotion" and decision == "promote_to_next_cohort"
        ),
        "source_attestation_current": source_attestation_current,
        "generation_timeout": {
            "conditions": timeout_by_condition,
            "timed_out_task_ids": timed_out_task_ids,
        },
        "scope": {
            "pair_count": len(pairs),
            "task_count": len(metrics),
            "task_ids": [item.task_id for item in metrics],
            "complete_pairs": True,
            "primary_evaluation_scope": declared_scope,
            "confirmatory_scope_complete": full_protocol,
            "manifest_performance_eligible": manifest_performance_eligible,
            "selection_attestation": (
                dict(scope_attestation) if scope_attestation is not None else None
            ),
        },
        "conditions": condition_metrics,
        "compliance": {
            "gating": False,
            "diagnostics": compliance_diagnostics,
        },
        "quality": quality,
        "efficiency_checkpoint": {
            "mode": (
                "early_primary_futility" if early_primary_checkpoint else "full_v3_enforcement"
            ),
            "final_thresholds_enforced": not early_primary_checkpoint,
            "visible_aggregate_benefit_required": early_primary_checkpoint,
            "visible_aggregate_benefit_passed": early_aggregate_benefit_passed
            if early_primary_checkpoint
            else None,
            "visible_paired_majority_benefit_required": early_primary_checkpoint,
            "visible_paired_majority_benefit_passed": early_paired_majority_benefit_passed
            if early_primary_checkpoint
            else None,
            "visible_benefit_passed": early_visible_benefit_passed
            if early_primary_checkpoint
            else None,
            "futile": efficiency_futile,
            "token_delta_optimistic_lower_bound": token_optimistic_lower,
            "operational_call_delta_optimistic_lower_bound": call_optimistic_lower,
            "diagnostic_final_criteria": (
                efficiency_criteria if early_primary_checkpoint else None
            ),
        },
        "paired": {
            "token_deltas": token_deltas,
            "raw_token_deltas": raw_token_deltas,
            "operational_call_deltas": call_deltas,
            "median_token_delta": statistics.median(token_deltas),
            "trimmed_mean_token_delta": _trimmed_mean(token_deltas),
            "median_operational_call_delta": statistics.median(call_deltas),
            "trimmed_mean_operational_call_delta": _trimmed_mean(call_deltas),
            "token_cheaper_pairs": token_cheaper,
            "operational_call_cheaper_pairs": call_cheaper,
            "pairs": pair_rows,
        },
        "benefit": {
            "token_reduction_fraction": token_reduction,
            "raw_token_reduction_fraction": raw_token_reduction,
            "treatment_retry_token_sensitivity": cast(
                int, treatment_summary["retry_token_sensitivity"]
            ),
            "baseline_retry_token_sensitivity_ignored_for_benefit": cast(
                int, baseline_summary["retry_token_sensitivity"]
            ),
            "operational_call_reduction_fraction": call_reduction,
            "mean_score_delta": quality_delta,
            "terminal_rate_delta": terminal_rate_delta,
        },
        "criteria": evaluated_criteria,
    }


def _generation_path(capture: Path, task: Mapping[str, object]) -> Path:
    arm = task.get("arm")
    if arm not in {"baseline", "treatment"}:
        raise TrialQAGateError(f"task has an invalid arm: {task.get('task_id')}")
    return (
        capture
        / "trialqa-local"
        / cast(str, task["pair_id"])
        / "arms"
        / arm
        / "outputs"
        / "generation.json"
    )


def _task_generation_timeout(
    ledger: demo.ResumableLedger,
    task_id: str,
) -> tuple[int, str]:
    values: set[tuple[int, str]] = set()
    for record in ledger.records():
        if record.get("task_id") != task_id or record.get("event") != "generation_started":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise TrialQAGateError(f"generation timeout payload is invalid: {task_id}")
        timeout = payload.get("wall_clock_timeout_seconds")
        policy = payload.get("timeout_policy")
        if timeout is None and policy is None:
            timeout = batch.DEFAULT_GENERATION_TIMEOUT_SECONDS
            policy = batch.DEFAULT_GENERATION_TIMEOUT_POLICY
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not isinstance(policy, str):
            raise TrialQAGateError(f"generation timeout policy is invalid: {task_id}")
        values.add((timeout, policy))
    if len(values) != 1:
        raise TrialQAGateError(f"generation timeout policy is ambiguous: {task_id}")
    return next(iter(values))


def _task_timed_out(ledger: demo.ResumableLedger, task_id: str) -> bool:
    for record in ledger.records():
        if record.get("task_id") != task_id or record.get("event") != "failed":
            continue
        payload = record.get("payload")
        if isinstance(payload, dict) and payload.get("timed_out") is True:
            return True
    return False


def _load_terminal_task_metrics(
    ledger: demo.ResumableLedger,
    capture: Path,
    task: Mapping[str, object],
) -> TaskMetrics | None:
    task_id = cast(str, task["task_id"])
    completion = ledger.event_record(task_id, "completed")
    completion_payload = completion.get("payload")
    if not isinstance(completion_payload, dict):
        raise TrialQAGateError(f"completed ledger payload is invalid for {task_id}")
    if completion_payload.get("terminal_error") is not True:
        return None
    failed_sha256 = completion_payload.get("failed_record_sha256")
    failures = [
        record
        for record in ledger.records()
        if record.get("task_id") == task_id
        and record.get("event") == "failed"
        and record.get("record_sha256") == failed_sha256
    ]
    if len(failures) != 1:
        raise TrialQAGateError(f"terminal task lacks one failed record: {task_id}")
    failure_payload = failures[0].get("payload")
    if (
        not isinstance(failure_payload, dict)
        or failure_payload.get("stage") != "generation"
        or failure_payload.get("terminal") is not True
    ):
        raise TrialQAGateError(f"terminal task has invalid failure provenance: {task_id}")
    timeout_seconds, timeout_policy = _task_generation_timeout(ledger, task_id)

    result_value = completion_payload.get("failure_result_path")
    quarantine_value = completion_payload.get("quarantined_attempt")
    if not isinstance(result_value, str) or not isinstance(quarantine_value, str):
        raise TrialQAGateError(f"terminal task lacks bound artifacts: {task_id}")
    result_path = Path(result_value).absolute()
    quarantine = Path(quarantine_value).absolute()
    if not demo._is_relative_to(result_path, capture) or not demo._is_relative_to(
        quarantine, capture
    ):
        raise TrialQAGateError(f"terminal artifacts escape the capture: {task_id}")
    if completion_payload.get("failure_result_sha256") != demo._sha256_file(result_path):
        raise TrialQAGateError(f"terminal result hash differs from ledger: {task_id}")
    result = demo.load_trial_result(result_path)
    if (
        result.status != "error"
        or result.error_stage != "generation"
        or result.manifest_id != ledger.manifest_id
        or result.task_id != task_id
        or result.pair_id != task.get("pair_id")
        or result.condition != task.get("condition")
    ):
        raise TrialQAGateError(f"terminal result differs from manifest task: {task_id}")
    if result.score != 0.0:
        raise TrialQAGateError(f"terminal result is not a zero-score ITT outcome: {task_id}")

    codex_events_path = quarantine / "outputs" / "switchyard-codex.stdout.log"
    artifact_sha256 = failure_payload.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict) or artifact_sha256.get(
        "switchyard-codex.stdout.log"
    ) != demo._sha256_file(codex_events_path):
        raise TrialQAGateError(f"terminal Codex log differs from ledger: {task_id}")
    session_proof = failure_payload.get("session_proof")
    if not isinstance(session_proof, dict):
        raise TrialQAGateError(f"terminal task lacks session proof: {task_id}")
    model_turns = session_proof.get("total_requests")
    if not isinstance(model_turns, int) or isinstance(model_turns, bool):
        raise TrialQAGateError(f"terminal task has invalid request stats: {task_id}")
    session_path_value = session_proof.get("session_path")
    if not isinstance(session_path_value, str):
        raise TrialQAGateError(f"terminal task lacks a session path: {task_id}")
    session_path = Path(session_path_value).absolute()
    if not demo._is_relative_to(session_path, capture):
        raise TrialQAGateError(f"terminal session escapes the capture: {task_id}")
    try:
        stats = demo._read_json_object(
            session_path / "stats.json", "terminal Switchyard session stats"
        )
        transport = demo._validate_openai_transport_stats(
            stats,
            total_requests=model_turns,
            require_priced=False,
        )
    except demo.TrialQADemoError as exc:
        raise TrialQAGateError(
            f"terminal task has invalid transport accounting: {task_id}"
        ) from exc
    if session_proof.get("openai_transport") != transport:
        raise TrialQAGateError(f"terminal transport proof differs from stats: {task_id}")
    sensitivity = cast(Mapping[str, object], transport["retry_token_sensitivity"])
    tool_metrics = demo.codex_tool_metrics(codex_events_path)
    return TaskMetrics(
        task_id=task_id,
        pair_id=result.pair_id,
        condition=result.condition,
        total_tokens=result.total_tokens,
        operational_calls=cast(int, tool_metrics["operational_calls"]),
        successful_operational_calls=cast(int, tool_metrics["successful_operational_calls"]),
        skill_load_calls=cast(int, tool_metrics["skill_load_calls"]),
        successful_skill_load_calls=cast(int, tool_metrics["successful_skill_load_calls"]),
        model_turns=model_turns,
        physical_attempts=cast(int, transport["physical_attempts"]),
        null_eof_retries=cast(int, transport["null_eof_retries"]),
        retry_usage_charges=cast(int, transport["retry_usage_charges"]),
        unpriced_null_eof_retries=cast(int, transport["unpriced_null_eof_retries"]),
        retry_token_sensitivity=cast(int, sensitivity["total"]),
        score=result.score,
        terminal=True,
        score_bound=True,
        timed_out=_task_timed_out(ledger, task_id),
        generation_timeout_seconds=timeout_seconds,
        generation_timeout_policy=timeout_policy,
    )


def load_task_metrics(
    manifest: Mapping[str, object],
    capture: Path,
    tasks: Sequence[Mapping[str, object]],
) -> tuple[TaskMetrics, ...]:
    """Load and re-attest complete generations for a fixed A/B scope."""

    ledger = demo.ResumableLedger(capture / "ledger.jsonl", manifest)
    states = ledger.states()
    loaded: list[TaskMetrics] = []
    for task in tasks:
        task_id = cast(str, task["task_id"])
        if states.get(task_id) not in {
            "generation_completed",
            "scored",
            "evidence_imported",
            "completed",
            "failed",
            "score_retry_started",
        }:
            raise TrialQAGateError(
                f"gate scope lacks a durable generation for {task_id}: {states.get(task_id)!r}"
            )
        if states.get(task_id) == "completed":
            terminal = _load_terminal_task_metrics(ledger, capture, task)
            if terminal is not None:
                loaded.append(terminal)
                continue
        try:
            generation = batch._load_completed_generation(ledger, task_id)
        except (RuntimeError, demo.TrialQADemoError) as exc:
            raise TrialQAGateError(f"gate scope lacks a reusable generation for {task_id}") from exc
        if any(
            (
                generation.pair_id != task.get("pair_id"),
                generation.row_id != task.get("row_id"),
                generation.condition != task.get("condition"),
                generation.repeat_index != task.get("repeat_index"),
            )
        ):
            raise TrialQAGateError(f"generation differs from manifest task {task_id}")
        expected_path = _generation_path(capture, task)
        if generation.generation_path.resolve() != expected_path.resolve():
            raise TrialQAGateError(f"generation is outside its task capture path: {task_id}")
        demo.validate_generation_for_import(generation, project_dir=capture)
        timeout_seconds, timeout_policy = _task_generation_timeout(ledger, task_id)
        tool_metrics = demo.codex_tool_metrics(generation.codex_events_path)
        total_tokens = generation.stats.get("total_tokens")
        if not isinstance(total_tokens, dict) or not isinstance(total_tokens.get("total"), int):
            raise TrialQAGateError(f"generation has invalid token stats: {task_id}")
        model_turns = generation.stats.get("total_requests")
        if not isinstance(model_turns, int) or isinstance(model_turns, bool):
            raise TrialQAGateError(f"generation has invalid request stats: {task_id}")
        try:
            transport = demo._validate_openai_transport_stats(
                generation.stats,
                total_requests=model_turns,
                require_priced=True,
            )
        except demo.TrialQADemoError as exc:
            raise TrialQAGateError(
                f"generation has invalid transport accounting: {task_id}"
            ) from exc
        sensitivity = cast(Mapping[str, object], transport["retry_token_sensitivity"])
        result_path = capture / "results" / f"{task_id}.json"
        score: float | None = None
        score_bound = False
        if result_path.is_file() and not result_path.is_symlink():
            result = demo.load_trial_result(result_path)
            if (
                result.status != "scored"
                or result.manifest_id != manifest.get("manifest_id")
                or result.task_id != task_id
            ):
                raise TrialQAGateError(f"score result differs from task {task_id}")
            if states.get(task_id) == "completed":
                completion = ledger.event_record(task_id, "completed")
                completion_payload = completion.get("payload")
                if not isinstance(completion_payload, dict):
                    raise TrialQAGateError(f"completed payload is invalid: {task_id}")
                if "result_sha256" in completion_payload:
                    if (
                        completion_payload.get("result_path") != str(result_path)
                        or completion_payload.get("result_sha256") != demo._sha256_file(result_path)
                        or completion_payload.get("score") != result.score
                        or completion_payload.get("evidence_id") != result.evidence_id
                    ):
                        raise TrialQAGateError(
                            f"score result has an invalid completion binding: {task_id}"
                        )
                    score_bound = True
            score = result.score
        loaded.append(
            TaskMetrics(
                task_id=task_id,
                pair_id=generation.pair_id,
                condition=generation.condition,
                total_tokens=cast(int, total_tokens["total"]),
                operational_calls=cast(int, tool_metrics["operational_calls"]),
                successful_operational_calls=cast(
                    int, tool_metrics["successful_operational_calls"]
                ),
                skill_load_calls=cast(int, tool_metrics["skill_load_calls"]),
                successful_skill_load_calls=cast(int, tool_metrics["successful_skill_load_calls"]),
                model_turns=model_turns,
                physical_attempts=cast(int, transport["physical_attempts"]),
                null_eof_retries=cast(int, transport["null_eof_retries"]),
                retry_usage_charges=cast(int, transport["retry_usage_charges"]),
                unpriced_null_eof_retries=cast(int, transport["unpriced_null_eof_retries"]),
                retry_token_sensitivity=cast(int, sensitivity["total"]),
                score=score,
                terminal=False,
                score_bound=score_bound,
                timed_out=_task_timed_out(ledger, task_id),
                generation_timeout_seconds=timeout_seconds,
                generation_timeout_policy=timeout_policy,
            )
        )
    return tuple(loaded)


def _source_attestation_current(manifest: Mapping[str, object]) -> bool:
    implementation = manifest.get("implementation")
    return (
        isinstance(implementation, dict)
        and implementation.get("source_sha256") == demo._execution_source_sha256()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--baseline-capture", type=Path)
    parser.add_argument("--gate", choices=("plumbing", "operational", "promotion"), required=True)
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--repeat-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = demo._read_json_object(args.manifest, "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    manifest_kind = cast(str, manifest.get("kind"))
    tasks = [dict(item) for item in cast(list[dict[str, object]], manifest["tasks"])]
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise TrialQAGateError("manifest protocol must be an object")
    raw_primary_scope = protocol.get("primary_evaluation_scope")
    if raw_primary_scope is not None and not isinstance(raw_primary_scope, dict):
        raise TrialQAGateError("manifest protocol primary_evaluation_scope must be an object")
    raw_performance_eligible = protocol.get("performance_eligible")
    if not isinstance(raw_performance_eligible, bool):
        raise TrialQAGateError("manifest protocol performance_eligible must be boolean")
    task_scope = batch._build_manifest_task_scope(
        manifest,
        tasks,
        limit=None,
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        condition="both",
    )
    scoped = list(task_scope.tasks)
    scope_attestation = task_scope.metadata(manifest["manifest_id"])
    control_design = "concurrent-paired"
    baseline_manifest_id: str | None = None
    if manifest_kind == "development":
        if args.baseline_manifest is None or args.baseline_capture is None:
            raise TrialQAGateError(
                "development gates require --baseline-manifest and --baseline-capture"
            )
        baseline_manifest = demo._read_json_object(
            args.baseline_manifest, "historical donor manifest"
        )
        demo.validate_manifest_pairing(baseline_manifest)
        if baseline_manifest.get("kind") != "donor":
            raise TrialQAGateError("development baseline must be a donor manifest")
        baseline_tasks = [
            dict(item) for item in cast(list[dict[str, object]], baseline_manifest["tasks"])
        ]
        baseline_scope = batch._build_task_scope(
            baseline_tasks,
            limit=None,
            question_start=args.question_start,
            question_limit=args.question_limit,
            repeat_limit=args.repeat_limit,
            condition="both",
        )
        baseline_scoped = list(baseline_scope.tasks)
        development_by_pair = {str(task["pair_id"]): task for task in scoped}
        baseline_by_pair = {str(task["pair_id"]): task for task in baseline_scoped}
        if set(development_by_pair) != set(baseline_by_pair):
            raise TrialQAGateError("development and donor scopes have different pair IDs")
        for pair_id, development_task in development_by_pair.items():
            baseline_task = baseline_by_pair[pair_id]
            for field in (
                "pair_id",
                "row_id",
                "dataset_row_index",
                "question_group_key",
                "repeat_index",
                "n_repeats",
            ):
                if development_task.get(field) != baseline_task.get(field):
                    raise TrialQAGateError(f"development baseline differs at {field}: {pair_id}")
        treatment_metrics = load_task_metrics(manifest, args.capture.absolute(), scoped)
        donor_metrics = load_task_metrics(
            baseline_manifest,
            args.baseline_capture.absolute(),
            baseline_scoped,
        )
        metrics = (
            tuple(replace(item, condition="baseline") for item in donor_metrics) + treatment_metrics
        )
        control_design = "historical-cached-donor"
        baseline_manifest_id = cast(str, baseline_manifest["manifest_id"])
    else:
        if args.baseline_manifest is not None or args.baseline_capture is not None:
            raise TrialQAGateError(
                "historical baseline arguments are valid only for development manifests"
            )
        metrics = load_task_metrics(manifest, args.capture.absolute(), scoped)
    report = build_gate_report(
        metrics,
        gate=cast(GateKind, args.gate),
        manifest_id=cast(str, manifest["manifest_id"]),
        manifest_kind=manifest_kind,
        manifest_task_count=len(tasks),
        candidate=cast(Mapping[str, object] | None, manifest.get("candidate")),
        source_attestation_current=_source_attestation_current(manifest),
        control_design=control_design,
        baseline_manifest_id=baseline_manifest_id,
        primary_evaluation_scope=cast(Mapping[str, object] | None, raw_primary_scope),
        manifest_performance_eligible=raw_performance_eligible,
        scope_attestation=scope_attestation,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report, exclusive=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
