# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

COMPARATOR = Path(__file__).parents[1] / "benchmark" / "nemo_gym" / "compare.py"
compare = runpy.run_path(str(COMPARATOR))["compare"]
RolloutKey = tuple[int, int]


def _model(calls: int, tokens: int, latency_ms: float) -> dict[str, Any]:
    return {
        "calls": calls,
        "errors": 0,
        "total_tokens": tokens,
        "model_call_latency": {"avg_ms": latency_ms},
    }


def _write_run(
    root: Path,
    rewards: dict[RolloutKey, float | None],
    *,
    classifier_tokens: int,
    input_suffix: str = "",
    provenance: str = "same-build",
    fail_open_reason: str | None = None,
) -> None:
    root.mkdir()
    rollouts: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for (task, rollout), reward in rewards.items():
        key = {"_ng_task_index": task, "_ng_rollout_index": rollout}
        inputs.append({**key, "prompt": f"task-{task}{input_suffix}"})
        if reward is None:
            continue
        rollouts.append(
            {
                **key,
                "reward": reward,
                "ng_model_call_capture": {
                    "metrics": {
                        "tokens_total": 10 + task + rollout,
                        "latency_total_ms": 20.0 + task + rollout,
                    }
                },
            }
        )
    answer_tokens = sum(
        10 + task + rollout for (task, rollout), reward in rewards.items() if reward is not None
    )
    (root / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts), encoding="utf-8"
    )
    (root / "rollouts_materialized_inputs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inputs), encoding="utf-8"
    )
    (root / "switchyard-stats-raw.json").write_text(
        json.dumps(
            {
                "classifier": {
                    "total_tokens": {"total": classifier_tokens},
                    "models": {"judge": _model(1, classifier_tokens, 4.0)}
                    if classifier_tokens
                    else {},
                },
                "routing_overhead": {"avg_ms": 3.5},
                "models": {"model-a": _model(len(rollouts), answer_tokens, 8.0)},
            }
        ),
        encoding="utf-8",
    )
    (root / "switchyard-condition.json").write_text(
        json.dumps(
            {
                "route": root.name,
                "mode": "attached",
                "proxy_provenance": {
                    "gym_revision": "gym-revision",
                    "switchyard_revision": provenance,
                },
            }
        ),
        encoding="utf-8",
    )
    metric = ""
    if fail_open_reason is not None:
        metric = (
            "switchyard_classifier_fail_open_total"
            f'{{judge_model="classifier",reason="{fail_open_reason}"}} 1\n'
        )
    (root / "switchyard-metrics.prom").write_text(metric, encoding="utf-8")
    (root / "routes.toml").write_text("same deployment", encoding="utf-8")


def test_compares_paired_quality_and_usage(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    routed = tmp_path / "routed"
    _write_run(baseline, {(0, 0): 0.0, (0, 1): 1.0}, classifier_tokens=0)
    _write_run(
        routed,
        {(0, 0): 1.0, (0, 1): None},
        classifier_tokens=17,
        fail_open_reason="parse_error",
    )

    result = compare(baseline, routed)

    assert result["coverage"]["paired"] == 1
    assert result["coverage"]["unpaired"] == {"baseline": 1, "routed": 0}
    assert result["coverage"]["missing"] == {"baseline": 0, "routed": 1}
    assert result["routed_vs_baseline"] == {"wins": 1, "ties": 0, "losses": 0}
    assert result["baseline"]["paired"]["mean_reward"] == 0.0
    assert result["routed"]["paired"]["mean_reward"] == 1.0
    assert result["baseline"]["paired"]["answer_model_tokens"] == 10
    assert result["baseline"]["paired"]["endpoint_latency_mean_ms"] == 20.0
    assert result["routed"]["condition_totals"]["classifier_tokens"] == 17
    assert result["routed"]["condition_totals"]["classifier_fail_opens"] == {"parse_error": 1}
    answer = result["routed"]["condition_totals"]["answer_models"]["model-a"]
    assert answer["model_call_latency_mean_ms"] == 8.0


def test_rejects_different_materialized_inputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    routed = tmp_path / "routed"
    _write_run(baseline, {(0, 0): 1.0}, classifier_tokens=0)
    _write_run(routed, {(0, 0): 1.0}, classifier_tokens=5, input_suffix="-changed")

    with pytest.raises(ValueError, match="different materialized inputs"):
        compare(baseline, routed)


def test_rejects_different_switchyard_builds(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    routed = tmp_path / "routed"
    _write_run(baseline, {(0, 0): 1.0}, classifier_tokens=0, provenance="build-a")
    _write_run(routed, {(0, 0): 1.0}, classifier_tokens=5, provenance="build-b")

    with pytest.raises(ValueError, match="different or incomplete Switchyard provenance"):
        compare(baseline, routed)


def test_rejects_non_finite_rewards(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    routed = tmp_path / "routed"
    _write_run(baseline, {(0, 0): float("nan")}, classifier_tokens=0)
    _write_run(routed, {(0, 0): 1.0}, classifier_tokens=5)

    with pytest.raises(ValueError, match="non-finite reward"):
        compare(baseline, routed)
