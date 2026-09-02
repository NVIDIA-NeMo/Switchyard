# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare paired quality and usage for two attached NeMo Gym runs.

Per-rollout measurements include only shared task/repeat pairs. Switchyard proxy
statistics remain condition-wide.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RolloutKey = tuple[int, int]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> dict[RolloutKey, dict[str, Any]]:
    rows: dict[RolloutKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            try:
                task_index = row["_ng_task_index"]
                rollout_index = row["_ng_rollout_index"]
            except KeyError as error:
                raise ValueError(f"{path}:{line_number}: missing {error.args[0]}") from error
            if type(task_index) is not int or type(rollout_index) is not int:
                raise ValueError(f"{path}:{line_number}: rollout indices must be integers")
            key = (task_index, rollout_index)
            if key in rows:
                raise ValueError(f"duplicate rollout key {key} in {path}")
            rows[key] = row
    return rows


def _fail_opens(run_dir: Path) -> dict[str, int]:
    """Aggregate Switchyard's fixed classifier fail-open metric by reason."""
    path = run_dir / "switchyard-metrics.prom"
    if not path.exists():
        raise ValueError(f"missing {path}; capture /metrics before stopping the proxy")

    counts: dict[str, int] = {}
    prefix = "switchyard_classifier_fail_open_total{"
    marker = 'reason="'
    for line in path.read_text().splitlines():
        if not line.startswith(prefix) or marker not in line:
            continue
        reason = line.split(marker, 1)[1].split('"', 1)[0]
        value = int(float(line.rsplit(maxsplit=1)[1]))
        counts[reason] = counts.get(reason, 0) + value
    return counts


def _models(stats: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    return {
        name: {
            "calls": model["calls"],
            "errors": model["errors"],
            "tokens": model["total_tokens"],
            "model_call_latency_mean_ms": model["model_call_latency"]["avg_ms"],
        }
        for name, model in stats.items()
    }


def _reward(row: dict[str, Any], key: RolloutKey) -> float:
    try:
        value = float(row["reward"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"rollout {key} has no numeric reward") from error
    if not math.isfinite(value):
        raise ValueError(f"rollout {key} has a non-finite reward")
    return value


def _condition_summary(
    run_dir: Path,
    rows: dict[RolloutKey, dict[str, Any]],
    shared: set[RolloutKey],
) -> dict[str, Any]:
    """Combine paired rollout measurements with condition-wide proxy statistics."""
    stats = _read_object(run_dir / "switchyard-stats-raw.json")
    rewards = [_reward(rows[key], key) for key in sorted(shared)]
    try:
        captures = [rows[key]["ng_model_call_capture"]["metrics"] for key in sorted(shared)]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "rollout is missing ng_model_call_capture.metrics; enable Gym observability"
        ) from error

    try:
        answer_model_tokens = sum(int(item["tokens_total"]) for item in captures)
        endpoint_latency_mean_ms = sum(float(item["latency_total_ms"]) for item in captures) / len(
            captures
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "rollout capture metrics must include numeric tokens and latency"
        ) from error

    classifier = stats["classifier"]
    return {
        "paired": {
            "mean_reward": sum(rewards) / len(rewards),
            "answer_model_tokens": answer_model_tokens,
            "endpoint_latency_mean_ms": endpoint_latency_mean_ms,
        },
        "condition_totals": {
            "classifier_tokens": classifier["total_tokens"]["total"],
            "routing_overhead_mean_ms": stats["routing_overhead"]["avg_ms"],
            "classifier_fail_opens": _fail_opens(run_dir),
            "answer_models": _models(stats["models"]),
            "classifier_models": _models(classifier["models"]),
        },
    }


def compare(baseline_dir: Path, routed_dir: Path) -> dict[str, Any]:
    """Return paired quality and usage summaries for two result directories."""
    conditions = {
        "baseline": _read_object(baseline_dir / "switchyard-condition.json"),
        "routed": _read_object(routed_dir / "switchyard-condition.json"),
    }
    if any(condition.get("mode") != "attached" for condition in conditions.values()):
        raise ValueError("this example compares attached Switchyard runs")
    provenances = {
        name: condition.get("proxy_provenance") for name, condition in conditions.items()
    }
    required_revisions = ("gym_revision", "switchyard_revision")
    if any(
        not isinstance(provenance, dict)
        or any(
            not isinstance(provenance.get(key), str) or not provenance[key]
            for key in required_revisions
        )
        for provenance in provenances.values()
    ):
        raise ValueError("the runs have incomplete Switchyard provenance")
    provenance = provenances["baseline"]
    if provenance != provenances["routed"]:
        raise ValueError("the runs used different or incomplete Switchyard provenance")
    if (baseline_dir / "routes.toml").read_bytes() != (routed_dir / "routes.toml").read_bytes():
        raise ValueError("the runs used different Switchyard deployments")

    runs = {
        "baseline": _read_jsonl(baseline_dir / "rollouts.jsonl"),
        "routed": _read_jsonl(routed_dir / "rollouts.jsonl"),
    }
    inputs = {
        "baseline": _read_jsonl(baseline_dir / "rollouts_materialized_inputs.jsonl"),
        "routed": _read_jsonl(routed_dir / "rollouts_materialized_inputs.jsonl"),
    }
    expected = set(inputs["baseline"])
    if expected != set(inputs["routed"]):
        raise ValueError("the runs materialized different rollout keys")
    # Matching indices are insufficient because the source dataset revision is not pinned.
    for key in sorted(expected):
        if inputs["baseline"][key] != inputs["routed"][key]:
            raise ValueError(f"paired rollout {key} used different materialized inputs")
    for name, rows in runs.items():
        if extra := set(rows) - expected:
            raise ValueError(f"{name} completed unknown rollout keys: {sorted(extra)}")

    shared = set(runs["baseline"]) & set(runs["routed"])
    if not shared:
        raise ValueError("the runs have no completed rollouts in common")
    rewards = {
        name: {key: _reward(rows[key], key) for key in shared} for name, rows in runs.items()
    }
    return {
        "provenance": provenance,
        "routes": {name: condition["route"] for name, condition in conditions.items()},
        "coverage": {
            "expected": len(expected),
            "paired": len(shared),
            "completed": {name: len(rows) for name, rows in runs.items()},
            "unpaired": {name: len(set(rows) - shared) for name, rows in runs.items()},
            "missing": {name: len(expected - set(rows)) for name, rows in runs.items()},
        },
        "routed_vs_baseline": {
            "wins": sum(rewards["routed"][key] > rewards["baseline"][key] for key in shared),
            "ties": sum(rewards["routed"][key] == rewards["baseline"][key] for key in shared),
            "losses": sum(rewards["routed"][key] < rewards["baseline"][key] for key in shared),
        },
        "baseline": _condition_summary(baseline_dir, runs["baseline"], shared),
        "routed": _condition_summary(routed_dir, runs["routed"], shared),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path, help="fixed-model condition directory")
    parser.add_argument("routed_dir", type=Path, help="routed condition directory")
    args = parser.parse_args()
    try:
        result = compare(args.baseline_dir, args.routed_dir)
    except KeyError as error:
        parser.exit(1, f"{parser.prog}: error: missing expected field {error}\n")
    except (OSError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
