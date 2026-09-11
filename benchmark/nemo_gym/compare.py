# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare complete, paired, single-turn MCQA runs from the hosted Gym tutorial."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, cast

GYM_COMMIT = "3a26c35fa90c243427378569511f7b06f503e0fd"
SWITCHYARD_VERSION = "0.2.0"
HOSTED_SCOPE = "this run (proxy hosted for exactly this run)"


def require(condition: bool, message: str) -> None:
    """Reject incomplete or incompatible evidence before calculating metrics."""
    if not condition:
        raise ValueError(message)


def number(value: Any, name: str) -> int | float:
    """Read a finite, nonnegative measurement without treating missing values as zero."""
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"{name} must be a finite, nonnegative number",
    )
    return cast(int | float, value)


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON artifact with an object at its root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return cast(dict[str, Any], value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL objects, ignoring empty lines."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    require(all(isinstance(row, dict) for row in rows), f"{path}: expected JSON objects")
    return rows


def index_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Index by Gym task and repeat, rejecting duplicate or malformed identities."""
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
        require(
            all(type(part) is int and part >= 0 for part in key),
            f"{path}: invalid task/repeat index",
        )
        key = cast(tuple[int, int], key)
        require(key not in indexed, f"{path}: duplicate task/repeat {key}")
        indexed[key] = row
    return indexed


def has_answer(response: Any) -> bool:
    """Require a completed Responses API message, not just reasoning or a tool call."""
    return (
        isinstance(response, dict)
        and response.get("status") == "completed"
        and any(
            item.get("type") == "message"
            and item.get("role") == "assistant"
            and any(
                part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
                and part["text"].strip()
                for part in (item.get("content") or [])
                if isinstance(part, dict)
            )
            for item in (response.get("output") or [])
            if isinstance(item, dict)
        )
    )


def load_run(path: Path) -> dict[str, Any]:
    """Load expected tasks, completed rollouts, failures, and hosted-proxy provenance."""
    failure_path = path / "rollouts_failures.jsonl"
    rollout_path = path / "rollouts.jsonl"
    condition_path = path / "switchyard-condition.json"
    stats_path = path / "switchyard-stats.json"
    require(
        condition_path.is_file(),
        f"Missing {condition_path}. This file is written when the model server starts. "
        "Set Terminal 1's OUT and condition_dir to the same run directory as Terminal 2. "
        "If another condition's metadata was overwritten, rerun that condition in a fresh directory.",
    )
    require(
        stats_path.is_file(),
        f"Missing {stats_path}. After evaluation finishes, press Ctrl-C in the terminal "
        "running gym env start and wait for shutdown to save the statistics. "
        "If the servers have already stopped, inspect server-logs/policy_model.log for snapshot errors.",
    )
    return {
        "inputs": index_rows(path / "rollouts_materialized_inputs.jsonl"),
        "rows": index_rows(rollout_path) if rollout_path.exists() else {},
        "failures": read_jsonl(failure_path) if failure_path.exists() else [],
        "condition": read_object(condition_path),
        "snapshot": read_object(stats_path),
        "gym_commit": (path / "gym-commit.txt").read_text(encoding="utf-8").strip(),
    }


def summarize(run: dict[str, Any], route: str) -> tuple[dict[str, int | float], Counter[str]]:
    """Calculate metrics only for this tutorial's complete, one-call-per-task runs."""
    condition, snapshot = run["condition"], run["snapshot"]
    require(
        condition["route"] == route,
        f"{route}: manifest records route {condition['route']!r}. "
        "The server's route and output directory must agree; rerun this condition in a fresh directory.",
    )
    require(condition["mode"] == snapshot["mode"] == "hosted", f"{route}: expected hosted mode")
    require(snapshot["scope"] == HOSTED_SCOPE, f"{route}: statistics are not run-scoped")
    require(
        condition["nemo_switchyard_version"] == SWITCHYARD_VERSION,
        f"{route}: unexpected Switchyard version",
    )
    require(run["gym_commit"] == GYM_COMMIT, f"{route}: unexpected Gym commit")
    stats = snapshot["stats"]
    classifier = stats["classifier"]
    model_errors = number(stats["total_errors"], "model errors")
    classifier_errors = number(classifier["total_errors"], "classifier errors")
    require(
        model_errors == classifier_errors == 0,
        f"{route}: model errors={model_errors}, classifier errors={classifier_errors}",
    )

    calls, rewards, latencies = [], [], []
    for key, row in sorted(run["rows"].items()):
        label = f"{route} {key}"
        require(not row.get("_ng_failure_class"), f"{label}: failed rollout")
        require(has_answer(row["response"]), f"{label}: missing or incomplete final answer")
        reward = number(row["reward"], f"{label}: reward")
        require(reward <= 1, f"{label}: MCQA reward must be between zero and one")
        capture = row["ng_model_call_capture"]
        require(isinstance(capture, dict), f"{label}: missing model-call capture")
        require(not capture.get("gaps"), f"{label}: incomplete model-call capture")
        require(len(capture["calls"]) == 1, f"{label}: expected exactly one captured model call")
        call = capture["calls"][0]
        require(
            call["status_code"] == 200 and not call.get("error_category"),
            f"{label}: model call failed",
        )
        require(call["response_status"] == "completed", f"{label}: captured call is incomplete")
        model = call.get("model")
        require(
            isinstance(model, str) and bool(model.strip()) and model not in {"fixed", "routed"},
            f"{label}: missing selected model",
        )
        calls.append(call)
        rewards.append(reward)
        latencies.append(number(row["ng_perf"]["total_latency_ms"], f"{label}: rollout latency"))

    selected_tokens = sum(number(call["tokens_total"], "captured model tokens") for call in calls)
    require(
        number(stats["total_requests"], "model requests") == len(calls),
        f"{route}: proxy/capture request counts differ",
    )
    require(
        number(stats["total_tokens"]["total"], "proxy model tokens") == selected_tokens,
        f"{route}: proxy/capture token totals differ",
    )
    classifier_calls = number(classifier["total_requests"], "classifier requests")
    require(
        classifier_calls == (0 if route == "fixed" else len(calls)),
        f"{route}: unexpected classifier request count",
    )
    classifier_tokens = number(classifier["total_tokens"]["total"], "classifier tokens")
    return {
        "Paired rollouts": len(rewards),
        "Mean reward": mean(rewards),
        "Selected-model tokens": selected_tokens,
        "Classifier tokens": classifier_tokens,
        "Combined reported tokens": selected_tokens + classifier_tokens,
        "Mean rollout latency (ms)": mean(latencies),
        "Mean routing overhead (ms)": number(
            stats["routing_overhead"]["avg_ms"], "routing overhead"
        ),
        "Classifier requests": classifier_calls,
        "Model errors": model_errors,
        "Classifier errors": classifier_errors,
    }, Counter(call["model"] for call in calls)


def compare(fixed: Path, routed: Path) -> None:
    """Report coverage first, then compare like-for-like complete runs."""
    runs = {"fixed": load_run(fixed), "routed": load_run(routed)}
    complete = True
    for name, run in runs.items():
        expected, actual = set(run["inputs"]), set(run["rows"])
        missing, unexpected = expected - actual, actual - expected
        print(
            f"{name}: expected={len(expected)}, completed={len(actual)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, failures={len(run['failures'])}"
        )
        complete &= bool(expected) and not missing and not unexpected and not run["failures"]
    fixed_keys, routed_keys = set(runs["fixed"]["rows"]), set(runs["routed"]["rows"])
    print(
        f"Pairing: matched={len(fixed_keys & routed_keys)}, "
        f"fixed-only={len(fixed_keys - routed_keys)}, routed-only={len(routed_keys - fixed_keys)}"
    )
    require(complete, "Incomplete runs: inspect the failure artifacts; no averages calculated")
    require(
        runs["fixed"]["inputs"] == runs["routed"]["inputs"],
        "Task inputs, verifier metadata, or generation settings differ",
    )
    hashes = [run["condition"].get("deployment_sha256") for run in runs.values()]
    require(
        all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in hashes
        )
        and hashes[0] == hashes[1],
        "Expected the same recorded deployment hash in both runs",
    )
    summaries = {name: summarize(run, name) for name, run in runs.items()}
    print(f"\n{'Metric':<29} {'fixed':>14} {'routed':>14}")
    for metric in summaries["fixed"][0]:
        values = [summaries[name][0][metric] for name in ("fixed", "routed")]
        formatted = [str(value) if type(value) is int else f"{value:.3f}" for value in values]
        print(f"{metric:<29} {formatted[0]:>14} {formatted[1]:>14}")
    for name, (_, models) in summaries.items():
        fallbacks = runs[name]["snapshot"]["stats"].get("routing_fallbacks", "unavailable")
        print(f"\n{name} selected models: {json.dumps(models, sort_keys=True)}")
        print(f"{name} reported fallbacks: {json.dumps(fallbacks, sort_keys=True)}")
    print(
        "\nClassifier fail-open decisions are not counted in these v0.2.0 statistics; "
        "inspect Gym's model-server log."
    )
    print(
        "Reported tokens are not dollar costs. A small run demonstrates the workflow, not a routing advantage."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the comparison without importing Gym or Switchyard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed", type=Path, help="Fixed run directory")
    parser.add_argument("routed", type=Path, help="Routed run directory")
    args = parser.parse_args(argv)
    try:
        compare(args.fixed, args.routed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot compare: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
