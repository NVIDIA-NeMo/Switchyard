# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one runner-produced experiment using Gym-captured usage and model attribution."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, cast


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read the JSONL objects written by Gym and the tutorial callback."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def index_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Index Gym rows by task and repeat, rejecting duplicate identities."""
    indexed = {}
    for row in read_jsonl(path):
        key = (row["_ng_task_index"], row["_ng_rollout_index"])
        require(key not in indexed, f"{path}: duplicate task/repeat {key}")
        indexed[key] = row
    return indexed


def has_answer_text(response: dict[str, Any]) -> bool:
    """Find answer text without mistaking reasoning or tool calls for the answer."""
    return any(
        part["type"] == "output_text" and bool(part["text"].strip())
        for item in response["output"]
        if item["type"] == "message" and item["role"] == "assistant"
        for part in item["content"]
    )


def load_run(path: Path) -> dict[str, Any]:
    """Load Gym's expected inputs, completed rollouts, and terminal failure sidecar."""
    failure_path, rollout_path = path / "rollouts_failures.jsonl", path / "rollouts.jsonl"
    return {
        "path": path,
        "inputs": index_rows(path / "rollouts_materialized_inputs.jsonl"),
        "rows": index_rows(rollout_path) if rollout_path.exists() else {},
        "failures": read_jsonl(failure_path) if failure_path.exists() else [],
    }


def token_total(calls: list[dict[str, Any]], field: str) -> int | str:
    """Sum reported token counts while explicitly retaining unknown contributions."""
    values = [call.get(field) for call in calls]
    known = [value for value in values if value is not None]
    require(all(type(value) is int and value >= 0 for value in known), f"Invalid captured {field}")
    if not known:
        return "unknown"
    total = sum(known)
    return total if len(known) == len(values) else f"{total}+unknown"


def summarize(run: dict[str, Any], route: str) -> tuple[dict[str, int | float | str], Counter[str]]:
    """Attribute final answers by response ID and count all Gym-captured calls."""
    model_path = run["path"] / "models.jsonl"
    require(model_path.is_file(), f"Missing {model_path}; use a fresh run with the updated runner")
    attribution: dict[str, str] = {}
    for record in read_jsonl(model_path):
        response_id, model = record["response_id"], record["model"]
        require(model not in ("fixed", "routed"), f"{route}: invalid model attribution")
        require(response_id not in attribution, f"{route}: ambiguous model attribution")
        attribution[response_id] = model

    calls, rewards, latencies, models, statuses = [], [], [], [], []
    response_ids: set[str] = set()
    for key, row in sorted(run["rows"].items()):
        label = f"{route} {key}"
        require(not row.get("_ng_failure_class"), f"{label}: failed rollout")
        response = row["response"]
        require(has_answer_text(response), f"{label}: missing final answer text")
        require(
            response["status"] in ("completed", "incomplete"), f"{label}: invalid response status"
        )
        response_id = response["id"]
        require(response_id in attribution, f"{label}: missing final-answer model attribution")
        require(response_id not in response_ids, f"{label}: response reused across rollouts")
        response_ids.add(response_id)
        capture = row["ng_model_call_capture"]
        require(not capture.get("gaps"), f"{label}: incomplete model-call capture")
        records = capture["calls"]
        terminal = [call for call in records if call.get("response_id") == response_id]
        require(len(terminal) == 1, f"{label}: missing or ambiguous terminal capture")
        call = terminal[0]
        require(
            call["status_code"] == 200 and not call.get("error_category"),
            f"{label}: terminal call failed",
        )
        require(
            call["response_status"] == response["status"],
            f"{label}: response and capture status differ",
        )
        require(call["model"] == route, f"{label}: captured response has the wrong model group")
        reward = number(row["reward"], f"{label}: reward")
        require(reward <= 1, f"{label}: MCQA reward must be between zero and one")
        rewards.append(reward)
        latencies.append(number(row["ng_perf"]["total_latency_ms"], f"{label}: rollout latency"))
        models.append(attribution[response_id])
        statuses.append(response["status"])
        calls.extend(records)
    return {
        "Paired rollouts": len(rewards),
        "Incomplete responses": statuses.count("incomplete"),
        "Mean reward": mean(rewards),
        "Captured input tokens": token_total(calls, "tokens_in"),
        "Captured output tokens": token_total(calls, "tokens_out"),
        "Mean rollout latency (ms)": mean(latencies),
        "Captured calls": len(calls),
        "Captured failed calls": sum(
            call.get("status_code") != 200 or bool(call.get("error_category")) for call in calls
        ),
        "Calls with unknown usage": sum(
            call.get("tokens_in") is None or call.get("tokens_out") is None for call in calls
        ),
    }, Counter(models)


def compare(results: Path) -> None:
    """Report coverage first, then compare both conditions from one experiment directory."""
    runs = {route: load_run(results / route) for route in ("fixed", "routed")}
    complete = True
    for name, run in runs.items():
        expected, actual = set(run["inputs"]), set(run["rows"])
        missing, unexpected = expected - actual, actual - expected
        print(
            f"{name}: expected={len(expected)}, completed={len(actual)}, missing={len(missing)}, unexpected={len(unexpected)}, failures={len(run['failures'])}"
        )
        complete &= bool(expected) and not missing and not unexpected and not run["failures"]
    fixed_keys, routed_keys = set(runs["fixed"]["rows"]), set(runs["routed"]["rows"])
    print(
        f"Pairing: matched={len(fixed_keys & routed_keys)}, fixed-only={len(fixed_keys - routed_keys)}, routed-only={len(routed_keys - fixed_keys)}"
    )
    require(complete, "Incomplete runs: inspect the failure artifacts; no averages calculated")
    require(
        runs["fixed"]["inputs"] == runs["routed"]["inputs"],
        "Task inputs, verifier metadata, or generation settings differ",
    )
    summaries = {name: summarize(run, name) for name, run in runs.items()}
    print(f"\n{'Metric':<29} {'fixed':>14} {'routed':>14}")
    for metric in summaries["fixed"][0]:
        values = [summaries[name][0][metric] for name in ("fixed", "routed")]
        formatted = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in values]
        print(f"{metric:<29} {formatted[0]:>14} {formatted[1]:>14}")
    for name, (summary, models) in summaries.items():
        print(f"\n{name} serving models: {json.dumps(models, sort_keys=True)}")
        if (
            summary["Incomplete responses"]
            or summary["Captured failed calls"]
            or summary["Calls with unknown usage"]
            or summary["Captured calls"] != summary["Paired rollouts"]
        ):
            print(
                f"WARNING: {name} has incomplete responses, recovered errors, additional calls, or unknown usage."
            )
    print(
        "\nGym-scored incomplete responses remain in the comparison. "
        "Gym-captured tokens include extra attempts; +unknown means some usage was not reported."
    )
    print(
        "Retries inside Gym's model adapter, LiteLLM, or the provider may not appear separately. Tokens are not dollar costs."
    )
    print(
        "Random makes no classifier calls. A small run demonstrates the integration, not a routing advantage."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the comparison without importing Gym or Switchyard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, help="Experiment directory containing fixed/ and routed/"
    )
    args = parser.parse_args(argv)
    try:
        compare(args.results)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot compare: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
