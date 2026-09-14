# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare complete, paired MCQA runs through LiteLLM and Switchyard Random routing."""

from __future__ import annotations

import argparse
import json
import math
import re
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


def count(value: Any, name: str) -> int:
    """Read a nonnegative integer counter."""
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return cast(int, value)


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
    """Require complete gateway request evidence alongside Gym's rollout artifacts."""
    failure_path, rollout_path = path / "rollouts_failures.jsonl", path / "rollouts.jsonl"
    for filename in ("run-provenance.json", "litellm-calls.jsonl"):
        require(
            (path / filename).is_file(),
            f"Missing {path / filename}. Inspect {path / 'gym.log'} and {path.parent / 'litellm.log'}.",
        )
    events: dict[str, dict[str, dict[str, Any]]] = {"start": {}, "finish": {}}
    for event in read_jsonl(path / "litellm-calls.jsonl"):
        kind, request_id = event.get("event"), event.get("request_id")
        require(kind in events, f"{path}: unknown gateway event")
        kind = cast(str, kind)
        require(isinstance(request_id, str) and bool(request_id), f"{path}: missing request ID")
        request_id = cast(str, request_id)
        require(request_id not in events[kind], f"{path}: duplicate gateway {kind} event")
        events[kind][request_id] = event
    require(
        bool(events["start"]) and events["start"].keys() == events["finish"].keys(),
        f"{path}: incomplete gateway request evidence",
    )
    return {
        "inputs": index_rows(path / "rollouts_materialized_inputs.jsonl"),
        "rows": index_rows(rollout_path) if rollout_path.exists() else {},
        "failures": read_jsonl(failure_path) if failure_path.exists() else [],
        "provenance": read_object(path / "run-provenance.json"),
        "events": events,
    }


def summarize(run: dict[str, Any], route: str) -> tuple[dict[str, int | float], Counter[str]]:
    """Join final answers by response ID while retaining all recorded gateway work."""
    runtime = run["provenance"]["runtime"]
    allowed_models = runtime["models"][route]
    finishes = list(run["events"]["finish"].values())
    responses: dict[str, dict[str, Any]] = {}
    tokens = []
    routing_times = []
    errors = unknown_usage = 0
    for phase in run["events"].values():
        for event in phase.values():
            require(event["route"] == route, f"{route}: gateway event belongs to another route")
            require(
                event["instance_id"] == runtime["instance_id"],
                f"{route}: gateway events belong to another proxy instance",
            )
    for event in finishes:
        failed = event.get("status_code") != 200 or bool(event.get("error_type"))
        errors += int(failed)
        if not failed:
            model, response_id = event.get("selected_model"), event.get("response_id")
            require(model in allowed_models, f"{route}: missing or invalid Switchyard selection")
            require(
                event.get("deployment_model") == model,
                f"{route}: selected model differs from the LiteLLM deployment",
            )
            require(
                isinstance(response_id, str) and bool(response_id),
                f"{route}: missing gateway response ID",
            )
            require(response_id not in responses, f"{route}: ambiguous gateway response ID")
            responses[response_id] = event
            number(event.get("tokens_total"), "gateway response tokens")
            number(event.get("routing_ms"), "routing decision time")
        if event.get("tokens_total") is None:
            unknown_usage += 1
        else:
            tokens.append(number(event["tokens_total"], "gateway tokens"))
        if event.get("routing_ms") is not None:
            routing_times.append(number(event["routing_ms"], "routing decision time"))

    calls, rewards, latencies, models = [], [], [], []
    used_response_ids: set[str] = set()
    captured_attempts = captured_errors = 0
    for key, row in sorted(run["rows"].items()):
        label = f"{route} {key}"
        require(not row.get("_ng_failure_class"), f"{label}: failed rollout")
        require(has_answer(row["response"]), f"{label}: missing or incomplete final answer")
        response_id = row["response"].get("id")
        require(isinstance(response_id, str) and bool(response_id), f"{label}: missing response id")
        require(response_id not in used_response_ids, f"{label}: response reused across rollouts")
        used_response_ids.add(response_id)
        reward = number(row["reward"], f"{label}: reward")
        require(reward <= 1, f"{label}: MCQA reward must be between zero and one")
        capture = row["ng_model_call_capture"]
        require(isinstance(capture, dict), f"{label}: missing model-call capture")
        require(not capture.get("gaps"), f"{label}: incomplete model-call capture")
        records = capture["calls"]
        require(
            isinstance(records, list) and all(isinstance(call, dict) for call in records),
            f"{label}: invalid captures",
        )
        terminal = [call for call in records if call.get("response_id") == response_id]
        require(len(terminal) == 1, f"{label}: missing or ambiguous terminal capture")
        call = terminal[0]
        require(
            call["status_code"] == 200 and not call.get("error_category"),
            f"{label}: terminal call failed",
        )
        require(call["response_status"] == "completed", f"{label}: captured call is incomplete")
        require(call.get("model") == route, f"{label}: captured response has the wrong model group")
        require(response_id in responses, f"{label}: final answer has no gateway evidence")
        event = responses[response_id]
        require(
            event.get("response_status") == "completed", f"{label}: gateway response is incomplete"
        )
        require(
            number(call["tokens_total"], "terminal-answer tokens") == event["tokens_total"],
            f"{label}: gateway/capture token totals differ",
        )
        captured_attempts += len(records)
        captured_errors += sum(
            record.get("status_code") != 200 or bool(record.get("error_category"))
            for record in records
        )
        calls.append(call)
        rewards.append(reward)
        latencies.append(number(row["ng_perf"]["total_latency_ms"], f"{label}: rollout latency"))
        models.append(event["selected_model"])

    return {
        "Paired rollouts": len(rewards),
        "Mean reward": mean(rewards),
        "Terminal-answer tokens": sum(
            number(call["tokens_total"], "terminal-answer tokens") for call in calls
        ),
        "Gateway-reported tokens": sum(tokens),
        "Gateway requests w/o usage": unknown_usage,
        "Mean rollout latency (ms)": mean(latencies),
        "Mean recorded routing (ms)": mean(routing_times),
        "Recorded routing decisions": len(routing_times),
        "Gateway requests": len(finishes),
        "Gateway errors": errors,
        "Captured model attempts": captured_attempts,
        "Captured failed attempts": captured_errors,
    }, Counter(models)


def compare(fixed: Path, routed: Path) -> None:
    """Report coverage first, then compare like-for-like complete runs."""
    runs = {"fixed": load_run(fixed), "routed": load_run(routed)}
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
    provenances = [run["provenance"] for run in runs.values()]
    for provenance in provenances:
        require(
            all(
                isinstance(provenance.get(key), str) and provenance[key].strip()
                for key in ("gym_revision", "switchyard_revision")
            ),
            "Missing Gym or Switchyard revision",
        )
        runtime = provenance["runtime"]
        require(runtime["mode"] == "litellm_libsy", "Expected the LiteLLM libsy integration")
        require(
            runtime["routing_plugin"] == "switchyard_litellm.RandomRoutingPlugin",
            "Expected Switchyard Random routing",
        )
        require(
            all(
                isinstance(runtime.get(key), str) and runtime[key].strip()
                for key in ("instance_id", "litellm_version", "switchyard_version")
            ),
            "Missing LiteLLM runtime identity",
        )
        require(
            all(
                re.fullmatch(r"[0-9a-f]{64}", str(runtime.get(key))) is not None
                for key in (
                    "profile_sha256",
                    "routing_sha256",
                    "callback_sha256",
                    "provider_base_sha256",
                )
            ),
            "Missing deployment fingerprint",
        )
        models = runtime["models"]
        require(
            all(
                isinstance(models.get(route), list)
                and all(isinstance(model, str) and model for model in models[route])
                for route in ("fixed", "routed")
            ),
            "Missing configured models",
        )
        require(
            len(models["fixed"]) == 1
            and len(set(models["routed"])) == 2
            and models["fixed"][0] in models["routed"],
            "Expected a fixed target and a routed pair containing it",
        )
    require(provenances[0] == provenances[1], "Gym, Switchyard, or deployment provenance differs")
    summaries = {name: summarize(run, name) for name, run in runs.items()}
    print(f"\nProvenance: {json.dumps(provenances[0], sort_keys=True)}")
    print(f"\n{'Metric':<29} {'fixed':>14} {'routed':>14}")
    for metric in summaries["fixed"][0]:
        values = [summaries[name][0][metric] for name in ("fixed", "routed")]
        formatted = [str(value) if type(value) is int else f"{value:.3f}" for value in values]
        print(f"{metric:<29} {formatted[0]:>14} {formatted[1]:>14}")
    for name, (summary, selected) in summaries.items():
        print(f"\n{name} selected models: {json.dumps(selected, sort_keys=True)}")
        if (
            summary["Gateway errors"]
            or summary["Gateway requests w/o usage"]
            or summary["Captured failed attempts"]
            or summary["Gateway requests"] != summary["Paired rollouts"]
            or summary["Captured model attempts"] != summary["Paired rollouts"]
        ):
            print(f"WARNING: {name} completed with recovered errors or additional work.")
    print("\nClassifier tokens: N/A (Random makes no classifier calls).")
    print(
        "Gateway totals include all recorded requests, including extra attempts; do not add terminal-answer tokens again."
    )
    print(
        "Unreported failed-request usage is unknown, not free. Gateway counts are not exhaustive provider-attempt or fallback telemetry."
    )
    print("Routing time is already included in rollout latency. Tokens are not dollar costs.")
    print("A small Random-routing run demonstrates the integration, not a routing advantage.")


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
