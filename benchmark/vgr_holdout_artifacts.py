# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an auditable summary from VGR hold-out run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_ARTIFACT_JSON_BYTES = 512 * 1024 * 1024
MAX_CONTROL_JSON_BYTES = 16 * 1024 * 1024
MAX_ROUTING_RECORD_BYTES = 1024 * 1024
LABELS = {"local_required", "cloud_required", "both_fail", "indeterminate"}
ROUTES = {"local", "cloud"}
BENCHMARK_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _file_digest(path: Path) -> bytes:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.digest()


def path_digest(path: Path) -> str:
    """Return a deterministic SHA-256 digest for a file or directory."""
    resolved = path.resolve()
    hasher = hashlib.sha256()
    if resolved.is_file():
        hasher.update(resolved.name.encode())
        hasher.update(_file_digest(resolved))
        return f"sha256:{hasher.hexdigest()}"
    if resolved.is_dir():
        for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
            relative = item.relative_to(resolved).as_posix()
            hasher.update(f"{relative}\n{_file_digest(item).hex()}\n".encode())
        return f"sha256:{hasher.hexdigest()}"
    return "sha256:missing"


def _read_json(path: Path, max_bytes: int = MAX_ARTIFACT_JSON_BYTES) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON artifact exceeds {max_bytes} bytes: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_key(record: dict[str, Any]) -> str | None:
    for field in ("task", "session_id"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _read_routing_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid = 0
    if not path.is_file():
        return records, invalid
    with path.open("rb") as file:
        for index, line in enumerate(file, start=1):
            if len(line) > MAX_ROUTING_RECORD_BYTES:
                raise ValueError(f"routing record {index} exceeds {MAX_ROUTING_RECORD_BYTES} bytes")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid += 1
                continue
            if not isinstance(value, dict):
                invalid += 1
                continue
            value["_record_index"] = index
            records.append(value)
    return records, invalid


def _counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = Counter(
        value
        for record in records
        if isinstance((value := record.get(field)), str) and value
    )
    return dict(sorted(values.items()))


def _token_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "prompt_tokens",
        "cached_tokens",
        "cache_creation_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    return {
        field: sum(
            value
            for record in records
            if isinstance((value := record.get(field)), int) and value >= 0
        )
        for field in fields
    }


def _tier_flips(records: list[dict[str, Any]]) -> int:
    sequence = [
        value
        for record in records
        if isinstance((value := record.get("vgr_served")), str) and value in ROUTES
    ]
    return sum(left != right for left, right in zip(sequence, sequence[1:], strict=False))


def _task_rows(
    records: list[dict[str, Any]],
    labels: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_task_key(record) or "unattributed"].append(record)

    rows: list[dict[str, Any]] = []
    for task, task_records in sorted(grouped.items()):
        label = labels.get(task)
        rows.append(
            {
                "task": task,
                "benchmark": label["benchmark"] if label else None,
                "counterfactual_label": label["required_route"] if label else None,
                "requests": len(task_records),
                "trial_ids": sorted(
                    {
                        value
                        for record in task_records
                        if isinstance((value := record.get("trial_id")), str) and value
                    }
                ),
                "predicted": _counts(task_records, "vgr_predicted"),
                "effective": _counts(task_records, "vgr_effective"),
                "served": _counts(task_records, "vgr_served"),
                "branches": _counts(task_records, "vgr_branch"),
                "readiness_gates": _counts(task_records, "vgr_readiness_gate"),
                "short_circuits": _counts(task_records, "vgr_short_circuit"),
                "models": _counts(task_records, "model"),
                "tiers": _counts(task_records, "tier"),
                "tier_flips": _tier_flips(task_records),
                "tokens": _token_totals(task_records),
            }
        )
    return rows


def _load_labels(path: Path | None, run_dir: Path) -> tuple[dict[str, dict[str, str]], Path | None]:
    if path is None:
        return {}, None
    value = _read_json(path, MAX_CONTROL_JSON_BYTES)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("counterfactual labels must be a schema_version 1 object")
    if value.get("decision_unit") != "request":
        raise ValueError("counterfactual labels decision_unit must be 'request'")
    entries = value.get("labels")
    if not isinstance(entries, list):
        raise ValueError("counterfactual labels must contain a labels array")

    labels: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"counterfactual label {index} must be an object")
        task = entry.get("task")
        benchmark = entry.get("benchmark")
        required_route = entry.get("required_route")
        if not isinstance(task, str) or not task or len(task) > 256:
            raise ValueError(f"counterfactual label {index} has an invalid task")
        if not isinstance(benchmark, str) or not BENCHMARK_RE.fullmatch(benchmark):
            raise ValueError(f"counterfactual label {index} has an invalid benchmark")
        if required_route not in LABELS:
            raise ValueError(f"counterfactual label {index} has an invalid required_route")
        if task in labels:
            raise ValueError(f"duplicate counterfactual label for task {task}")
        labels[task] = {"benchmark": benchmark, "required_route": required_route}

    snapshot = run_dir / "inputs/counterfactual_labels.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != snapshot.resolve():
        shutil.copyfile(path, snapshot)
    return labels, snapshot


def _wilson_interval(numerator: int, denominator: int) -> list[float] | None:
    if denominator == 0:
        return None
    z = 1.959963984540054
    proportion = numerator / denominator
    z_squared = z * z
    scale = 1 + z_squared / denominator
    center = (proportion + z_squared / (2 * denominator)) / scale
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / denominator
            + z_squared / (4 * denominator * denominator)
        )
        / scale
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "wilson_95_confidence_interval": _wilson_interval(numerator, denominator),
    }


def _gate_summary(
    records: list[dict[str, Any]],
    labels: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if not labels:
        return {
            "status": "blocked",
            "decision_unit": "request",
            "blockers": ["counterfactual_labels_missing"],
            "note": "Run TC-EVAL-01/02 and supply their frozen labels to compute FPR/FNR.",
        }

    by_benchmark: dict[str, Counter[str]] = defaultdict(Counter)
    unlabeled_records = 0
    unusable_routes = 0
    seen_tasks: set[str] = set()
    for record in records:
        task = _task_key(record)
        label = labels.get(task or "")
        if label is None:
            unlabeled_records += 1
            continue
        seen_tasks.add(task or "")
        counts = by_benchmark[label["benchmark"]]
        required_route = label["required_route"]
        counts[f"label_{required_route}"] += 1
        if required_route in {"both_fail", "indeterminate"}:
            counts["excluded"] += 1
            continue
        effective = record.get("vgr_effective")
        if effective not in ROUTES:
            counts["unusable_route"] += 1
            unusable_routes += 1
            continue
        if required_route == "cloud_required":
            counts["fpr_denominator"] += 1
            counts["fpr_numerator"] += effective == "local"
        else:
            counts["fnr_denominator"] += 1
            counts["fnr_numerator"] += effective == "cloud"

    missing_labeled_tasks = sorted(set(labels) - seen_tasks)
    benchmarks = {
        benchmark: {
            "fpr": _rate(counts["fpr_numerator"], counts["fpr_denominator"]),
            "fnr": _rate(counts["fnr_numerator"], counts["fnr_denominator"]),
            "local_required_records": counts["label_local_required"],
            "cloud_required_records": counts["label_cloud_required"],
            "both_fail_records": counts["label_both_fail"],
            "indeterminate_records": counts["label_indeterminate"],
            "excluded_records": counts["excluded"],
            "unusable_route_records": counts["unusable_route"],
        }
        for benchmark, counts in sorted(by_benchmark.items())
    }
    blockers = []
    if unlabeled_records:
        blockers.append("routing_records_without_labels")
    if missing_labeled_tasks:
        blockers.append("labels_without_routing_records")
    if unusable_routes:
        blockers.append("records_without_effective_route")
    return {
        "status": "computed" if not blockers else "incomplete",
        "decision_unit": "request",
        "blockers": blockers,
        "unlabeled_routing_records": unlabeled_records,
        "labeled_tasks_without_routing_records": missing_labeled_tasks,
        "benchmarks": benchmarks,
        "threshold_evaluation": "not_scored",
        "note": (
            "Rates are reproduced from frozen labels and readiness-effective routes. "
            "QA owns the acceptance rule for confidence intervals."
        ),
    }


def _native_summaries(run_dir: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    automation = run_dir / "automationbench-simple.json"
    if automation.is_file():
        value = _read_json(automation)
        if isinstance(value, dict):
            summaries["automationbench_simple"] = {
                "meta": value.get("meta"),
                "summary": value.get("summary"),
                "task_rows": len(value.get("tasks", []))
                if isinstance(value.get("tasks"), list)
                else None,
            }

    harbor_results = sorted((run_dir / "tb21/jobs").rglob("result.json"))
    summaries["tb21"] = {
        "result_files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "digest": path_digest(path),
            }
            for path in harbor_results
        ]
    }
    appworld_results = sorted((run_dir / "appworld-output").rglob("evaluations/dev_easy.json"))
    summaries["appworld_easy"] = {
        "evaluation_files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "digest": path_digest(path),
                "aggregate": (
                    value.get("aggregate")
                    if isinstance((value := _read_json(path)), dict)
                    else None
                ),
            }
            for path in appworld_results
        ]
    }
    return summaries


def _artifact_index(run_dir: Path, labels_snapshot: Path | None) -> dict[str, Any]:
    candidates = {
        "server_config": (run_dir / "inputs/server-config.toml", True),
        "routing_records": (run_dir / "routing_requests.jsonl", True),
        "routing_stats": (run_dir / "routing_stats_final.json", True),
        "server_metrics": (run_dir / "server_metrics_final.prom", True),
        "tb21_native_outputs": (run_dir / "tb21/jobs", True),
        "automationbench_native_output": (run_dir / "automationbench-simple.json", True),
        "appworld_native_outputs": (run_dir / "appworld-output", True),
        "counterfactual_labels": (
            labels_snapshot or run_dir / "inputs/counterfactual_labels.json",
            False,
        ),
        "per_task_routing": (run_dir / "per_task_routing.jsonl", True),
        "machine_readable_summary": (run_dir / "machine_readable_summary.json", True),
    }
    return {
        name: {
            "path": path.relative_to(run_dir).as_posix(),
            "required": required,
            "status": "present" if path.exists() else "missing",
            "digest": path_digest(path),
        }
        for name, (path, required) in candidates.items()
    }


def finalize_artifacts(
    run_dir: Path,
    labels_path: Path | None = None,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Write per-task routing, a machine summary, and a digest index."""
    resolved = run_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"hold-out run directory not found: {resolved}")
    manifest_path = resolved / "run_manifest.json"
    manifest = _read_json(manifest_path, MAX_CONTROL_JSON_BYTES)
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be a JSON object")

    labels, labels_snapshot = _load_labels(labels_path, resolved)
    records, invalid_records = _read_routing_records(resolved / "routing_requests.jsonl")
    task_rows = _task_rows(records, labels)
    with (resolved / "per_task_routing.jsonl").open("w", encoding="utf-8") as file:
        for row in task_rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "routing": {
            "records": len(records),
            "invalid_records": invalid_records,
            "attributed_records": sum(_task_key(record) is not None for record in records),
            "predicted": _counts(records, "vgr_predicted"),
            "effective": _counts(records, "vgr_effective"),
            "served": _counts(records, "vgr_served"),
            "branches": _counts(records, "vgr_branch"),
            "readiness_gates": _counts(records, "vgr_readiness_gate"),
            "short_circuits": _counts(records, "vgr_short_circuit"),
            "models": _counts(records, "model"),
            "tiers": _counts(records, "tier"),
            "tokens": _token_totals(records),
            "tasks": len(task_rows),
            "tier_flips": sum(row["tier_flips"] for row in task_rows),
        },
        "counterfactual_gate": _gate_summary(records, labels),
        "benchmark_native": _native_summaries(resolved),
        "metric_coverage": {
            "available": [
                "predicted_effective_served_route_counts",
                "model_and_tier_token_counts",
                "vgr_branch_and_readiness_gate_counts",
                "short_circuit_counts",
                "tier_flips",
                "benchmark_native_scores_and_errors",
            ],
            "conditional": ["fpr_fnr_with_95_percent_ci_when_frozen_labels_are_supplied"],
            "not_emitted_by_current_sources": [
                "local_decode_tokens_per_second",
                "time_to_first_token",
                "router_only_wallclock",
                "task_typing_abstain_rate",
                "verifier_cost_separated_from_cloud_model_cost",
            ],
        },
    }
    summary_path = resolved / "machine_readable_summary.json"
    _write_json(summary_path, summary)
    index = _artifact_index(resolved, labels_snapshot)
    _write_json(resolved / "artifact_index.json", {"schema_version": 1, "artifacts": index})

    missing = [name for name, item in index.items() if item["required"] and item["status"] != "present"]
    if require_complete and missing:
        raise RuntimeError(f"required hold-out artifacts are missing: {', '.join(missing)}")
    return {
        "summary": summary_path.relative_to(resolved).as_posix(),
        "per_task_routing": "per_task_routing.jsonl",
        "artifact_index": "artifact_index.json",
        "counterfactual_labels": (
            labels_snapshot.relative_to(resolved).as_posix() if labels_snapshot else None
        ),
        "missing_required": missing,
    }
