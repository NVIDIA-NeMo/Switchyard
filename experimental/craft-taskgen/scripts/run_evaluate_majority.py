#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the evaluate step multiple times and apply per-task majority votes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

VERDICT_PRIORITY = ["accept", "reject", "ERROR", "error", "unknown"]
EVALUATE_STAGE_BY_VERDICT = {
    "accept": "promising",
    "reject": "rejected",
    "ERROR": "evaluated",
    "error": "rejected",
}
EVALUATE_ERA_STAGES = {"candidate", "evaluated", "promising", "rejected"}


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _task_instance_id(task: dict[str, Any]) -> str:
    candidate = task.get("candidate_data") or {}
    source_meta = candidate.get("source_metadata") or {}
    return str(source_meta.get("instance_id") or candidate.get("source_task_id") or "").strip()


def _matches_filters(task_id: str, task: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.task_id and task_id not in args.task_id and str(task.get("task_id") or "") not in args.task_id:
        return False
    if args.repo and str(task.get("repo") or "") != args.repo:
        return False
    if args.instance_id and _task_instance_id(task) != args.instance_id:
        return False
    return True


def _clear_trial_eval_fields(task: dict[str, Any]) -> None:
    task["stage"] = "candidate"
    task["eval_verdict"] = ""
    task["eval_reason"] = ""
    task["eval_instruction_sketch"] = ""
    task["eval_verifier_notes"] = ""
    task["in_progress_step"] = ""
    task["iteration_log"] = [
        entry for entry in task.get("iteration_log", []) if entry.get("step") != "evaluate"
    ]
    usage = task.get("llm_usage")
    if isinstance(usage, dict):
        usage.pop("evaluate", None)


def prepare_trial_state(input_state: Path, output_state: Path, args: argparse.Namespace) -> list[str]:
    state = json.loads(input_state.read_text())
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{input_state}: expected top-level 'tasks' dict")

    selected: list[str] = []
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            continue
        task["in_progress_step"] = ""
        if _matches_filters(task_id, task, args):
            selected.append(task_id)

    if args.limit:
        selected = selected[: args.limit]
    selected_set = set(selected)
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            continue
        if task_id in selected_set:
            _clear_trial_eval_fields(task)
        elif task.get("stage") == "candidate":
            task["stage"] = "rejected"
            task["eval_verdict"] = "skip"
            task["eval_reason"] = "Skipped by run_evaluate_majority filter"

    if not selected:
        raise RuntimeError("no tasks matched the requested filters")

    output_state.parent.mkdir(parents=True, exist_ok=True)
    output_state.write_text(json.dumps(state, indent=2, default=str) + "\n")
    return selected


def _latest_evaluate_usage(task: dict[str, Any]) -> dict[str, Any]:
    usage = ((task.get("llm_usage") or {}).get("evaluate") or [])
    if isinstance(usage, list) and usage:
        latest = usage[-1]
        if isinstance(latest, dict):
            return latest
    return {}


def extract_trial_rows(
    state_path: Path,
    selected_task_ids: list[str],
    run_number: int,
) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text())
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{state_path}: expected top-level 'tasks' dict")

    rows: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        task = tasks.get(task_id)
        if not isinstance(task, dict):
            continue
        rows.append(
            {
                "run_number": run_number,
                "task_id": str(task.get("task_id") or task_id),
                "source_task_id": _task_instance_id(task),
                "repo": task.get("repo", ""),
                "commit_sha": task.get("commit_sha", ""),
                "stage": task.get("stage", ""),
                "eval_verdict": task.get("eval_verdict", ""),
                "eval_reason": task.get("eval_reason", ""),
                "eval_instruction_sketch": task.get("eval_instruction_sketch", ""),
                "eval_verifier_notes": task.get("eval_verifier_notes", ""),
                "usage": _latest_evaluate_usage(task),
            }
        )
    return rows


def _vote_key(row: dict[str, Any]) -> str:
    for key in ("source_task_id", "task_id", "commit_sha"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"evaluate row has no source_task_id/task_id/commit_sha: {row}")


def _vote_label(row: dict[str, Any]) -> str:
    return str(row.get("eval_verdict") or "unknown").strip() or "unknown"


def _choose_from_tie(
    votes: list[dict[str, Any]],
    tied_labels: set[str],
    tie_policy: str,
) -> dict[str, Any] | None:
    if tie_policy == "keep-current":
        return None
    if tie_policy == "first":
        return next(row for row in votes if _vote_label(row) in tied_labels)
    if tie_policy == "priority":
        for verdict in VERDICT_PRIORITY:
            if verdict in tied_labels:
                return next(row for row in votes if _vote_label(row) == verdict)
        return next(row for row in votes if _vote_label(row) in tied_labels)
    raise ValueError(f"unknown tie policy: {tie_policy}")


def aggregate_votes(
    run_rows: list[list[dict[str, Any]]],
    *,
    tie_policy: str = "keep-current",
    require_strict_majority: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    key_order: dict[str, int] = {}
    for rows in run_rows:
        for row in rows:
            key = _vote_key(row)
            key_order.setdefault(key, len(key_order))
            grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, votes in grouped.items():
        counts = Counter(_vote_label(row) for row in votes)
        top_count = max(counts.values())
        top_labels = {label for label, count in counts.items() if count == top_count}
        strict_majority = top_count > len(votes) / 2
        tie = len(top_labels) > 1
        if tie:
            chosen = _choose_from_tie(votes, top_labels, tie_policy)
        else:
            chosen = next(row for row in votes if _vote_label(row) in top_labels)
        if require_strict_majority and not strict_majority:
            chosen = None

        first = votes[0]
        summary = {
            "key": key,
            "task_id": first.get("task_id", ""),
            "source_task_id": first.get("source_task_id", ""),
            "repo": first.get("repo", ""),
            "commit_sha": first.get("commit_sha", ""),
            "counts": dict(sorted(counts.items())),
            "total_votes": len(votes),
            "winner_count": top_count,
            "strict_majority": strict_majority,
            "tie": tie,
            "applied": False,
            "votes": votes,
        }
        if chosen is None:
            summary.update(
                {
                    "winner": "",
                    "eval_verdict": "",
                    "eval_reason": "",
                    "eval_instruction_sketch": "",
                    "eval_verifier_notes": "",
                    "chosen_run_number": 0,
                }
            )
        else:
            summary.update(
                {
                    "winner": _vote_label(chosen),
                    "eval_verdict": chosen.get("eval_verdict", ""),
                    "eval_reason": chosen.get("eval_reason", ""),
                    "eval_instruction_sketch": chosen.get("eval_instruction_sketch", ""),
                    "eval_verifier_notes": chosen.get("eval_verifier_notes", ""),
                    "chosen_run_number": chosen.get("run_number", 0),
                }
            )
        summaries.append(summary)

    return sorted(summaries, key=lambda row: key_order[row["key"]])


def _task_key_values(task_id: str, task: dict[str, Any]) -> set[str]:
    candidate = task.get("candidate_data") or {}
    source_meta = candidate.get("source_metadata") or {}
    values = {
        task_id,
        str(task.get("task_id") or ""),
        str(task.get("commit_sha") or ""),
        str(candidate.get("sha") or ""),
        str(candidate.get("source_task_id") or ""),
        str(source_meta.get("instance_id") or ""),
    }
    return {value for value in values if value}


def apply_majority_to_state(
    state_path: Path,
    summaries: list[dict[str, Any]],
    *,
    backup_path: Path | None = None,
    force_stage: bool = False,
) -> int:
    if backup_path is not None:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(state_path, backup_path)

    state = json.loads(state_path.read_text())
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{state_path}: expected top-level 'tasks' dict")

    task_index: dict[str, str] = {}
    for task_id, task in tasks.items():
        if isinstance(task, dict):
            for value in _task_key_values(task_id, task):
                task_index[value] = task_id

    applied = 0
    for summary in summaries:
        verdict = str(summary.get("eval_verdict") or "")
        if not verdict:
            continue
        task_id = task_index.get(str(summary.get("task_id") or ""))
        task_id = task_id or task_index.get(str(summary.get("source_task_id") or ""))
        task_id = task_id or task_index.get(str(summary.get("commit_sha") or ""))
        task_id = task_id or task_index.get(str(summary.get("key") or ""))
        if not task_id:
            continue

        task = tasks[task_id]
        task["eval_verdict"] = verdict
        task["eval_reason"] = str(summary.get("eval_reason") or "")
        task["eval_instruction_sketch"] = str(summary.get("eval_instruction_sketch") or "")
        task["eval_verifier_notes"] = str(summary.get("eval_verifier_notes") or "")

        final_stage = EVALUATE_STAGE_BY_VERDICT.get(verdict, "rejected")
        if force_stage or str(task.get("stage") or "") in EVALUATE_ERA_STAGES:
            task["stage"] = final_stage

        task.setdefault("iteration_log", []).append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "evaluate_majority",
                "verdict": verdict,
                "reason": task["eval_reason"],
                "counts": summary.get("counts", {}),
                "total_votes": summary.get("total_votes", 0),
                "strict_majority": summary.get("strict_majority", False),
                "chosen_run_number": summary.get("chosen_run_number", 0),
            }
        )
        summary["applied"] = True
        summary["applied_task_id"] = task_id
        applied += 1

    state["last_updated"] = datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")
    return applied


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _profile_path(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.profile:
        return args.profile
    state = json.loads(args.state_json.read_text())
    profile_data = state.get("profile_data") or {}
    if not isinstance(profile_data, dict) or not profile_data:
        return None
    profile_path = output_dir / "profile_from_state.toml"
    lines = ["[profile]"]
    for key, value in sorted(profile_data.items()):
        if isinstance(value, str | int | float | bool):
            lines.append(f"{key} = {_toml_value(value)}")
    profile_path.write_text("\n".join(lines) + "\n")
    return profile_path


def _default_output_dir(state_json: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return state_json.parent / f"evaluate-majority-{timestamp}"


def _evaluate_command(args: argparse.Namespace, state_json: Path, profile_path: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "craft_taskgen.pipeline",
        "--resume",
        str(state_json),
        "--from-step",
        "evaluate",
        "--concurrency",
        str(args.concurrency),
    ]
    if profile_path is not None:
        cmd.extend(["--profile", str(profile_path)])
    return cmd


def _run_streamed(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"evaluate run failed with exit code {rc}; see {log_path}")


def run_trials(args: argparse.Namespace) -> tuple[list[list[dict[str, Any]]], Path]:
    output_dir = args.output_dir or _default_output_dir(args.state_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = _profile_path(args, output_dir)

    run_rows: list[list[dict[str, Any]]] = []
    selected_task_ids: list[str] | None = None
    for run_number in range(1, args.runs + 1):
        run_label = f"run_{run_number:02d}"
        trial_state = output_dir / f"{run_label}_state.json"
        trial_log = output_dir / f"{run_label}.log"
        trial_rows_path = output_dir / f"{run_label}.jsonl"
        selected = prepare_trial_state(args.state_json, trial_state, args)
        if selected_task_ids is None:
            selected_task_ids = selected
        elif selected != selected_task_ids:
            raise RuntimeError("selected task set changed between runs")

        cmd = _evaluate_command(args, trial_state, profile_path)
        print(f"\n=== Evaluate {run_number}/{args.runs}: {' '.join(cmd)} ===")
        _run_streamed(cmd, trial_log)
        rows = extract_trial_rows(trial_state, selected, run_number)
        for row in rows:
            row["run_state"] = str(trial_state)
        _jsonl_write(trial_rows_path, rows)
        run_rows.append(rows)

    return run_rows, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_json", type=Path, help="Pipeline state.json to re-evaluate")
    parser.add_argument("-n", "--runs", type=int, default=5, help="Number of evaluate trials to run")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrency passed to each evaluate trial",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Optional pipeline profile TOML")
    parser.add_argument("--repo", default="", help="Filter by short repo name")
    parser.add_argument("--instance-id", default="", help="Filter by source_task_id / instance_id")
    parser.add_argument("--task-id", action="append", default=[], help="Filter by task_id; may be repeated")
    parser.add_argument("--limit", type=int, default=0, help="Limit selected tasks after filters")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for per-run artifacts")
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=None,
        help="Majority summary JSONL path (default: <output-dir>/majority.jsonl)",
    )
    parser.add_argument(
        "--tie-policy",
        choices=["keep-current", "first", "priority"],
        default="keep-current",
        help="How to handle tied top vote counts",
    )
    parser.add_argument(
        "--allow-plurality",
        action="store_true",
        help="Apply the top non-tied verdict even if it is not a strict majority",
    )
    parser.add_argument(
        "--force-stage",
        action="store_true",
        help="Update task stage even for tasks that have progressed beyond evaluate-era stages",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Only write trial outputs and majority summary; do not update state_json",
    )
    args = parser.parse_args()

    if args.runs < 1:
        print("ERROR: --runs must be >= 1", file=sys.stderr)
        return 1

    try:
        run_rows, output_dir = run_trials(args)
        summaries = aggregate_votes(
            run_rows,
            tie_policy=args.tie_policy,
            require_strict_majority=not args.allow_plurality,
        )
        summary_jsonl = args.summary_jsonl or output_dir / "majority.jsonl"
        applied = 0
        if not args.no_apply:
            backup_path = output_dir / f"{args.state_json.name}.before-evaluate-majority"
            applied = apply_majority_to_state(
                args.state_json,
                summaries,
                backup_path=backup_path,
                force_stage=args.force_stage,
            )
        _jsonl_write(summary_jsonl, summaries)
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    unresolved = sum(1 for row in summaries if not row.get("applied"))
    print(f"\nWrote majority summary: {summary_jsonl}")
    if args.no_apply:
        print("State update skipped (--no-apply)")
    else:
        print(f"Applied majority results to {args.state_json}: {applied} task(s)")
        backup_path = output_dir / (args.state_json.name + ".before-evaluate-majority")
        print(f"Backup before final apply: {backup_path}")
    if unresolved:
        print(f"Unapplied/tied/no-majority tasks: {unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
