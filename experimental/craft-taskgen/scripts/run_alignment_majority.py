#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run SWE-bench alignment multiple times and apply per-instance majority votes."""

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

VERDICT_PRIORITY = [
    "ok",
    "narrow_tests",
    "vague",
    "misaligned",
    "leaked",
    "skipped",
    "context_error",
    "judge_error",
    "unknown",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSONL row: {e}") from e
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _row_key(row: dict[str, Any]) -> str:
    for key in ("source_task_id", "task_id", "commit_sha"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"alignment row has no source_task_id/task_id/commit_sha: {row}")


def _vote_label(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").strip()
    if status == "ok":
        return str(row.get("verdict") or "ok").strip() or "ok"
    return status or "unknown"


def _vote_reason(row: dict[str, Any]) -> str:
    if str(row.get("status") or "") == "ok":
        return str(row.get("reason") or "")
    return str(row.get("error") or "")


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
        for label in VERDICT_PRIORITY:
            if label in tied_labels:
                return next(row for row in votes if _vote_label(row) == label)
        return next(row for row in votes if _vote_label(row) in tied_labels)
    raise ValueError(f"unknown tie policy: {tie_policy}")


def aggregate_votes(
    run_rows: list[list[dict[str, Any]]],
    *,
    tie_policy: str = "keep-current",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    key_order: dict[str, int] = {}

    for run_number, rows in enumerate(run_rows, start=1):
        for row in rows:
            annotated = dict(row)
            annotated["run_number"] = run_number
            key = _row_key(annotated)
            key_order.setdefault(key, len(key_order))
            grouped.setdefault(key, []).append(annotated)

    summaries: list[dict[str, Any]] = []
    for key, votes in grouped.items():
        counts = Counter(_vote_label(row) for row in votes)
        top_count = max(counts.values())
        top_labels = {label for label, count in counts.items() if count == top_count}
        tie = len(top_labels) > 1
        if tie:
            chosen = _choose_from_tie(votes, top_labels, tie_policy)
        else:
            chosen = next(row for row in votes if _vote_label(row) in top_labels)

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
            "strict_majority": top_count > len(votes) / 2,
            "tie": tie,
            "applied": False,
            "votes": votes,
        }
        if chosen is not None:
            label = _vote_label(chosen)
            summary.update(
                {
                    "winner": label,
                    "alignment_verdict": label,
                    "alignment_reason": _vote_reason(chosen),
                    "chosen_run_number": chosen.get("run_number", 0),
                }
            )
        else:
            summary.update(
                {
                    "winner": "",
                    "alignment_verdict": "",
                    "alignment_reason": "",
                    "chosen_run_number": 0,
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


def _attempt_from_vote(row: dict[str, Any], *, selected: bool) -> dict[str, Any]:
    attempt = {
        "attempt": row.get("run_number", 0),
        "run_number": row.get("run_number", 0),
        "status": row.get("status", ""),
        "verdict": _vote_label(row),
        "reason": _vote_reason(row),
        "selected_by_majority": selected,
    }
    for key in (
        "v4_audit",
        "leakage_evidence",
        "tokens_in",
        "tokens_out",
        "latency_s",
        "model",
        "reference_test_count",
        "reference_test_paths",
    ):
        if key in row:
            attempt[key] = row[key]
    return attempt


def apply_majority_to_state(
    state_path: Path,
    summaries: list[dict[str, Any]],
    *,
    backup_path: Path | None = None,
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
        if not summary.get("winner"):
            continue
        task_id = task_index.get(str(summary.get("task_id") or ""))
        task_id = task_id or task_index.get(str(summary.get("source_task_id") or ""))
        task_id = task_id or task_index.get(str(summary.get("commit_sha") or ""))
        task_id = task_id or task_index.get(str(summary.get("key") or ""))
        if not task_id:
            continue

        task = tasks[task_id]
        winner = str(summary["winner"])
        chosen_run = summary.get("chosen_run_number")
        chosen_vote = next(
            (
                row
                for row in summary["votes"]
                if row.get("run_number") == chosen_run and _vote_label(row) == winner
            ),
            None,
        )
        task["alignment_verdict"] = winner
        task["alignment_reason"] = str(summary.get("alignment_reason") or "")
        task["alignment_v4_audit"] = chosen_vote.get("v4_audit", {}) if chosen_vote else {}
        task["alignment_attempts"] = []
        for row in summary["votes"]:
            selected = _vote_label(row) == winner
            task["alignment_attempts"].append(_attempt_from_vote(row, selected=selected))
        summary["applied"] = True
        summary["applied_task_id"] = task_id
        applied += 1

    state["last_updated"] = datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")
    return applied


def _default_output_dir(state_json: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return state_json.parent / f"alignment-majority-{timestamp}"


def _alignment_command(args: argparse.Namespace, state_json: Path, output_jsonl: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "craft_taskgen.swebench_alignment",
        "--state-json",
        str(state_json),
        "--repos-dir",
        str(args.repos_dir),
        "--output",
        str(output_jsonl),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.repo:
        cmd.extend(["--repo", args.repo])
    if args.instance_id:
        cmd.extend(["--instance-id", args.instance_id])
    if args.include_interface:
        cmd.append("--include-interface")
    if args.include_requirements:
        cmd.append("--include-requirements")
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
        raise RuntimeError(f"alignment run failed with exit code {rc}; see {log_path}")


def run_trials(args: argparse.Namespace) -> tuple[list[list[dict[str, Any]]], Path]:
    output_dir = args.output_dir or _default_output_dir(args.state_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[list[dict[str, Any]]] = []
    for run_number in range(1, args.runs + 1):
        run_label = f"run_{run_number:02d}"
        trial_state = output_dir / f"{run_label}_state.json"
        trial_output = output_dir / f"{run_label}.jsonl"
        trial_log = output_dir / f"{run_label}.log"
        shutil.copy2(args.state_json, trial_state)
        cmd = _alignment_command(args, trial_state, trial_output)
        print(f"\n=== Alignment {run_number}/{args.runs}: {' '.join(cmd)} ===")
        _run_streamed(cmd, trial_log)
        rows = _read_jsonl(trial_output)
        for row in rows:
            row["run_output"] = str(trial_output)
        run_rows.append(rows)

    return run_rows, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "state_json",
        type=Path,
        help="Pipeline state.json to read and update with final votes",
    )
    parser.add_argument("-n", "--runs", type=int, default=3, help="Number of alignment trials to run")
    parser.add_argument("--repos-dir", type=Path, default=Path("repos"), help="Local git clone directory")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrency passed to each alignment trial",
    )
    parser.add_argument("--model", default="", help="Optional alignment model override")
    parser.add_argument("--limit", type=int, default=0, help="Maximum candidates to process per trial")
    parser.add_argument("--repo", default="", help="Filter by short repo name")
    parser.add_argument("--instance-id", default="", help="Filter by source_task_id")
    parser.add_argument(
        "--include-interface",
        action="store_true",
        help="Pass through to craft-taskgen-swebench-align",
    )
    parser.add_argument(
        "--include-requirements",
        action="store_true",
        help="Pass through to craft-taskgen-swebench-align",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-run JSONL/log/state artifacts",
    )
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
        summaries = aggregate_votes(run_rows, tie_policy=args.tie_policy)
        summary_jsonl = args.summary_jsonl or output_dir / "majority.jsonl"
        applied = 0
        if not args.no_apply:
            backup_path = output_dir / f"{args.state_json.name}.before-majority"
            applied = apply_majority_to_state(args.state_json, summaries, backup_path=backup_path)
        _write_jsonl(summary_jsonl, summaries)
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    unresolved = sum(1 for row in summaries if not row.get("applied"))
    print(f"\nWrote majority summary: {summary_jsonl}")
    if args.no_apply:
        print("State update skipped (--no-apply)")
    else:
        print(f"Applied majority results to {args.state_json}: {applied} instance(s)")
        print(f"Backup before final apply: {output_dir / (args.state_json.name + '.before-majority')}")
    if unresolved:
        print(f"Unapplied/tied/missing instances: {unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
