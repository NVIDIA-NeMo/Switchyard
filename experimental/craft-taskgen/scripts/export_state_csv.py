#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a pipeline state.json to an Excel-friendly CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def _load_state(path: Path, retries: int = 8, delay_s: float = 0.4) -> dict:
    """Retry reads because state.json may be updated while the pipeline runs."""
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with path.open() as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(delay_s)
    raise RuntimeError(f"Could not read a stable JSON snapshot from {path}: {last_error}") from last_error


def _clean_text(value: object, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _join_list(value: object, max_items: int = 12) -> str:
    if not isinstance(value, list):
        return ""
    items = [str(v) for v in value[:max_items]]
    if len(value) > max_items:
        items.append(f"...(+{len(value) - max_items} more)")
    return "; ".join(items)


def _row_for_task(task_id: str, task: dict, *, max_text: int) -> dict[str, object]:
    candidate = task.get("candidate_data") or {}
    source_meta = candidate.get("source_metadata") or {}

    return {
        "task_id": task_id,
        "repo": task.get("repo", ""),
        "commit_sha": task.get("commit_sha", ""),
        "base_sha": task.get("base_sha", ""),
        "merge_base_sha": task.get("merge_base_sha", ""),
        "title": _clean_text(task.get("description") or candidate.get("subject", ""), max_text),
        "stage": task.get("stage", ""),
        "eval_verdict": task.get("eval_verdict", ""),
        "eval_reason": _clean_text(task.get("eval_reason", ""), max_text),
        "hardness_verdict": task.get("hardness_verdict", ""),
        "hardness_band": task.get("hardness_band", ""),
        "needs_human_review": task.get("needs_human_review", False),
        "human_review_reason": _clean_text(task.get("human_review_reason", ""), max_text),
        "oracle_resolved": task.get("oracle_resolved", False),
        "oracle_flagged": task.get("oracle_flagged", False),
        "oracle_flag_reason": _clean_text(task.get("oracle_flag_reason", ""), max_text),
        "oracle_f2p_score": task.get("oracle_f2p_score", ""),
        "oracle_p2p_score": task.get("oracle_p2p_score", ""),
        "opus_score": task.get("opus_score", ""),
        "haiku_score": task.get("haiku_score", ""),
        "instruction_words": task.get("instruction_words", ""),
        "task_dir": task.get("task_dir", ""),
        "swebench_source_task_id": candidate.get("source_task_id", ""),
        "swebench_instance_id": source_meta.get("instance_id", ""),
        "author": candidate.get("author", ""),
        "date": candidate.get("date", ""),
        "score": candidate.get("score", ""),
        "has_test_patch": candidate.get("has_test_patch", ""),
        "is_multi_file": candidate.get("is_multi_file", ""),
        "is_multi_package": candidate.get("is_multi_package", ""),
        "is_refactoring": candidate.get("is_refactoring", ""),
        "packages_touched": candidate.get("packages_touched", ""),
        "package_names": _join_list(candidate.get("package_names", [])),
        "source_files": _join_list(candidate.get("source_files", [])),
        "test_files": _join_list(candidate.get("test_files", [])),
        "other_files": _join_list(candidate.get("other_files", [])),
        "source_lines_changed": candidate.get("source_lines_changed", ""),
        "test_lines_changed": candidate.get("test_lines_changed", ""),
        "f2p_tests": _join_list(task.get("f2p_tests", [])),
        "p2p_tests": _join_list(task.get("p2p_tests", [])),
        "summary": _clean_text(task.get("summary", ""), max_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export state.json to a spreadsheet-friendly CSV.")
    parser.add_argument("input_state", type=Path, help="Source pipeline state.json")
    parser.add_argument("output_csv", type=Path, help="Destination CSV path")
    parser.add_argument(
        "--max-text",
        type=int,
        default=500,
        help="Maximum length for free-text fields like title/reasons/summary",
    )
    args = parser.parse_args()

    state = _load_state(args.input_state)
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{args.input_state}: expected top-level 'tasks' dict")

    rows = [_row_for_task(task_id, task, max_text=args.max_text) for task_id, task in tasks.items()]
    rows.sort(key=lambda row: (str(row["repo"]), str(row["stage"]), str(row["task_id"])))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
