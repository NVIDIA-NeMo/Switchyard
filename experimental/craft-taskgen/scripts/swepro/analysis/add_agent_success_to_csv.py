#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add per-task agent success to the CRAFT analysis CSV.

The CSV uses full SWE-bench instance ids. The aggregate result.json stores
reward buckets by trial directory name, and combined_manifest.json maps those
trial names back to the full task ids.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_CSV = Path("docs/analyses/data/swebench-pro/swebench-pro-craft-analysis.csv")
DEFAULT_RESULTS = Path("tmp/swebench-results/combined_non_error/result.json")
DEFAULT_MANIFEST = Path("tmp/swebench-results/combined_non_error/combined_manifest.json")
SUCCESS_COL = "agent_success"


def load_trial_success(result_path: Path) -> dict[str, bool]:
    with result_path.open() as f:
        result = json.load(f)

    evals = result["stats"]["evals"]
    if len(evals) != 1:
        raise ValueError(f"Expected exactly one eval entry, found {len(evals)}")

    eval_result = next(iter(evals.values()))
    reward_buckets = eval_result["reward_stats"]["reward"]

    trial_success: dict[str, bool] = {}
    for reward_text, trial_names in reward_buckets.items():
        success = float(reward_text) == 1.0
        for trial_name in trial_names:
            if trial_name in trial_success:
                raise ValueError(f"Duplicate trial in reward stats: {trial_name}")
            trial_success[trial_name] = success

    return trial_success


def load_task_success(manifest_path: Path, trial_success: dict[str, bool]) -> dict[str, bool]:
    with manifest_path.open() as f:
        manifest = json.load(f)

    task_success: dict[str, bool] = {}
    for trial in manifest["selected_trials"]:
        task_id = trial["task_id"]
        trial_name = trial["trial_name"]
        if trial_name not in trial_success:
            raise ValueError(f"Manifest trial missing from result reward stats: {trial_name}")
        if task_id in task_success:
            raise ValueError(f"Duplicate task_id in manifest: {task_id}")
        task_success[task_id] = trial_success[trial_name]

    return task_success


def find_task_id_column(header: list[str]) -> int:
    for col_name in ("swebench_instance_id", "new_swebench_instance_id"):
        if col_name in header:
            return header.index(col_name)
    raise ValueError("CSV has neither swebench_instance_id nor new_swebench_instance_id")


def add_success_column(csv_path: Path, task_success: dict[str, bool]) -> tuple[int, int, int]:
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    header = rows[0]
    task_id_col = find_task_id_column(header)

    if SUCCESS_COL in header:
        success_col = header.index(SUCCESS_COL)
    else:
        header.append(SUCCESS_COL)
        success_col = len(header) - 1

    matched = 0
    missing = 0
    for row in rows[1:]:
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))

        success = task_success.get(row[task_id_col])
        if success is None:
            row[success_col] = ""
            missing += 1
        else:
            row[success_col] = "TRUE" if success else "FALSE"
            matched += 1

    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    return matched, missing, len(rows) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    trial_success = load_trial_success(args.results)
    task_success = load_task_success(args.manifest, trial_success)
    matched, missing, total = add_success_column(args.csv, task_success)

    print(f"Updated {args.csv}")
    print(f"Rows matched: {matched}/{total}")
    print(f"Rows missing result: {missing}")


if __name__ == "__main__":
    main()
