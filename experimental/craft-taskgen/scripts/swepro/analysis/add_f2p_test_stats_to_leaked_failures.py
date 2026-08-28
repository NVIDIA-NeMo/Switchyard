#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add verifier F2P pass counts to leaked-agent-failure findings.

Harbor's verifier/output.json records the required tests for each task with
per-test statuses. In this dataset those required tests are the F2P/gold tests
checked by the verifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_CSV = Path("docs/analyses/data/swebench-pro/findings/leaked_agent_failures.csv")
DEFAULT_MANIFEST = Path("docs/analyses/data/swebench-pro/runs/combined_non_error/combined_manifest.json")
DEFAULT_RUNS_DIR = Path("docs/analyses/data/swebench-pro/runs/combined_non_error")

PASSED_COL = "agent_f2p_tests_passed"
TOTAL_COL = "agent_f2p_tests_total"


def load_task_to_trial(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open() as f:
        manifest = json.load(f)
    return {trial["task_id"]: trial["trial_name"] for trial in manifest["selected_trials"]}


def f2p_counts_from_stdout(runs_dir: Path, trial_name: str) -> tuple[int, int] | None:
    stdout_path = runs_dir / trial_name / "verifier" / "test-stdout.txt"
    if not stdout_path.exists():
        return None

    text = stdout_path.read_text(errors="replace")
    total_match = re.search(r"^Required tests:\s+(\d+)$", text, re.MULTILINE)
    passed_match = re.search(r"^Required tests that passed:\s+(\d+)$", text, re.MULTILINE)
    if total_match is None or passed_match is None:
        return None

    return int(passed_match.group(1)), int(total_match.group(1))


def f2p_counts_from_output_json(runs_dir: Path, trial_name: str) -> tuple[int, int]:
    output_path = runs_dir / trial_name / "verifier" / "output.json"
    with output_path.open() as f:
        verifier_output = json.load(f)

    statuses = Counter(test.get("status") for test in verifier_output.get("tests", []))
    total = sum(statuses.values())
    passed = statuses["PASSED"]
    return passed, total


def f2p_counts(runs_dir: Path, trial_name: str) -> tuple[int, int]:
    stdout_counts = f2p_counts_from_stdout(runs_dir, trial_name)
    if stdout_counts is not None:
        return stdout_counts
    return f2p_counts_from_output_json(runs_dir, trial_name)


def ensure_column(header: list[str], name: str) -> int:
    if name in header:
        return header.index(name)
    header.append(name)
    return len(header) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    task_to_trial = load_task_to_trial(args.manifest)

    with args.csv.open(newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {args.csv}")

    header = rows[0]
    task_id_col = header.index("swebench_instance_id")
    passed_col = ensure_column(header, PASSED_COL)
    total_col = ensure_column(header, TOTAL_COL)

    matched = 0
    missing = 0
    for row in rows[1:]:
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))

        trial_name = task_to_trial.get(row[task_id_col])
        if trial_name is None:
            row[passed_col] = ""
            row[total_col] = ""
            missing += 1
            continue

        output_path = args.runs_dir / trial_name / "verifier" / "output.json"
        if not output_path.exists():
            row[passed_col] = ""
            row[total_col] = ""
            missing += 1
            continue

        passed, total = f2p_counts(args.runs_dir, trial_name)
        row[passed_col] = str(passed)
        row[total_col] = str(total)
        matched += 1

    with args.csv.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Updated {args.csv}")
    print(f"Rows matched: {matched}/{len(rows) - 1}")
    print(f"Rows missing verifier output: {missing}")
    print(f"Added/updated columns: {PASSED_COL}, {TOTAL_COL}")


if __name__ == "__main__":
    main()
