# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter a recent run's state.json to rejected tasks and emit a flat CSV
in the shape ``scripts/calibrate-alignment.py`` expects.

Usage:
    # Alignment-rejected tasks (default)
    uv run python scripts/state_to_rejected_csv.py path/to/state.json \\
        --output candidates/rejected_cohort.csv

    # Eval-rejected tasks (rejected before they reached build+align)
    uv run python scripts/state_to_rejected_csv.py path/to/state.json \\
        --output candidates/eval_rejected.csv \\
        --filter eval

Filters:
- ``--filter alignment`` (default): ``stage == "rejected"`` AND
  ``alignment_verdict`` in ``--verdicts`` list.
- ``--filter eval``: ``eval_verdict == "reject"`` (no alignment ever ran).
- All filters require non-empty repo + commit_sha (to fetch git context).

Output columns match the rerun-accepts-v2 calibration_input.csv shape:
    task_id, repo, commit_sha, base_sha, subject, pr_url, instruction_md,
    in_v1a, in_mother_run_41, accepted_in_runs, n_runs_accepted

(``in_v1a`` etc. are filled with empty strings — there's no manifest to
join against; preserved for column-shape compatibility with the calibrate
script's CSV reader.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_VERDICTS = "leaked,narrow_tests,vague,misaligned"

OUTPUT_FIELDS = [
    "task_id",
    "repo",
    "commit_sha",
    "base_sha",
    "subject",
    "pr_url",
    "instruction_md",
    "in_v1a",
    "in_mother_run_41",
    "accepted_in_runs",
    "n_runs_accepted",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("state_json", type=Path, help="path to a state.json file")
    parser.add_argument("--output", type=Path, default=Path("candidates/rejected_cohort.csv"))
    parser.add_argument("--max-rows", type=int, default=0, help="cap output rows (0 = no cap)")
    parser.add_argument(
        "--filter",
        choices=["alignment", "eval"],
        default="alignment",
        help="which rejection stage to filter by (default: alignment)",
    )
    parser.add_argument(
        "--verdicts",
        default=DEFAULT_VERDICTS,
        help=f"comma-separated alignment_verdict values to include "
        f"(only used when --filter=alignment; default {DEFAULT_VERDICTS})",
    )
    args = parser.parse_args()

    if not args.state_json.is_file():
        print(f"ERROR: not a file: {args.state_json}", file=sys.stderr)
        return 1

    target_verdicts = {v.strip() for v in args.verdicts.split(",") if v.strip()}
    if args.filter == "alignment":
        print(
            f"Filter: stage=rejected AND alignment_verdict in {sorted(target_verdicts)}",
            file=sys.stderr,
        )
    else:
        print("Filter: eval_verdict='reject' (rejected at evaluate step)", file=sys.stderr)

    with args.state_json.open() as f:
        state = json.load(f)
    tasks = state.get("tasks", {}) or {}
    print(f"Loaded {len(tasks)} tasks from {args.state_json}", file=sys.stderr)

    out_rows: list[dict] = []
    for tid, task in tasks.items():
        if not task.get("repo") or not task.get("commit_sha"):
            continue
        if args.filter == "alignment":
            if task.get("stage") != "rejected":
                continue
            if task.get("alignment_verdict") not in target_verdicts:
                continue
        else:  # eval
            if task.get("eval_verdict") != "reject":
                continue
        out_rows.append(
            {
                "task_id": tid,
                "repo": task.get("repo", ""),
                "commit_sha": task.get("commit_sha", ""),
                "base_sha": task.get("base_sha", ""),
                "subject": task.get("description", ""),
                "pr_url": "",
                "instruction_md": "",
                "in_v1a": "",
                "in_mother_run_41": "",
                "accepted_in_runs": "",
                "n_runs_accepted": "",
            }
        )

    if args.max_rows > 0 and len(out_rows) > args.max_rows:
        out_rows = out_rows[: args.max_rows]
        print(f"Truncated to first {args.max_rows} rows", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
