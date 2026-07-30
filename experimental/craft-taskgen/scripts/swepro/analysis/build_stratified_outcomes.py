#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an enriched task outcome table and stratified pass-rate report."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_CSV = Path("docs/analyses/data/swebench-pro/swebench-pro-craft-analysis.csv")
DEFAULT_MANIFEST = Path("docs/analyses/data/swebench-pro/runs/combined_non_error/combined_manifest.json")
DEFAULT_RUNS_DIR = Path("docs/analyses/data/swebench-pro/runs/combined_non_error")
DEFAULT_OUTPUT_CSV = Path("docs/analyses/data/swebench-pro/findings/task_outcomes_enriched.csv")
DEFAULT_REPORT = Path("docs/analyses/data/swebench-pro/findings/stratified_pass_rate_report.md")

ALIGNMENT_COL = "new_alignment_verdict"
ALIGNMENT_REASON_COL = "new_alignment_reason"


def populated_column_index(rows: list[list[str]], header: list[str], name: str) -> int:
    candidates = [idx for idx, col_name in enumerate(header) if col_name == name]
    if not candidates:
        raise ValueError(f"Missing required column: {name}")
    return max(
        candidates,
        key=lambda idx: sum(1 for row in rows[1:] if len(row) > idx and row[idx].strip()),
    )


def col(header: list[str], name: str) -> int:
    if name not in header:
        raise ValueError(f"Missing required column: {name}")
    return header.index(name)


def split_paths(value: str) -> set[str]:
    return {part.strip() for part in value.split(";") if part.strip()}


def normalize_edit_path(path: str) -> str:
    if path.startswith("/app/"):
        return path[len("/app/") :]
    return path


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


def f2p_total_bucket(total: int) -> str:
    if total <= 5:
        return "01_1-5"
    if total <= 20:
        return "02_6-20"
    if total <= 100:
        return "03_21-100"
    return "04_101+"


def f2p_result_bucket(passed: int, total: int) -> str:
    if total == 0:
        return "no_tests"
    if passed == total:
        return "all_passed"
    if passed == 0:
        return "zero_passed"
    if total - passed <= 3:
        return "near_miss"
    return "partial"


def trajectory_metrics(runs_dir: Path, trial_name: str) -> dict[str, object]:
    trajectory_path = runs_dir / trial_name / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return {
            "edit_count": 0,
            "edited_files": [],
            "bash_count": 0,
            "ran_pytest": False,
            "ran_tests": False,
        }

    with trajectory_path.open() as f:
        trajectory = json.load(f)

    edited_files: list[str] = []
    bash_commands: list[str] = []
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls") or []:
            function_name = call.get("function_name")
            args = call.get("arguments") or {}
            file_path = args.get("file_path")
            if function_name in {"Edit", "Write"} and file_path:
                edited_files.append(normalize_edit_path(file_path))
            if function_name == "Bash" and args.get("command"):
                bash_commands.append(args["command"])

    joined_commands = "\n".join(bash_commands).lower()
    return {
        "edit_count": len(edited_files),
        "edited_files": sorted(set(edited_files)),
        "bash_count": len(bash_commands),
        "ran_pytest": "pytest" in joined_commands,
        "ran_tests": any(token in joined_commands for token in ("pytest", "tox", "ansible-test", "test_")),
    }


def pct(successes: int, total: int) -> str:
    if total == 0:
        return ""
    return f"{successes / total * 100:.1f}%"


def pct_float(successes: float, total: int) -> str:
    if total == 0:
        return ""
    return f"{successes / total * 100:.1f}%"


def as_bool(value: str) -> bool:
    return value.strip().upper() == "TRUE"


def group_counts(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups[key][0] += 1
        groups[key][1] += 1 if row["agent_success"] else 0

    result: list[dict[str, object]] = []
    for key, (total, successes) in sorted(groups.items()):
        result.append(
            {
                **dict(zip(keys, key)),
                "n": total,
                "passed": successes,
                "pass_rate": pct(successes, total),
            }
        )
    return result


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def matched_ok_vs_leaked(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    cell_groups: dict[tuple[object, ...], dict[object, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    cell_keys = ("repo", "new_eval_verdict", "f2p_total_bucket")
    for row in rows:
        if row["alignment_verdict"] not in {"ok", "leaked"}:
            continue
        cell = tuple(row[k] for k in cell_keys)
        slot = cell_groups[cell][row["alignment_verdict"]]
        slot[0] += 1
        slot[1] += 1 if row["agent_success"] else 0

    matched_rows: list[dict[str, object]] = []
    leaked_actual_passes = 0
    leaked_total = 0
    leaked_expected_passes_at_ok_rate = 0.0
    for cell, by_alignment in sorted(cell_groups.items()):
        if "ok" not in by_alignment or "leaked" not in by_alignment:
            continue
        ok_n, ok_pass = by_alignment["ok"]
        leaked_n, leaked_pass = by_alignment["leaked"]
        ok_rate = ok_pass / ok_n if ok_n else 0.0
        leaked_rate = leaked_pass / leaked_n if leaked_n else 0.0
        leaked_actual_passes += leaked_pass
        leaked_total += leaked_n
        leaked_expected_passes_at_ok_rate += leaked_n * ok_rate
        matched_rows.append(
            {
                "repo": cell[0],
                "new_eval_verdict": cell[1],
                "f2p_total_bucket": cell[2],
                "ok": f"{ok_pass}/{ok_n}",
                "ok_rate": pct(ok_pass, ok_n),
                "leaked": f"{leaked_pass}/{leaked_n}",
                "leaked_rate": pct(leaked_pass, leaked_n),
                "gap_pp": f"{(leaked_rate - ok_rate) * 100:.1f}",
            }
        )

    summary = {
        "matched_leaked_passed": leaked_actual_passes,
        "matched_leaked_total": leaked_total,
        "matched_leaked_rate": pct(leaked_actual_passes, leaked_total),
        "expected_leaked_passed_at_ok_rate": f"{leaked_expected_passes_at_ok_rate:.1f}",
        "expected_leaked_rate_at_ok_rate": pct_float(leaked_expected_passes_at_ok_rate, leaked_total),
    }
    return matched_rows, summary


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    with args.csv.open(newline="") as f:
        csv_rows = list(csv.reader(f))

    if not csv_rows:
        raise ValueError(f"CSV is empty: {args.csv}")

    header = csv_rows[0]
    alignment_idx = populated_column_index(csv_rows, header, ALIGNMENT_COL)
    alignment_reason_idx = populated_column_index(csv_rows, header, ALIGNMENT_REASON_COL)

    indices = {
        "task_id": col(header, "task_id"),
        "repo": col(header, "repo"),
        "github_pr": col(header, "github_pr"),
        "title": col(header, "title"),
        "new_eval_verdict": col(header, "new_eval_verdict"),
        "new_eval_reason": col(header, "new_eval_reason"),
        "swebench_instance_id": col(header, "swebench_instance_id"),
        "source_files": col(header, "source_files"),
        "test_files": col(header, "test_files"),
        "other_files": col(header, "other_files"),
        "source_lines_changed": col(header, "source_lines_changed"),
        "test_lines_changed": col(header, "test_lines_changed"),
        "has_test_patch": col(header, "has_test_patch"),
        "is_multi_file": col(header, "is_multi_file"),
        "is_multi_package": col(header, "is_multi_package"),
        "is_refactoring": col(header, "is_refactoring"),
        "agent_success": col(header, "agent_success"),
    }

    task_to_trial = load_task_to_trial(args.manifest)
    output_rows: list[dict[str, object]] = []
    for raw in csv_rows[1:]:
        swebench_id = raw[indices["swebench_instance_id"]]
        trial_name = task_to_trial.get(swebench_id, "")
        passed, total = f2p_counts(args.runs_dir, trial_name) if trial_name else (0, 0)
        trajectory = (
            trajectory_metrics(args.runs_dir, trial_name)
            if trial_name
            else trajectory_metrics(args.runs_dir, "")
        )

        source_files = split_paths(raw[indices["source_files"]])
        test_files = split_paths(raw[indices["test_files"]])
        other_files = split_paths(raw[indices["other_files"]])
        edited_files = set(trajectory["edited_files"])
        expected_files = source_files | test_files | other_files

        row = {
            "task_id": raw[indices["task_id"]],
            "repo": raw[indices["repo"]],
            "github_pr": raw[indices["github_pr"]],
            "title": raw[indices["title"]],
            "new_eval_verdict": raw[indices["new_eval_verdict"]],
            "alignment_verdict": raw[alignment_idx],
            "alignment_reason": raw[alignment_reason_idx],
            "swebench_instance_id": swebench_id,
            "trial_name": trial_name,
            "agent_success": as_bool(raw[indices["agent_success"]]),
            "agent_f2p_tests_passed": passed,
            "agent_f2p_tests_total": total,
            "agent_f2p_pass_rate": f"{passed / total:.4f}" if total else "",
            "f2p_total_bucket": f2p_total_bucket(total),
            "f2p_result_bucket": f2p_result_bucket(passed, total),
            "source_lines_changed": raw[indices["source_lines_changed"]],
            "test_lines_changed": raw[indices["test_lines_changed"]],
            "has_test_patch": raw[indices["has_test_patch"]],
            "is_multi_file": raw[indices["is_multi_file"]],
            "is_multi_package": raw[indices["is_multi_package"]],
            "is_refactoring": raw[indices["is_refactoring"]],
            "source_file_count": len(source_files),
            "test_file_count": len(test_files),
            "edit_count": trajectory["edit_count"],
            "edited_file_count": len(edited_files),
            "edited_expected_file_count": len(edited_files & expected_files),
            "edited_source_file_count": len(edited_files & source_files),
            "edited_test_file_count": len(edited_files & test_files),
            "bash_count": trajectory["bash_count"],
            "agent_ran_pytest": trajectory["ran_pytest"],
            "agent_ran_tests": trajectory["ran_tests"],
            "edited_files": "; ".join(sorted(edited_files)),
        }
        output_rows.append(row)

    return output_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    alignment_rows = group_counts(rows, ("alignment_verdict",))
    alignment_eval_rows = group_counts(rows, ("alignment_verdict", "new_eval_verdict"))
    alignment_repo_rows = group_counts(rows, ("alignment_verdict", "repo"))
    alignment_f2p_rows = group_counts(rows, ("alignment_verdict", "f2p_total_bucket"))
    detailed_rows = group_counts(rows, ("alignment_verdict", "new_eval_verdict", "repo", "f2p_total_bucket"))
    failure_shape_rows = group_counts(rows, ("alignment_verdict", "f2p_result_bucket"))
    matched_rows, matched_summary = matched_ok_vs_leaked(rows)
    by_alignment = {row["alignment_verdict"]: row for row in alignment_rows}
    leaked_row = by_alignment.get("leaked", {"passed": 0, "n": 0})
    ok_row = by_alignment.get("ok", {"passed": 0, "n": 0})
    leaked_rate = leaked_row["passed"] / leaked_row["n"] if leaked_row["n"] else 0.0
    ok_rate = ok_row["passed"] / ok_row["n"] if ok_row["n"] else 0.0
    leaked_trailing_gap = (ok_rate - leaked_rate) * 100

    zero_leaked = [
        {
            "task_id": row["task_id"],
            "repo": row["repo"],
            "passed_total": f"{row['agent_f2p_tests_passed']}/{row['agent_f2p_tests_total']}",
            "title": str(row["title"])[:80],
        }
        for row in rows
        if row["alignment_verdict"] == "leaked" and row["f2p_result_bucket"] == "zero_passed"
    ]

    lines: list[str] = [
        "# Stratified Pass-Rate Report",
        "",
        f"Rows analyzed: {len(rows)}",
        "",
        "## Key Observations",
        "",
        f"- Overall, `leaked` trails `ok` by {leaked_trailing_gap:.1f} percentage points "
        f"({leaked_row['passed']}/{leaked_row['n']} vs {ok_row['passed']}/{ok_row['n']}).",
        "- The gap is highly confounded by eval verdict, repo mix, and F2P test-count bucket.",
        f"- In matched cells with the same repo, eval verdict, and F2P bucket, leaked tasks pass at "
        f"{matched_summary['matched_leaked_rate']} versus an expected "
        f"{matched_summary['expected_leaked_rate_at_ok_rate']} if they followed the OK-cell rates.",
        "- That matched comparison suggests the raw leaked-vs-ok gap is mostly compositional in this run.",
        "- The worst leaked failures are concentrated in Ansible zero-pass cases; see the zero-pass deep dive.",
        "",
        "## Overall By Alignment",
        "",
        *markdown_table(alignment_rows, ["alignment_verdict", "passed", "n", "pass_rate"]),
        "",
        "## Alignment x Eval Verdict",
        "",
        *markdown_table(
            alignment_eval_rows, ["alignment_verdict", "new_eval_verdict", "passed", "n", "pass_rate"]
        ),
        "",
        "## Alignment x Repo",
        "",
        *markdown_table(alignment_repo_rows, ["alignment_verdict", "repo", "passed", "n", "pass_rate"]),
        "",
        "## Alignment x F2P Total Bucket",
        "",
        *markdown_table(
            alignment_f2p_rows, ["alignment_verdict", "f2p_total_bucket", "passed", "n", "pass_rate"]
        ),
        "",
        "## Failure Shape By Alignment",
        "",
        *markdown_table(
            failure_shape_rows, ["alignment_verdict", "f2p_result_bucket", "passed", "n", "pass_rate"]
        ),
        "",
        "## Matched OK vs Leaked",
        "",
        "Matched cells use the same `repo`, `new_eval_verdict`, and `f2p_total_bucket`.",
        "",
        *markdown_table(
            [
                {
                    "matched_leaked": f"{matched_summary['matched_leaked_passed']}/{matched_summary['matched_leaked_total']}",
                    "matched_leaked_rate": matched_summary["matched_leaked_rate"],
                    "expected_leaked_passes_at_ok_rate": matched_summary["expected_leaked_passed_at_ok_rate"],
                    "expected_leaked_rate_at_ok_rate": matched_summary["expected_leaked_rate_at_ok_rate"],
                }
            ],
            [
                "matched_leaked",
                "matched_leaked_rate",
                "expected_leaked_passes_at_ok_rate",
                "expected_leaked_rate_at_ok_rate",
            ],
        ),
        "",
        "### Matched Cells",
        "",
        *markdown_table(
            matched_rows,
            [
                "repo",
                "new_eval_verdict",
                "f2p_total_bucket",
                "ok",
                "ok_rate",
                "leaked",
                "leaked_rate",
                "gap_pp",
            ],
        ),
        "",
        "## Full Stratification: Alignment x Eval x Repo x F2P Bucket",
        "",
        *markdown_table(
            detailed_rows,
            [
                "alignment_verdict",
                "new_eval_verdict",
                "repo",
                "f2p_total_bucket",
                "passed",
                "n",
                "pass_rate",
            ],
        ),
        "",
        "## Zero-Pass Leaked Failures",
        "",
        *markdown_table(zero_leaked, ["task_id", "repo", "passed_total", "title"]),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    write_report(args.report, rows)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
