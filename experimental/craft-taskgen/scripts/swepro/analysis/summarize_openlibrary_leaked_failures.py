#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize OpenLibrary leaked failures with verifier and trajectory context."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_ENRICHED = Path("docs/analyses/data/swebench-pro/findings/task_outcomes_enriched.csv")
DEFAULT_RUNS_DIR = Path("docs/analyses/data/swebench-pro/runs/combined_non_error")
DEFAULT_OUTPUT = Path("docs/analyses/data/swebench-pro/findings/openlibrary_leaked_failures_raw.md")


def short(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_failure_excerpt(text: str, max_chars: int = 5000) -> str:
    markers = [
        "=================================== FAILURES ===================================",
        "==================================== ERRORS ====================================",
        "FAILED",
        "ERROR",
    ]
    starts = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if not starts:
        return text[-max_chars:]
    start = min(starts)
    return text[start : start + max_chars]


def interesting_failure_lines(text: str, limit: int = 30) -> list[str]:
    pattern = re.compile(
        r"(^E\s+|AssertionError|AttributeError|TypeError|KeyError|ValueError|FAILED|ERROR|assert )"
    )
    lines = [line for line in text.splitlines() if pattern.search(line)]
    return lines[-limit:]


def trajectory_summary(run_dir: Path) -> tuple[list[str], list[str], str]:
    trajectory_path = run_dir / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return [], [], ""

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
                edited_files.append(file_path.removeprefix("/app/"))
            if function_name == "Bash" and args.get("command"):
                bash_commands.append(args["command"])

    final_message = ""
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") == "agent" and step.get("message"):
            final_message = step["message"]
            break

    return sorted(set(edited_files)), bash_commands[-8:], final_message


def verifier_tests(run_dir: Path) -> list[dict[str, str]]:
    output_path = run_dir / "verifier" / "output.json"
    if not output_path.exists():
        return []
    with output_path.open() as f:
        output = json.load(f)
    return output.get("tests", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.enriched.open(newline="") as f:
        rows = list(csv.DictReader(f))

    failures = [
        row
        for row in rows
        if row["repo"] == "openlibrary"
        and row["alignment_verdict"] == "leaked"
        and row["agent_success"] == "False"
    ]

    lines: list[str] = [
        "# OpenLibrary Leaked Failures Raw Summary",
        "",
        f"Count: {len(failures)}",
        "",
    ]

    for row in failures:
        run_dir = args.runs_dir / row["trial_name"]
        test_stdout = (run_dir / "verifier" / "test-stdout.txt").read_text(errors="replace")
        run_stdout = (run_dir / "verifier" / "run-script-stdout.txt").read_text(errors="replace")
        edited_files, recent_bash, final_message = trajectory_summary(run_dir)
        tests = verifier_tests(run_dir)
        failed_tests = [test for test in tests if test.get("status") != "PASSED"]

        lines.extend(
            [
                f"## {row['task_id']} ({row['agent_f2p_tests_passed']}/{row['agent_f2p_tests_total']})",
                "",
                f"- Eval verdict: `{row['new_eval_verdict']}`",
                f"- F2P result bucket: `{row['f2p_result_bucket']}`",
                f"- Trial: `{row['trial_name']}`",
                f"- Title: {short(row['title'], 300)}",
                f"- Alignment reason: {short(row['alignment_reason'], 800)}",
                f"- Edited files: {', '.join(edited_files) or '(none)'}",
                "",
                "### Failed Tests",
                "",
            ]
        )
        for test in failed_tests:
            lines.append(f"- `{test.get('status')}` `{test.get('name')}`")

        lines.extend(
            [
                "",
                "### Verifier Summary",
                "",
                "```text",
                test_stdout.strip(),
                "```",
                "",
                "### Interesting Failure Lines",
                "",
                "```text",
                "\n".join(interesting_failure_lines(run_stdout)),
                "```",
                "",
                "### Failure Excerpt",
                "",
                "```text",
                extract_failure_excerpt(run_stdout).strip(),
                "```",
                "",
                "### Recent Bash Commands",
                "",
            ]
        )
        for command in recent_bash:
            lines.append(f"- `{short(command, 250)}`")

        lines.extend(
            [
                "",
                "### Final Agent Message",
                "",
                "```text",
                final_message[:2200],
                "```",
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
