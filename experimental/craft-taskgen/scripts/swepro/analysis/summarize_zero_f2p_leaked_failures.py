#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize leaked failures where no verifier F2P tests passed."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_CSV = Path("docs/analyses/data/swebench-pro/findings/leaked_agent_failures.csv")
DEFAULT_MANIFEST = Path("docs/analyses/data/swebench-pro/runs/combined_non_error/combined_manifest.json")
DEFAULT_RUNS_DIR = Path("docs/analyses/data/swebench-pro/runs/combined_non_error")
DEFAULT_OUTPUT = Path("docs/analyses/data/swebench-pro/findings/zero_f2p_leaked_failures_raw.md")


def load_task_to_trial(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open() as f:
        manifest = json.load(f)
    return {trial["task_id"]: trial["trial_name"] for trial in manifest["selected_trials"]}


def extract_failure_excerpt(text: str, max_chars: int = 5000) -> str:
    markers = [
        "=================================== FAILURES ===================================",
        "ERROR",
        "FAILED",
    ]
    starts = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if not starts:
        return text[-max_chars:]
    start = min(starts)
    return text[start : start + max_chars]


def extract_stdout_counts(text: str) -> tuple[str, str]:
    passed = re.search(r"^Required tests that passed:\s+(\d+)$", text, re.MULTILINE)
    total = re.search(r"^Required tests:\s+(\d+)$", text, re.MULTILINE)
    return (
        passed.group(1) if passed else "",
        total.group(1) if total else "",
    )


def trajectory_summary(path: Path) -> tuple[list[str], list[str], str]:
    with path.open() as f:
        trajectory = json.load(f)

    edit_paths: list[str] = []
    bash_commands: list[str] = []
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls") or []:
            args = call.get("arguments") or {}
            file_path = args.get("file_path")
            if call.get("function_name") in {"Edit", "Write"} and file_path:
                edit_paths.append(file_path)
            if call.get("function_name") == "Bash" and args.get("command"):
                bash_commands.append(args["command"])

    final_message = ""
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") == "agent" and step.get("message"):
            final_message = step["message"]
            break

    return edit_paths, bash_commands[-8:], final_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    task_to_trial = load_task_to_trial(args.manifest)
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    zero_rows = [row for row in rows if row["agent_f2p_tests_passed"] == "0"]

    lines: list[str] = [
        "# Zero-F2P Leaked Failures Raw Summary",
        "",
        f"Count: {len(zero_rows)}",
        "",
    ]

    for row in zero_rows:
        trial_name = task_to_trial[row["swebench_instance_id"]]
        run_dir = args.runs_dir / trial_name
        verifier_output = json.loads((run_dir / "verifier" / "output.json").read_text())
        test_stdout = (run_dir / "verifier" / "test-stdout.txt").read_text(errors="replace")
        run_stdout = (run_dir / "verifier" / "run-script-stdout.txt").read_text(errors="replace")
        passed, total = extract_stdout_counts(test_stdout)
        edits, recent_bash, final_message = trajectory_summary(run_dir / "agent" / "trajectory.json")

        lines.extend(
            [
                f"## {row['task_id']} ({passed}/{total})",
                "",
                f"- Repo: `{row['repo']}`",
                f"- Trial: `{trial_name}`",
                f"- Title: {row['title']}",
                f"- SWE-bench id: `{row['swebench_instance_id']}`",
                f"- Edited files: {', '.join(dict.fromkeys(edits)) or '(none)'}",
                "",
                "### Failed/Missing Tests",
                "",
            ]
        )
        for test in verifier_output.get("tests", []):
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
            lines.append(f"- `{command[:300]}`")

        lines.extend(
            [
                "",
                "### Final Agent Message",
                "",
                "```text",
                final_message[:2500],
                "```",
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
