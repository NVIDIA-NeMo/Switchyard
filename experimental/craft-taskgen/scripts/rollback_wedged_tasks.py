#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roll back NEEDS_FIX tasks whose failure looks infra-caused (not quality).

One-shot ops tool for recovering a pipeline run where disk pressure / stuck
containers caused false-positive NEEDS_FIX's. Run on a quiesced state.json
(kill the pipeline first) so there's no write race. After --apply, relaunch
the pipeline with --resume.

Categories:

INFRA (default --apply set): reason looks caused by environment, not task quality
- "Docker build failed after N attempts"         -> tests_discovered
- "Classification: Docker timed out ..."         -> dockerfile_built
- "Opus smoke infra/no_trial ..."                -> oracle_checked
- "Opus smoke timed out ..."                     -> oracle_checked
- "build_dockerfile Claude error: error_max_turns" -> tests_discovered
- "" (empty reason)                              -> promising

BORDERLINE (needs --include-borderline to roll back): could be quality OR infra
- "Classification: oracle run passed 0 tests ..."  -> f2p_p2p_classified

QUALITY (always skipped):
- "Build failed after N attempts"                — LLM Build output unusable
- "Classification: patch failed to apply ..."    — solve.sh is wrong
- "Opus deep dive failed: ..."                   — judge parse error
- "easiness=..."                                 — reviewer verdict
- "No test files found in commit diff"

Usage:

    # dry-run: print table of candidate rollbacks
    uv run python scripts/rollback_wedged_tasks.py PATH/state.json

    # apply with borderline oracle=0 cases also rolled back
    uv run python scripts/rollback_wedged_tasks.py PATH/state.json \\
        --apply --include-borderline
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

INFRA_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^Docker build failed"), "tests_discovered", "infra"),
    (re.compile(r"^Classification: Docker timed out"), "dockerfile_built", "infra"),
    (re.compile(r"^Opus smoke infra/no_trial"), "oracle_checked", "infra"),
    (re.compile(r"^Opus smoke timed out"), "oracle_checked", "infra"),
    (re.compile(r"^build_dockerfile Claude error: error_max_turns"), "tests_discovered", "infra"),
]
BORDERLINE_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^Classification: oracle run passed 0 tests"), "f2p_p2p_classified", "borderline"),
]
EMPTY_REASON_TARGET = ("promising", "infra")  # empty reason → fresh start

QUALITY_RULES: list[re.Pattern[str]] = [
    re.compile(r"^Build failed after"),
    re.compile(r"^Classification: patch failed to apply"),
    re.compile(r"^Opus deep dive failed"),
    re.compile(r"^easiness="),
    re.compile(r"^No test files found"),
]


def classify(reason: str, include_borderline: bool) -> tuple[str, str] | None:
    """Return (target_stage, category) or None if not eligible."""
    reason = reason or ""
    if not reason.strip():
        return EMPTY_REASON_TARGET
    for rx in QUALITY_RULES:
        if rx.search(reason):
            return None
    rules = INFRA_RULES + (BORDERLINE_RULES if include_borderline else [])
    for rx, target, category in rules:
        if rx.search(reason):
            return target, category
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("state_json", type=Path, help="path to state.json to mutate")
    p.add_argument("--apply", action="store_true", help="write back to state.json (default: dry-run print)")
    p.add_argument(
        "--include-borderline",
        action="store_true",
        help="also roll back oracle-run-passed-0-tests (borderline cases)",
    )
    args = p.parse_args()

    if not args.state_json.is_file():
        print(f"ERROR: {args.state_json} not found", file=sys.stderr)
        return 2

    with args.state_json.open() as f:
        state = json.load(f)

    tasks = state.get("tasks", {})
    rollbacks: list[
        tuple[str, str, str, str, str]
    ] = []  # (tid, reason_short, target, category, current_stage)
    skipped_quality = 0
    skipped_nonterminal = 0

    for tid, task in tasks.items():
        if task.get("stage") != "needs_fix":
            skipped_nonterminal += 1
            continue
        reason = task.get("human_review_reason", "")
        verdict = classify(reason, args.include_borderline)
        if verdict is None:
            skipped_quality += 1
            continue
        target_stage, category = verdict
        reason_short = (reason or "(empty)").splitlines()[0][:60]
        rollbacks.append((tid, reason_short, target_stage, category, task.get("stage", "?")))

    # Print summary
    rollbacks.sort(key=lambda r: (r[3], r[2]))
    print(f"State: {args.state_json}")
    print(f"  Total tasks:            {len(tasks)}")
    print(f"  NEEDS_FIX eligible:     {len(rollbacks)}")
    print(f"  NEEDS_FIX skipped (quality): {skipped_quality}")
    print()

    if not rollbacks:
        print("Nothing to roll back.")
        return 0

    print(f"{'task_id':<38} {'cat':<11} {'target_stage':<22} {'reason':<60}")
    print("-" * 135)
    for tid, reason, target, category, _ in rollbacks:
        print(f"{tid:<38} {category:<11} {target:<22} {reason:<60}")
    print()

    if not args.apply:
        print("Dry-run. Re-run with --apply to write changes.")
        return 0

    # Backup before mutating
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = args.state_json.with_suffix(f".json.bak-{ts}")
    shutil.copy2(args.state_json, backup)
    print(f"Backup: {backup}")

    for tid, _, target_stage, _, _ in rollbacks:
        task = tasks[tid]
        task["stage"] = target_stage
        task["needs_human_review"] = False
        task["human_review_reason"] = ""
        task["fix_attempts"] = 0

    tmp = args.state_json.with_suffix(args.state_json.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(args.state_json)
    print(f"Applied {len(rollbacks)} rollbacks to {args.state_json}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
