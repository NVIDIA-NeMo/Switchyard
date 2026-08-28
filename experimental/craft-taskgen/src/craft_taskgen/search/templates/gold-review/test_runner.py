#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifier for gold-review batch verdicts.

Reads /app/verdicts.json (agent output) and /tests/tasks_to_review.json (expected tasks).
Validates schema and completeness. Writes reward based on fraction of tasks reviewed.
"""

from __future__ import annotations

import json
import os
import sys

REQUIRED_FIELDS = {"task_id", "function_actions", "overall_recommendation"}
VALID_ACTIONS = {"KEEP", "DEMOTE", "REMOVE"}
VALID_RECOMMENDATIONS = {"ACCEPT", "FLAG", "REJECT"}


def load_json(path: str) -> dict | list | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def validate_verdict(verdict: dict) -> list[str]:
    """Validate a single verdict dict. Returns list of issues."""
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in verdict:
            issues.append(f"missing field: {field}")

    if "function_actions" in verdict:
        for action in verdict["function_actions"]:
            if "action" in action and action["action"] not in VALID_ACTIONS:
                issues.append(f"invalid function action: {action['action']}")

    if "file_actions" in verdict:
        for action in verdict["file_actions"]:
            if "action" in action and action["action"] not in VALID_ACTIONS:
                issues.append(f"invalid file action: {action['action']}")

    rec = verdict.get("overall_recommendation", "")
    if rec and rec not in VALID_RECOMMENDATIONS:
        issues.append(f"invalid recommendation: {rec}")

    return issues


def main():
    os.makedirs("/logs/verifier", exist_ok=True)

    # Load expected tasks
    expected = load_json("/tests/tasks_to_review.json")
    if expected is None:
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("0.0")
        json.dump(
            {"error": "tasks_to_review.json missing", "reward": 0.0},
            open("/logs/verifier/reward.json", "w"),
            indent=2,
        )
        return

    expected_ids = set()
    if isinstance(expected, list):
        expected_ids = {t.get("id", t.get("task_id", "")) for t in expected}
    n_expected = len(expected_ids)

    # Load agent verdicts
    verdicts = load_json("/app/verdicts.json")
    if verdicts is None:
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("0.0")
        json.dump(
            {"error": "verdicts.json missing or malformed", "reward": 0.0},
            open("/logs/verifier/reward.json", "w"),
            indent=2,
        )
        return

    if not isinstance(verdicts, list):
        verdicts = [verdicts]

    # Validate each verdict
    reviewed_ids = set()
    valid_verdicts = []
    all_issues = []

    for v in verdicts:
        if not isinstance(v, dict):
            all_issues.append("non-dict verdict entry")
            continue
        issues = validate_verdict(v)
        tid = v.get("task_id", "")
        if tid:
            reviewed_ids.add(tid)
        if issues:
            all_issues.extend([f"{tid}: {i}" for i in issues])
        else:
            valid_verdicts.append(v)

    # Compute reward: fraction of expected tasks that got valid verdicts
    covered = reviewed_ids & expected_ids
    reward = len(covered) / n_expected if n_expected > 0 else 0.0

    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(round(reward, 4)))

    json.dump(
        {
            "reward": round(reward, 4),
            "n_expected": n_expected,
            "n_reviewed": len(reviewed_ids),
            "n_valid": len(valid_verdicts),
            "n_covered": len(covered),
            "missing_tasks": sorted(expected_ids - reviewed_ids),
            "issues": all_issues[:20],
            "verdicts": valid_verdicts,
        },
        open("/logs/verifier/reward.json", "w"),
        indent=2,
    )

    print(f"Reward: {reward:.4f} ({len(covered)}/{n_expected} tasks reviewed)")
    if all_issues:
        print(f"Issues: {len(all_issues)}", file=sys.stderr)
        for issue in all_issues[:5]:
            print(f"  {issue}", file=sys.stderr)


if __name__ == "__main__":
    main()
