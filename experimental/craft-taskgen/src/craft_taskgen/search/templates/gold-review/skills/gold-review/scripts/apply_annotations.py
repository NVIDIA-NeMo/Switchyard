#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply annotations from the gold-reviewer UI to task JSONs.

Reads annotations.json (exported from the UI), applies changes to
task files in tasks/accepted/search/, and logs each change to a JSONL
audit trail.

Usage:
    uv run python .claude/skills/gold-review/scripts/apply_annotations.py annotations.json

Options:
    --dry-run   Show what would change without writing files
    --log-file  Path to JSONL audit log (default: tools/search/annotation_log.jsonl)
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ACCEPTED_DIR = ROOT / "tasks" / "accepted" / "search"


def find_task_file(task_id: str) -> Path | None:
    """Find the task JSON file for a craft-{repo}-{uuid8} task ID."""
    # Format: craft-{repo}-{uuid8} where repo may contain hyphens (e.g. pre-commit)
    # uuid8 is always the last segment and is 8 hex chars
    parts = task_id.split("-")
    if len(parts) < 3:
        return None
    uuid8 = parts[-1]
    repo = "-".join(parts[1:-1])
    repo_dir = ACCEPTED_DIR / repo
    if not repo_dir.exists():
        return None
    for f in repo_dir.glob("*.json"):
        if f.stem.startswith(uuid8):
            return f
    return None


def apply_one(task_id: str, annotation: dict, dry_run: bool) -> list[dict]:
    """Apply one annotation to a task file. Returns list of log entries."""
    task_file = find_task_file(task_id)
    if not task_file:
        print(f"  SKIP {task_id}: task file not found", file=sys.stderr)
        return []

    with open(task_file) as f:
        data = json.load(f)

    gold = data["task"]["gold_answer"]
    original = deepcopy(gold)
    logs = []
    now = datetime.now(timezone.utc).isoformat()

    def log(action: str, kind: str, item: str, detail: str = ""):
        logs.append(
            {
                "timestamp": now,
                "task_id": task_id,
                "action": action,
                "kind": kind,
                "item": item,
                "detail": detail,
            }
        )

    # Remove files
    for f in annotation.get("removed_files", []):
        if f in gold["files"]:
            gold["files"].remove(f)
            log("remove", "file", f)

    # Remove functions
    for f in annotation.get("removed_functions", []):
        if f in gold["functions"]:
            gold["functions"].remove(f)
            log("remove", "function", f)

    # Demote files (primary -> alt)
    for f in annotation.get("demoted_files", []):
        if f in gold["files"]:
            gold["files"].remove(f)
            if f not in gold.get("alt_files", []):
                gold.setdefault("alt_files", []).append(f)
            log("demote", "file", f)

    # Demote functions (primary -> alt)
    for f in annotation.get("demoted_functions", []):
        if f in gold["functions"]:
            gold["functions"].remove(f)
            if f not in gold.get("alt_functions", []):
                gold.setdefault("alt_functions", []).append(f)
            log("demote", "function", f)

    # Promote alt files (alt -> primary)
    for f in annotation.get("promoted_alt_files", []):
        if f in gold.get("alt_files", []):
            gold["alt_files"].remove(f)
        if f not in gold["files"]:
            gold["files"].append(f)
        log("promote", "file", f)

    # Promote alt functions (alt -> primary)
    for f in annotation.get("promoted_alt_functions", []):
        if f in gold.get("alt_functions", []):
            gold["alt_functions"].remove(f)
        if f not in gold["functions"]:
            gold["functions"].append(f)
        log("promote", "function", f)

    # Remove assertions (highest index first to preserve indices)
    removed_assertions = sorted(annotation.get("removed_assertions", []), reverse=True)
    for idx in removed_assertions:
        if 0 <= idx < len(gold["assertions"]):
            removed_text = gold["assertions"].pop(idx)
            log("remove", "assertion", str(idx), removed_text)

    # Edit assertions
    for idx_str, new_text in annotation.get("edited_assertions", {}).items():
        idx = int(idx_str)
        if 0 <= idx < len(gold["assertions"]):
            old_text = gold["assertions"][idx]
            gold["assertions"][idx] = new_text
            log("edit", "assertion", str(idx), f"{old_text} -> {new_text}")

    # Edit explanation
    if "edited_explanation" in annotation and annotation["edited_explanation"] is not None:
        old_expl = gold.get("explanation", "")
        gold["explanation"] = annotation["edited_explanation"]
        log("edit", "explanation", "", f"len {len(old_expl)} -> {len(annotation['edited_explanation'])}")

    # Check if anything changed
    if gold == original:
        return []

    # Add notes to log
    if annotation.get("notes"):
        log("note", "annotation", "", annotation["notes"])

    if not dry_run:
        with open(task_file, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    return logs


def main():
    parser = argparse.ArgumentParser(description="Apply gold-reviewer annotations to task JSONs")
    parser.add_argument("annotations_file", help="Path to annotations.json exported from UI")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument(
        "--log-file",
        default=str(ROOT / "tools" / "search" / "annotation_log.jsonl"),
        help="Path to JSONL audit log",
    )
    args = parser.parse_args()

    with open(args.annotations_file) as f:
        annotations = json.load(f)

    all_logs = []
    applied = 0
    skipped = 0

    for task_id, annotation in sorted(annotations.items()):
        # Skip tasks with no actionable changes
        has_changes = any(
            annotation.get(k)
            for k in [
                "removed_files",
                "removed_functions",
                "removed_assertions",
                "demoted_files",
                "demoted_functions",
                "promoted_alt_files",
                "promoted_alt_functions",
                "edited_assertions",
                "edited_explanation",
            ]
        )
        if not has_changes:
            skipped += 1
            continue

        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"{prefix}Applying {task_id} (status: {annotation.get('status', '?')})...")
        logs = apply_one(task_id, annotation, args.dry_run)
        for log_entry in logs:
            print(f"  {log_entry['action']:>7} {log_entry['kind']:>12}: {log_entry['item']}")
            if log_entry["detail"] and log_entry["action"] != "note":
                detail = log_entry["detail"]
                if len(detail) > 100:
                    detail = detail[:100] + "..."
                print(f"{'':>22}  {detail}")
        all_logs.extend(logs)
        if logs:
            applied += 1

    # Write audit log
    if all_logs and not args.dry_run:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            for entry in all_logs:
                f.write(json.dumps(entry) + "\n")
        print(f"\nWrote {len(all_logs)} log entries to {log_path}")

    print(f"\n{applied} tasks modified, {skipped} skipped (no changes)")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
