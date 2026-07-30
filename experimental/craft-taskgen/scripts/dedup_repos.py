#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedup a craft-repos-style CSV and remove any repos that appear in swebench-repos.csv.

Usage:
    python3 scripts/dedup_repos.py references/repo_list_v2.csv
    python3 scripts/dedup_repos.py references/repo_list_v2.csv --out references/repo_list_v2-clean.csv
    python3 scripts/dedup_repos.py references/repo_list_v2.csv --swebench references/swebench-repos.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_EXCLUDED = Path("references/excluded-repos.csv")
FIELDS = ["short_name", "github_repo", "github_url", "stars", "license", "domain", "description"]


def run_dedup(rows: list[dict], excluded_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (kept, duplicates_removed, excluded_removed)."""
    seen: set[str] = set()
    deduped, duplicates = [], []
    for r in rows:
        key = r["github_repo"]
        if key in seen:
            duplicates.append(r)
        else:
            seen.add(key)
            deduped.append(r)

    excluded: set[str] = set()
    if excluded_path.exists():
        with excluded_path.open() as f:
            excluded = {r["github_repo"] for r in csv.DictReader(f)}

    kept = [r for r in deduped if r["github_repo"] not in excluded]
    excluded_removed = [r for r in deduped if r["github_repo"] in excluded]
    return kept, duplicates, excluded_removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup repo CSV and remove swebench repos.")
    parser.add_argument("csv_file", type=Path, help="Input CSV (craft-repos-style)")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: overwrites input)")
    parser.add_argument(
        "--exclude",
        type=Path,
        default=DEFAULT_EXCLUDED,
        help="Excluded repos CSV (default: references/excluded-repos.csv)",
    )
    args = parser.parse_args()

    if not args.csv_file.exists():
        sys.exit(f"ERROR: input file not found: {args.csv_file}")

    with args.csv_file.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit(f"ERROR: {args.csv_file} is empty or contains no data rows.")

    kept, duplicates, excluded_removed = run_dedup(rows, args.exclude)

    out_file = args.out or args.csv_file
    tmp = out_file.with_suffix(".tmp")
    try:
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(kept)
        tmp.rename(out_file)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"ERROR: failed to write {out_file}: {e}")

    print(f"Input:            {len(rows)}")
    if duplicates:
        print(f"Duplicates removed: -{len(duplicates)}  ({[r['short_name'] for r in duplicates]})")
    else:
        print("Duplicates removed:  0")
    if excluded_removed:
        names = [r["short_name"] for r in excluded_removed]
        print(f"Excluded removed:   -{len(excluded_removed)}  ({names})")
    else:
        print("Excluded removed:    0")
    print(f"Final:             {len(kept)}  → {out_file}")


if __name__ == "__main__":
    main()
