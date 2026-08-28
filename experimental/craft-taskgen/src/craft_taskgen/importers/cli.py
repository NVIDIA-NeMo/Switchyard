#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importer CLI for external PR-reference datasets."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from craft_taskgen.importers import pr_refs, swebench_pro


def _print_summary(summary: dict[str, int]) -> None:
    print(
        "Summary: "
        f"rows={summary['rows_total']} "
        f"unresolved={summary['rows_unresolved']} "
        f"repos={summary['repos_written']} "
        f"prs_scanned={summary['prs_scanned']} "
        f"candidates={summary['candidates_kept']} "
        f"skipped_unmerged={summary['skipped_unmerged_or_missing']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import external PR-reference datasets into craft-taskgen candidate JSON files."
    )
    parser.add_argument(
        "--format",
        choices=["pr-refs", "swebench-pro"],
        default="pr-refs",
        help="Input dataset format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file (.json/.jsonl/.csv/.tsv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for miner-compatible candidate JSON files.",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("repos"),
        help="Directory with local cloned repos (missing repos are cloned).",
    )
    parser.add_argument(
        "--top-per-repo",
        type=int,
        default=0,
        help="Keep top N candidates per repo after scoring (0 = keep all).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Discard candidates below this score.",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Only include PRs merged on/after YYYY-MM-DD.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of input records/rows (0 = all).",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Source label (written as source_dataset).",
    )
    args = parser.parse_args()

    if args.after and args.format == "pr-refs":
        try:
            datetime.strptime(args.after, "%Y-%m-%d")
        except ValueError:
            parser.error("--after must be in YYYY-MM-DD format")

    if args.format == "swebench-pro":
        summary = swebench_pro.run_import(
            input_path=args.input,
            out_dir=args.out_dir or (Path("candidates") / "swebench-pro"),
            repos_dir=args.repos_dir,
            top_per_repo=args.top_per_repo,
            min_score=args.min_score,
            limit=args.limit,
            source_name=args.source_name or "swebench-pro",
        )
    else:
        summary = pr_refs.run_import(
            input_path=args.input,
            out_dir=args.out_dir or (Path("candidates") / "pr-refs"),
            repos_dir=args.repos_dir,
            top_per_repo=args.top_per_repo,
            min_score=args.min_score,
            after=args.after,
            limit=args.limit,
            source_name=args.source_name or "generic-pr-refs",
        )

    _print_summary(summary)


if __name__ == "__main__":
    main()
