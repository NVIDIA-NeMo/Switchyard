#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import generic PR-reference records into miner-compatible candidate JSON files.

Supports keyed JSON maps like:
{
  "craft-click-1234abcd": {
    "pr_url": "https://github.com/pallets/click/pull/3152",
    "repo": "pallets/click",
    "task_type": "bug_fix"
  }
}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from craft_taskgen.importers.common import extract_pr_ref, fetch_merged_pr, load_records
from craft_taskgen.miner import _clone_or_find_repo, analyze_pr


def _load_records_with_ids(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = load_records(path, allow_json_object_map=True, record_id_key="source_record_id")
    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        r = dict(row)
        if not r.get("source_record_id"):
            r["source_record_id"] = r.get("task_id") or r.get("id") or f"row-{i:06d}"
        normalized.append(r)
    if limit > 0:
        normalized = normalized[:limit]
    return normalized


def _record_to_metadata(record: dict[str, Any], github_repo: str, pr_number: int) -> dict[str, Any]:
    metadata = dict(record)
    metadata.setdefault("repo", github_repo)
    metadata.setdefault("pr_number", pr_number)
    metadata.setdefault("pr_url", f"https://github.com/{github_repo}/pull/{pr_number}")
    return metadata


def run_import(
    input_path: Path,
    out_dir: Path,
    repos_dir: Path,
    top_per_repo: int,
    min_score: float,
    after: str | None = None,
    limit: int = 0,
    source_name: str = "generic-pr-refs",
) -> dict[str, int]:
    rows = _load_records_with_ids(input_path, limit=limit)

    records_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved = 0
    for row in rows:
        ref = extract_pr_ref(row)
        if not ref:
            unresolved += 1
            continue
        github_repo, pr_number = ref
        records_by_repo[github_repo].append(
            {
                "source_record_id": str(row["source_record_id"]),
                "pr_number": pr_number,
                "record": row,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    repos_written = 0
    prs_scanned = 0
    candidates_kept = 0
    skipped_unmerged_or_missing = 0

    for github_repo in sorted(records_by_repo):
        repo_path = _clone_or_find_repo(github_repo, repos_dir)
        if not repo_path:
            print(f"SKIP: could not find or clone {github_repo}", file=sys.stderr)
            continue

        short_name = repo_path.name
        records = records_by_repo[github_repo]
        unique_prs = sorted({r["pr_number"] for r in records})
        prs_scanned += len(unique_prs)

        pr_cache: dict[int, dict[str, Any] | None] = {}
        cand_cache: dict[int, dict[str, Any] | None] = {}
        repo_candidates: list[dict[str, Any]] = []

        for item in records:
            pr_number = int(item["pr_number"])
            record = item["record"]

            if pr_number not in pr_cache:
                try:
                    pr_cache[pr_number] = fetch_merged_pr(github_repo, pr_number)
                except RuntimeError as e:
                    print(f"WARN: {e}", file=sys.stderr)
                    pr_cache[pr_number] = None
                if pr_cache[pr_number] is None:
                    skipped_unmerged_or_missing += 1

            pr = pr_cache[pr_number]
            if not pr:
                continue
            if after and pr["date"][:10] < after[:10]:
                continue

            if pr_number not in cand_cache:
                try:
                    candidate = analyze_pr(repo_path, pr)
                except RuntimeError:
                    cand_cache[pr_number] = None
                    continue
                except Exception as e:
                    print(
                        f"WARN: analyze_pr failed for {github_repo}#{pr_number}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    cand_cache[pr_number] = None
                    continue

                if candidate.score < min_score:
                    cand_cache[pr_number] = None
                    continue

                cand_cache[pr_number] = asdict(candidate)

            cached = cand_cache[pr_number]
            if not cached:
                continue

            row = dict(cached)
            row["pr_number"] = pr_number
            row["source_task_id"] = item["source_record_id"]
            if "task_type" in record:
                row["source_task_type"] = record.get("task_type")
            if "matched_practice" in record:
                row["source_matched_practice"] = record.get("matched_practice")
            row["source_metadata"] = _record_to_metadata(record, github_repo, pr_number)
            repo_candidates.append(row)

        repo_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        if top_per_repo > 0:
            repo_candidates = repo_candidates[:top_per_repo]

        output = {
            "repo": short_name,
            "github_repo": github_repo,
            "source_dataset": source_name,
            "after": after,
            "n_input_records": len(records),
            "n_prs_scanned": len(unique_prs),
            "n_candidates": len(repo_candidates),
            "candidates": repo_candidates,
        }
        out_file = out_dir / f"{short_name}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        repos_written += 1
        candidates_kept += len(repo_candidates)
        print(f"Wrote {out_file} ({len(repo_candidates)} candidates)", file=sys.stderr)

    return {
        "rows_total": len(rows),
        "rows_unresolved": unresolved,
        "repos_written": repos_written,
        "prs_scanned": prs_scanned,
        "candidates_kept": candidates_kept,
        "skipped_unmerged_or_missing": skipped_unmerged_or_missing,
    }
