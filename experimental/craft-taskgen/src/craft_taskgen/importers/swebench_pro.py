#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import SWE-bench Pro-style rows into miner-compatible candidate JSON files."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from craft_taskgen.importers.common import load_records, normalize_repo_slug
from craft_taskgen.miner import Candidate

INSTANCE_COMMIT_RE = re.compile(r"^instance_[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-(?P<sha>[0-9a-f]{40})(?:-.*)?$")


def _load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = load_records(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(dict(row))
    if limit > 0:
        normalized = normalized[:limit]
    return normalized


def _row_id(row: dict[str, Any]) -> str:
    instance_id = row.get("instance_id")
    return str(instance_id) if instance_id is not None else "<missing-instance_id>"


def _require_nonempty_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _require_patch_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when present")
    return value


def _commit_sha_from_instance_id(instance_id: str) -> str:
    match = INSTANCE_COMMIT_RE.fullmatch(instance_id)
    if not match:
        raise ValueError("instance_id must match instance_<owner>__<repo>-<40hex> with any optional suffix")
    return match.group("sha")


def _derive_subject(problem_statement: Any, instance_id: Any) -> str:
    if problem_statement is None:
        return str(instance_id)
    if not isinstance(problem_statement, str):
        raise ValueError("problem_statement must be a string when present")
    text = problem_statement
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return str(instance_id)


def _normalize_patch_path(path: str | None) -> str:
    if not path:
        return ""
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _parse_patch(patch_text: Any) -> tuple[list[str], int, int]:
    if not isinstance(patch_text, str):
        raise ValueError("patch text must be a string")
    text = patch_text
    if not text:
        return [], 0, 0

    try:
        from unidiff import PatchSet
    except ImportError as e:
        raise RuntimeError("unidiff is required for --format swebench-pro") from e

    patch = PatchSet(text)
    files: list[str] = []
    seen: set[str] = set()
    added = 0
    deleted = 0

    for patched_file in patch:
        path = _normalize_patch_path(getattr(patched_file, "path", None))
        if not path or path == "/dev/null":
            source = _normalize_patch_path(getattr(patched_file, "source_file", None))
            target = _normalize_patch_path(getattr(patched_file, "target_file", None))
            path = target if target and target != "/dev/null" else source
        if path and path not in seen:
            seen.add(path)
            files.append(path)
        added += int(getattr(patched_file, "added", 0))
        deleted += int(getattr(patched_file, "removed", 0))

    return files, added, deleted


def _package_names(source_files: list[str]) -> list[str]:
    packages = {path.split("/", 1)[0] for path in source_files if path}
    return sorted(packages)


def _repo_clone_path(repos_dir: Path, github_repo: str) -> Path:
    _, name = github_repo.split("/", 1)
    return repos_dir / name


def _compute_merge_base(repo_path: Path, base_sha: str, commit_sha: str) -> str:
    if not repo_path.is_dir() or not (repo_path / ".git").is_dir():
        raise ValueError(f"expected git clone at {repo_path}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "merge-base", base_sha, commit_sha],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"git merge-base timed out in {repo_path}") from e
    except OSError as e:
        raise RuntimeError(f"git merge-base failed to start in {repo_path}: {e}") from e
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ValueError(
            f"git merge-base failed for base_sha={base_sha} commit_sha={commit_sha} in {repo_path}: {stderr}"
        )
    merge_base = result.stdout.strip()
    if not merge_base:
        raise ValueError(f"git merge-base returned empty output in {repo_path}")
    return merge_base


def _build_candidate_from_row(row: dict[str, Any], repos_dir: Path) -> dict[str, Any]:
    repo = _require_nonempty_str(row, "repo")
    github_repo = normalize_repo_slug(repo)
    if not github_repo:
        raise ValueError("repo must be a valid GitHub owner/repo slug")

    instance_id = _require_nonempty_str(row, "instance_id")
    base_commit = _require_nonempty_str(row, "base_commit")
    commit_sha = _commit_sha_from_instance_id(instance_id)
    patch = _require_patch_text(row, "patch")
    test_patch = _require_patch_text(row, "test_patch")
    if not patch and not test_patch:
        raise ValueError("at least one of patch or test_patch must be present")

    source_files, src_added, src_deleted = _parse_patch(patch)
    test_files, test_added, test_deleted = _parse_patch(test_patch)
    package_names = _package_names(source_files)
    merge_base_sha = _compute_merge_base(_repo_clone_path(repos_dir, github_repo), base_commit, commit_sha)

    candidate = Candidate(
        sha=commit_sha,
        base_sha=base_commit,
        merge_base_sha=merge_base_sha,
        subject=_derive_subject(row.get("problem_statement"), instance_id),
        author="swebench-pro",
        date="",
        source_files=source_files,
        test_files=test_files,
        other_files=[],
        source_lines_changed=src_added + src_deleted,
        test_lines_changed=test_added + test_deleted,
        packages_touched=len(package_names),
        package_names=package_names,
        has_test_patch=bool(test_files),
        is_multi_file=len(source_files) >= 3,
        is_multi_package=len(package_names) >= 2,
        is_nontrivial_source=(src_added + src_deleted) >= 30,
        is_nontrivial_tests=(test_added + test_deleted) >= 10,
        is_refactoring=False,
        has_iteration_signal=False,
    )
    candidate.compute_score()
    result = asdict(candidate)
    result["source_task_id"] = instance_id
    result["source_metadata"] = dict(row)
    return result


def run_import(
    input_path: Path,
    out_dir: Path,
    repos_dir: Path,
    top_per_repo: int,
    min_score: float,
    limit: int = 0,
    source_name: str = "swebench-pro",
) -> dict[str, int]:
    rows = _load_rows(input_path, limit=limit)
    records_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved = 0

    for row in rows:
        try:
            repo = _require_nonempty_str(row, "repo")
            github_repo = normalize_repo_slug(repo)
            if not github_repo:
                raise ValueError("repo must be a valid GitHub owner/repo slug")
        except Exception as e:
            unresolved += 1
            print(f"WARN: failed to import {_row_id(row)}: {e}", file=sys.stderr)
            continue
        records_by_repo[github_repo].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)

    repos_written = 0
    prs_scanned = 0
    candidates_kept = 0

    for github_repo in sorted(records_by_repo):
        repo_rows = records_by_repo[github_repo]
        repo_candidates: list[dict[str, Any]] = []
        prs_scanned += len(repo_rows)

        for row in repo_rows:
            try:
                candidate = _build_candidate_from_row(row, repos_dir)
            except Exception as e:
                unresolved += 1
                print(f"WARN: failed to import {_row_id(row)}: {e}", file=sys.stderr)
                continue

            if candidate["score"] < min_score:
                continue
            repo_candidates.append(candidate)

        repo_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        if top_per_repo > 0:
            repo_candidates = repo_candidates[:top_per_repo]

        short_name = github_repo.split("/")[-1]
        output = {
            "repo": short_name,
            "github_repo": github_repo,
            "source_dataset": source_name,
            "after": None,
            "n_input_records": len(repo_rows),
            "n_prs_scanned": len(repo_rows),
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
        "skipped_unmerged_or_missing": 0,
    }
