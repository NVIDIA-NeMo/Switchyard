#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clone all repos referenced by a SWE-bench JSONL into repos/{repo}."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from craft_taskgen.importers.common import load_records, normalize_repo_slug


def collect_unique_repos(input_path: Path) -> list[str]:
    rows = load_records(input_path)
    repos: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{input_path}: expected every row to be a JSON object")
        repo = normalize_repo_slug(row.get("repo"))
        if not repo:
            instance_id = row.get("instance_id", "<missing-instance_id>")
            raise ValueError(f"{instance_id}: repo must be a valid GitHub owner/repo slug")
        repos.add(repo)
    return sorted(repos)


def repo_clone_path(repos_dir: Path, github_repo: str) -> Path:
    _, name = github_repo.split("/", 1)
    return repos_dir / name


def is_git_clone(path: Path) -> bool:
    return path.is_dir() and (path / ".git").is_dir()


def clone_repo(github_repo: str, repos_dir: Path, timeout: int = 300) -> tuple[str, Path]:
    target = repo_clone_path(repos_dir, github_repo)
    if is_git_clone(target):
        return "exists", target
    if target.exists():
        raise RuntimeError(f"target exists but is not a git clone: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", f"https://github.com/{github_repo}.git", str(target)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed for {github_repo}: {result.stderr[:200].strip()}")
    return "cloned", target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone all repos referenced by a SWE-bench JSONL into repos/{repo}."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input SWE-bench .jsonl file.")
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("repos"),
        help="Target repos directory (default: repos).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which repos would be cloned without running git clone.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-clone timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    repos = collect_unique_repos(args.input)
    print(f"Found {len(repos)} unique repos in {args.input}")

    cloned = 0
    existing = 0
    for github_repo in repos:
        target = repo_clone_path(args.repos_dir, github_repo)
        if args.dry_run:
            status = "exists" if is_git_clone(target) else "clone"
            print(f"{status}: {github_repo} -> {target}")
            continue

        status, target = clone_repo(github_repo, args.repos_dir, timeout=args.timeout)
        print(f"{status}: {github_repo} -> {target}")
        if status == "cloned":
            cloned += 1
        else:
            existing += 1

    if not args.dry_run:
        print(f"Done: cloned={cloned} existing={existing} total={len(repos)}")


if __name__ == "__main__":
    main()
