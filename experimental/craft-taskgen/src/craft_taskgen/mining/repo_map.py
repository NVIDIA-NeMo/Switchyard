# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Aider-style repo maps from a repository.

Produces a compact text representation showing file paths with class/function
signatures (no bodies), ranked by structural importance using PageRank on
the cross-reference graph (via vendored Aider repomap).
"""

from __future__ import annotations

import os

from .schemas import RepoGroundTruth
from .tree_sitter_extractor import PRUNE_DIRS
from .vendor.aider_repomap import RepoMap


def build_repo_map(
    ground_truth: RepoGroundTruth,
    max_chars: int = 80_000,
    include_tests: bool = False,
    priority_files: list[str] | None = None,
) -> str:
    """Build a ranked repo map from mined ground truth.

    Parameters
    ----------
    ground_truth:
        Complete ground truth from RepoIndexer.
    max_chars:
        Approximate character budget for the output.
    include_tests:
        Whether to include test files.
    priority_files:
        Repo-relative paths the caller wants to prioritize in the output. The
        underlying Aider repomap promotes these via PageRank personalization so
        they're the last to be truncated when the char budget is tight.

    Returns a text string showing file->class->function signatures.
    """
    repo_path = ground_truth.repo_path
    all_files = _collect_python_files(repo_path, include_tests)

    priority_abs: list[str] = []
    if priority_files:
        for rel in priority_files:
            abs_path = os.path.join(repo_path, rel)
            if os.path.isfile(abs_path):
                priority_abs.append(abs_path)

    # Priority files should be ranked + included, even if they weren't in
    # all_files (e.g. a test-dir file when include_tests=False).
    other_set = set(all_files) | set(priority_abs)

    rm = RepoMap(
        map_tokens=max_chars // 4,  # chars -> approx tokens
        root=repo_path,
    )
    return (
        rm.get_ranked_tags_map(
            chat_fnames=priority_abs,
            other_fnames=sorted(other_set - set(priority_abs)),
        )
        or ""
    )


def _collect_python_files(repo_path: str, include_tests: bool) -> list[str]:
    """Walk the repo and collect Python file paths, respecting PRUNE_DIRS."""
    result = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in PRUNE_DIRS]

        rel_dir = os.path.relpath(dirpath, repo_path)
        if not include_tests and (rel_dir.startswith("test") or "/test" in rel_dir):
            continue

        for fname in filenames:
            if fname.endswith(".py"):
                if not include_tests and fname.startswith("test_"):
                    continue
                result.append(os.path.join(dirpath, fname))
    return sorted(result)
