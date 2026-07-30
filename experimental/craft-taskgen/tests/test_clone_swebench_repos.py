# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/clone_swebench_repos.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.clone_swebench_repos import clone_repo, collect_unique_repos, repo_clone_path


def test_collect_unique_repos_deduplicates_and_sorts(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    rows = [
        {"repo": "gravitational/teleport", "instance_id": "a"},
        {"repo": "NodeBB/NodeBB", "instance_id": "b"},
        {"repo": "gravitational/teleport", "instance_id": "c"},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert collect_unique_repos(input_path) == ["NodeBB/NodeBB", "gravitational/teleport"]


def test_collect_unique_repos_fails_for_invalid_repo(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(json.dumps({"repo": "", "instance_id": "bad"}) + "\n")

    with pytest.raises(ValueError, match="bad: repo must be a valid GitHub owner/repo slug"):
        collect_unique_repos(input_path)


def test_repo_clone_path_uses_short_repo_layout(tmp_path) -> None:
    assert repo_clone_path(tmp_path, "gravitational/teleport") == tmp_path / "teleport"


def test_clone_repo_skips_existing_clone(tmp_path) -> None:
    target = tmp_path / "teleport"
    (target / ".git").mkdir(parents=True)

    status, cloned_path = clone_repo("gravitational/teleport", tmp_path)

    assert status == "exists"
    assert cloned_path == target


def test_clone_repo_runs_git_clone_into_short_repo_layout(tmp_path) -> None:
    fake_result = MagicMock(returncode=0, stderr="")

    with patch("scripts.clone_swebench_repos.subprocess.run", return_value=fake_result) as mock_run:
        status, cloned_path = clone_repo("gravitational/teleport", tmp_path)

    assert status == "cloned"
    assert cloned_path == tmp_path / "teleport"
    cmd = mock_run.call_args[0][0]
    assert cmd == [
        "git",
        "clone",
        "https://github.com/gravitational/teleport.git",
        str(tmp_path / "teleport"),
    ]
