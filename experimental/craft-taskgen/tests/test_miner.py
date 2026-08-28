# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for miner.py — PR-first candidate mining."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from craft_taskgen.miner import (
    Candidate,
    analyze_pr,
    get_diff_stats,
    get_prs,
    mine_repo,
)

# ---------------------------------------------------------------------------
# get_diff_stats
# ---------------------------------------------------------------------------


def test_get_diff_stats_uses_two_sha_diff(tmp_path):
    """get_diff_stats must find merge-base then diff against it (not raw base_sha)."""
    with patch("craft_taskgen.miner.git") as mock_git:
        mock_git.side_effect = [
            "mergebase0\n",  # first call: merge-base base000 head111
            "5\t2\tsrc/foo.py\n3\t1\ttests/test_foo.py\n",  # second call: diff --numstat mergebase0 head111
        ]
        result = get_diff_stats(tmp_path, "base000", "head111")

    calls = mock_git.call_args_list
    assert calls[0][0] == (tmp_path, "merge-base", "base000", "head111")
    assert calls[1][0] == (tmp_path, "diff", "--numstat", "mergebase0", "head111")
    assert len(result["files"]) == 2
    assert result["files"][0] == {"path": "src/foo.py", "added": 5, "deleted": 2}


# ---------------------------------------------------------------------------
# Candidate.base_sha
# ---------------------------------------------------------------------------


def test_candidate_requires_base_sha():
    """Candidate must not accept construction without base_sha."""
    with pytest.raises(TypeError):
        Candidate(sha="abc123", subject="fix: thing", author="alice", date="2025-01-01T00:00:00Z")


def test_candidate_with_base_sha():
    """Candidate constructs normally when base_sha and merge_base_sha are provided."""
    c = Candidate(
        sha="abc123",
        base_sha="def456",
        merge_base_sha="ghi789",
        subject="fix: thing",
        author="alice",
        date="2025-01-01T00:00:00Z",
    )
    assert c.base_sha == "def456"
    assert c.merge_base_sha == "ghi789"


# ---------------------------------------------------------------------------
# analyze_pr
# ---------------------------------------------------------------------------


def test_analyze_pr_populates_both_shas(tmp_path):
    """analyze_pr must set sha, base_sha, and merge_base_sha from the PR dict and diff stats."""
    pr = {
        "sha": "merge111",
        "base_sha": "base000",
        "subject": "feat: add widget",
        "author": "alice",
        "date": "2025-06-01T10:00:00Z",
        "pr_number": 42,
    }
    with patch("craft_taskgen.miner.get_diff_stats") as mock_stats:
        mock_stats.return_value = {
            "files": [
                {"path": "src/widget.py", "added": 50, "deleted": 5},
                {"path": "src/core.py", "added": 30, "deleted": 2},
                {"path": "tests/test_widget.py", "added": 40, "deleted": 0},
            ],
            "merge_base_sha": "mergebase000",
        }
        candidate = analyze_pr(tmp_path, pr)

    mock_stats.assert_called_once_with(tmp_path, "base000", "merge111")
    assert candidate.sha == "merge111"
    assert candidate.base_sha == "base000"
    assert candidate.merge_base_sha == "mergebase000"
    assert candidate.has_test_patch is True


# ---------------------------------------------------------------------------
# get_prs
# ---------------------------------------------------------------------------

_FAKE_PR_PAGE = [
    {
        "number": 42,
        "title": "feat: add widget",
        "user": {"login": "alice"},
        "merged_at": "2025-06-01T10:00:00Z",
        "merge_commit_sha": "merge111aaa",
        "base": {"sha": "base000bbb"},
    },
    {
        "number": 41,
        "title": "fix: rename typo",
        "user": {"login": "bob"},
        "merged_at": "2025-05-15T08:00:00Z",
        "merge_commit_sha": "merge222ccc",
        "base": {"sha": "base333ddd"},
    },
    {
        "number": 40,
        "title": "fix: something",
        "user": {"login": "bob"},
        "merged_at": None,  # not merged — should be skipped
        "merge_commit_sha": None,
        "base": {"sha": "base444eee"},
    },
]

_OLD_PAGE = [
    {
        "number": 10,
        "title": "old fix",
        "user": {"login": "alice"},
        "merged_at": "2024-12-01T00:00:00Z",
        "merge_commit_sha": "oldsha",
        "base": {"sha": "oldbase"},
    }
]


def _gh_responses(*pages: list) -> list[MagicMock]:
    """Build subprocess.run mocks for paged gh api calls, ending with empty page."""
    results = []
    for page in pages:
        m = MagicMock()
        m.returncode = 0
        m.stdout = json.dumps(page)
        results.append(m)
    empty = MagicMock()
    empty.returncode = 0
    empty.stdout = json.dumps([])
    results.append(empty)
    return results


def test_get_prs_returns_merged_only():
    """get_prs skips PRs where merged_at is None."""
    with patch("subprocess.run", side_effect=_gh_responses(_FAKE_PR_PAGE)):
        prs = get_prs("owner/repo")

    assert len(prs) == 2  # PR 40 (unmerged) excluded
    assert prs[0]["pr_number"] == 42
    assert prs[0]["sha"] == "merge111aaa"
    assert prs[0]["base_sha"] == "base000bbb"
    assert prs[0]["subject"] == "feat: add widget"
    assert prs[0]["author"] == "alice"


def test_get_prs_raises_on_gh_failure():
    """get_prs raises RuntimeError if gh api returns non-zero."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "authentication required"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="gh api failed"):
            get_prs("owner/repo")


def test_get_prs_stops_paging_at_after_date():
    """get_prs stops fetching pages when first merged PR on a page predates --after."""
    with patch("subprocess.run", side_effect=_gh_responses(_FAKE_PR_PAGE, _OLD_PAGE)) as mock_run:
        prs = get_prs("owner/repo", after="2025-01-01")

    # Only 2 merged PRs from page 1 pass the date filter; page 2 (2024) triggers stop
    assert len(prs) == 2
    assert all(p["date"] >= "2025-01-01" for p in prs)
    # Exactly 2 subprocess calls: page 1 + page 2 (which triggered stop; no page 3)
    assert mock_run.call_count == 2


# ---------------------------------------------------------------------------
# mine_repo
# ---------------------------------------------------------------------------


def test_mine_repo_returns_candidates_and_pr_count(tmp_path):
    """mine_repo returns (candidates, n_prs_scanned). Candidates have base_sha and merge_base_sha."""
    fake_prs = [
        {
            "sha": "merge111",
            "base_sha": "base000",
            "subject": "feat: add lots of files",
            "author": "alice",
            "date": "2025-06-01T10:00:00Z",
            "pr_number": 1,
        },
        {
            "sha": "merge222",
            "base_sha": "base111",
            "subject": "docs: update readme",  # will score 0 — no tests
            "author": "bob",
            "date": "2025-05-01T10:00:00Z",
            "pr_number": 2,
        },
    ]
    good_stats = {
        "files": [
            {"path": "src/a.py", "added": 60, "deleted": 0},
            {"path": "src/b.py", "added": 40, "deleted": 0},
            {"path": "src/c.py", "added": 30, "deleted": 0},
            {"path": "tests/test_a.py", "added": 50, "deleted": 0},
        ],
        "merge_base_sha": "mergebase000",
    }
    no_test_stats = {
        "files": [{"path": "README.md", "added": 5, "deleted": 0}],
        "merge_base_sha": "mergebase111",
    }

    def fake_diff_stats(_repo, base_sha, _sha):
        return good_stats if base_sha == "base000" else no_test_stats

    with (
        patch("craft_taskgen.miner.get_prs", return_value=fake_prs),
        patch("craft_taskgen.miner.get_diff_stats", side_effect=fake_diff_stats),
    ):
        results, n_prs = mine_repo(tmp_path, "owner/repo", after=None, top_n=5)

    assert n_prs == 2  # total PRs fetched
    assert len(results) == 1  # only PR 1 scored > 0
    assert results[0]["sha"] == "merge111"
    assert results[0]["base_sha"] == "base000"
    assert results[0]["merge_base_sha"] == "mergebase000"
