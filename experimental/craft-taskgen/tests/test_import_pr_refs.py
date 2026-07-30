# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic PR-reference importer."""

from __future__ import annotations

import json
from pathlib import Path

from craft_taskgen.importers.pr_refs import run_import
from craft_taskgen.miner import Candidate


def test_run_import_handles_keyed_json_map(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "task_map.json"
    payload = {
        "craft-click-17b8cad6": {
            "pr_url": "https://github.com/pallets/click/pull/3152",
            "repo": "pallets/click",
            "task_type": "bug_fix",
            "matched_practice": "systematic-debugging",
        },
        "craft-click-32d9d2a4": {
            "pr_url": "https://github.com/pallets/click/pull/3152",
            "repo": "pallets/click",
            "task_type": "feature",
            "matched_practice": "brainstorming",
        },
        "craft-arrow-91008b60": {
            "pr_url": "https://github.com/arrow-py/arrow/pull/1222",
            "repo": "arrow-py/arrow",
            "task_type": "feature",
            "matched_practice": "test-driven-development",
        },
        "bad-record": {"repo": "missing/pr"},
    }
    input_path.write_text(json.dumps(payload))

    repos_dir = tmp_path / "repos"
    out_dir = tmp_path / "candidates"
    click_repo = repos_dir / "click"
    arrow_repo = repos_dir / "arrow"
    click_repo.mkdir(parents=True)
    arrow_repo.mkdir(parents=True)

    def fake_clone(github_repo: str, _repos_dir: Path) -> Path | None:
        if github_repo == "pallets/click":
            return click_repo
        if github_repo == "arrow-py/arrow":
            return arrow_repo
        return None

    def fake_fetch(github_repo: str, pr_number: int):
        if github_repo == "pallets/click" and pr_number == 3152:
            return {
                "sha": "merge-click",
                "base_sha": "base-click",
                "subject": "Fix parser edge case",
                "author": "alice",
                "date": "2026-04-01T12:00:00Z",
                "pr_number": 3152,
            }
        if github_repo == "arrow-py/arrow" and pr_number == 1222:
            return {
                "sha": "merge-arrow",
                "base_sha": "base-arrow",
                "subject": "Add datetime helper",
                "author": "bob",
                "date": "2026-04-02T12:00:00Z",
                "pr_number": 1222,
            }
        return None

    def fake_analyze(_repo_path: Path, pr: dict) -> Candidate:
        score = 6.0 if pr["sha"] == "merge-click" else 4.0
        return Candidate(
            sha=pr["sha"],
            base_sha=pr["base_sha"],
            merge_base_sha=f"mb-{pr['sha']}",
            subject=pr["subject"],
            author=pr["author"],
            date=pr["date"],
            source_files=["pkg/mod.py"],
            test_files=["tests/test_mod.py"],
            source_lines_changed=25,
            test_lines_changed=10,
            has_test_patch=True,
            score=score,
            score_breakdown={"fake": score},
        )

    monkeypatch.setattr("craft_taskgen.importers.pr_refs._clone_or_find_repo", fake_clone)
    monkeypatch.setattr("craft_taskgen.importers.pr_refs.fetch_merged_pr", fake_fetch)
    monkeypatch.setattr("craft_taskgen.importers.pr_refs.analyze_pr", fake_analyze)

    summary = run_import(
        input_path=input_path,
        out_dir=out_dir,
        repos_dir=repos_dir,
        top_per_repo=0,
        min_score=0.0,
    )

    assert summary["rows_total"] == 4
    assert summary["rows_unresolved"] == 1
    assert summary["repos_written"] == 2
    assert summary["prs_scanned"] == 2
    assert summary["candidates_kept"] == 3

    click_json = json.loads((out_dir / "click.json").read_text())
    assert click_json["n_input_records"] == 2
    assert click_json["n_prs_scanned"] == 1
    assert click_json["n_candidates"] == 2
    ids = {c["source_task_id"] for c in click_json["candidates"]}
    assert ids == {"craft-click-17b8cad6", "craft-click-32d9d2a4"}
    for cand in click_json["candidates"]:
        assert cand["source_task_type"] in {"bug_fix", "feature"}
        assert "source_matched_practice" in cand
        assert cand["source_metadata"]["repo"] == "pallets/click"
        assert cand["source_metadata"]["pr_number"] == 3152
