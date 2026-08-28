# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SWE-bench Pro importer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from craft_taskgen.importers.swebench_pro import run_import
from craft_taskgen.steps import select_candidates


def _row(
    *,
    repo: str = "gravitational/teleport",
    instance_id: str = "instance_gravitational__teleport-0123456789abcdef0123456789abcdef01234567",
    base_commit: str = "base123",
    patch: str = "",
    test_patch: str = "",
    problem_statement: str = "# Title: Fix kube login",
) -> dict[str, str]:
    return {
        "repo": repo,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "patch": patch,
        "test_patch": test_patch,
        "problem_statement": problem_statement,
    }


def test_run_import_basic_and_select_compatible(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        json.dumps(
            _row(
                patch=(
                    "diff --git a/tool/tsh/kube.py b/tool/tsh/kube.py\n"
                    "--- a/tool/tsh/kube.py\n"
                    "+++ b/tool/tsh/kube.py\n"
                    "@@ -1,1 +1,2 @@\n"
                    "-old\n"
                    "+new\n"
                    "+new2\n"
                ),
                test_patch=(
                    "diff --git a/tool/tsh/test_kube.py b/tool/tsh/test_kube.py\n"
                    "--- a/tool/tsh/test_kube.py\n"
                    "+++ b/tool/tsh/test_kube.py\n"
                    "@@ -1,1 +1,11 @@\n"
                    "-old\n"
                    "+new\n+1\n+2\n+3\n+4\n+5\n+6\n+7\n+8\n+9\n+10\n"
                ),
            )
        )
        + "\n"
    )

    out_dir = tmp_path / "candidates"
    repos_dir = tmp_path / "repos"
    repo_path = repos_dir / "teleport"
    (repo_path / ".git").mkdir(parents=True)
    fake_result = MagicMock(returncode=0, stdout="mergebase123\n", stderr="")
    with patch("craft_taskgen.importers.swebench_pro.subprocess.run", return_value=fake_result):
        summary = run_import(
            input_path=input_path,
            out_dir=out_dir,
            repos_dir=repos_dir,
            top_per_repo=0,
            min_score=0.0,
        )

    assert summary["rows_total"] == 1
    assert summary["rows_unresolved"] == 0
    data = json.loads((out_dir / "teleport.json").read_text())
    assert data["repo"] == "teleport"
    candidate = data["candidates"][0]
    assert candidate["sha"] == "0123456789abcdef0123456789abcdef01234567"
    assert candidate["base_sha"] == "base123"
    assert candidate["merge_base_sha"] == "mergebase123"
    assert candidate["subject"] == "Title: Fix kube login"
    assert (
        candidate["source_task_id"]
        == "instance_gravitational__teleport-0123456789abcdef0123456789abcdef01234567"
    )
    assert candidate["source_metadata"]["repo"] == "gravitational/teleport"

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        selected = select_candidates([str(out_dir / "teleport.json")], top_per_repo=5, max_total=10)
        assert len(selected) == 1
        assert selected[0]["repo"] == "teleport"


def test_run_import_uses_commit_sha_from_instance_id_and_counts(tmp_path) -> None:
    source_patch = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "-one\n-two\n"
        "+one\n+two\n+three\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n"
        "--- a/pkg/b.py\n"
        "+++ b/pkg/b.py\n"
        "@@ -1 +1,2 @@\n"
        "-x\n+y\n+z\n"
    )
    test_patch = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n"
        "--- a/tests/test_a.py\n"
        "+++ b/tests/test_a.py\n"
        "@@ -1 +1,3 @@\n"
        "-old\n+new\n+new2\n+new3\n"
    )
    row = _row(
        instance_id="instance_gravitational__teleport-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        patch=source_patch,
        test_patch=test_patch,
        problem_statement="Fix issue",
    )
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    out_dir = tmp_path / "candidates"
    repos_dir = tmp_path / "repos"
    repo_path = repos_dir / "teleport"
    (repo_path / ".git").mkdir(parents=True)
    fake_result = MagicMock(returncode=0, stdout="mergebase123\n", stderr="")
    with patch("craft_taskgen.importers.swebench_pro.subprocess.run", return_value=fake_result):
        run_import(
            input_path=input_path,
            out_dir=out_dir,
            repos_dir=repos_dir,
            top_per_repo=0,
            min_score=0.0,
        )
    data = json.loads((out_dir / "teleport.json").read_text())

    assert len(data["candidates"]) == 2
    first, second = data["candidates"]
    assert first["sha"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert second["sha"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert first["source_files"] == ["src/a.py", "pkg/b.py"]
    assert first["test_files"] == ["tests/test_a.py"]
    assert first["source_lines_changed"] == 8
    assert first["test_lines_changed"] == 4
    assert first["package_names"] == ["pkg", "src"]
    assert first["packages_touched"] == 2


def test_run_import_ignores_suffix_after_commit_sha_in_instance_id(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    row = _row(
        repo="NodeBB/NodeBB",
        instance_id="instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
        patch=(
            "diff --git a/src/a.py b/src/a.py\n"
            "--- a/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -1 +1,2 @@\n"
            "-old\n+new\n+new2\n"
        ),
        test_patch=(
            "diff --git a/tests/test_a.py b/tests/test_a.py\n"
            "--- a/tests/test_a.py\n"
            "+++ b/tests/test_a.py\n"
            "@@ -1 +1,12 @@\n"
            "-old\n" + "".join(f"+line{i}\n" for i in range(12))
        ),
    )
    input_path.write_text(json.dumps(row) + "\n")

    out_dir = tmp_path / "candidates"
    repos_dir = tmp_path / "repos"
    repo_path = repos_dir / "NodeBB"
    (repo_path / ".git").mkdir(parents=True)
    fake_result = MagicMock(returncode=0, stdout="mergebase123\n", stderr="")
    with patch("craft_taskgen.importers.swebench_pro.subprocess.run", return_value=fake_result):
        run_import(input_path=input_path, out_dir=out_dir, repos_dir=repos_dir, top_per_repo=0, min_score=0.0)
    data = json.loads((out_dir / "NodeBB.json").read_text())
    assert data["candidates"][0]["sha"] == "04998908ba6721d64eba79ae3b65a351dcfbc5b5"


def test_run_import_fails_fast_for_missing_required_fields(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        json.dumps(_row(repo="", patch="diff --git a/a b/a", test_patch=""))
        + "\n"
        + json.dumps(_row(base_commit="", patch="diff --git a/a b/a", test_patch=""))
        + "\n"
        + json.dumps(_row(patch="", test_patch=""))
        + "\n"
    )

    out_dir = tmp_path / "candidates"
    summary = run_import(
        input_path=input_path,
        out_dir=out_dir,
        repos_dir=tmp_path / "repos",
        top_per_repo=0,
        min_score=0.0,
    )
    assert summary["rows_total"] == 3
    assert summary["rows_unresolved"] == 3


def test_run_import_fails_fast_for_non_string_patch(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    row = _row(patch="diff --git a/src/a.py b/src/a.py\n")
    row["test_patch"] = {"not": "a string"}
    input_path.write_text(json.dumps(row) + "\n")

    out_dir = tmp_path / "candidates"
    summary = run_import(
        input_path=input_path,
        out_dir=out_dir,
        repos_dir=tmp_path / "repos",
        top_per_repo=0,
        min_score=0.0,
    )
    assert summary["rows_total"] == 1
    assert summary["rows_unresolved"] == 1


def test_run_import_fails_fast_when_instance_id_has_no_commit_sha(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    row = _row(instance_id="teleport-no-sha", patch="diff --git a/src/a.py b/src/a.py\n")
    input_path.write_text(json.dumps(row) + "\n")

    out_dir = tmp_path / "candidates"
    summary = run_import(
        input_path=input_path,
        out_dir=out_dir,
        repos_dir=tmp_path / "repos",
        top_per_repo=0,
        min_score=0.0,
    )
    assert summary["rows_total"] == 1
    assert summary["rows_unresolved"] == 1


def test_run_import_fails_fast_when_repo_clone_missing(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    row = _row(patch="diff --git a/src/a.py b/src/a.py\n")
    input_path.write_text(json.dumps(row) + "\n")

    out_dir = tmp_path / "candidates"
    summary = run_import(
        input_path=input_path,
        out_dir=out_dir,
        repos_dir=tmp_path / "repos",
        top_per_repo=0,
        min_score=0.0,
    )
    assert summary["rows_total"] == 1
    assert summary["rows_unresolved"] == 1


def test_run_import_fails_fast_when_merge_base_fails(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    row = _row(patch="diff --git a/src/a.py b/src/a.py\n")
    input_path.write_text(json.dumps(row) + "\n")

    out_dir = tmp_path / "candidates"
    repos_dir = tmp_path / "repos"
    repo_path = repos_dir / "teleport"
    (repo_path / ".git").mkdir(parents=True)
    summary = run_import(
        input_path=input_path,
        out_dir=out_dir,
        repos_dir=repos_dir,
        top_per_repo=0,
        min_score=0.0,
    )
    assert summary["rows_total"] == 1
    assert summary["rows_unresolved"] == 1


def test_run_import_zero_score_without_test_patch_and_groups_by_repo(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    row1 = _row(
        repo="gravitational/teleport",
        instance_id="instance_gravitational__teleport-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        patch=(
            "diff --git a/src/a.py b/src/a.py\n"
            "--- a/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -1 +1,40 @@\n"
            "-old\n" + "".join(f"+line{i}\n" for i in range(40))
        ),
        test_patch="",
        problem_statement="No tests here",
    )
    row2 = _row(
        repo="pallets/click",
        instance_id="instance_pallets__click-cccccccccccccccccccccccccccccccccccccccc",
        patch=(
            "diff --git a/src/b.py b/src/b.py\n"
            "--- a/src/b.py\n"
            "+++ b/src/b.py\n"
            "@@ -1 +1,2 @@\n"
            "-old\n+new\n+new2\n"
        ),
        test_patch=(
            "diff --git a/tests/test_b.py b/tests/test_b.py\n"
            "--- a/tests/test_b.py\n"
            "+++ b/tests/test_b.py\n"
            "@@ -1 +1,12 @@\n"
            "-old\n" + "".join(f"+line{i}\n" for i in range(12))
        ),
        problem_statement="Click tests",
    )
    input_path.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")

    out_dir = tmp_path / "candidates"
    repos_dir = tmp_path / "repos"
    for name in ["teleport", "click"]:
        ((repos_dir / name) / ".git").mkdir(parents=True)
    fake_result = MagicMock(returncode=0, stdout="mergebase123\n", stderr="")
    with patch("craft_taskgen.importers.swebench_pro.subprocess.run", return_value=fake_result):
        summary = run_import(
            input_path=input_path,
            out_dir=out_dir,
            repos_dir=repos_dir,
            top_per_repo=0,
            min_score=0.0,
        )

    assert summary["repos_written"] == 2
    teleport = json.loads((out_dir / "teleport.json").read_text())
    click = json.loads((out_dir / "click.json").read_text())
    assert teleport["candidates"][0]["has_test_patch"] is False
    assert teleport["candidates"][0]["score"] == 0.0
    assert click["candidates"][0]["has_test_patch"] is True
