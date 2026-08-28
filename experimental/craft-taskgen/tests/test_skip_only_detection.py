# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the skip-only fast-path detection used by the triage step.

The fix-agent triage path snapshots the task dir before the agent runs, then
diffs after. If only f2p_skip.txt / p2p_skip.txt actually changed, the pipeline
fast-paths to _rescore_trial instead of re-running alignment / classify / oracle
/ smoke. The diagnostics/ dir must be excluded because the pipeline writes its
own fix log there, which would otherwise defeat the fast-path.
"""

from __future__ import annotations

import os

from craft_taskgen.steps import (
    SKIP_ONLY_FILES,
    _diff_task_files,
    _is_skip_only_change,
    _load_skipped_tests,
    _snapshot_task_files,
)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _build_task_dir(tmp_path) -> str:
    task_dir = str(tmp_path / "task")
    _write(os.path.join(task_dir, "instruction.md"), "do the thing")
    _write(os.path.join(task_dir, "task.toml"), "[task]\nname = 'x'\n")
    _write(os.path.join(task_dir, "environment", "Dockerfile"), "FROM python:3.12\n")
    _write(os.path.join(task_dir, "tests", "fail_to_pass.txt"), "test_a\ntest_b\n")
    _write(os.path.join(task_dir, "tests", "pass_to_pass.txt"), "test_c\n")
    return task_dir


def test_snapshot_captures_all_files(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    snap = _snapshot_task_files(task_dir)
    assert "instruction.md" in snap
    assert "task.toml" in snap
    assert os.path.join("environment", "Dockerfile") in snap
    assert os.path.join("tests", "fail_to_pass.txt") in snap


def test_snapshot_empty_task_dir(tmp_path):
    assert _snapshot_task_files("") == {}
    assert _snapshot_task_files(str(tmp_path / "nonexistent")) == {}


def test_diff_detects_new_skip_file(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    pre = _snapshot_task_files(task_dir)
    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "test_a | format only\n")
    changed = _diff_task_files(task_dir, pre)
    assert changed == {"f2p_skip.txt"}


def test_diff_detects_modified_file(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    pre = _snapshot_task_files(task_dir)
    _write(os.path.join(task_dir, "instruction.md"), "do the thing, but differently")
    changed = _diff_task_files(task_dir, pre)
    assert changed == {"instruction.md"}


def test_diff_ignores_diagnostics_dir(tmp_path):
    """Pipeline writes fix logs to diagnostics/ — must not defeat skip-only fast-path."""
    task_dir = _build_task_dir(tmp_path)
    pre = _snapshot_task_files(task_dir)
    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "test_a | format only\n")
    _write(os.path.join(task_dir, "diagnostics", "006_fix.md"), "# Fix Attempt 1\n...")
    changed = _diff_task_files(task_dir, pre)
    assert changed == {"f2p_skip.txt"}


def test_diff_ignores_existing_diagnostic_changes(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    _write(os.path.join(task_dir, "diagnostics", "001_eval.md"), "original")
    pre = _snapshot_task_files(task_dir)
    _write(os.path.join(task_dir, "diagnostics", "001_eval.md"), "modified")
    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "test_a | reason\n")
    changed = _diff_task_files(task_dir, pre)
    assert changed == {"f2p_skip.txt"}


def test_is_skip_only_change_true_for_f2p(tmp_path):
    assert _is_skip_only_change({"f2p_skip.txt"}) is True


def test_is_skip_only_change_true_for_p2p(tmp_path):
    assert _is_skip_only_change({"p2p_skip.txt"}) is True


def test_is_skip_only_change_true_for_both(tmp_path):
    assert _is_skip_only_change({"f2p_skip.txt", "p2p_skip.txt"}) is True


def test_is_skip_only_change_false_when_instruction_changed():
    assert _is_skip_only_change({"f2p_skip.txt", "instruction.md"}) is False


def test_is_skip_only_change_false_when_dockerfile_changed():
    assert _is_skip_only_change({"Dockerfile"}) is False


def test_is_skip_only_change_false_when_empty():
    """Fix agent claimed success but changed nothing — don't fast-path."""
    assert _is_skip_only_change(set()) is False


def test_full_skip_only_flow(tmp_path):
    """End-to-end: fix agent writes skip file + pipeline writes diagnostic."""
    task_dir = _build_task_dir(tmp_path)
    pre = _snapshot_task_files(task_dir)

    # Simulate fix agent: writes f2p_skip.txt
    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "test_a | format only\n")
    # Simulate pipeline: writes diagnostic bookkeeping (must not defeat fast-path)
    _write(os.path.join(task_dir, "diagnostics", "006_fix.md"), "# Fix Attempt 1\n")

    changed = _diff_task_files(task_dir, pre)
    assert _is_skip_only_change(changed) is True


def test_full_flow_instruction_change_breaks_fast_path(tmp_path):
    """When fix agent also edits instruction.md, must NOT fast-path."""
    task_dir = _build_task_dir(tmp_path)
    pre = _snapshot_task_files(task_dir)

    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "test_a | reason\n")
    _write(os.path.join(task_dir, "instruction.md"), "tightened instruction")
    _write(os.path.join(task_dir, "diagnostics", "006_fix.md"), "log")

    changed = _diff_task_files(task_dir, pre)
    assert _is_skip_only_change(changed) is False


def test_skip_only_files_constant():
    """Sanity: the constant used by the production code matches what we test against."""
    assert SKIP_ONLY_FILES == frozenset({"f2p_skip.txt", "p2p_skip.txt"})


# ---------------------------------------------------------------------------
# _load_skipped_tests — used to filter deep-dive failures that are already
# excluded via f2p_skip.txt / p2p_skip.txt. Without this filter the pipeline
# loops: pytest still runs skipped tests (they show FAILED in output), deep
# dive flags them, reviewer rubber-stamps, fix agent polishes the skip file
# cosmetically, full rebuild runs, repeat.
# ---------------------------------------------------------------------------


def test_load_skipped_tests_empty_task_dir():
    assert _load_skipped_tests("") == set()


def test_load_skipped_tests_no_skip_files(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    assert _load_skipped_tests(task_dir) == set()


def test_load_skipped_tests_f2p_only(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    _write(
        os.path.join(task_dir, "tests", "f2p_skip.txt"),
        "tests/test_x.py::test_a | format only\ntests/test_x.py::test_b | not relevant\n",
    )
    skipped = _load_skipped_tests(task_dir)
    assert skipped == {
        "tests/test_x.py::test_a",
        "test_a",
        "tests/test_x.py::test_b",
        "test_b",
    }


def test_load_skipped_tests_merges_f2p_and_p2p(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "tests/test_x.py::test_a | r\n")
    _write(os.path.join(task_dir, "tests", "p2p_skip.txt"), "tests/test_y.py::test_b | r\n")
    skipped = _load_skipped_tests(task_dir)
    assert skipped == {
        "tests/test_x.py::test_a",
        "test_a",
        "tests/test_y.py::test_b",
        "test_b",
    }


def test_load_skipped_tests_ignores_comments_and_blanks(tmp_path):
    task_dir = _build_task_dir(tmp_path)
    _write(
        os.path.join(task_dir, "tests", "f2p_skip.txt"),
        "# header comment\n\ntests/test_x.py::test_a | reason\n   \n# another\n",
    )
    assert _load_skipped_tests(task_dir) == {"tests/test_x.py::test_a", "test_a"}


def test_load_skipped_tests_without_pipe(tmp_path):
    """Bare test_id lines (no pipe / reason) are still valid."""
    task_dir = _build_task_dir(tmp_path)
    _write(os.path.join(task_dir, "tests", "f2p_skip.txt"), "tests/test_x.py::test_a\n")
    assert _load_skipped_tests(task_dir) == {"tests/test_x.py::test_a", "test_a"}


def test_filter_deep_dive_failures_by_skip_set(tmp_path):
    """The triage filter: drop failures whose test_name is already in skip files."""
    task_dir = _build_task_dir(tmp_path)
    _write(
        os.path.join(task_dir, "tests", "f2p_skip.txt"),
        "tests/test_x.py::test_skipped_a | reason\ntests/test_x.py::test_skipped_b | reason\n",
    )
    already_skipped = _load_skipped_tests(task_dir)

    raw_failures = [
        {"test_name": "tests/test_x.py::test_skipped_a", "classification": "instruction_scope"},
        {"test_name": "tests/test_x.py::test_real_gap", "classification": "genuine_gap"},
        {"test_name": "tests/test_x.py::test_skipped_b", "classification": "test_format_only"},
    ]
    filtered = [f for f in raw_failures if f.get("test_name") not in already_skipped]

    assert len(filtered) == 1
    assert filtered[0]["test_name"] == "tests/test_x.py::test_real_gap"


def test_filter_empty_when_all_already_skipped(tmp_path):
    """Reward=1.0 loop case: deep dive flags only tests already in skip files."""
    task_dir = _build_task_dir(tmp_path)
    _write(
        os.path.join(task_dir, "tests", "f2p_skip.txt"),
        "tests/test_x.py::a | r\ntests/test_x.py::b | r\ntests/test_x.py::c | r\n",
    )
    already_skipped = _load_skipped_tests(task_dir)

    raw_failures = [
        {"test_name": f"tests/test_x.py::{t}", "classification": "instruction_scope"} for t in ("a", "b", "c")
    ]
    filtered = [f for f in raw_failures if f.get("test_name") not in already_skipped]

    # Empty failures → triage loop's all_genuine stays True → task accepts, no fix cycle.
    assert filtered == []


def test_filter_deep_dive_short_names_matched_by_skip(tmp_path):
    """Deep dive often reports short test names; skip file has full paths."""
    task_dir = _build_task_dir(tmp_path)
    _write(
        os.path.join(task_dir, "tests", "f2p_skip.txt"),
        "tests/test_x.py::test_skipped_a | reason\ntests/test_x.py::test_skipped_b | reason\n",
    )
    already_skipped = _load_skipped_tests(task_dir)

    raw_failures = [
        {"test_name": "test_skipped_a", "classification": "instruction_scope"},
        {"test_name": "test_real_gap", "classification": "genuine_gap"},
        {"test_name": "test_skipped_b", "classification": "test_format_only"},
    ]
    filtered = [f for f in raw_failures if f.get("test_name") not in already_skipped]

    assert len(filtered) == 1
    assert filtered[0]["test_name"] == "test_real_gap"


# (No tests for _remaining_failures_all_genuine — that helper no
# longer exists; `_run_triage_one` decides rebuild-or-accept directly
# from the reviewer severity and post-skip F2P count.)
