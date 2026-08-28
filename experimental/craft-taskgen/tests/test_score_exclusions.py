# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for F2P exclusion support in SCORE_PY_TEMPLATE."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from craft_taskgen.prompts import SCORE_PY_TEMPLATE


def _setup_score_env(tmp_path, f2p_tests, p2p_tests, passed_tests, exclusions=None):
    """Write score.py + supporting files into tmp_path, mimicking the container layout.

    Paths inside the template are absolute (/tests/*, /logs/verifier/*), so we
    rewrite them to point at tmp_path before executing.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    logs_dir = tmp_path / "logs" / "verifier"
    logs_dir.mkdir(parents=True)

    # Write F2P / P2P lists
    (tests_dir / "fail_to_pass.txt").write_text("\n".join(f2p_tests) + "\n" if f2p_tests else "")
    (tests_dir / "pass_to_pass.txt").write_text("\n".join(p2p_tests) + "\n" if p2p_tests else "")

    # Write exclusions file (or leave it absent)
    if exclusions is not None:
        (tests_dir / "f2p_skip.txt").write_text(exclusions)

    # Build verify_full_output.txt from the passed list
    lines = []
    for t in passed_tests:
        lines.append(f"{t} PASSED")
    (logs_dir / "verify_full_output.txt").write_text("\n".join(lines) + "\n")

    # Rewrite absolute paths in the template so they resolve inside tmp_path
    script = SCORE_PY_TEMPLATE
    script = script.replace('Path("/logs/verifier")', f'Path("{logs_dir}")')
    script = script.replace('"/tests/fail_to_pass.txt"', f'"{tests_dir / "fail_to_pass.txt"}"')
    script = script.replace('"/tests/pass_to_pass.txt"', f'"{tests_dir / "pass_to_pass.txt"}"')
    script = script.replace('"/tests/f2p_skip.txt"', f'"{tests_dir / "f2p_skip.txt"}"')
    script = script.replace('"/tests/p2p_skip.txt"', f'"{tests_dir / "p2p_skip.txt"}"')

    score_py = tmp_path / "score.py"
    score_py.write_text(script)
    return score_py, logs_dir


def _run_score(score_py):
    """Execute score.py and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(score_py)],
        capture_output=True,
        text=True,
    )
    return result


def _load_reward(logs_dir):
    return json.loads((logs_dir / "reward.json").read_text())


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #


class TestNoExclusionFile:
    """score.py with no f2p_skip.txt — unchanged behavior."""

    def test_all_pass(self, tmp_path):
        f2p = ["tests/test_a.py::test_one", "tests/test_a.py::test_two"]
        p2p = ["tests/test_b.py::test_existing"]
        passed = f2p + p2p

        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed)
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr

        reward = _load_reward(logs_dir)
        assert reward["reward"] == 1.0
        assert reward["f2p_passed"] == 2
        assert reward["f2p_total"] == 2
        assert reward["f2p_skipped"] == 0
        assert reward["f2p_total_before_skips"] == 2

    def test_partial_pass(self, tmp_path):
        f2p = ["tests/test_a.py::test_one", "tests/test_a.py::test_two"]
        p2p = []
        passed = ["tests/test_a.py::test_one"]

        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed)
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr

        reward = _load_reward(logs_dir)
        assert reward["reward"] == 0.0
        assert reward["f2p_passed"] == 1
        assert reward["f2p_total"] == 2
        assert reward["f2p_score"] == 0.5


class TestEmptyExclusionFile:
    """score.py with empty f2p_skip.txt — no effect."""

    def test_empty_file(self, tmp_path):
        f2p = ["tests/test_a.py::test_one"]
        p2p = []
        passed = ["tests/test_a.py::test_one"]

        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed, exclusions="")
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr

        reward = _load_reward(logs_dir)
        assert reward["f2p_total"] == 1
        assert reward["f2p_skipped"] == 0
        assert reward["f2p_total_before_skips"] == 1


class TestValidExclusions:
    """score.py with exclusions that match F2P tests — f2p_total reduced."""

    def test_one_excluded(self, tmp_path):
        f2p = ["tests/test_a.py::test_one", "tests/test_a.py::test_two", "tests/test_a.py::test_three"]
        p2p = ["tests/test_b.py::test_existing"]
        passed = [
            "tests/test_a.py::test_one",
            "tests/test_a.py::test_three",
            "tests/test_b.py::test_existing",
        ]

        exclusions = "tests/test_a.py::test_two | not relevant to feature\n"
        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed, exclusions=exclusions)
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr

        reward = _load_reward(logs_dir)
        # test_two excluded, test_one and test_three passed => 2/2 => resolved
        assert reward["reward"] == 1.0
        assert reward["f2p_passed"] == 2
        assert reward["f2p_total"] == 2
        assert reward["f2p_skipped"] == 1
        assert reward["f2p_total_before_skips"] == 3

    def test_exclusion_changes_reward(self, tmp_path):
        """Without exclusion: 1/2 = 0.0. With exclusion of the failed test: 1/1 = 1.0."""
        f2p = ["tests/test_a.py::test_pass", "tests/test_a.py::test_fail"]
        p2p = []
        passed = ["tests/test_a.py::test_pass"]

        # Without exclusions: reward should be 0
        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed)
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr
        reward = _load_reward(logs_dir)
        assert reward["reward"] == 0.0

        # With exclusion of the failing test: reward should be 1.0
        tmp2 = tmp_path / "with_exclusion"
        tmp2.mkdir()
        exclusions = "tests/test_a.py::test_fail | regression test, not new feature\n"
        score_py2, logs_dir2 = _setup_score_env(tmp2, f2p, p2p, passed, exclusions=exclusions)
        result2 = _run_score(score_py2)
        assert result2.returncode == 0, result2.stderr
        reward2 = _load_reward(logs_dir2)
        assert reward2["reward"] == 1.0
        assert reward2["f2p_total"] == 1
        assert reward2["f2p_skipped"] == 1


class TestExclusionNotMatching:
    """Exclusion that doesn't match any F2P test — no-op."""

    def test_nonmatching_exclusion(self, tmp_path):
        f2p = ["tests/test_a.py::test_one"]
        p2p = []
        passed = ["tests/test_a.py::test_one"]

        exclusions = "tests/test_z.py::test_nonexistent | does not exist\n"
        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed, exclusions=exclusions)
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr

        reward = _load_reward(logs_dir)
        assert reward["f2p_total"] == 1
        assert reward["f2p_passed"] == 1
        # Exclusion exists in the file but doesn't match any F2P test
        assert reward["f2p_skipped"] == 0
        assert reward["f2p_total_before_skips"] == 1


class TestCommentsAndBlankLines:
    """Comment lines and blank lines in exclusion file are skipped."""

    def test_comments_and_blanks(self, tmp_path):
        f2p = ["tests/test_a.py::test_one", "tests/test_a.py::test_two"]
        p2p = []
        passed = ["tests/test_a.py::test_one"]

        exclusions = textwrap.dedent("""\
            # This is a comment

            # Another comment
            tests/test_a.py::test_two | bundled regression test

            # trailing comment
        """)
        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed, exclusions=exclusions)
        result = _run_score(score_py)
        assert result.returncode == 0, result.stderr

        reward = _load_reward(logs_dir)
        # test_two excluded, only test_one remains and passed => 1/1 => resolved
        assert reward["reward"] == 1.0
        assert reward["f2p_total"] == 1
        assert reward["f2p_skipped"] == 1
        assert reward["f2p_total_before_skips"] == 2


class TestAllF2PExcluded:
    """All F2P tests skipped — raises ValueError."""

    def test_all_excluded_raises(self, tmp_path):
        f2p = ["tests/test_a.py::test_one", "tests/test_a.py::test_two"]
        p2p = []
        passed = []

        exclusions = textwrap.dedent("""\
            tests/test_a.py::test_one | not relevant
            tests/test_a.py::test_two | also not relevant
        """)
        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed, exclusions=exclusions)
        result = _run_score(score_py)
        assert result.returncode != 0
        assert "All F2P tests skipped" in result.stderr

    def test_empty_f2p_no_exclusions_original_error(self, tmp_path):
        """Empty fail_to_pass.txt with no exclusion file gives original error message."""
        f2p = []
        p2p = []
        passed = []

        score_py, logs_dir = _setup_score_env(tmp_path, f2p, p2p, passed)
        result = _run_score(score_py)
        assert result.returncode != 0
        assert "fail_to_pass.txt is empty" in result.stderr
