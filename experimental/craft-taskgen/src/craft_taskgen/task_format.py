# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task format utilities — validates task directory structure and strips disallowed boilerplate."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from craft_taskgen.config import PipelineContext

_ENV_SECTION_RE = re.compile(r"(?:^|\n)#+\s*Environment\b", re.IGNORECASE | re.MULTILINE)


def strip_instruction_boilerplate(path: str) -> None:
    """Strip any '# Environment' heading (any depth) and everything after it.

    The model occasionally appends this section despite prompt instructions not to.
    Matches case-insensitively across any number of leading '#' characters,
    including a heading at byte offset 0. No-ops if the file is missing,
    unreadable, or the pattern is absent.
    """
    try:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    m = _ENV_SECTION_RE.search(text)
    if m:
        stripped = text[: m.start()].rstrip() + "\n"
        try:
            p.write_text(stripped, encoding="utf-8")
            chars_removed = len(text) - len(stripped)
            print(f"    [strip_instruction_boilerplate] removed {chars_removed} chars from {path}")
        except OSError as e:
            print(f"    [strip_instruction_boilerplate] WARNING: failed to write {path}: {e}")


def validate_task_dir(task_dir: str) -> list[str]:
    """Validate a task directory against the CRAFT task format spec.

    Supports two formats:
    - **New format (F2P/P2P):** fail_to_pass.txt + score.py + test.sh
    - **Old format (gold reference):** gold_reference_tests.py + test_runner.py + verify_*.sh

    Returns a list of error messages. Empty list = valid.
    """
    errors: list[str] = []

    # Detect format: new if fail_to_pass.txt exists
    f2p_path = os.path.join(task_dir, "tests", "fail_to_pass.txt")
    is_new_format = os.path.isfile(f2p_path)

    # Common required files (both formats)
    common_required = [
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "tests/test.sh",
        "solution/solve.sh",
    ]
    for rel_path in common_required:
        full_path = os.path.join(task_dir, rel_path)
        if not os.path.isfile(full_path):
            errors.append(f"Missing required file: {rel_path}")

    if is_new_format:
        # New format validation
        with open(f2p_path) as f:
            f2p_lines = [line.strip() for line in f if line.strip()]
        if not f2p_lines:
            errors.append("tests/fail_to_pass.txt is empty (must list at least one test)")

        score_path = os.path.join(task_dir, "tests", "score.py")
        if not os.path.isfile(score_path):
            errors.append("Missing required file: tests/score.py")

        # Optional: validate f2p_skip.txt format if present
        f2p_skip_path = os.path.join(task_dir, "tests", "f2p_skip.txt")
        if os.path.isfile(f2p_skip_path):
            with open(f2p_skip_path) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped and stripped.startswith("#"):
                        continue  # comment lines are OK
                    if stripped and "::" not in stripped:
                        errors.append(
                            f"tests/f2p_skip.txt line {lineno}: "
                            f"expected pytest node ID (file::test), got: {stripped!r}"
                        )

        # Optional: validate p2p_skip.txt format if present
        p2p_skip_path = os.path.join(task_dir, "tests", "p2p_skip.txt")
        if os.path.isfile(p2p_skip_path):
            with open(p2p_skip_path) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped and stripped.startswith("#"):
                        continue  # comment lines are OK
                    if stripped and "::" not in stripped:
                        errors.append(
                            f"tests/p2p_skip.txt line {lineno}: "
                            f"expected pytest node ID (file::test), got: {stripped!r}"
                        )
    else:
        # Old format validation
        old_required = [
            "tests/gold_reference_tests.py",
            "tests/test_runner.py",
        ]
        for rel_path in old_required:
            full_path = os.path.join(task_dir, rel_path)
            if not os.path.isfile(full_path):
                errors.append(f"Missing required file: {rel_path}")

        # Verify script exists (tests/verify_*.sh)
        tests_dir = os.path.join(task_dir, "tests")
        if os.path.isdir(tests_dir):
            verify_scripts = [
                f for f in os.listdir(tests_dir) if f.startswith("verify_") and f.endswith(".sh")
            ]
            if not verify_scripts:
                errors.append("Missing verify script: tests/verify_*.sh")

        # Gold reference tests have at least one test function
        gold_path = os.path.join(task_dir, "tests", "gold_reference_tests.py")
        if os.path.isfile(gold_path):
            with open(gold_path) as f:
                content = f.read()
            test_count = content.count("def test_")
            if test_count == 0:
                errors.append("gold_reference_tests.py has no test functions (def test_*)")

        # test_runner.py has EXPECTED_REF_TESTS
        runner_path = os.path.join(task_dir, "tests", "test_runner.py")
        if os.path.isfile(runner_path):
            with open(runner_path) as f:
                runner_content = f.read()
            if "EXPECTED_REF_TESTS" not in runner_content:
                errors.append("test_runner.py missing EXPECTED_REF_TESTS constant")

    # Required directories
    for subdir in ["environment", "tests", "solution"]:
        full_path = os.path.join(task_dir, subdir)
        if not os.path.isdir(full_path):
            errors.append(f"Missing required directory: {subdir}/")

    # Instruction format and word count
    preamble = PipelineContext().instruction_preamble
    instruction_path = os.path.join(task_dir, "instruction.md")
    if os.path.isfile(instruction_path):
        with open(instruction_path) as f:
            raw = f.read()
        if raw.startswith(preamble):
            body = raw.split("\n", 1)[1].strip() if "\n" in raw else ""
            words = len(body.split())
        else:
            words = len(raw.split())
        if words < 50:
            errors.append(f"instruction.md too short: {words} words (minimum 50)")
        if words > 200:
            errors.append(f"instruction.md too long: {words} words (maximum 200)")

    return errors


def main() -> None:
    """CLI entry point: craft-taskgen validate <task_dir>."""
    if len(sys.argv) < 2:
        print("Usage: craft-taskgen validate <task_dir>")
        sys.exit(1)

    task_dir = sys.argv[1]
    if not os.path.isdir(task_dir):
        print(f"Error: {task_dir} is not a directory")
        sys.exit(1)

    errors = validate_task_dir(task_dir)
    if errors:
        print(f"FAIL: {len(errors)} issue(s) in {task_dir}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"OK: {task_dir} passes all checks")
