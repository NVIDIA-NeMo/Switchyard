# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-LLM candidate filtering — reject obvious non-tasks before expensive eval calls.

Applied before sending candidates to Claude for evaluation. Each filter is cheap
(string matching on commit metadata) and saves ~$0.10-0.30 per rejected candidate.
"""

from __future__ import annotations

import re

# Patterns that indicate docs-only, CI-only, or formatting-only commits
_DOCS_ONLY_PATTERNS = re.compile(
    r"^(docs?|readme|changelog|license|contributing|authors|history|news)",
    re.IGNORECASE,
)
_CI_ONLY_PATTERNS = re.compile(
    r"^(\.(github|gitlab|circleci|travis)|Jenkinsfile|\.pre-commit|tox\.ini|noxfile)",
    re.IGNORECASE,
)
_FORMAT_ONLY_SUBJECTS = re.compile(
    r"^(style|format|lint|black|isort|ruff|autopep8|yapf)\b",
    re.IGNORECASE,
)
_VERSION_BUMP_SUBJECTS = re.compile(
    r"^(bump|release|version|v?\d+\.\d+)",
    re.IGNORECASE,
)
_PYTHON_EXTENSIONS = frozenset({".py", ".pyi"})


def is_docs_only(source_files: list[str]) -> bool:
    """True if all changed source files are documentation. Empty/missing = unknown, not docs-only."""
    if not source_files:
        return False
    return all(_DOCS_ONLY_PATTERNS.match(f.split("/")[0]) for f in source_files)


def is_ci_only(source_files: list[str]) -> bool:
    """True if all changed source files are CI/config."""
    if not source_files:
        return False
    return all(_CI_ONLY_PATTERNS.match(f) for f in source_files)


def is_formatting_only(subject: str) -> bool:
    """True if the commit subject suggests a formatting-only change."""
    return bool(_FORMAT_ONLY_SUBJECTS.match(subject.strip()))


def is_version_bump(subject: str) -> bool:
    """True if the commit subject suggests a version bump."""
    return bool(_VERSION_BUMP_SUBJECTS.match(subject.strip()))


def is_non_python(source_files: list[str]) -> bool:
    """True if ALL source files are non-Python (Rust, C, Go, etc.)."""
    if not source_files:
        return False
    return all(not any(f.endswith(ext) for ext in _PYTHON_EXTENSIONS) for f in source_files)


def has_no_source_changes(source_files: list[str]) -> bool:
    """True if there are no source file changes (test-only or config-only commit)."""
    return len(source_files) == 0


def prefilter_candidate(candidate: dict) -> str | None:
    """Apply all pre-filters to a candidate. Returns rejection reason or None if it passes."""
    subject = candidate.get("subject", "")
    source_files = candidate.get("source_files", [])

    if not candidate.get("has_test_patch", False):
        return "no test files in commit"

    if is_docs_only(source_files):
        return "docs-only change"

    if is_ci_only(source_files):
        return "CI/config-only change"

    if is_formatting_only(subject):
        return "formatting-only change"

    if is_version_bump(subject):
        return "version bump"

    if is_non_python(source_files):
        return "non-Python change (Rust/C/Go/etc.)"

    if has_no_source_changes(source_files):
        return "no source file changes"

    return None
