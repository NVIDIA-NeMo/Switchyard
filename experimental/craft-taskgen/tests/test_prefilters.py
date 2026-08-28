# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pre-LLM candidate filtering."""

from __future__ import annotations

from craft_taskgen.prefilters import (
    has_no_source_changes,
    is_ci_only,
    is_docs_only,
    is_formatting_only,
    is_non_python,
    is_version_bump,
    prefilter_candidate,
)


def test_docs_only_rejects():
    assert is_docs_only(["docs/guide.md", "README.md"])
    assert is_docs_only(["CHANGELOG.md"])


def test_docs_only_unknown_when_empty():
    assert not is_docs_only([])  # empty = unknown, not docs-only


def test_docs_only_passes():
    assert not is_docs_only(["src/main.py", "docs/guide.md"])
    assert not is_docs_only(["tests/test_foo.py"])


def test_ci_only_rejects():
    assert is_ci_only([".github/workflows/ci.yml"])
    assert is_ci_only([".gitlab-ci.yml", ".pre-commit-config.yaml"])
    assert is_ci_only(["tox.ini"])


def test_ci_only_passes():
    assert not is_ci_only(["src/main.py", ".github/workflows/ci.yml"])
    assert not is_ci_only([])  # empty is not CI


def test_formatting_only():
    assert is_formatting_only("style: run black formatter")
    assert is_formatting_only("format: autopep8")
    assert is_formatting_only("ruff: fix import ordering")
    assert not is_formatting_only("feat: add streaming support")
    assert not is_formatting_only("fix: formatting bug in output")


def test_version_bump():
    assert is_version_bump("bump version to 2.0")
    assert is_version_bump("release 1.5.0")
    assert is_version_bump("v2.0.0")
    assert not is_version_bump("feat: add version display")


def test_prefilter_no_tests():
    c = {"subject": "feat: add X", "has_test_patch": False, "source_files": ["src/x.py"]}
    assert prefilter_candidate(c) == "no test files in commit"


def test_prefilter_docs_only():
    c = {"subject": "docs: update readme", "has_test_patch": True, "source_files": ["docs/guide.md"]}
    assert prefilter_candidate(c) == "docs-only change"


def test_non_python_rejects():
    assert is_non_python(["src/lib.rs", "src/main.rs"])
    assert is_non_python(["pkg/handler.go"])
    assert is_non_python(["ext/module.c", "ext/module.h"])


def test_non_python_passes():
    assert not is_non_python(["src/main.py", "src/lib.rs"])  # mixed = not non-Python
    assert not is_non_python(["src/main.py"])
    assert not is_non_python([])  # empty = unknown, not non-Python


def test_no_source_changes():
    assert has_no_source_changes([])
    assert not has_no_source_changes(["src/main.py"])


def test_prefilter_non_python():
    c = {"subject": "feat: add X", "has_test_patch": True, "source_files": ["src/lib.rs", "src/util.rs"]}
    assert prefilter_candidate(c) == "non-Python change (Rust/C/Go/etc.)"


def test_prefilter_no_source_files():
    c = {"subject": "test: add tests", "has_test_patch": True, "source_files": []}
    assert prefilter_candidate(c) == "no source file changes"


def test_prefilter_passes_good_candidate():
    c = {
        "subject": "feat: add streaming",
        "has_test_patch": True,
        "source_files": ["src/stream.py", "tests/test_stream.py"],
    }
    assert prefilter_candidate(c) is None
