# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for path-based local validation selection and its contributor guidance."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "select_validation.py"
GUIDE_PATH = REPO_ROOT / "scripts" / "ci" / "README.md"


def _load_script() -> dict[str, Any]:
    """Load the validation selector without making scripts a Python package."""
    return runpy.run_path(str(SCRIPT_PATH))


def _names(*paths: str) -> set[str]:
    """Return validation names selected for the supplied paths."""
    module = _load_script()
    return {item.name for item in module["select_validations"](paths)}


def test_fern_workflow_selects_docs_and_security_validation() -> None:
    """Fern workflow edits need both product checks and workflow security checks."""
    names = _names(".github/workflows/fern-docs-preview-comment.yml")
    assert names == {"Fern docs", "Workflow syntax and security"}


def test_native_or_package_changes_select_all_affected_gates() -> None:
    """Native package metadata must cover Python, Rust, tests, and package construction."""
    names = _names("Cargo.lock", "switchyard_rust/__init__.py")
    assert names == {
        "Offline test suite",
        "Package build",
        "Python quality gates",
        "Rust quality gates",
    }


def test_docs_only_change_does_not_select_runtime_or_live_tests() -> None:
    """Ordinary docs changes stay focused and never opt into provider calls."""
    validations = _load_script()["select_validations"](["docs/operations/context_window.mdx"])
    assert [item.name for item in validations] == ["Fern docs"]
    assert all("integration" not in item.command for item in validations)


def test_ci_guide_records_required_security_invariants() -> None:
    """The guide must retain the cross-repository workflow design lessons."""
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    for invariant in (
        "Stable required aggregation",
        "Trust boundaries",
        "Concurrency",
        "Permissions and secrets",
        "Supply chain and toolchains",
        "same-repository PRs",
        "CI Success",
        "actionlint",
        "zizmor",
        "NeMo Curator",
        "NeMo Data Designer",
    ):
        assert invariant in guide
