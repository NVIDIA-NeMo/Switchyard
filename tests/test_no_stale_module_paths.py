# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prevent deleted Python server and compatibility paths from returning."""

from __future__ import annotations

from pathlib import Path

import pytest

_STALE_PATH_PATTERNS = (
    "switchyard.core.",
    "switchyard.foundation",
    "switchyard.lib.",
    "switchyard.server.",
    "switchyard_rust.components",
    "switchyard_rust.core",
    "switchyard_rust.translation",
    "nemo_switchyard.",
    "SwitchyardV2",
    "switchyard_v2",
    "build_switchyard_v2_app",
)

# Repo root: tests/<this file> → tests/ → repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "switchyard"


def _all_source_files() -> list[Path]:
    """Every ``.py`` file under the ``switchyard/`` package."""
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("pattern", _STALE_PATH_PATTERNS)
def test_no_stale_module_paths_in_package(pattern: str) -> None:
    """No source file under ``switchyard/`` may contain *pattern*.

    Failure message lists every offending ``file:line`` so the fix is a
    one-shot edit pass, not a repeated test-fail / fix / test-fail cycle.
    """
    offenders: list[str] = []
    for path in _all_source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        f"Found {len(offenders)} stale reference(s) to {pattern!r}:\n  "
        + "\n  ".join(offenders)
    )
