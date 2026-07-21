#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select local validation commands from changed repository paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Validation:
    """One local validation command and the CI behavior it covers."""

    name: str
    command: str
    reason: str


VALIDATIONS = {
    "python": Validation(
        "Python quality gates",
        "uv run ruff check . && uv run mypy switchyard",
        "Python, tests, scripts, or package metadata changed.",
    ),
    "tests": Validation(
        "Offline test suite",
        'env -u OPENROUTER_API_KEY -u NVIDIA_API_KEY -u OPENAI_API_KEY '
        '-u ANTHROPIC_API_KEY uv run pytest tests/ -v -m "not integration"',
        "Runtime code, tests, or package metadata changed.",
    ),
    "rust": Validation(
        "Rust quality gates",
        "cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings "
        "&& cargo test --workspace",
        "Rust sources or Cargo metadata changed.",
    ),
    "fern": Validation(
        "Fern docs",
        "make -C docs check && uv run pytest tests/test_fern_docs.py -v -o addopts=",
        "Fern content, configuration, or docs workflows changed.",
    ),
    "readme": Validation(
        "README examples",
        "OPENAI_API_KEY=sk-test NVIDIA_API_KEY=nvapi-test ANTHROPIC_API_KEY=sk-ant-test "
        "uv run pytest tests/readme --markdown-docs README.md -v",
        "README examples or their executable coverage changed.",
    ),
    "getting-started": Validation(
        "Getting Started examples",
        "OPENAI_API_KEY=sk-test NVIDIA_API_KEY=nvapi-test ANTHROPIC_API_KEY=sk-ant-test "
        "uv run pytest tests/getting_started --markdown-docs docs/getting_started.mdx -v",
        "The Getting Started guide or its executable coverage changed.",
    ),
    "workflow-security": Validation(
        "Workflow syntax and security",
        "actionlint && zizmor --pedantic .github/workflows",
        "A GitHub Actions workflow changed.",
    ),
    "package": Validation(
        "Package build",
        "uv build",
        "Package metadata, lockfiles, native code, or release automation changed.",
    ),
}


def _normalize(path: str) -> str:
    """Return a repository-relative path with POSIX separators."""
    value = Path(path).as_posix().removeprefix("./")
    return value.rstrip("/")


def select_validations(paths: Iterable[str]) -> list[Validation]:
    """Return the stable, deduplicated validation plan for ``paths``."""
    normalized = {_normalize(path) for path in paths if _normalize(path)}
    selected: set[str] = set()

    python_changed = any(
        path.endswith(".py")
        or path in {"pyproject.toml", "uv.lock"}
        or path.startswith("switchyard/")
        or path.startswith("switchyard_rust/")
        for path in normalized
    )
    if python_changed:
        selected.update({"python", "tests"})

    if any(
        path in {"Cargo.toml", "Cargo.lock"}
        or path.startswith("crates/")
        or path.startswith("switchyard_rust/")
        for path in normalized
    ):
        selected.add("rust")

    if any(path == "README.md" or path.startswith("tests/readme/") for path in normalized):
        selected.add("readme")

    if any(
        path == "docs/getting_started.mdx" or path.startswith("tests/getting_started/")
        for path in normalized
    ):
        selected.add("getting-started")

    fern_workflows = {
        ".github/workflows/ci.yml",
        ".github/workflows/fern-docs-ci.yml",
        ".github/workflows/fern-docs-preview-build.yml",
        ".github/workflows/fern-docs-preview-comment.yml",
        ".github/workflows/publish-fern-docs.yml",
    }
    if any(
        path.startswith("docs/")
        or path == "tests/test_fern_docs.py"
        or path in fern_workflows
        for path in normalized
    ):
        selected.add("fern")

    if any(path.startswith(".github/workflows/") for path in normalized):
        selected.add("workflow-security")

    if any(
        path in {"pyproject.toml", "uv.lock", "Cargo.toml", "Cargo.lock"}
        or path.startswith("crates/")
        or path.startswith("scripts/release/")
        or path in {
            ".github/workflows/package-portability.yml",
            ".github/workflows/publish.yml",
        }
        for path in normalized
    ):
        selected.add("package")

    return [validation for key, validation in VALIDATIONS.items() if key in selected]


def _git_paths(*args: str) -> set[str]:
    """Return paths printed by one Git command."""
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def changed_paths(base: str | None) -> set[str]:
    """Return committed, staged, unstaged, and untracked paths for this checkout."""
    paths: set[str] = set()
    if base is not None:
        paths.update(_git_paths("diff", "--name-only", f"{base}...HEAD"))
    paths.update(_git_paths("diff", "--name-only", "HEAD"))
    paths.update(_git_paths("ls-files", "--others", "--exclude-standard"))
    return paths


def main(argv: list[str] | None = None) -> int:
    """Print a focused validation plan for explicit or Git-derived paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[], help="Changed path; repeat as needed")
    parser.add_argument(
        "--changed",
        action="store_true",
        help="Include staged, unstaged, and untracked paths",
    )
    parser.add_argument(
        "--base",
        help="Also include committed changes since the merge base with this ref",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    if not args.path and not args.changed and args.base is None:
        parser.error("pass --path, --changed, or --base")

    paths = {_normalize(path) for path in args.path}
    try:
        if args.changed or args.base is not None:
            paths.update(changed_paths(args.base))
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or "git path discovery failed", file=sys.stderr)
        return 2

    validations = select_validations(paths)
    if args.json:
        print(
            json.dumps(
                {
                    "paths": sorted(paths),
                    "validations": [asdict(validation) for validation in validations],
                },
                indent=2,
            )
        )
        return 0

    if not paths:
        print("No changed paths found.")
        return 0

    print("Changed paths:")
    for path in sorted(paths):
        print(f"  - {path}")
    print("\nRecommended validation:")
    if not validations:
        print("  - No focused gate mapped; review scripts/ci/README.md before pushing.")
    for validation in validations:
        print(f"  - {validation.name}: {validation.command}")
        print(f"    {validation.reason}")
    print("\nLive provider tests are never selected automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
