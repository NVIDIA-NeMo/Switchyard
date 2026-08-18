#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronize documented Rust dependencies with the workspace release version."""

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]
_START = "<!-- BEGIN GENERATED: Rust dependencies -->"
_END = "<!-- END GENERATED: Rust dependencies -->"
_TARGETS = {
    Path("README.md"): "core",
    Path("docs/getting_started.md"): "libsy",
    Path("docs/reference/rust_api.md"): "core",
    Path("crates/libsy/README.md"): "libsy",
    Path("crates/protocol/README.md"): "protocol",
}


def _workspace_version(repo_root: Path) -> str:
    cargo = tomllib.loads((repo_root / "Cargo.toml").read_text(encoding="utf-8"))
    version = cargo.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("Cargo.toml must define workspace.package.version")
    return version


def _render(profile: str, version: str) -> str:
    switchyard = [
        f'switchyard-libsy = "={version}"',
        f'switchyard-protocol = "={version}"',
    ]
    if profile == "core":
        dependencies = switchyard
    elif profile == "libsy":
        dependencies = [
            'async-trait = "0.1"',
            'futures = "0.3"',
            *switchyard,
            'tokio = { version = "1", features = ["macros", "rt"] }',
        ]
    elif profile == "protocol":
        dependencies = [switchyard[1], 'serde_json = "1"']
    else:
        raise ValueError(f"unknown dependency profile: {profile}")

    return "\n".join(
        [
            _START,
            "```toml",
            "[dependencies]",
            *dependencies,
            "```",
            _END,
        ]
    )


def _synchronize(text: str, replacement: str, path: Path) -> str:
    pattern = re.compile(rf"{re.escape(_START)}.*?{re.escape(_END)}", re.DOTALL)
    updated, count = pattern.subn(lambda _match: replacement, text)
    if count != 1:
        raise ValueError(f"{path}: expected one generated dependency block, found {count}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated dependency examples are stale",
    )
    args = parser.parse_args()

    version = _workspace_version(_REPO_ROOT)
    stale: list[Path] = []

    for relative_path, profile in _TARGETS.items():
        path = _REPO_ROOT / relative_path
        current = path.read_text(encoding="utf-8")
        expected = _synchronize(current, _render(profile, version), relative_path)
        if current == expected:
            continue
        if args.check:
            stale.append(relative_path)
            continue
        path.write_text(expected, encoding="utf-8")
        print(f"updated {relative_path}")

    if stale:
        print(
            "Rust dependency examples are stale: "
            + ", ".join(str(path) for path in stale),
            file=sys.stderr,
        )
        print(
            "Run: uv run python scripts/docs/sync_rust_dependency_examples.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
