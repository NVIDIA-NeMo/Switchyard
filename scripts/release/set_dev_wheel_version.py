#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp temporary Python package metadata for dev wheel artifact builds."""

import argparse
import re
import sys
from pathlib import Path

DEV_VERSION_RE = re.compile(r"^(?P<release>\d+\.\d+\.\d+)\.dev(?P<number>\d*)$")
PACKAGE_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(".*)$')


def parse_dev_wheel_version(version: str) -> str:
    """Return the normalized PEP 440 `.dev` version or raise `ValueError`."""

    match = DEV_VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError("dev wheel versions must look like 0.0.1.dev0")

    number = match.group("number") or "0"
    return f"{match.group('release')}.dev{number}"


def update_pyproject(path: Path, version: str) -> bool:
    """Set `[project].version` in `pyproject.toml`."""

    lines = path.read_text().splitlines(keepends=True)
    in_project = False
    changed = False
    found_version = False
    output: list[str] = []

    for line in lines:
        section = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?$", line)
        if section is not None:
            in_project = section.group(1) == "project"

        updated = line
        if in_project:
            updated, count = PACKAGE_VERSION_RE.subn(rf"\g<1>{version}\g<3>", updated, count=1)
            if count:
                found_version = True

        changed = changed or updated != line
        output.append(updated)

    if not found_version:
        raise ValueError(f"{path}: missing [project] version")
    if changed:
        path.write_text("".join(output))
    return changed


def apply_version(version: str) -> None:
    """Set the wheel version in `pyproject.toml`."""

    changed = update_pyproject(Path("pyproject.toml"), version)
    if changed:
        print("Set dev wheel metadata:")
        print(f"  Version: {version}")
        print("  updated pyproject.toml")
    else:
        print(f"dev wheel metadata already set to {version}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="PEP 440 .dev version, such as 0.0.1.dev0")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print only the normalized version",
    )
    args = parser.parse_args(argv)

    try:
        version = parse_dev_wheel_version(args.version)
        if args.print_version:
            print(version)
            return 0
        apply_version(version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
