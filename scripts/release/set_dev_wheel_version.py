#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp temporary Python package metadata for dev wheel artifact builds."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

DEV_VERSION_RE = re.compile(r"^(?P<release>\d+\.\d+\.\d+)\.dev(?P<number>\d*)$")
PACKAGE_NAME_RE = re.compile(r'^(name\s*=\s*")([^"]+)(".*)$')
PACKAGE_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(".*)$')
PYTHON_VERSION_RE = re.compile(r'^(__version__\s*=\s*")([^"]+)(".*)$', re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class DevWheelVersion:
    """Normalized metadata used for a short-lived dev wheel artifact build."""

    version: str


def parse_dev_wheel_version(version: str) -> DevWheelVersion:
    """Return the normalized PEP 440 `.dev` version or raise `ValueError`."""

    match = DEV_VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError("dev wheel versions must look like 0.0.1.dev0")

    number = match.group("number") or "0"
    return DevWheelVersion(version=f"{match.group('release')}.dev{number}")


def update_pyproject(path: Path, *, package_name: str, version: str) -> bool:
    """Set `[project]` name and version in `pyproject.toml`."""

    lines = path.read_text().splitlines(keepends=True)
    in_project = False
    changed = False
    found_name = False
    found_version = False
    output: list[str] = []

    for line in lines:
        section = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?$", line)
        if section is not None:
            in_project = section.group(1) == "project"

        updated = line
        if in_project:
            updated, count = PACKAGE_NAME_RE.subn(rf"\g<1>{package_name}\g<3>", updated, count=1)
            if count:
                found_name = True
            updated, count = PACKAGE_VERSION_RE.subn(rf"\g<1>{version}\g<3>", updated, count=1)
            if count:
                found_version = True

        changed = changed or updated != line
        output.append(updated)

    if not found_name:
        raise ValueError(f"{path}: missing [project] name")
    if not found_version:
        raise ValueError(f"{path}: missing [project] version")
    if changed:
        path.write_text("".join(output))
    return changed


def update_python_init(path: Path, version: str) -> bool:
    """Set `switchyard.__version__` for the dev wheel artifact."""

    text = path.read_text()
    updated, count = PYTHON_VERSION_RE.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise ValueError(f"{path}: missing __version__")
    if updated != text:
        path.write_text(updated)
        return True
    return False


def apply_version(version: DevWheelVersion, *, package_name: str) -> None:
    """Update package metadata files used by maturin wheel builds."""

    changes = [
        (
            "pyproject.toml",
            update_pyproject(
                Path("pyproject.toml"),
                package_name=package_name,
                version=version.version,
            ),
        ),
        ("switchyard/__init__.py", update_python_init(Path("switchyard/__init__.py"), version.version)),
    ]

    changed = [path for path, did_change in changes if did_change]
    if changed:
        print("Set dev wheel metadata:")
        print(f"  Package: {package_name}")
        print(f"  Version: {version.version}")
        for path in changed:
            print(f"  updated {path}")
    else:
        print(f"dev wheel metadata already set for {package_name} {version.version}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="PEP 440 .dev version, such as 0.0.1.dev0")
    parser.add_argument(
        "--package-name",
        default="nemo-switchyard",
        help="Distribution name to stamp into wheel metadata",
    )
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print only the normalized version",
    )
    args = parser.parse_args(argv)

    try:
        version = parse_dev_wheel_version(args.version)
        if args.print_version:
            print(version.version)
            return 0
        apply_version(version, package_name=args.package_name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
