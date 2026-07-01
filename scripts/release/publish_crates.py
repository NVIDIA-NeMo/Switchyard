# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package or publish Switchyard Rust crates in dependency order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PUBLIC_CRATES = (
    "switchyard-core",
    "switchyard-translation",
    "switchyard-components",
    "switchyard-components-v2-macros",
    "switchyard-components-v2",
    "switchyard-server",
)

PRIVATE_CRATES = ("switchyard-py",)


def crate_manifest(crate: str) -> Path:
    """Return the manifest path for a workspace crate."""
    return ROOT / "crates" / crate / "Cargo.toml"


def load_toml(path: Path) -> dict[str, object]:
    """Load a TOML file with Python's standard parser."""
    with path.open("rb") as file:
        return tomllib.load(file)


def package_table(crate: str) -> dict[str, object]:
    """Return the `[package]` table for a workspace crate."""
    manifest = crate_manifest(crate)
    try:
        data = load_toml(manifest)
        package = data["package"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"{manifest}: failed to read package metadata: {error}") from error
    if not isinstance(package, dict):
        raise SystemExit(f"{manifest}: [package] must be a table")
    return package


def dependency_table(crate: str) -> dict[str, object]:
    """Return the `[dependencies]` table for a workspace crate."""
    data = load_toml(crate_manifest(crate))
    dependencies = data.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise SystemExit(f"{crate}: [dependencies] must be a table")
    return dependencies


def crate_versions() -> dict[str, str]:
    """Return the version of each crate managed by this release script."""
    versions: dict[str, str] = {}
    for crate in (*PUBLIC_CRATES, *PRIVATE_CRATES):
        version = package_table(crate).get("version")
        if not isinstance(version, str):
            raise SystemExit(f"{crate}: package.version must be a string")
        versions[crate] = version
    return versions


def validate_publish_metadata() -> None:
    """Validate manifest details required before packaging or publishing crates."""
    versions = crate_versions()
    public_versions = {versions[crate] for crate in PUBLIC_CRATES}
    if len(public_versions) != 1:
        details = ", ".join(f"{crate}={versions[crate]}" for crate in PUBLIC_CRATES)
        raise SystemExit(f"public crate versions must match: {details}")

    for crate in PUBLIC_CRATES:
        publish = package_table(crate).get("publish")
        if publish is False:
            raise SystemExit(f"{crate}: package.publish=false but crate is in PUBLIC_CRATES")

        for dep_name, dep_spec in dependency_table(crate).items():
            if dep_name not in PUBLIC_CRATES:
                continue
            if not isinstance(dep_spec, dict):
                raise SystemExit(f"{crate}: dependency {dep_name} must use a table")
            if "path" not in dep_spec:
                raise SystemExit(f"{crate}: dependency {dep_name} must keep a local path")
            if dep_spec.get("version") != versions[dep_name]:
                raise SystemExit(
                    f"{crate}: dependency {dep_name} must pin version {versions[dep_name]}"
                )

    for crate in PRIVATE_CRATES:
        if package_table(crate).get("publish") is not False:
            raise SystemExit(f"{crate}: private release crate must set package.publish=false")


def run(command: list[str]) -> None:
    """Run one subprocess command from the repository root."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def local_patch_config(crate: str) -> list[str]:
    """Return dry-run-only crates.io patches for unpublished workspace dependencies."""
    config: list[str] = []
    for dependency in dependency_table(crate):
        if dependency not in PUBLIC_CRATES:
            continue
        config.extend(
            [
                "--config",
                f'patch.crates-io.{dependency}.path="crates/{dependency}"',
            ]
        )
    return config


def cargo_publish(crate: str, *, publish: bool, allow_dirty: bool) -> None:
    """Run `cargo publish` for one crate, optionally as a real upload."""
    command = ["cargo", "publish", "--locked", "--package", crate]
    if allow_dirty:
        command.append("--allow-dirty")
    if not publish:
        command.append("--dry-run")
        command.extend(local_patch_config(crate))
    run(command)


def publish_crates(*, publish: bool, pause_seconds: int, allow_dirty: bool = False) -> None:
    """Package or publish all public crates in dependency order."""
    validate_publish_metadata()
    if publish and allow_dirty:
        raise SystemExit("--allow-dirty is only allowed for dry-run packaging")
    if publish and not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise SystemExit("CARGO_REGISTRY_TOKEN is required for crates.io publishing")

    for index, crate in enumerate(PUBLIC_CRATES):
        cargo_publish(crate, publish=publish, allow_dirty=allow_dirty)
        if publish and pause_seconds > 0 and index < len(PUBLIC_CRATES) - 1:
            # Let the crates.io index settle before the next dependent crate resolves.
            time.sleep(pause_seconds)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload crates to crates.io. Without this flag, run cargo publish --dry-run.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=int,
        default=20,
        help="Seconds to wait between real publishes so registry metadata can settle.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow uncommitted files during local dry-runs. Rejected with --publish.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Package or publish the public Rust crates."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    publish_crates(
        publish=args.publish,
        pause_seconds=args.pause_seconds,
        allow_dirty=args.allow_dirty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
