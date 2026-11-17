#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package Go bindings and prebuilt libraries without publishing them."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = {"linux_amd64", "linux_arm64", "darwin_amd64", "darwin_arm64"}
MAX_ZIP_SIZE = 500 << 20


def git(root: Path, *args: str) -> str:
    """Read the source revision without modifying Git state."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def release_version(python_version: str, rust_version: str) -> str:
    """Require the same stable SemVer used by the Python and Rust release."""
    if (
        python_version != rust_version
        or re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", rust_version) is None
    ):
        raise ValueError("release requires matching stable Python and Rust versions")
    return f"v{rust_version}"


def source_metadata(root: Path, mode: str) -> dict[str, str]:
    """Describe the checked-out source and choose its candidate module version."""
    python_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    rust_version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"][
        "version"
    ]
    source = git(root, "rev-parse", "HEAD")
    if git(root, "status", "--porcelain"):
        raise ValueError("packaging requires a clean source checkout")
    timestamp = datetime.fromtimestamp(
        int(git(root, "show", "-s", "--format=%ct", "HEAD")), timezone.utc
    )
    version = f"v0.0.0-preview.g{source[:12]}"
    if mode == "release":
        version = release_version(python_version, rust_version)
        if version not in git(root, "tag", "--points-at", "HEAD").splitlines():
            raise ValueError(f"release requires the existing root tag {version} at HEAD")
    return {
        "mode": mode,
        "version": version,
        "source_commit": source,
        "source_time": timestamp.isoformat().replace("+00:00", "Z"),
        "python_version": python_version,
        "rust_version": rust_version,
        "go_tag": f"bindings/go/{version}" if mode == "release" else "",
    }


def package(root: Path, artifacts: Path, output: Path, mode: str) -> dict[str, object]:
    """Stage a self-contained Go module and a local module-proxy candidate."""
    metadata: dict[str, object] = dict(source_metadata(root, mode))
    native = sorted(artifacts.glob("native-*"))
    if not native:
        raise ValueError("at least one native artifact is required")
    for directory in [artifacts / "wasm", *native]:
        if (directory / "SOURCE_COMMIT").read_text().strip() != metadata["source_commit"]:
            raise ValueError(f"{directory}: artifact source does not match HEAD")
    platforms = [directory.name.removeprefix("native-") for directory in native]
    if not set(platforms) <= PLATFORMS:
        raise ValueError(f"unsupported native platforms: {platforms}")
    output.mkdir(parents=True)
    module = output / "module"
    module.mkdir()
    for name in git(root, "ls-files", "bindings/go").splitlines():
        relative = Path(name).relative_to("bindings/go")
        destination = module / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    header = root / "crates/switchyard-ffi/include/switchyard.h"
    shutil.copyfile(header, module / "switchyard.h")
    abi = re.search(r"^#define SY_ABI_VERSION (\d+)$", header.read_text(), re.MULTILINE)
    if abi is None:
        raise ValueError("C header does not declare SY_ABI_VERSION")
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(root / name, module / name)
    wasm = module / "internal/wasm/switchyard.wasm"
    wasm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifacts / "wasm/switchyard_ffi.wasm", wasm)
    flags = []
    for directory, platform in zip(native, platforms, strict=True):
        # Keep archives beside Go files so go mod vendor retains them.
        library = f"libswitchyard_ffi_{platform}.a"
        shutil.copyfile(directory / "libswitchyard_ffi.a", module / library)
        if (directory / "build-info.txt").exists():
            shutil.copyfile(directory / "build-info.txt", module / f"build-info-{platform}.txt")
        flags.append(f"#cgo {platform.replace('_', ',')} LDFLAGS: ${{SRCDIR}}/{library}")
    (module / "link_native.go").write_text(
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n\n"
        "//go:build cgo && switchyard_native\n\npackage switchyard\n\n/*\n"
        + "\n".join(flags)
        + "\n#cgo linux LDFLAGS: -ldl -lpthread -lm\n"
        "#cgo darwin LDFLAGS: -framework Security -framework CoreFoundation -liconv\n"
        '*/\nimport "C"\n'
    )
    files = sorted(path for path in module.rglob("*") if path.is_file())
    if sum(path.stat().st_size for path in files) > MAX_ZIP_SIZE:
        raise ValueError("Go module exceeds the 500 MiB uncompressed limit")
    metadata.update(
        abi_version=int(abi[1]),
        native_platforms=platforms,
        files={
            path.relative_to(module).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    )
    module_name = re.search(r"^module (\S+)$", (module / "go.mod").read_text(), re.MULTILINE)
    if module_name is None:
        raise ValueError("missing Go module path")
    name = module_name[1]
    metadata["module"] = name
    escaped = "".join("!" + char.lower() if char.isupper() else char for char in name)
    proxy = output / "proxy" / escaped / "@v"
    proxy.mkdir(parents=True)
    version = str(metadata["version"])
    archive_path = proxy / f"{version}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            entry = zipfile.ZipInfo(f"{name}@{version}/{path.relative_to(module).as_posix()}")
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    if archive_path.stat().st_size > MAX_ZIP_SIZE:
        raise ValueError("Go module exceeds the 500 MiB compressed limit")
    shutil.copyfile(module / "go.mod", proxy / f"{version}.mod")
    (proxy / f"{version}.info").write_text(
        json.dumps({"Version": version, "Time": metadata["source_time"]}) + "\n"
    )
    (proxy / "list").write_text(version + "\n")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}\n"
            for path in [output / "manifest.json", *sorted(proxy.iterdir())]
        )
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Build local package artifacts without creating a release or tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("preview", "release"), default="preview")
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.print_version:
            print(source_metadata(ROOT, args.mode)["version"])
        elif args.artifacts is None or args.output is None:
            parser.error("--artifacts and --output are required")
        else:
            result = package(ROOT, args.artifacts, args.output, args.mode)
            print(f"Packaged {result['version']} at {args.output}")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
