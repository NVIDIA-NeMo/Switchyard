#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a tested Go distribution without changing the source branch."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from package_go_bindings import ROOT, source_metadata


def git(
    root: Path, *args: str, data: bytes | None = None, env: dict[str, str] | None = None
) -> str:
    """Run Git with an optional isolated index or binary input."""
    return (
        subprocess.check_output(["git", "-C", str(root), *args], input=data, env=env)
        .decode()
        .strip()
    )


def publish(root: Path, candidate: Path, *, remote: str = "origin", push: bool = False) -> str:
    """Prepare a release-only commit; publish its immutable Go tag only when requested."""
    metadata = source_metadata(root, "release")
    manifest = json.loads((candidate / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError("candidate does not match this stable release checkout")
    match = re.search(r"^module (\S+)$", (root / "bindings/go/go.mod").read_text(), re.MULTILINE)
    if match is None or manifest["module"] != match[1]:
        raise ValueError("candidate module does not match the source module")
    module, version = match[1], metadata["version"]
    escaped = "".join("!" + char.lower() if char.isupper() else char for char in module)
    archive_path = candidate / "proxy" / escaped / "@v" / f"{version}.zip"
    prefix = f"{module}@{version}/"
    files = {}
    with zipfile.ZipFile(archive_path) as archive:
        expected = {prefix + name for name in manifest["files"]}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("candidate archive does not match its file inventory")
        for name, digest in manifest["files"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
                raise ValueError(f"invalid module path: {name}")
            content = archive.read(prefix + name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"candidate checksum mismatch: {name}")
            if len(content) > 100 << 20:
                raise ValueError(f"{name} exceeds GitHub's 100 MiB file limit")
            files[name] = content

    source, ref = metadata["source_commit"], f"refs/tags/{metadata['go_tag']}"
    with tempfile.TemporaryDirectory(prefix="switchyard-go-index-") as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        git(root, "read-tree", source, env=env)
        git(root, "rm", "--cached", "-r", "--ignore-unmatch", "--", "bindings/go", env=env)
        entries = []
        for name, content in sorted(files.items()):
            blob = git(root, "hash-object", "-w", "--stdin", data=content)
            entries.append(f"100644 {blob}\tbindings/go/{name}\0".encode())
        git(root, "update-index", "-z", "--index-info", data=b"".join(entries), env=env)
        tree = git(root, "write-tree", env=env)

    existing = subprocess.run(
        ["git", "-C", str(root), "ls-remote", "--exit-code", remote, ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        git(root, "fetch", "--no-tags", remote, ref)
        commit = git(root, "rev-parse", "FETCH_HEAD^{commit}")
        if (
            git(root, "rev-list", "--parents", "-n", "1", commit).split() != [commit, source]
            or git(root, "rev-parse", f"{commit}^{{tree}}") != tree
        ):
            raise ValueError(
                f"{ref} already exists with different release contents; refusing to replace it"
            )
        print(f"{ref} already contains this distribution")
        return commit
    if existing.returncode != 2:
        raise subprocess.CalledProcessError(
            existing.returncode, existing.args, stderr=existing.stderr
        )

    identity = "github-actions[bot]"
    email = "41898282+github-actions[bot]@users.noreply.github.com"
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": identity,
        "GIT_COMMITTER_NAME": identity,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_EMAIL": email,
        "GIT_AUTHOR_DATE": metadata["source_time"],
        "GIT_COMMITTER_DATE": metadata["source_time"],
    }
    message = f"build(go): package {version}\n\nSigned-off-by: {identity} <{email}>\n"
    commit = git(root, "commit-tree", tree, "-p", source, data=message.encode(), env=env)
    if push:
        git(root, "push", remote, f"{commit}:{ref}")
        print(f"Published {ref} at {commit}")
    else:
        print(f"Dry run: would publish {ref} at {commit}; no refs changed")
    return commit


def main() -> int:
    """Default to a dry run; never create a local tag or modify the source index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    try:
        publish(ROOT, args.candidate, remote=args.remote, push=args.push)
    except (
        ValueError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
