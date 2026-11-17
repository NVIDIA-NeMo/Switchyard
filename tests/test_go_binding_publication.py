# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/release"))
try:
    packaging = importlib.import_module("package_go_bindings")
    publication = importlib.import_module("publish_go_bindings")
finally:
    sys.path.pop(0)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, candidate, remote = tmp_path / "source", tmp_path / "candidate", tmp_path / "origin.git"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Local Publication Test")
    git(root, "config", "user.email", "publication-test@example.invalid")
    files = {
        "pyproject.toml": '[project]\nversion = "0.4.0"\n',
        "Cargo.toml": '[workspace.package]\nversion = "0.4.0"\n',
        "bindings/go/go.mod": "module github.com/NVIDIA-NeMo/Switchyard/bindings/go\n\ngo 1.25.0\n",
        "bindings/go/.gitignore": "*.a\n",
        "bindings/go/switchyard.go": 'package switchyard\nimport _ "embed"\n//go:embed internal/wasm/switchyard.wasm\nvar Guest []byte\n',
        "bindings/go/testdata/example.txt": "preserve testdata\n",
        "crates/switchyard-ffi/include/switchyard.h": "#define SY_ABI_VERSION 1\n",
        "LICENSE": "license\n",
        "NOTICE": "notice\n",
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    git(root, "add", ".")
    subprocess.run(
        ["git", "-C", str(root), "commit", "-s", "-m", "test: create release source"],
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-11-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-11-01T00:00:00Z",
        },
        check=True,
        capture_output=True,
    )
    git(root, "tag", "v0.4.0")
    git(root, "clone", "--bare", str(root), str(remote))
    git(root, "remote", "add", "origin", str(remote))
    artifacts = tmp_path / "artifacts"
    for directory, filename in [
        ("wasm", "switchyard_ffi.wasm"),
        ("native-linux_amd64", "libswitchyard_ffi.a"),
    ]:
        path = artifacts / directory
        path.mkdir(parents=True)
        (path / "SOURCE_COMMIT").write_text(git(root, "rev-parse", "HEAD"))
        (path / filename).write_bytes(b"compiled artifact")
    packaging.package(root, artifacts, candidate, "release")
    return root, candidate, remote


def test_dry_run_preserves_checkout_and_refs(source: tuple[Path, Path, Path]) -> None:
    root, candidate, remote = source
    before = [
        git(root, *args) for args in [("rev-parse", "HEAD"), ("ls-files", "--stage"), ("show-ref",)]
    ]
    commit = publication.publish(root, candidate)
    assert git(root, "rev-parse", f"{commit}^") == before[0]
    assert [
        git(root, *args) for args in [("rev-parse", "HEAD"), ("ls-files", "--stage"), ("show-ref",)]
    ] == before
    assert not git(root, "status", "--porcelain")
    assert not git(remote, "for-each-ref", "refs/tags/bindings/go")


def test_publish_is_immutable_and_idempotent(source: tuple[Path, Path, Path]) -> None:
    root, candidate, remote = source
    original = git(remote, "rev-parse", "main")
    commit = publication.publish(root, candidate, push=True)
    assert publication.publish(root, candidate, push=True) == commit
    assert git(remote, "rev-parse", "refs/tags/bindings/go/v0.4.0") == commit
    assert git(remote, "rev-parse", "main") == original
    assert git(remote, "rev-parse", "refs/tags/v0.4.0") == original
    manifest = json.loads((candidate / "manifest.json").read_text())
    paths = git(remote, "ls-tree", "-r", "--name-only", commit, "bindings/go").splitlines()
    assert {path.removeprefix("bindings/go/") for path in paths} == set(manifest["files"])
    assert all(
        path.startswith("bindings/go/")
        for path in git(
            root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).splitlines()
    )
    assert not git(root, "for-each-ref", "refs/tags/bindings/go")
    assert not git(root, "status", "--porcelain")


def test_existing_different_tag_is_rejected(source: tuple[Path, Path, Path]) -> None:
    root, candidate, remote = source
    original = git(remote, "rev-parse", "main")
    git(remote, "update-ref", "refs/tags/bindings/go/v0.4.0", original)
    with pytest.raises(ValueError):
        publication.publish(root, candidate, push=True)
    assert git(remote, "rev-parse", "refs/tags/bindings/go/v0.4.0") == original


@pytest.mark.parametrize("change", ["preview", "source", "hash", "path"])
def test_invalid_candidate_cannot_publish(source: tuple[Path, Path, Path], change: str) -> None:
    root, candidate, remote = source
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "preview":
        manifest["mode"] = "preview"
    elif change == "source":
        manifest["source_commit"] = "f" * 40
    elif change == "hash":
        manifest["files"]["LICENSE"] = "0" * 64
    else:
        with zipfile.ZipFile(next((candidate / "proxy").rglob("*.zip")), "a") as archive:
            archive.writestr(f"{manifest['module']}@{manifest['version']}/../escape", "bad")
        manifest["files"]["../escape"] = hashlib.sha256(b"bad").hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid module path" if change == "path" else None):
        publication.publish(root, candidate, push=True)
    assert not git(remote, "for-each-ref", "refs/tags/bindings/go")
    assert not git(root, "status", "--porcelain")
