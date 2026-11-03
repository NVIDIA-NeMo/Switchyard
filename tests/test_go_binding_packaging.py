# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_go_bindings",
    Path(__file__).resolve().parents[1] / "scripts/release/package_go_bindings.py",
)
assert SPEC is not None and SPEC.loader is not None
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


@pytest.mark.parametrize("python,rust", [("0.4.0a1", "0.4.0a1"), ("0.4.0", "0.3.0")])
def test_release_version_requires_matching_stable_versions(python: str, rust: str) -> None:
    with pytest.raises(ValueError, match="matching stable"):
        packaging.release_version(python, rust)
    assert packaging.release_version("0.4.0", "0.4.0") == "v0.4.0"


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root, artifacts = tmp_path / "source", tmp_path / "artifacts"
    files = {
        "pyproject.toml": '[project]\nversion = "0.4.0a1"\n',
        "Cargo.toml": '[workspace.package]\nversion = "0.3.0"\n',
        "bindings/go/go.mod": "module github.com/NVIDIA-NeMo/Switchyard/bindings/go\n\ngo 1.25.0\n",
        "bindings/go/link_native.go": "source-only linker setup",
        "crates/switchyard-ffi/include/switchyard.h": "#define SY_ABI_VERSION 1\n",
        "LICENSE": "license",
        "NOTICE": "notice",
    }
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    replies = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("status", "--porcelain"): "",
        ("show", "-s", "--format=%ct", "HEAD"): "1762128000",
        ("ls-files", "bindings/go"): "bindings/go/go.mod\nbindings/go/link_native.go",
    }
    monkeypatch.setattr(packaging, "git", lambda _root, *args: replies[args])
    for directory, filename in [
        ("wasm", "switchyard_ffi.wasm"),
        ("native-linux_amd64", "libswitchyard_ffi.a"),
    ]:
        path = artifacts / directory
        path.mkdir(parents=True)
        (path / "SOURCE_COMMIT").write_text("a" * 40)
        (path / filename).write_bytes(b"fake compiled artifact")
    return root, artifacts


def test_package_is_self_contained(source: tuple[Path, Path], tmp_path: Path) -> None:
    root, artifacts = source
    result = packaging.package(root, artifacts, tmp_path / "output", "preview")
    assert result["version"] == "v0.0.0-preview.gaaaaaaaaaaaa"
    assert result["rust_version"] == "0.3.0" and result["python_version"] == "0.4.0a1"
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["source_commit"] == "a" * 40 and manifest["abi_version"] == 1
    archive_path = next((tmp_path / "output/proxy").rglob("*.zip"))
    assert "!n!v!i!d!i!a-!ne!mo/!switchyard" in str(archive_path)
    prefix = f"github.com/NVIDIA-NeMo/Switchyard/bindings/go@{result['version']}/"
    with zipfile.ZipFile(archive_path) as archive:
        for name in [
            "go.mod",
            "switchyard.h",
            "LICENSE",
            "NOTICE",
            "internal/wasm/switchyard.wasm",
            "libswitchyard_ffi_linux_amd64.a",
        ]:
            assert archive.read(prefix + name)
            assert name in manifest["files"]
        linker = archive.read(prefix + "link_native.go").decode()
        assert "linux,amd64 LDFLAGS: ${SRCDIR}/libswitchyard_ffi_linux_amd64.a" in linker
        assert "../../" not in linker


def test_package_rejects_artifacts_from_other_source(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    root, artifacts = source
    (artifacts / "native-linux_amd64/SOURCE_COMMIT").write_text("b" * 40)
    with pytest.raises(ValueError, match="does not match HEAD"):
        packaging.package(root, artifacts, tmp_path / "output", "preview")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("status,tag", [(" M go.mod", "v0.4.0"), ("", ""), ("", "v0.4.0")])
def test_release_requires_clean_tagged_source(
    source: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, status: str, tag: str
) -> None:
    root, _ = source
    (root / "pyproject.toml").write_text('[project]\nversion = "0.4.0"\n')
    (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.4.0"\n')
    previous = packaging.git
    monkeypatch.setattr(
        packaging,
        "git",
        lambda root, *args: (
            {"status": status, "tag": tag}.get(args[0])
            if args[0] in {"status", "tag"}
            else previous(root, *args)
        ),
    )
    if status or not tag:
        with pytest.raises(ValueError, match="clean source|existing root tag"):
            packaging.source_metadata(root, "release")
    else:
        assert packaging.source_metadata(root, "release")["go_tag"] == "bindings/go/v0.4.0"
