# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/release/set_devzone_prerelease_version.py"
_SPEC = importlib.util.spec_from_file_location("set_devzone_prerelease_version", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
set_devzone_prerelease_version = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = set_devzone_prerelease_version
_SPEC.loader.exec_module(set_devzone_prerelease_version)


@pytest.mark.parametrize(
    ("raw_version", "normalized"),
    [
        ("0.0.1.dev", "0.0.1.dev0"),
        ("0.0.1.dev0", "0.0.1.dev0"),
        ("1.2.3.dev42", "1.2.3.dev42"),
    ],
)
def test_parse_devzone_prerelease_version(raw_version: str, normalized: str) -> None:
    version = set_devzone_prerelease_version.parse_devzone_prerelease_version(raw_version)

    assert version.version == normalized


@pytest.mark.parametrize(
    "raw_version",
    [
        "0.0.1",
        "0.0.1rc1",
        "0.0.1.dev.a",
        "internal/v0.0.1-dev.0",
        "v0.0.1.dev0",
    ],
)
def test_parse_devzone_prerelease_version_rejects_non_dev_versions(raw_version: str) -> None:
    with pytest.raises(ValueError):
        set_devzone_prerelease_version.parse_devzone_prerelease_version(raw_version)


def test_metadata_file_updates(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[build-system]\nrequires = []\n\n[project]\nname = "switchyard"\nversion = "0.1.0"\n'
    )
    init = tmp_path / "__init__.py"
    init.write_text('__all__ = []\n\n__version__ = "0.1.0"\n')

    assert set_devzone_prerelease_version.update_pyproject(
        pyproject,
        package_name="nemo-switchyard",
        version="0.0.1.dev0",
    )
    assert set_devzone_prerelease_version.update_python_init(init, "0.0.1.dev0")

    assert 'name = "nemo-switchyard"' in pyproject.read_text()
    assert 'version = "0.0.1.dev0"' in pyproject.read_text()
    assert '__version__ = "0.0.1.dev0"' in init.read_text()
