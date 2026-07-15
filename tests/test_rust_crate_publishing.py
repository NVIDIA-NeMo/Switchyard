# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Rust crate release metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "release" / "publish_crates.py"


def load_publish_script() -> ModuleType:
    """Load the release helper as a regular Python module."""
    spec = importlib.util.spec_from_file_location("publish_crates", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load publish_crates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_crates_are_ordered_by_dependency() -> None:
    """Public crates should be published before crates that depend on them."""
    publish_crates = load_publish_script()

    assert publish_crates.PUBLIC_CRATES == (
        "switchyard-core",
        "switchyard-translation",
        "switchyard-components",
        "switchyard-components-v2-macros",
        "switchyard-components-v2",
        "switchyard-server",
    )


def test_rust_crate_publish_metadata_is_valid() -> None:
    """Release metadata should be valid before cargo publish runs."""
    publish_crates = load_publish_script()

    publish_crates.validate_publish_metadata()
