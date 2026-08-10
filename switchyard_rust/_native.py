# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the private native extension shared by the public wrappers."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from typing import Any


def _ensure_switchyard_version_env() -> None:
    if os.environ.get("SWITCHYARD_VERSION", "").strip():
        return
    try:
        version = importlib.metadata.version("nemo-switchyard")
    except importlib.metadata.PackageNotFoundError:
        return
    if version.strip():
        os.environ["SWITCHYARD_VERSION"] = version


def load_native() -> Any:
    """Load the extension or report how to repair a source checkout."""
    _ensure_switchyard_version_env()
    try:
        return importlib.import_module("switchyard_rust._switchyard_rust")
    except ImportError as exc:  # pragma: no cover - broken install guard
        raise RuntimeError(
            "The Switchyard native extension is required. Run `uv run maturin develop` "
            "or install a built switchyard wheel."
        ) from exc
