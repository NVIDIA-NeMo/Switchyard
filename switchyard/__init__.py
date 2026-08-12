# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard's Python launcher and libsy bindings."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("nemo-switchyard")
except _metadata.PackageNotFoundError:
    # A source checkout may not have installed distribution metadata.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
