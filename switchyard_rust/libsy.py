# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-owned libsy target bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from switchyard_rust.core import _load_native

_EXPORTS = frozenset(
    {
        "LlmTarget",
        "LlmTargetSet",
    }
)

if TYPE_CHECKING:
    LlmTarget: type[Any]
    LlmTargetSet: type[Any]


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        return getattr(_load_native().libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
