# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal bindings for Rust-owned libsy algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from switchyard_rust.core import _load_native

_EXPORTS = frozenset({"Algorithm", "LibsyError", "LlmTarget", "noop", "random"})

if TYPE_CHECKING:
    Algorithm: type[Any]
    LibsyError: type[RuntimeError]
    LlmTarget: type[Any]


class LlmClient(Protocol):
    """Structural interface for a Python-hosted model client."""

    async def call(
        self,
        request: Mapping[str, object],
        *,
        target: str,
    ) -> Mapping[str, object]:
        """Call the selected target and return an aggregate neutral response."""
        ...


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        native: Any = _load_native()
        return getattr(native.libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*sorted(_EXPORTS), "LlmClient"]
