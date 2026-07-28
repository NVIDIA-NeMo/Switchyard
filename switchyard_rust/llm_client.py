# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python binding for the Rust Switchyard LLM client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from switchyard_rust.core import _load_native

if TYPE_CHECKING:
    LlmClient: type[Any]


def __getattr__(name: str) -> object:
    if name == "LlmClient":
        return _load_native().LlmClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LlmClient"]
