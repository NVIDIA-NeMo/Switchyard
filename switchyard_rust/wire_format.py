# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in wire formats exported by the native protocol bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from switchyard_rust._native import load_native

if TYPE_CHECKING:
    from enum import Enum

    class WireFormat(str, Enum):
        """Provider protocols accepted by configured routes."""

        OPENAI_CHAT = "openai_chat"
        OPENAI_RESPONSES = "openai_responses"
        ANTHROPIC_MESSAGES = "anthropic_messages"


def __getattr__(name: str) -> object:
    if name == "WireFormat":
        return cast(object, load_native().WireFormat)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["WireFormat"]
