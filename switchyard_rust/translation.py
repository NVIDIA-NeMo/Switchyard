# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native bindings for buffered request and response translation."""

from enum import Enum
from typing import TYPE_CHECKING, Any, final

from switchyard_rust._native import load_native


class WireFormat(str, Enum):
    """Supported provider JSON formats."""

    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


if TYPE_CHECKING:

    class TranslationError(Exception):
        """Translation failure with a readable message and structured details."""

        kind: str
        details: dict[str, Any]

    @final
    class TranslationEngine:
        """Translate buffered provider JSON using the default translation policy.

        Calls are independent. Request context is not retained for responses.
        """

        def __init__(self) -> None: ...

        def translate_request(
            self, source: WireFormat | str, target: WireFormat | str, body: dict[str, Any]
        ) -> dict[str, Any]:
            """Return a dictionary with body and diagnostics entries."""
            ...

        def translate_response(
            self, source: WireFormat | str, target: WireFormat | str, body: dict[str, Any]
        ) -> dict[str, Any]:
            """Translate a complete response, returning body and diagnostics."""
            ...


def __getattr__(name: str) -> object:
    if name in {"TranslationEngine", "TranslationError"}:
        native: Any = load_native()
        return getattr(native.translation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TranslationEngine", "TranslationError", "WireFormat"]
