# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native bindings for request, response, and stream translation."""

from enum import Enum
from typing import TYPE_CHECKING, Any, final

from switchyard_rust._native import load_native


class WireFormat(str, Enum):
    """Supported provider JSON formats."""

    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


if TYPE_CHECKING:
    from collections.abc import AsyncIterable

    from switchyard_rust._translation_stream import _TranslatedStream, _TranslationStream

    class TranslationError(Exception):
        """Translation failure with a readable message and structured details."""

        kind: str
        details: dict[str, Any]

    @final
    class TranslationEngine:
        """Translate provider JSON and streams.

        Calls are independent. Request context is not retained for responses.
        """

        def __init__(self) -> None: ...

        def decode_request(
            self, source: WireFormat | str, body: dict[str, Any]
        ) -> dict[str, Any]:
            """Return request IR and diagnostics under request and diagnostics keys."""
            ...

        def encode_request(
            self, target: WireFormat | str, request: dict[str, Any]
        ) -> dict[str, Any]:
            """Encode request IR, returning body and diagnostics."""
            ...

        def decode_response(
            self, source: WireFormat | str, body: dict[str, Any]
        ) -> dict[str, Any]:
            """Return response IR and diagnostics under response and diagnostics keys."""
            ...

        def encode_response(
            self,
            target: WireFormat | str,
            response: dict[str, Any],
            *,
            request_extensions: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Return body and diagnostics, optionally using decoded request extensions."""
            ...

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

        def decode_stream(
            self, source: WireFormat | str, byte_stream: AsyncIterable[bytes]
        ) -> _TranslationStream:
            """Decode SSE bytes into IR events; prefer consumption inside async with."""
            ...

        def encode_stream(
            self,
            target: WireFormat | str,
            chunks: AsyncIterable[dict[str, Any]],
            *,
            served_model: str | None = None,
            request_extensions: dict[str, Any] | None = None,
        ) -> _TranslationStream:
            """Encode IR events into provider dictionaries; the caller owns input cleanup."""
            ...

        def translate_stream(
            self,
            source: WireFormat | str,
            target: WireFormat | str,
            byte_stream: AsyncIterable[bytes],
            *,
            served_model: str | None = None,
            request_extensions: dict[str, Any] | None = None,
        ) -> _TranslatedStream:
            """Convenience method for calling decode_stream and encode_stream."""
            ...


def __getattr__(name: str) -> object:
    if name in {"TranslationEngine", "TranslationError"}:
        native: Any = load_native()
        return getattr(native.translation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TranslationEngine", "TranslationError", "WireFormat"]
