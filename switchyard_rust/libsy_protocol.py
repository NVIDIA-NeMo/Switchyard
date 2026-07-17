# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Rust-owned Switchyard protocol bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from switchyard_rust.core import _load_native

_EXPORTS = frozenset(
    {
        "AggLlmResponse",
        "ContentBlock",
        "Context",
        "Decision",
        "FileSource",
        "FormatId",
        "ImageSource",
        "InstructionBlock",
        "LlmRequest",
        "LlmResponseChunk",
        "LlmResponseStream",
        "MediaSource",
        "Message",
        "Metadata",
        "OutputParams",
        "PreservationMetadata",
        "ProviderExtensions",
        "ReasoningParams",
        "Request",
        "Response",
        "ResponseOutput",
        "Role",
        "SamplingParams",
        "StopReason",
        "ToolCall",
        "ToolChoice",
        "ToolDefinition",
        "ToolResult",
        "Usage",
        "WireFormat",
    }
)

if TYPE_CHECKING:
    AggLlmResponse: type[Any]
    ContentBlock: type[Any]
    Context: type[Any]
    Decision: type[Any]
    FileSource: type[Any]
    FormatId: type[Any]
    ImageSource: type[Any]
    InstructionBlock: type[Any]
    LlmRequest: type[Any]
    LlmResponseChunk: type[Any]
    LlmResponseStream: type[Any]
    MediaSource: type[Any]
    Message: type[Any]
    Metadata: type[Any]
    OutputParams: type[Any]
    PreservationMetadata: type[Any]
    ProviderExtensions: type[Any]
    ReasoningParams: type[Any]
    Request: type[Any]
    Response: type[Any]
    ResponseOutput: type[Any]
    Role: type[Any]
    SamplingParams: type[Any]
    StopReason: type[Any]
    ToolCall: type[Any]
    ToolChoice: type[Any]
    ToolDefinition: type[Any]
    ToolResult: type[Any]
    Usage: type[Any]
    WireFormat: type[Any]


class RoutedLlmClient(Protocol):
    """Host implementation that serves a routed model call."""

    async def call(
        self,
        context: Any,
        request: Any,
        decision: Any,
        /,
    ) -> Any:
        """Return a typed protocol response for the selected model."""
        ...


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        return getattr(_load_native().libsy_protocol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS | {"RoutedLlmClient"})
