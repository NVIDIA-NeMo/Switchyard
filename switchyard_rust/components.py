# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct bindings for Rust-owned Switchyard components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from switchyard_rust.core import _load_native

_COMPONENT_EXPORTS = frozenset(
    {
        "AnthropicNativeBackend",
        "BackendFormat",
        "ContextSignals",
        "DimensionCollector",
        "DimensionScore",
        "EndpointConfig",
        "IntakeQueueFullPolicy",
        "IntakeRequestMetadata",
        "IntakeRequestProcessor",
        "IntakeResponseProcessor",
        "IntakeSinkConfig",
        "LlmTarget",
        "LlmTargetBackend",
        "MultiLlmBackend",
        "OpenAiNativeBackend",
        "OpenAiPassthroughBackend",
        "RandomRoutingProcessorConfig",
        "RequestMetadata",
        "ResponseFlag",
        "ResponseSignalCollector",
        "ResponseSignals",
        "ScoringConfig",
        "ToolResultSignal",
        "extract_response_signals",
        "get_context_signals",
        "get_response_signals",
        "get_tool_result_signal",
        "set_stats_route_label",
    }
)

if TYPE_CHECKING:
    AnthropicNativeBackend: type[Any]
    BackendFormat: type[Any]
    ContextSignals: type[Any]
    DimensionCollector: type[Any]
    DimensionScore: type[Any]
    EndpointConfig: type[Any]
    IntakeQueueFullPolicy: type[Any]
    IntakeRequestMetadata: type[Any]
    IntakeRequestProcessor: type[Any]
    IntakeResponseProcessor: type[Any]
    IntakeSinkConfig: type[Any]
    LlmTarget: type[Any]
    LlmTargetBackend: type[Any]
    MultiLlmBackend: type[Any]
    OpenAiNativeBackend: type[Any]
    OpenAiPassthroughBackend: type[Any]
    RandomRoutingProcessorConfig: type[Any]
    RequestMetadata: type[Any]
    ResponseFlag: type[Any]
    ResponseSignalCollector: type[Any]
    ResponseSignals: type[Any]
    ScoringConfig: type[Any]
    ToolResultSignal: type[Any]
    extract_response_signals: Any
    get_context_signals: Any
    get_response_signals: Any
    get_tool_result_signal: Any
    set_stats_route_label: Any


def __getattr__(name: str) -> object:
    if name in _COMPONENT_EXPORTS:
        return getattr(_load_native(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_COMPONENT_EXPORTS)
