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
        "DimensionCollector",
        "EndpointConfig",
        "LlmTarget",
        "LlmTargetBackend",
        "MultiLlmBackend",
        "OpenAiNativeBackend",
        "OpenAiPassthroughBackend",
        "PickOutcome",
        "RandomRoutingProcessorConfig",
        "ResponseFlag",
        "ResponseSignalCollector",
        "ResponseSignals",
        "StatsAccumulator",
        "StatsLlmBackend",
        "StatsRequestProcessor",
        "StatsResponseProcessor",
        "ToolResultSignal",
        "extract_response_signals",
        "get_response_signals",
        "get_tool_result_signal",
        "set_stats_route_label",
        "stage_pick_tier",
        "stage_score_signal",
    }
)
_BACKEND_EXPORTS = frozenset(
    {
        "AnthropicNativeBackend",
        "MultiLlmBackend",
        "OpenAiNativeBackend",
        "OpenAiPassthroughBackend",
        "StatsLlmBackend",
    }
)

if TYPE_CHECKING:
    AnthropicNativeBackend: type[Any]
    BackendFormat: type[Any]
    DimensionCollector: type[Any]
    EndpointConfig: type[Any]
    LlmTarget: type[Any]
    LlmTargetBackend: type[Any]
    MultiLlmBackend: type[Any]
    OpenAiNativeBackend: type[Any]
    OpenAiPassthroughBackend: type[Any]
    PickOutcome: type[Any]
    RandomRoutingProcessorConfig: type[Any]
    ResponseFlag: type[Any]
    ResponseSignalCollector: type[Any]
    ResponseSignals: type[Any]
    StatsAccumulator: type[Any]
    StatsLlmBackend: type[Any]
    StatsRequestProcessor: type[Any]
    StatsResponseProcessor: type[Any]
    ToolResultSignal: type[Any]
    extract_response_signals: Any
    get_response_signals: Any
    get_tool_result_signal: Any
    set_stats_route_label: Any
    stage_pick_tier: Any
    stage_score_signal: Any


def __getattr__(name: str) -> object:
    if name in _COMPONENT_EXPORTS:
        value = getattr(_load_native(), name)
        if name in _BACKEND_EXPORTS:
            from switchyard_rust.core import LLMBackend

            LLMBackend.register(value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_COMPONENT_EXPORTS)
