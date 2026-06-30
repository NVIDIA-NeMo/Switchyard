# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python companion wrappers for the ``crates/switchyard-py`` bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from switchyard_rust.translation import (
    TranslationEngine,
    is_native_translation_available,
)

if TYPE_CHECKING:
    from switchyard_rust.components import AnthropicNativeBackend as AnthropicNativeBackend
    from switchyard_rust.components import BackendFormat as BackendFormat
    from switchyard_rust.components import EndpointConfig as EndpointConfig
    from switchyard_rust.components import IntakeQueueFullPolicy as IntakeQueueFullPolicy
    from switchyard_rust.components import IntakeRequestMetadata as IntakeRequestMetadata
    from switchyard_rust.components import IntakeRequestProcessor as IntakeRequestProcessor
    from switchyard_rust.components import IntakeResponseProcessor as IntakeResponseProcessor
    from switchyard_rust.components import IntakeSinkConfig as IntakeSinkConfig
    from switchyard_rust.components import LlmTarget as LlmTarget
    from switchyard_rust.components import LlmTargetBackend as LlmTargetBackend
    from switchyard_rust.components import MultiLlmBackend as MultiLlmBackend
    from switchyard_rust.components import OpenAiNativeBackend as OpenAiNativeBackend
    from switchyard_rust.components import OpenAiPassthroughBackend as OpenAiPassthroughBackend
    from switchyard_rust.components import (
        RandomRoutingProcessorConfig as RandomRoutingProcessorConfig,
    )
    from switchyard_rust.components import RequestMetadata as RequestMetadata
    from switchyard_rust.core import ChatRequest as ChatRequest
    from switchyard_rust.core import ChatRequestType as ChatRequestType
    from switchyard_rust.core import ChatResponse as ChatResponse
    from switchyard_rust.core import ChatResponseStream as ChatResponseStream
    from switchyard_rust.core import ChatResponseType as ChatResponseType
    from switchyard_rust.core import LLMBackend as LLMBackend
    from switchyard_rust.core import ProxyContext as ProxyContext
    from switchyard_rust.core import ProxyMetadata as ProxyMetadata
    from switchyard_rust.core import SwitchyardBackendError as SwitchyardBackendError
    from switchyard_rust.core import SwitchyardConfigError as SwitchyardConfigError
    from switchyard_rust.core import (
        SwitchyardDuplicateRegistrationError as SwitchyardDuplicateRegistrationError,
    )
    from switchyard_rust.core import SwitchyardInvalidIdError as SwitchyardInvalidIdError
    from switchyard_rust.core import SwitchyardModelNotFoundError as SwitchyardModelNotFoundError
    from switchyard_rust.core import SwitchyardProcessorError as SwitchyardProcessorError
    from switchyard_rust.core import SwitchyardRuntimeError as SwitchyardRuntimeError
    from switchyard_rust.core import (
        SwitchyardUnsupportedRequestTypeError as SwitchyardUnsupportedRequestTypeError,
    )
    from switchyard_rust.core import SwitchyardUpstreamError as SwitchyardUpstreamError


def __getattr__(name: str) -> object:
    if name in {
        "AnthropicNativeBackend",
        "BackendFormat",
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
    }:
        from switchyard_rust import components

        return getattr(components, name)
    if name == "ChatRequest":
        from switchyard_rust.core import ChatRequest

        return ChatRequest
    if name == "ChatRequestType":
        from switchyard_rust.core import ChatRequestType

        return ChatRequestType
    if name == "ChatResponse":
        from switchyard_rust.core import ChatResponse

        return ChatResponse
    if name == "ChatResponseStream":
        from switchyard_rust.core import ChatResponseStream

        return ChatResponseStream
    if name == "ChatResponseType":
        from switchyard_rust.core import ChatResponseType

        return ChatResponseType
    if name == "LLMBackend":
        from switchyard_rust.core import LLMBackend

        return LLMBackend
    if name == "ProxyMetadata":
        from switchyard_rust.core import ProxyMetadata

        return ProxyMetadata
    if name == "ProxyContext":
        from switchyard_rust.core import ProxyContext

        return ProxyContext
    if name in {
        "SwitchyardRuntimeError",
        "SwitchyardConfigError",
        "SwitchyardInvalidIdError",
        "SwitchyardDuplicateRegistrationError",
        "SwitchyardModelNotFoundError",
        "SwitchyardUnsupportedRequestTypeError",
        "SwitchyardProcessorError",
        "SwitchyardBackendError",
        "SwitchyardUpstreamError",
    }:
        from switchyard_rust import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicNativeBackend",
    "BackendFormat",
    "ChatRequest",
    "ChatRequestType",
    "ChatResponse",
    "ChatResponseStream",
    "ChatResponseType",
    "EndpointConfig",
    "IntakeQueueFullPolicy",
    "IntakeRequestMetadata",
    "IntakeRequestProcessor",
    "IntakeResponseProcessor",
    "IntakeSinkConfig",
    "LLMBackend",
    "LlmTarget",
    "LlmTargetBackend",
    "MultiLlmBackend",
    "OpenAiNativeBackend",
    "OpenAiPassthroughBackend",
    "ProxyMetadata",
    "ProxyContext",
    "RandomRoutingProcessorConfig",
    "RequestMetadata",
    "SwitchyardBackendError",
    "SwitchyardConfigError",
    "SwitchyardDuplicateRegistrationError",
    "SwitchyardInvalidIdError",
    "SwitchyardModelNotFoundError",
    "SwitchyardProcessorError",
    "SwitchyardRuntimeError",
    "SwitchyardUnsupportedRequestTypeError",
    "SwitchyardUpstreamError",
    "TranslationEngine",
    "is_native_translation_available",
]
