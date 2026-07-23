# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Switchyard - Typed LLM routing and orchestration.

This library provides a composable, type-safe foundation for routing
requests across multiple LLM backends with intelligent tier selection,
format translation, and extensible middleware.
"""

from typing import TYPE_CHECKING, Any

from switchyard.lib.backends import (
    AnthropicNativeBackend,
    EndpointHealth,
    EndpointHealthStatus,
    HealthPoller,
    LatencyServiceLLMBackend,
    OpenAiNativeBackend,
)
from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)
from switchyard.lib.chat_request import (
    AnthropicChatRequest,
    OpenAIChatRequest,
    ResponsesChatRequest,
)
from switchyard.lib.chat_response import (
    AnthropicChatResponse,
    AnthropicResponseStream,
    AnthropicStreamingChatResponse,
    AnyResponseStream,
    CompletionChatResponse,
    ResponsesApiChatResponse,
    ResponsesApiStream,
    ResponsesApiStreamingChatResponse,
    ResponseStream,
    StreamingChatResponse,
)
from switchyard.lib.config import (
    IntakeSinkConfig,
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.processors.intake_payload_builder import IntakePayloadBuilder
from switchyard.lib.processors.intake_request_processor import IntakeRequestProcessor
from switchyard.lib.processors.intake_response_processor import IntakeResponseProcessor
from switchyard.lib.processors.rl_logging_request_processor import RlLoggingRequestProcessor
from switchyard.lib.processors.rl_logging_response_processor import RlLoggingResponseProcessor
from switchyard.lib.profiles import (
    ClassifierConfig,
    ContextAwareProfile,
    DeterministicRoutingConfig,
    DeterministicRoutingPresets,
    DeterministicRoutingProfileConfig,
    EscalationRouterConfig,
    EscalationRouterProfileConfig,
    LatencyServiceProfileConfig,
    NoopProfile,
    NoopProfileConfig,
    PassthroughProfileConfig,
    PlanExecuteConfig,
    PlanExecutePresets,
    PlanExecuteProfileConfig,
    Profile,
    ProfileConfig,
    ProfileConfigError,
    ProfileHooks,
    ProfileInput,
    ProfileLifecycle,
    ProfileRunner,
    ProfileSwitchyard,
    RandomRoutingConfig,
    RandomRoutingPresets,
    RandomRoutingProfileConfig,
    StageRouterConfig,
    StageRouterProfileConfig,
    TranslateProfileConfig,
    build_profile,
    profile_config,
    profile_config_type,
)
from switchyard.lib.roles import (
    LLMBackend,
)
from switchyard.lib.route_table import RouteTable
from switchyard.lib.switchyard import Switchyard
from switchyard_rust.components import RandomRoutingProcessorConfig
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
)
from switchyard_rust.translation import TranslationEngine

if TYPE_CHECKING:
    from switchyard.lib.endpoints.anthropic_messages_endpoint import (
        AnthropicMessagesEndpoint,
    )
    from switchyard.lib.endpoints.models_endpoint import ModelsEndpoint
    from switchyard.lib.endpoints.openai_chat_endpoint import (
        OpenAIChatEndpoint,
    )
    from switchyard.lib.endpoints.responses_endpoint import ResponsesEndpoint
    from switchyard.server.switchyard_app import build_switchyard_app


def __getattr__(name: str) -> Any:
    """Lazy-load optional server exports that require the ``server`` extra."""
    if name == "OpenAIChatEndpoint":
        from switchyard.lib.endpoints.openai_chat_endpoint import (
            OpenAIChatEndpoint,
        )

        return OpenAIChatEndpoint
    if name == "AnthropicMessagesEndpoint":
        from switchyard.lib.endpoints.anthropic_messages_endpoint import (
            AnthropicMessagesEndpoint,
        )

        return AnthropicMessagesEndpoint
    if name == "ResponsesEndpoint":
        from switchyard.lib.endpoints.responses_endpoint import ResponsesEndpoint

        return ResponsesEndpoint
    if name == "ModelsEndpoint":
        from switchyard.lib.endpoints.models_endpoint import ModelsEndpoint

        return ModelsEndpoint
    if name == "build_switchyard_app":
        from switchyard.server.switchyard_app import build_switchyard_app

        return build_switchyard_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # ChatRequest types
    "AnthropicChatRequest",
    "ChatRequest",
    "ChatRequestType",
    "OpenAIChatRequest",
    "ResponsesChatRequest",
    # Chain infrastructure
    "Switchyard",
    "LLMBackend",
    "NoopProfile",
    "NoopProfileConfig",
    "StageRouterConfig",
    "StageRouterProfileConfig",
    "ClassifierConfig",
    "ContextAwareProfile",
    "DeterministicRoutingConfig",
    "DeterministicRoutingProfileConfig",
    "DeterministicRoutingPresets",
    "EscalationRouterConfig",
    "EscalationRouterProfileConfig",
    "LatencyServiceProfileConfig",
    "PassthroughProfileConfig",
    "PlanExecuteConfig",
    "PlanExecuteProfileConfig",
    "PlanExecutePresets",
    "Profile",
    "ProfileConfig",
    "ProfileConfigError",
    "ProfileHooks",
    "ProfileInput",
    "ProfileLifecycle",
    "ProfileRunner",
    "ProfileSwitchyard",
    "RandomRoutingConfig",
    "RandomRoutingPresets",
    "RandomRoutingProfileConfig",
    "TranslateProfileConfig",
    "build_profile",
    "profile_config",
    "profile_config_type",
    "AnthropicNativeBackend",
    "OpenAiNativeBackend",
    "OpenAIChatEndpoint",
    "AnthropicMessagesEndpoint",
    "ResponsesEndpoint",
    "ModelsEndpoint",
    "build_switchyard_app",
    # Route dispatch table
    "RouteTable",
    # Latency Service usage case
    "EndpointHealth",
    "EndpointHealthStatus",
    "HealthPoller",
    "LatencyServiceBackendConfig",
    "LatencyServiceEndpoint",
    "LatencyServiceLLMBackend",
    "IntakePayloadBuilder",
    "IntakeRequestProcessor",
    "IntakeResponseProcessor",
    "IntakeSinkConfig",
    "RlLoggingRequestProcessor",
    "RlLoggingResponseProcessor",
    # Random Routing usage case
    "BackendFormat",
    "RandomRoutingProcessorConfig",
    "LlmTarget",
    # Deterministic (LLM-classifier) routing usage case
    # Plan-execute (strong planner + weak executor) usage case
    # Translation engine
    "TranslationEngine",
    # ChatResponse types
    "AnthropicChatResponse",
    "ChatResponse",
    "ChatResponseType",
    "CompletionChatResponse",
    "StreamingChatResponse",
    "ResponsesApiChatResponse",
    "ResponsesApiStreamingChatResponse",
    "AnthropicStreamingChatResponse",
    "ResponseStream",
    "ResponsesApiStream",
    "AnthropicResponseStream",
    "AnyResponseStream",
]

__version__ = "0.1.0"
