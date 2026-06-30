# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete :class:`LLMBackend` implementations + colocated backend config.

Each file defines one ``LLMBackend``. Re-exports live here for ergonomic imports like
``from switchyard.lib.backends import OpenAiNativeBackend``.
"""

from switchyard.lib.backends.backend_format_resolver import (
    BackendFormatResolution,
    BackendFormatResolver,
)
from switchyard.lib.backends.health_poller import (
    EndpointHealth,
    EndpointHealthStatus,
    HealthPoller,
)
from switchyard.lib.backends.latency_service_llm_backend import (
    LatencyServiceLLMBackend,
)
from switchyard_rust.components import (
    AnthropicNativeBackend,
    OpenAiNativeBackend,
    OpenAiPassthroughBackend,
)

__all__ = [
    "AnthropicNativeBackend",
    "BackendFormatResolution",
    "BackendFormatResolver",
    "EndpointHealth",
    "EndpointHealthStatus",
    "HealthPoller",
    "LatencyServiceLLMBackend",
    "OpenAiPassthroughBackend",
    "OpenAiNativeBackend",
]
