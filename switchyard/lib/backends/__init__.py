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
from switchyard.lib.backends.stats_llm_backend import (
    StatsLlmBackend,
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
    "OpenAiPassthroughBackend",
    "OpenAiNativeBackend",
    "StatsLlmBackend",
]
