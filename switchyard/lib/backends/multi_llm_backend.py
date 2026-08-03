# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a native backend for a single LLM target."""

from __future__ import annotations

import logging

from switchyard.lib.backends.backend_format_resolver import BackendFormatResolver
from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
    llm_target_with_format,
    llm_target_with_runtime_defaults,
)
from switchyard.lib.roles import LLMBackend
from switchyard_rust.components import (
    AnthropicNativeBackend,
    OpenAiNativeBackend,
)

log = logging.getLogger(__name__)


def resolve_llm_target(target: LlmTarget) -> LlmTarget:
    """Resolve ``BackendFormat.AUTO`` into the concrete native backend format."""
    if target.format != BackendFormat.AUTO:
        return target
    resolution = BackendFormatResolver.resolve(target)
    log.debug(
        "resolved LLM target id=%s model=%s format=%s: %s",
        target.id,
        target.model,
        resolution.format.value,
        resolution.reason,
    )
    return llm_target_with_format(target, resolution.format)


def build_native_backend(target: LlmTarget) -> LLMBackend:
    """Build the native Rust backend for one resolved or auto ``LlmTarget``."""
    target = llm_target_with_runtime_defaults(resolve_llm_target(target))
    if target.format in (BackendFormat.OPENAI, BackendFormat.RESPONSES):
        return OpenAiNativeBackend(target)
    if target.format == BackendFormat.ANTHROPIC:
        return AnthropicNativeBackend(target)
    raise ValueError(f"Unsupported backend format: {target.format!r}")


__all__ = [
    "build_native_backend",
    "resolve_llm_target",
]
