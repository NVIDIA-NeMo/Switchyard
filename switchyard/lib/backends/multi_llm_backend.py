# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-owned multi-LLM backend helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from switchyard.lib.backends.backend_format_resolver import BackendFormatResolver
from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
    coerce_llm_target,
    llm_target_with_format,
    llm_target_with_runtime_defaults,
)
from switchyard.lib.roles import LLMBackend
from switchyard_rust.components import (
    AnthropicNativeBackend,
    GeminiNativeBackend,
    LlmTargetBackend,
    MultiLlmBackend,
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
    if target.format == BackendFormat.GEMINI:
        return GeminiNativeBackend(target)
    raise ValueError(f"Unsupported backend format: {target.format!r}")


def build_target_backend(target: LlmTarget) -> LlmTargetBackend:
    """Build one target/backend pair for ``MultiLlmBackend``."""
    target = llm_target_with_runtime_defaults(resolve_llm_target(target))
    return LlmTargetBackend(target, build_native_backend(target))


def build_multi_llm_backend(
    targets: Iterable[LlmTarget] | Mapping[str, LlmTarget],
    *,
    default_target_id: str | None = None,
) -> MultiLlmBackend:
    """Build a Rust ``MultiLlmBackend`` from configured targets."""
    if isinstance(targets, Mapping):
        target_values: Iterable[LlmTarget] = [
            coerce_llm_target(target, default_id=str(target_id))
            for target_id, target in targets.items()
        ]
    else:
        target_values = targets
    return MultiLlmBackend(
        [build_target_backend(target) for target in target_values],
        default_target_id=default_target_id,
    )


__all__ = [
    "MultiLlmBackend",
    "build_multi_llm_backend",
    "build_native_backend",
    "build_target_backend",
    "resolve_llm_target",
]
