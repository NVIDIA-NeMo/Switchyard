# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction helpers for the Rust Switchyard LLM client."""

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
from switchyard_rust.llm_client import LlmClient

log = logging.getLogger(__name__)


def resolve_llm_target(target: LlmTarget) -> LlmTarget:
    """Resolve ``BackendFormat.AUTO`` into a concrete provider wire format."""
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


def prepare_llm_target(target: LlmTarget) -> LlmTarget:
    """Apply runtime defaults and resolve one target for the Rust client."""
    return llm_target_with_runtime_defaults(resolve_llm_target(target))


def build_target_llm_client(target: LlmTarget) -> LlmClient:
    """Build an LLM client for one configured target."""
    return LlmClient([prepare_llm_target(target)])


def build_llm_client(
    targets: Iterable[LlmTarget] | Mapping[str, LlmTarget],
    *,
    default_target_id: str | None = None,
) -> LlmClient:
    """Build an LLM client over configured routing targets."""
    if isinstance(targets, Mapping):
        target_values: Iterable[LlmTarget] = [
            coerce_llm_target(target, default_id=str(target_id))
            for target_id, target in targets.items()
        ]
    else:
        target_values = targets
    return LlmClient(
        [prepare_llm_target(target) for target in target_values],
        default_target_id=default_target_id,
    )


__all__ = [
    "build_llm_client",
    "build_target_llm_client",
    "prepare_llm_target",
    "resolve_llm_target",
]
