# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared strong/weak tier target preparation and backend construction.

The deterministic (classifier) router and the escalation router build the same
two-tier shape: apply the default tier timeout, apply DeepSeek
benchmark-gateway overrides, resolve ``format='auto'`` once, build the native
backend, and wrap Anthropic targets with cache breakpoints. Owning those rules
here keeps the two profiles from drifting as the rules change.
"""

from __future__ import annotations

from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
    maybe_wrap_anthropic_cache,
)
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.backends.multi_llm_backend import (
    build_native_backend,
    resolve_llm_target,
)
from switchyard.lib.processors.reasoning_hint import model_accepts_reasoning_hint
from switchyard.lib.profiles.deterministic_routing_config import (
    DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
)
from switchyard.lib.roles import LLMBackend


def apply_deepseek_overrides(target: LlmTarget) -> LlmTarget:
    """Apply benchmark-specific DeepSeek extras without clobbering callers.

    The thinking-off default is a vLLM-side hint (``chat_template_kwargs``);
    serving stacks on the :func:`model_accepts_reasoning_hint` deny list
    reject the field outright (HTTP 400 ``Extra inputs are not permitted``),
    so the default is gated on the same model-id check the LLM classifier
    already uses for its own calls. The batch-priority header default is
    provider-neutral and stays unconditional.
    """
    default_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if "deepseek-v4" in target.model and model_accepts_reasoning_hint(target.model)
        else None
    )
    default_headers = (
        {"X-Inference-Priority": "batch"}
        if "deepseek" in target.model
        else None
    )
    if default_body is None and default_headers is None:
        return target

    existing_body = target.extra_body
    # LlmTarget normalizes omitted and explicit empty headers to the same
    # empty dict, so keep current defaulting behavior for normal DeepSeek
    # targets until the target type preserves "headers were provided" state.
    existing_headers = target.extra_headers or None
    merged_body = existing_body if existing_body is not None else default_body
    merged_headers = existing_headers if existing_headers is not None else default_headers
    if merged_body == existing_body and merged_headers == existing_headers:
        return target

    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target.format,
        base_url=target.base_url,
        api_key=target.api_key,
        timeout_secs=target.endpoint.timeout_secs,
        extra_body=merged_body,
        extra_headers=merged_headers,
    )


def apply_default_tier_timeout(
    target: LlmTarget,
    timeout_s: float | None = DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
) -> LlmTarget:
    """Apply the default tier timeout when a target has no explicit timeout."""
    if timeout_s is None or target.endpoint.timeout_secs is not None:
        return target
    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target.format,
        base_url=target.base_url,
        api_key=target.api_key,
        timeout_secs=timeout_s,
        extra_body=target.extra_body,
        extra_headers=target.extra_headers,
    )


def build_tier_backend(
    target: LlmTarget,
    tier_timeout_s: float | None,
) -> tuple[LlmTarget, LLMBackend]:
    """Prepare one tier target and build its wrapped backend.

    Resolves ``format='auto'`` once after the tier defaults are applied so
    backend selection and Anthropic cache wrapping see the same concrete
    target. Returns ``(resolved_target, backend)`` — profiles need the
    resolved target for the tier's model id.
    """
    resolved = resolve_llm_target(
        apply_deepseek_overrides(apply_default_tier_timeout(target, tier_timeout_s)),
    )
    return resolved, maybe_wrap_anthropic_cache(build_native_backend(resolved), resolved)


__all__ = [
    "apply_deepseek_overrides",
    "apply_default_tier_timeout",
    "build_tier_backend",
]
