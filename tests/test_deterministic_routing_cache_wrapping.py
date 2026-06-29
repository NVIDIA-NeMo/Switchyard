# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deterministic profile wraps Anthropic tiers for prompt caching.

An OpenAI-shaped harness (Codex, OpenAI SDK) routed onto a Claude tier never
sends Anthropic ``cache_control`` markers, so without the wrapper the
translated ``/v1/messages`` request misses caching entirely. The profile must
wrap any tier that resolves to Anthropic format in
:class:`AnthropicCacheBreakpointBackend`, and leave OpenAI tiers bare.
"""

from __future__ import annotations

from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
    AnthropicCacheBreakpointBackend,
)
from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles.deterministic_routing_config import DeterministicRoutingConfig
from switchyard.lib.profiles.deterministic_routing_profile_config import (
    DeterministicRoutingProfileConfig,
)


def _config(
    *,
    strong_format: BackendFormat,
    weak_format: BackendFormat = BackendFormat.OPENAI,
    enable_stats: bool = False,
) -> DeterministicRoutingConfig:
    # Weak model follows weak_format so we can exercise an Anthropic *weak*
    # tier (e.g. Opus as the weak model) as well as the usual OpenAI one.
    weak_model = (
        "aws/anthropic/bedrock-claude-opus-4-7"
        if weak_format == BackendFormat.ANTHROPIC
        else "azure/openai/gpt-5.5"
    )
    return DeterministicRoutingConfig(
        strong=LlmTarget(
            id="strong",
            model="aws/anthropic/bedrock-claude-opus-4-7",
            format=strong_format,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        weak=LlmTarget(
            id="weak",
            model=weak_model,
            format=weak_format,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        classifier=LlmTarget(
            id="classifier",
            model="gcp/google/gemini-3.5-flash",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        profile_name="coding_agent",  # type: ignore[arg-type]
        classifier_min_confidence=0.0,
        enable_stats=enable_stats,
        fallback_target_on_evict="strong",
    )


def _backend(config: DeterministicRoutingConfig) -> DeterministicRoutingLLMBackend:
    """Build the deterministic profile and return its dispatch backend."""
    profile = (
        DeterministicRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(enable_stats=config.enable_stats)
    )
    backend = next(
        component for component in profile.iter_components()
        if isinstance(component, DeterministicRoutingLLMBackend)
    )
    return backend


def test_anthropic_strong_tier_is_cache_wrapped() -> None:
    backend = _backend(_config(strong_format=BackendFormat.ANTHROPIC))
    assert isinstance(backend._backends["strong"], AnthropicCacheBreakpointBackend)


def test_openai_weak_tier_is_not_wrapped() -> None:
    backend = _backend(_config(strong_format=BackendFormat.ANTHROPIC))
    assert not isinstance(
        backend._backends["weak"], AnthropicCacheBreakpointBackend,
    )


def test_anthropic_weak_tier_is_cache_wrapped() -> None:
    # Opus as the *weak* model (format=auto/anthropic) is wrapped too — the
    # fix keys on resolved format, not the strong/weak tier role.
    backend = _backend(
        _config(
            strong_format=BackendFormat.OPENAI,
            weak_format=BackendFormat.ANTHROPIC,
        ),
    )
    assert isinstance(backend._backends["weak"], AnthropicCacheBreakpointBackend)
    assert not isinstance(
        backend._backends["strong"], AnthropicCacheBreakpointBackend,
    )


def test_openai_strong_tier_is_not_wrapped() -> None:
    backend = _backend(_config(strong_format=BackendFormat.OPENAI))
    assert not isinstance(
        backend._backends["strong"], AnthropicCacheBreakpointBackend,
    )
