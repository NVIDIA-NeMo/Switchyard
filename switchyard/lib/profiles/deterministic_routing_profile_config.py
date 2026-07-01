# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned deterministic LLM-classifier routing construction."""

from __future__ import annotations

from typing import Any, Self

from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.processors.llm_classifier.presets import (
    PROFILE_FACTORIES,
    resolve_classifier_prompt,
)
from switchyard.lib.processors.llm_classifier.signals import RouteTier
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.deterministic_routing_config import (
    DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
    DeterministicRoutingConfig,
)
from switchyard.lib.profiles.table import profile_config

_TIER_STRONG = "strong"
_TIER_WEAK = "weak"


@profile_config("deterministic")
class DeterministicRoutingProfileConfig:
    """Profile config wrapper for content-aware deterministic routing."""

    config: DeterministicRoutingConfig

    @classmethod
    def from_config(cls, config: DeterministicRoutingConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the deterministic routing profile runtime."""
        from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
            maybe_wrap_anthropic_cache,
        )
        from switchyard.lib.backends.deterministic_routing_llm_backend import (
            DeterministicRoutingLLMBackend,
        )
        from switchyard.lib.backends.multi_llm_backend import (
            build_native_backend,
            resolve_llm_target,
        )
        from switchyard.lib.processors.llm_classifier import (
            LLMClassifierRequestProcessor,
            SignalTierSelectorRequestProcessor,
        )
        from switchyard.lib.processors.reasoning_effort_normalizer import (
            ReasoningEffortNormalizer,
        )
        from switchyard.lib.session_affinity import SessionAffinity

        config = self.config
        tier_targets = config.resolved_tiers()
        default_tier = (
            config.default_tier
            or (_TIER_STRONG if _TIER_STRONG in tier_targets else next(iter(tier_targets)))
        )
        weak_label = _TIER_WEAK
        strong_label = default_tier
        if config.tier_mapping:
            weak_label = config.tier_mapping.get(RouteTier.SIMPLE, _TIER_WEAK)
            strong_label = config.tier_mapping.get(RouteTier.REASONING, default_tier)
        profile = PROFILE_FACTORIES[config.profile_name](
            weak=weak_label,
            strong=strong_label,
        )

        request_processors: list[Any] = [ReasoningEffortNormalizer()]

        # One affinity coordinator shared by the classifier and tier selector:
        # the classifier gates its LLM call on it (classify once per task) and
        # the tier selector records / reuses the per-conversation tier pin.
        affinity = SessionAffinity(
            enabled=config.session_affinity,
            max_sessions=config.affinity_max_sessions,
            warmup_turns=config.affinity_warmup_turns,
        )

        classifier_config = profile.make_classifier_config(
            model=config.classifier.model,
            api_key=config.classifier.api_key,
            base_url=config.classifier.base_url,
            timeout_s=config.classifier_timeout_s,
            max_request_chars=config.classifier_max_request_chars,
            fail_open=config.classifier_fail_open,
            recent_turn_window=config.classifier_recent_turn_window,
            system_prompt=resolve_classifier_prompt(
                config.profile_name,
                config.classifier_system_prompt,
            ),
        ).model_copy(update={"dump_signals_to_stderr": False})
        request_processors.append(
            LLMClassifierRequestProcessor(
                classifier_config,
                signal_schema=profile.signal_schema,
                affinity=affinity,
            )
        )
        request_processors.append(
            SignalTierSelectorRequestProcessor(
                profile.make_tier_selector_config(
                    tier_mapping=config.tier_mapping,
                    default_tier=default_tier,
                    min_confidence=config.classifier_min_confidence,
                ),
                affinity=affinity,
            )
        )

        # Resolve format='auto' once after deterministic tier defaults are
        # applied so backend selection and Anthropic cache wrapping see the
        # same concrete target.
        resolved_targets = {
            label: resolve_llm_target(
                _apply_deepseek_overrides(
                    _apply_default_tier_timeout(target, config.tier_timeout_s),
                ),
            )
            for label, target in tier_targets.items()
        }

        backend = DeterministicRoutingLLMBackend(
            tiers={
                label: (
                    maybe_wrap_anthropic_cache(
                        build_native_backend(target),
                        target,
                    ),
                    target.model,
                )
                for label, target in resolved_targets.items()
            },
            default_tier=default_tier,
        )
        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


def _apply_deepseek_overrides(target: LlmTarget) -> LlmTarget:
    """Apply benchmark-specific DeepSeek extras without clobbering callers."""
    default_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if "deepseek-v4" in target.model
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


def _apply_default_tier_timeout(
    target: LlmTarget,
    timeout_s: float | None = DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
) -> LlmTarget:
    """Apply deterministic tier timeout when a target has no explicit timeout."""
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


__all__ = ["DeterministicRoutingProfileConfig"]
