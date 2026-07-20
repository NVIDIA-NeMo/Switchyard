# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned deterministic LLM-classifier routing construction."""

from __future__ import annotations

from typing import Any, Self

from switchyard.lib.processors.llm_classifier.presets import (
    PROFILE_FACTORIES,
    resolve_classifier_prompt,
)
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.deterministic_routing_config import (
    DeterministicRoutingConfig,
)
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.profiles.tier_target_builders import build_tier_backend


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
        from switchyard.lib.backends.deterministic_routing_llm_backend import (
            DeterministicRoutingLLMBackend,
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
        # Tier labels are the configured target ids ("strong"/"weak" unless
        # overridden). The tier selector stamps them, the backend keys its
        # tier dict by them, and the chain's evict-and-retry rewrites
        # selected_target to fallback_target_on_evict — which the config
        # validates against these same ids — so an overflow reroute always
        # lands on a registered tier.
        strong_id = config.strong.id
        weak_id = config.weak.id
        profile = PROFILE_FACTORIES[config.profile_name](
            weak=weak_id,
            strong=strong_id,
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
                    min_confidence=config.classifier_min_confidence,
                ),
                affinity=affinity,
            )
        )

        strong_target, strong_backend = build_tier_backend(
            config.strong, config.tier_timeout_s,
        )
        weak_target, weak_backend = build_tier_backend(
            config.weak, config.tier_timeout_s,
        )

        backend = DeterministicRoutingLLMBackend(
            tiers={
                strong_id: (strong_backend, strong_target.model),
                weak_id: (weak_backend, weak_target.model),
            },
            default_tier=strong_id,
        )
        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


__all__ = ["DeterministicRoutingProfileConfig"]
