# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned escalation-router construction (judge-latched strong/weak)."""

from __future__ import annotations

from typing import Any, Self

from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.escalation_router_config import EscalationRouterConfig
from switchyard.lib.profiles.table import profile_config

_TIER_STRONG = "strong"
_TIER_WEAK = "weak"


@profile_config("escalation_router")
class EscalationRouterProfileConfig:
    """Profile config wrapper for judge-latched escalation routing."""

    config: EscalationRouterConfig

    @classmethod
    def from_config(cls, config: EscalationRouterConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the escalation-router profile runtime."""
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
        from switchyard.lib.processors.escalation_judge_request_processor import (
            ESCALATION_JUDGE_SYSTEM_PROMPT,
            EscalationJudgeConfig,
            EscalationJudgeRequestProcessor,
        )
        from switchyard.lib.processors.reasoning_effort_normalizer import (
            ReasoningEffortNormalizer,
        )
        from switchyard.lib.profiles.deterministic_routing_profile_config import (
            _apply_deepseek_overrides,
            _apply_default_tier_timeout,
        )
        from switchyard.lib.session_affinity import SessionAffinity

        config = self.config

        # The affinity store IS the escalation latch, so it is always on.
        # Warmup gating lives in the judge processor (min_judge_turn), not
        # here — affinity warmup would delay a decided escalation from
        # taking effect, not just from being decided.
        affinity = SessionAffinity(
            enabled=True,
            max_sessions=config.affinity_max_sessions,
            warmup_turns=0,
        )

        judge_config = EscalationJudgeConfig(
            model=config.judge.model,
            api_key=config.judge.api_key,
            base_url=config.judge.base_url,
            timeout_s=config.judge.endpoint.timeout_secs or 5.0,
            system_prompt=config.judge_system_prompt or ESCALATION_JUDGE_SYSTEM_PROMPT,
            min_judge_turn=config.judge_min_turn,
            recent_turn_window=config.judge_recent_turn_window,
            max_request_chars=config.judge_max_request_chars,
            extra_headers=config.judge.extra_headers or None,
        )
        request_processors: list[Any] = [
            ReasoningEffortNormalizer(),
            EscalationJudgeRequestProcessor(
                judge_config,
                affinity=affinity,
                session_key_depth=config.session_key_depth,
            ),
        ]

        # Resolve format='auto' once after tier defaults are applied so
        # backend selection and Anthropic cache wrapping see the same
        # concrete target (mirrors the deterministic profile).
        strong_target = resolve_llm_target(
            _apply_deepseek_overrides(
                _apply_default_tier_timeout(config.strong, config.tier_timeout_s),
            ),
        )
        weak_target = resolve_llm_target(
            _apply_deepseek_overrides(
                _apply_default_tier_timeout(config.weak, config.tier_timeout_s),
            ),
        )
        strong_backend = maybe_wrap_anthropic_cache(
            build_native_backend(strong_target),
            strong_target,
        )
        weak_backend = maybe_wrap_anthropic_cache(
            build_native_backend(weak_target),
            weak_target,
        )

        backend = DeterministicRoutingLLMBackend(
            tiers={
                _TIER_STRONG: (strong_backend, strong_target.model),
                _TIER_WEAK: (weak_backend, weak_target.model),
            },
            # Weak is the resting state; the judge processor stamps a tier on
            # every turn, so the default only covers malformed metadata.
            default_tier=_TIER_WEAK,
        )
        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


__all__ = ["EscalationRouterProfileConfig"]
