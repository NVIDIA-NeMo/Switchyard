# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned escalation-router construction (judge-latched strong/weak)."""

from __future__ import annotations

from typing import Any, Self

from switchyard.lib.affinity_pin_store import AffinityPinStore
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
from switchyard.lib.processors.reasoning_hint import model_accepts_reasoning_hint
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.deterministic_routing_profile_config import (
    _apply_deepseek_overrides,
    _apply_default_tier_timeout,
)
from switchyard.lib.profiles.escalation_router_config import EscalationRouterConfig
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.session_affinity import SessionAffinity


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
        config = self.config

        # The affinity store IS the escalation latch, so it is always on.
        # Warmup gating lives in the judge processor (min_judge_turn), not
        # here — affinity warmup would delay a decided escalation from
        # taking effect, not just from being decided. The optional Redis L2
        # keeps the latch across workers and pod churn; without it a worker
        # change silently restarts an escalated conversation on weak.
        affinity = SessionAffinity(
            enabled=True,
            max_sessions=config.affinity_max_sessions,
            warmup_turns=0,
            l2=_build_affinity_l2(config),
        )

        # DeepSeek judges get the same benchmark-gateway defaults as DeepSeek
        # tiers (``X-Inference-Priority: batch``) so their calls land on the
        # relaxed-timeout gateway alongside the routed tier traffic.
        judge_target = _apply_deepseek_overrides(config.judge)
        # Claude/Bedrock judges reject the vLLM-only chat_template_kwargs
        # hint outright — every judged turn would fail open to weak — so
        # the hint is only sent to models that tolerate it (same gate as
        # the classifier presets and the stage router).
        judge_disable_reasoning = (
            config.judge_disable_reasoning
            and model_accepts_reasoning_hint(judge_target.model)
        )
        judge_config = EscalationJudgeConfig(
            model=judge_target.model,
            api_key=judge_target.api_key,
            base_url=judge_target.base_url,
            timeout_s=judge_target.endpoint.timeout_secs or config.judge_timeout_s,
            system_prompt=config.judge_system_prompt or ESCALATION_JUDGE_SYSTEM_PROMPT,
            min_judge_turn=config.judge_min_turn,
            escalate_confirmations=config.judge_escalate_confirmations,
            confirmation_window=config.judge_confirmation_window,
            disable_reasoning=judge_disable_reasoning,
            # Reasoning tokens and the JSON verdict share one budget: a
            # thinking judge needs the classifier's larger ceiling or it
            # truncates mid-reasoning and silently fails open every turn.
            max_completion_tokens=(
                config.judge_max_completion_tokens
                if config.judge_max_completion_tokens is not None
                else (128 if judge_disable_reasoning else 4096)
            ),
            dump_verdicts_to_stderr=config.judge_dump_verdicts,
            recent_turn_window=config.judge_recent_turn_window,
            window_message_chars=config.judge_window_message_chars,
            max_request_chars=config.judge_max_request_chars,
            extra_headers=judge_target.extra_headers or None,
        )
        # Tier labels are the configured target ids ("strong"/"weak" unless
        # overridden). The judge stamps them, the backend keys its tier dict
        # by them, and the chain's evict-and-retry rewrites selected_target to
        # fallback_target_on_evict — which the config validates against these
        # same ids — so an overflow reroute always lands on a registered tier.
        strong_id = config.strong.id
        weak_id = config.weak.id
        request_processors: list[Any] = [
            ReasoningEffortNormalizer(),
            EscalationJudgeRequestProcessor(
                judge_config,
                affinity=affinity,
                session_key_depth=config.session_key_depth,
                strong_tier=strong_id,
                weak_tier=weak_id,
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
                strong_id: (strong_backend, strong_target.model),
                weak_id: (weak_backend, weak_target.model),
            },
            # Weak is the resting state; the judge processor stamps a tier on
            # every turn, so the default only covers malformed metadata.
            default_tier=weak_id,
        )
        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


def _build_affinity_l2(config: EscalationRouterConfig) -> AffinityPinStore | None:
    """Build the optional shared L2 latch store (``None`` = L1-only).

    Mirrors the latency route's ``affinity_store`` semantics; ``"redis"``
    imports :class:`RedisPinStore` lazily so the ``redis`` dependency stays
    optional. Config validation guarantees a URL when the store is Redis.
    """
    if config.affinity_store != "redis":
        return None
    from switchyard.lib.redis_pin_store import RedisPinStore

    assert config.affinity_store_url is not None  # enforced by config validator
    return RedisPinStore(
        config.affinity_store_url,
        ttl_seconds=config.affinity_store_ttl_seconds,
        key_prefix=config.affinity_key_prefix,
    )


__all__ = ["EscalationRouterProfileConfig"]
