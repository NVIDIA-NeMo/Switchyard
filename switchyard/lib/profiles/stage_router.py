# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned signal stage_router routing construction."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Self

from switchyard.lib.processors.reasoning_hint import model_accepts_reasoning_hint
from switchyard.lib.processors.stage_router import StageRouterDecisionLog, TierClassifier
from switchyard.lib.processors.stage_router.handoff_notes import HandoffNoteInjector
from switchyard.lib.processors.stage_router_request_processor import (
    BUILTIN_PICKERS,
    StageRouterRequestProcessor,
    TierPicker,
)
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.stage_router_config import (
    ClassifierConfig,
    HandoffNoteConfig,
    StageRouterConfig,
)
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.roles import LLMBackend

if TYPE_CHECKING:
    from switchyard.lib.backends.llm_target import LlmTarget


@profile_config("stage_router")
class StageRouterProfileConfig:
    """Profile config wrapper for signal-driven capable/efficient stage_router profiles."""

    config: StageRouterConfig

    @classmethod
    def from_config(cls, config: StageRouterConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the stage_router profile runtime."""
        from switchyard.lib.backends.deterministic_routing_llm_backend import (
            DeterministicRoutingLLMBackend,
        )
        from switchyard_rust.components import DimensionCollector

        config = self.config
        request_processors: list[Any] = []
        request_processors.append(
            DimensionCollector(recent_window=config.signal_recent_window)
        )
        decision_log = StageRouterDecisionLog()
        classifier = _build_classifier(config.classifier)
        request_processors.append(
            StageRouterRequestProcessor(
                targets=(config.efficient, config.capable),
                picker=_build_tier_picker(config, decision_log, classifier),
                classifier=classifier,
                decision_log=decision_log,
                handoff_injector=_build_handoff_injector(config.handoff_notes),
                strong_system_prompt=config.strong_system_prompt,
                weak_system_prompt=config.weak_system_prompt,
            )
        )

        # Host each tier in the Python routing backend rather than the Rust
        # MultiLlmBackend, which rejects Python-only backends: an Anthropic
        # capable tier must be wrapped so cache_control breakpoints reach the
        # model for every inbound client format. The picker stamps the tier id
        # on ctx.selected_target, which this backend routes on.
        efficient_id, efficient_tier = _build_tier(config.efficient)
        capable_id, capable_tier = _build_tier(config.capable)
        backend: LLMBackend = DeterministicRoutingLLMBackend(
            tiers={efficient_id: efficient_tier, capable_id: capable_tier},
            default_tier=efficient_id,
        )

        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


def _build_tier(target: LlmTarget) -> tuple[str, tuple[LLMBackend, str]]:
    """Build one tier's ``DeterministicRoutingLLMBackend`` entry.

    Returns ``(tier_id, (backend, model))``. The tier id is the target's id —
    the same value the picker stamps on ``ctx.selected_target``. Anthropic
    tiers are wrapped so ``cache_control`` breakpoints reach the model for any
    inbound client format; non-Anthropic tiers pass through unwrapped.
    """
    from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
        maybe_wrap_anthropic_cache,
    )
    from switchyard.lib.backends.multi_llm_backend import (
        build_native_backend,
        resolve_llm_target,
    )

    resolved = resolve_llm_target(target)
    backend = maybe_wrap_anthropic_cache(build_native_backend(resolved), resolved)
    return target.id, (backend, resolved.model)


def _build_tier_picker(
    config: StageRouterConfig,
    decision_log: StageRouterDecisionLog,
    classifier: TierClassifier | None,
) -> TierPicker:
    """Resolve the named stage_router picker and bind its runtime knobs."""
    picker_fn = BUILTIN_PICKERS.get(config.picker)
    if picker_fn is None:
        allowed = ", ".join(sorted(BUILTIN_PICKERS))
        raise ValueError(f"unknown picker {config.picker!r}; allowed: {allowed}")
    return functools.partial(
        picker_fn,
        confidence_threshold=config.confidence_threshold,
        classifier=classifier,
        decision_log=decision_log,
    )


def _build_handoff_injector(config: HandoffNoteConfig | None) -> HandoffNoteInjector | None:
    """Build the optional tier-transition note injector; ``None`` when disabled."""
    if config is None or not config.enabled:
        return None
    return HandoffNoteInjector(
        escalation_note=config.escalation_note,
        deescalation_note=config.deescalation_note,
        only_on_wrong_signal_escalation=config.only_on_wrong_signal_escalation,
    )


def _build_classifier(config: ClassifierConfig | None) -> TierClassifier | None:
    """Build the optional LLM fallback classifier for stage_router routing."""
    if config is None:
        return None
    return TierClassifier(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_secs=config.timeout_secs,
        recent_turn_window=config.recent_turn_window,
        disable_reasoning=model_accepts_reasoning_hint(config.model),
    )


__all__ = ["StageRouterProfileConfig"]
