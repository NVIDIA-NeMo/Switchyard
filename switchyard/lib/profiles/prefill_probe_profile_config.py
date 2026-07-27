# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned construction for learned prefill-probe routing."""

from __future__ import annotations

from typing import Self

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.prefill_probe_config import PrefillProbeConfig
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.profiles.tier_target_builders import build_tier_backend

DEFAULT_PREFILL_PROBE_BASE_URL = "http://localhost:8000/v1"


@profile_config("prefill_probe")
class PrefillProbeProfileConfig:
    """Profile config wrapper for learned prompt hidden-state routing."""

    config: PrefillProbeConfig

    @classmethod
    def from_config(cls, config: PrefillProbeConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the prefill-probe profile runtime and validate its checkpoint."""
        from switchyard_rust.components import PrefillProbeRequestProcessor

        config = self.config
        strong_id = config.strong.id
        weak_id = config.weak.id
        policy = config.routing_policy

        processor = PrefillProbeRequestProcessor(
            probe_base_url=config.probe.base_url or DEFAULT_PREFILL_PROBE_BASE_URL,
            probe_model=config.probe.model,
            hidden_states_dir=config.hidden_states_dir,
            checkpoint_dir=config.checkpoint_dir,
            strong_checkpoint_head=config.strong_checkpoint_head,
            weak_checkpoint_head=config.weak_checkpoint_head,
            strong_target_id=strong_id,
            weak_target_id=weak_id,
            routing_lambda=policy.lambda_,
            weak_cost=policy.weak_cost,
            strong_cost=policy.strong_cost,
        )

        strong_target, strong_backend = build_tier_backend(
            config.strong,
            config.tier_timeout_s,
        )
        weak_target, weak_backend = build_tier_backend(
            config.weak,
            config.tier_timeout_s,
        )
        backend = DeterministicRoutingLLMBackend(
            tiers={
                strong_id: (strong_backend, strong_target.model),
                weak_id: (weak_backend, weak_target.model),
            },
            default_tier=strong_id,
        )
        return ComponentChainProfile(
            request_processors=[processor],
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


__all__ = [
    "DEFAULT_PREFILL_PROBE_BASE_URL",
    "PrefillProbeProfileConfig",
]
