# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned random-routing construction."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.roles import LLMBackend
from switchyard_rust.components import RandomRoutingProcessorConfig


@profile_config("random_routing")
class RandomRoutingProfileConfig:
    """Profile config wrapper for weighted strong/weak random routing."""

    config: RandomRoutingConfig

    @classmethod
    def from_config(cls, config: RandomRoutingConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the random-routing profile runtime."""
        from switchyard.lib.llm_client_builder import build_llm_client
        from switchyard.lib.processors.random_routing_request_processor import (
            RandomRoutingRequestProcessor,
        )

        config = self.config
        backend: LLMBackend = build_llm_client((config.strong, config.weak))

        return ComponentChainProfile(
            request_processors=[RandomRoutingRequestProcessor(config.processor_config)],
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


class RandomRoutingConfig(BaseModel):
    """Validated parsing config for weighted strong/weak random routing."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    strong: LlmTarget
    weak: LlmTarget
    fallback_target_on_evict: str
    strong_probability: float = 0.5
    enable_stats: bool = True
    rng_seed: int | None = None
    preset: str | None = None

    @field_validator("strong", "weak", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        """Accept dicts and existing ``LlmTarget`` instances for tiers."""
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("strong", "weak")
    @classmethod
    def _target_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        """Reject empty model names before the first request."""
        if not tier.model:
            raise ValueError("target.model must be a non-empty string")
        return tier

    @field_validator("strong_probability")
    @classmethod
    def _strong_prob_range(cls, value: float) -> float:
        """Validate the weighted coin range."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"strong_probability must be in [0.0, 1.0], got {value!r}"
            )
        return value

    @field_validator("fallback_target_on_evict")
    @classmethod
    def _fallback_matches_existing_target(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        """Ensure evict-and-retry can only fall back to a configured tier."""
        valid_ids = {info.data[key].id for key in ("strong", "weak") if key in info.data}
        if value not in valid_ids:
            raise ValueError(
                f"fallback_target_on_evict={value!r} must match one of "
                f"{sorted(valid_ids)} (the configured target ids)"
            )
        return value

    @property
    def processor_config(self) -> RandomRoutingProcessorConfig:
        """Return the Rust request-processor config for this validated config."""
        return RandomRoutingProcessorConfig(
            strong=self.strong,
            weak=self.weak,
            strong_probability=self.strong_probability,
            rng_seed=self.rng_seed,
        )

    @classmethod
    def model_validate_config(cls, raw: Any) -> RandomRoutingConfig:
        """Coerce dicts or Pydantic models into ``RandomRoutingConfig``."""
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, BaseModel):
            return cls(**raw.model_dump())
        return cls.model_validate(raw)


__all__ = [
    "RandomRoutingConfig",
    "RandomRoutingProfileConfig",
]
