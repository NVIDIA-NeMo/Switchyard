# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated configuration for learned prefill-probe routing."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.profiles.deterministic_routing_config import (
    DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
)


class PrefillProbeRoutingPolicyConfig(BaseModel):
    """Cost-aware policy applied to two checkpoint correctness heads."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        allow_inf_nan=False,
    )

    type: Literal["cost_aware"] = "cost_aware"
    lambda_: float = Field(alias="lambda", ge=0.0, le=1.0)
    weak_cost: float = Field(ge=0.0)
    strong_cost: float = Field(ge=0.0)


class PrefillProbeConfig(BaseModel):
    """Configuration for hidden-state routing between strong and weak tiers.

    The probe target produces prompt hidden states but never handles completion
    traffic. The external checkpoint maps those hidden states to correctness
    probabilities, and ``routing_policy.lambda_`` balances correctness against
    the configured completion costs.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    probe: LlmTarget
    strong: LlmTarget
    weak: LlmTarget
    strong_checkpoint_head: str
    weak_checkpoint_head: str
    hidden_states_dir: str
    checkpoint_dir: str
    routing_policy: PrefillProbeRoutingPolicyConfig
    fallback_target_on_evict: str
    tier_timeout_s: float | None = Field(
        default=DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
        gt=0.0,
    )
    enable_stats: bool = True

    @field_validator("probe", "strong", "weak", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        """Accept mappings and existing ``LlmTarget`` instances."""
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("probe", "strong", "weak")
    @classmethod
    def _target_model_non_empty(cls, target: LlmTarget) -> LlmTarget:
        """Reject targets without a usable model name."""
        if not target.model:
            raise ValueError("target.model must be a non-empty string")
        return target

    @field_validator(
        "strong_checkpoint_head",
        "weak_checkpoint_head",
        "hidden_states_dir",
        "checkpoint_dir",
        "fallback_target_on_evict",
    )
    @classmethod
    def _string_non_blank(cls, value: str, info: ValidationInfo) -> str:
        """Reject blank artifact, routing, and filesystem identifiers."""
        if not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate_cross_field_invariants(self) -> Self:
        """Validate target, checkpoint-head, and eviction relationships."""
        if self.strong.id == self.weak.id:
            raise ValueError(
                f"strong.id and weak.id must differ (both are {self.strong.id!r}); "
                "they key the routing tiers"
            )
        if self.strong_checkpoint_head == self.weak_checkpoint_head:
            raise ValueError(
                "strong_checkpoint_head and weak_checkpoint_head must be distinct"
            )
        valid_ids = {self.strong.id, self.weak.id}
        if self.fallback_target_on_evict not in valid_ids:
            raise ValueError(
                f"fallback_target_on_evict={self.fallback_target_on_evict!r} "
                f"must match one of {sorted(valid_ids)} "
                "(the configured strong/weak target ids)"
            )
        return self


__all__ = [
    "PrefillProbeConfig",
    "PrefillProbeRoutingPolicyConfig",
]
