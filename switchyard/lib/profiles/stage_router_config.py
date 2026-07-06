# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative config for stage_router profiles. See ``docs/stage_router_routing.md``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target

#: Picker mode accepted in YAML. The profile resolves it to a :class:`TierPicker`.
#: The name describes the *default tier* — what the picker returns when the
#: scorer is ambiguous and no classifier is configured.
StageRouterPickerMode = Literal["capable_first", "efficient_first"]


class ClassifierConfig(BaseModel):
    """Optional LLM classifier invoked on low-confidence scorer outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    api_key: str
    base_url: str | None = None
    timeout_secs: float = Field(default=30.0, gt=0.0)
    #: Number of trailing conversation messages (from the inbound chat
    #: request) to include in the classifier prompt. ``0`` sends only the
    #: aggregate :class:`ToolResultSignal` summary. Default matches the
    #: Rust extractor's ``RECENT_WINDOW`` so the classifier sees the same
    #: span the ``recent_*`` signal fields cover.
    recent_turn_window: int = Field(default=3, ge=0)


class StageRouterConfig(BaseModel):
    """Configuration for the stage-router-routing profile."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    capable: LlmTarget
    efficient: LlmTarget
    #: Target ID the post-routing guard rewrites picks to when the picked
    #: target has been evicted from the pool (e.g. after a context-window
    #: overflow). Must match either ``capable.id`` or ``efficient.id``.
    fallback_target_on_evict: str
    picker: StageRouterPickerMode = "capable_first"
    #: Scorer confidence in ``[0, 1]`` below which the picker consults the
    #: classifier (if configured) or returns its default tier. ``0.0`` forces
    #: pure-deterministic routing; ``1.0`` forces every turn through the
    #: classifier (equivalent to the legacy ``coding_agent`` profile).
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Sliding-window size for the Rust signal extractor's ``recent_*``
    #: counts (``recent_write_count``, ``recent_edit_count``, etc.).
    #: Smaller = more reactive to the very last tool call; larger smooths
    #: turn-by-turn fluctuations. Default 3 matches
    #: ``DEFAULT_RECENT_WINDOW`` in ``tool_signals.rs``.
    signal_recent_window: int = Field(default=3, ge=1)
    classifier: ClassifierConfig | None = None
    enable_stats: bool = True

    @field_validator("capable", "efficient", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("capable", "efficient")
    @classmethod
    def _target_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        if not tier.model:
            raise ValueError("target.model must be a non-empty string")
        return tier

    @field_validator("fallback_target_on_evict")
    @classmethod
    def _fallback_matches_existing_target(cls, value: str, info: ValidationInfo) -> str:
        valid_ids = {info.data[key].id for key in ("capable", "efficient") if key in info.data}
        if value not in valid_ids:
            raise ValueError(
                f"fallback_target_on_evict={value!r} must match one of "
                f"{sorted(valid_ids)} (the configured target ids)"
            )
        return value
