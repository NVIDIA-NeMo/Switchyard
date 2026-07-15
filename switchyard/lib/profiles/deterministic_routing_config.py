# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config model for the deterministic LLM-classifier routing profile."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.processors.llm_classifier import DEFAULT_MAX_REQUEST_CHARS

ProfileName = Literal["general", "coding_agent", "openclaw"]
DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S = 600.0


class DeterministicRoutingConfig(BaseModel):
    """Configuration for the deterministic LLM-classifier routing profile.

    A strong/weak tier pair plus a classifier LLM target. The classifier
    inspects each request and decides which tier to dispatch to via the
    profile's tier mapping; routing is content-aware, not a coin flip.

    Attributes:
        strong: Strong tier (typically the higher-quality model).
            Selected when the classifier collapses
            ``RouteTier.COMPLEX`` / ``RouteTier.REASONING`` (and per
            profile rules, also on tool-planning escalation).
        weak: Weak tier (typically the cheap / fast model). Selected
            when the classifier returns ``RouteTier.SIMPLE`` (and,
            depending on profile, ``RouteTier.MEDIUM``).
        classifier: Target for the classifier LLM call itself.
            ``model`` / ``base_url`` / ``api_key`` are extracted at
            build time; ``format``, ``extra_body``, ``extra_headers``
            are ignored (the classifier client owns its own structured-
            output mechanics and the DeepSeek
            ``chat_template_kwargs.enable_thinking=False`` knob is
            applied via :class:`LLMClassifierConfig.disable_reasoning`).
        profile_name: Selects which :class:`LLMClassifierPresets` preset
            bundles the classifier prompt, JSON schema, and tier mapping.
        classifier_min_confidence: Below this floor the tier selector
            falls back to the profile's ``default_tier`` (strong).
            ``0.0`` (default) honors every non-abstain classification.
        classifier_fail_open: When ``True`` (default), classifier errors
            stamp abstain signals and fall back to the default tier.
            When ``False``, classifier errors propagate as 5xx.
        classifier_recent_turn_window: Number of trailing turns the
            classifier sees in addition to the system + first-user
            anchors. ``4`` matches the validated benchmark default for
            agent-loop traffic.
        classifier_system_prompt: Optional prompt override for the
            selected classifier profile. ``None`` uses the profile's
            built-in prompt.
        classifier_max_request_chars: Maximum serialized request-summary
            characters sent to the classifier before truncation.
        classifier_timeout_s: Per-call timeout for the classifier LLM
            (seconds). Defaults to ``30`` — the classifier fails open at
            timeout, so a short floor bounds tail latency without losing
            availability.
        tier_timeout_s: Default per-call timeout for strong/weak tier
            LLM calls when a target does not set its own
            ``timeout_secs``. ``None`` disables the default and leaves
            the provider client unbounded.
        enable_stats: Wire stats request/response processors and per-
            tier :class:`StatsLlmBackend` wrappers. Default ``True``.
        preset: Optional name of the
            :class:`DeterministicRoutingPresets` preset builder that produced
            this config — surfaced via ``GET /v1/routing/stats`` so
            saved stats files self-document which shipping bundle was
            used.
        affinity_warmup_turns: Initial conversation turns that remain
            non-sticky even when ``session_affinity`` is enabled. The first
            confident verdict after this warmup can pin the session.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    strong: LlmTarget
    weak: LlmTarget
    classifier: LlmTarget
    #: Target id the chain executor reroutes to when the picked target is
    #: evicted (e.g. context-window overflow). Must match either
    #: ``strong.id`` or ``weak.id``. The classifier target is not a routing
    #: candidate, so it cannot be selected as the fallback.
    fallback_target_on_evict: str
    profile_name: ProfileName = "coding_agent"
    classifier_min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classifier_fail_open: bool = True
    classifier_recent_turn_window: int = Field(default=4, ge=0)
    classifier_system_prompt: str | None = Field(default=None, min_length=1)
    classifier_max_request_chars: int = Field(
        default=DEFAULT_MAX_REQUEST_CHARS,
        ge=256,
    )
    classifier_timeout_s: float = Field(default=30.0, gt=0.0)
    tier_timeout_s: float | None = Field(
        default=DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
        gt=0.0,
    )
    enable_stats: bool = True
    session_affinity: bool = False
    affinity_max_sessions: int = Field(default=10_000, ge=0)
    affinity_warmup_turns: int = Field(default=0, ge=0)
    preset: str | None = None

    @field_validator("strong", "weak", "classifier", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("strong", "weak", "classifier")
    @classmethod
    def _target_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        if not tier.model:
            raise ValueError("target.model must be a non-empty string")
        return tier

    @field_validator("weak")
    @classmethod
    def _tier_ids_distinct(cls, value: LlmTarget, info: ValidationInfo) -> LlmTarget:
        strong = info.data.get("strong")
        if isinstance(strong, LlmTarget) and strong.id == value.id:
            raise ValueError(
                f"strong.id and weak.id must differ (both are {value.id!r}); "
                "they key the routing tiers"
            )
        return value

    @field_validator("classifier_system_prompt", mode="before")
    @classmethod
    def _blank_classifier_prompt_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("fallback_target_on_evict")
    @classmethod
    def _fallback_matches_existing_target(cls, value: str, info: ValidationInfo) -> str:
        valid_ids = {info.data[key].id for key in ("strong", "weak") if key in info.data}
        if value not in valid_ids:
            raise ValueError(
                f"fallback_target_on_evict={value!r} must match one of "
                f"{sorted(valid_ids)} (the configured strong/weak target ids; "
                f"the classifier target is not a routing candidate)"
            )
        return value

    @model_validator(mode="after")
    def _affinity_capacity_nonzero_when_enabled(self) -> DeterministicRoutingConfig:
        # A zero-capacity affinity store retains nothing, which would silently
        # disable stickiness while still paying the per-request key cost.
        if self.session_affinity and self.affinity_max_sessions == 0:
            raise ValueError(
                "affinity_max_sessions must be > 0 when session_affinity is enabled"
            )
        return self


__all__ = [
    "DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S",
    "DeterministicRoutingConfig",
    "ProfileName",
]
