# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config model for the escalation-router profile (judge-latched strong/weak)."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.profiles.deterministic_routing_config import (
    DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
)


class EscalationRouterConfig(BaseModel):
    """Configuration for the escalation-router profile.

    Every conversation starts on the ``weak`` tier. An LLM ``judge`` watches
    the trajectory each turn and, on a clear pattern of trouble, escalates the
    conversation to the ``strong`` tier — one-way for the rest of the task
    (session-affinity latch). A new conversation resets to weak.

    Attributes:
        strong: Escalation target (frontier / expensive model).
        weak: Starting tier (cheap / efficient model).
        judge: Target for the judge LLM call. ``model`` / ``base_url`` /
            ``api_key`` / ``timeout_secs`` are extracted at build time;
            other target fields are ignored.
        fallback_target_on_evict: Target id the chain executor reroutes to on
            eviction (context-window overflow). Must match ``strong.id`` or
            ``weak.id``; the judge target is not a routing candidate.
        judge_min_turn: First conversation turn on which the judge runs.
        judge_recent_turn_window: Trailing messages shown to the judge on top
            of the system + first-user anchors.
        judge_max_request_chars: Cap on the assembled judge transcript.
        judge_system_prompt: Optional judge prompt override. ``None`` uses the
            built-in prompt.
        judge_timeout_s: Per-call judge timeout (seconds); the judge fails
            open to the weak tier at timeout.
        session_key_depth: ``0`` (default) keys conversations on system +
            first user message (the shared Rust session key). ``N > 0``
            extends the key with the first ``N`` post-first-user messages so
            repeated runs of the identical task (k>1 benchmark trials against
            one server) diverge via early model responses instead of sharing
            an escalation latch. Requires nonzero sampling temperature to be
            effective; production traffic should keep ``0``.
        tier_timeout_s: Default per-call timeout for strong/weak tier calls
            when a target does not set its own ``timeout_secs``.
        enable_stats: Wire stats processors and per-tier stats wrappers.
        affinity_max_sessions: LRU capacity of the escalation latch store.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    strong: LlmTarget
    weak: LlmTarget
    judge: LlmTarget
    fallback_target_on_evict: str
    judge_min_turn: int = Field(default=3, ge=1)
    judge_recent_turn_window: int = Field(default=14, ge=1)
    judge_max_request_chars: int = Field(default=12_000, ge=1_000)
    judge_system_prompt: str | None = Field(default=None, min_length=1)
    judge_timeout_s: float = Field(default=5.0, gt=0.0)
    session_key_depth: int = Field(default=0, ge=0)
    tier_timeout_s: float | None = Field(
        default=DEFAULT_DETERMINISTIC_TIER_TIMEOUT_S,
        gt=0.0,
    )
    enable_stats: bool = True
    affinity_max_sessions: int = Field(default=10_000, gt=0)

    @field_validator("strong", "weak", "judge", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("strong", "weak", "judge")
    @classmethod
    def _target_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        if not tier.model:
            raise ValueError("target.model must be a non-empty string")
        return tier

    @field_validator("judge_system_prompt", mode="before")
    @classmethod
    def _blank_judge_prompt_is_unset(cls, value: object) -> object:
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
                f"the judge target is not a routing candidate)"
            )
        return value


__all__ = ["EscalationRouterConfig"]
