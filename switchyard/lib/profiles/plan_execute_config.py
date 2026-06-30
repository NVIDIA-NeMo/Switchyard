# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config model for the plan-execute profile.

A plan-execute chain pairs a **strong planner** (calls the planner LLM,
emits a :class:`PlannerDecision`, optionally prefills ``plan_text`` into
the executor's request as an assistant turn) with a **weak executor**
that carries out each plan step. The cost-asymmetry is the whole point
— see ``switchyard/lib/profiles/plan_execute_presets.py``
for the rationale.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target


class PlanExecuteConfig(BaseModel):
    """Configuration for the plan-execute profile.

    Attributes:
        planner: Strong planner tier. Must support
            ``response_format={"type": "json_object"}`` to honour the
            :class:`PlanningConfig` default. ``model`` / ``api_key`` /
            ``base_url`` / ``timeout_secs`` are forwarded into the
            :class:`PlanningConfig` constructed by the profile.
        executor: Weak executor tier. Routes the actual user-visible
            chat completion. The planner output is prefilled as an
            ``assistant`` turn at the end of this tier's request so the
            executor continues from the plan rather than re-planning.
        cadence_n: Throttle the planner LLM call to roughly every
            ``cadence_n`` assistant turns. Default ``2`` empirically beat
            both ``1`` (every turn, expensive + noisy) and ``4`` (too
            sparse) on the TB-Lite Nemotron-Nano sweep — see commit
            ``c9339748``. Set to ``1`` for plan-on-every-turn.
        disable_reasoning: Pass through to :class:`PlanningConfig`. ``True``
            sends ``extra_body={"chat_template_kwargs":
            {"enable_thinking": False}}`` — required for DeepSeek
            planners on NVIDIA Inference Hub, harmful for Anthropic-on-
            Bedrock planners.
        fail_open: When ``True`` (default), planner errors degrade
            gracefully — the request flows to the executor unchanged.
            When ``False``, planner errors surface as 5xx.
        enable_stats: Retained config key (accepted, no longer wires anything).
            Per-request metrics are emitted via OpenTelemetry; see
            :mod:`switchyard.lib.observability`. Default ``True``.
        preset: Optional name of the :class:`PlanExecutePresets` preset builder
            that produced this config.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    planner: LlmTarget
    executor: LlmTarget
    #: Target id the chain executor reroutes to when the picked target is
    #: evicted (e.g. context-window overflow). Must match either
    #: ``planner.id`` or ``executor.id``.
    fallback_target_on_evict: str
    cadence_n: int = Field(default=2, ge=1)
    disable_reasoning: bool = False
    fail_open: bool = True
    enable_stats: bool = True
    preset: str | None = None

    @field_validator("planner", "executor", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("planner", "executor")
    @classmethod
    def _target_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        if not tier.model:
            raise ValueError("target.model must be a non-empty string")
        return tier

    @field_validator("fallback_target_on_evict")
    @classmethod
    def _fallback_matches_existing_target(cls, value: str, info: ValidationInfo) -> str:
        valid_ids = {info.data[key].id for key in ("planner", "executor") if key in info.data}
        if value not in valid_ids:
            raise ValueError(
                f"fallback_target_on_evict={value!r} must match one of "
                f"{sorted(valid_ids)} (the configured planner/executor target ids)"
            )
        return value


__all__ = ["PlanExecuteConfig"]
