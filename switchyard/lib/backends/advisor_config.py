# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config model for the advisor review gate.

An advisor chain pairs an **executor** (the base model under test) with a
stronger **advisor**. No advisor tool is injected: the executor works the task
with its own tools; when it first produces a no-tool-call turn — a plan, or a
claim of "done" — the backend consults the advisor once to APPROVE or send it
back (REDO) with an optimized plan. See
``switchyard/lib/backends/advisor_loop_backend.py``.

Both tiers are ordinary targets; each tier's ``format`` selects its wire
independently and tiers mix freely. ``anthropic`` targets are served native
Anthropic-Messages with the body passed through verbatim (the client's prompt
caching survives); ``openai`` targets (Qwen, DeepSeek, vLLM/NIM, OpenAI) are
served OpenAI Chat Completions, likewise verbatim. ``responses`` targets are
rejected (the advisor loop is Chat-shaped).
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from switchyard.lib.backends.advisor_prompts import (
    ADVISOR_SYSTEM_PROMPT,
    REDO_FEEDBACK_PREFIX,
    REVIEWER_SYSTEM_PROMPT,
    SEED_ADVICE_PREFIX,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget, coerce_llm_target


class AdvisorConfig(BaseModel):
    """Configuration for the advisor review gate.

    Attributes:
        executor: The base model under test. Runs the user-visible chat
            completion with the client's own tools.
        advisor: The stronger advisor model. Must be at least as capable.

        reviewer_system_prompt: System prompt for the advisor's review call;
            instructs the APPROVE / REDO contract.
        redo_feedback_prefix: Prepended to the advisor's REDO plan when it is
            injected back to the executor as a user turn. Tune per executor
            family (e.g. append "continue using tool calls only" for small
            OSS executors).
        gate_trigger: What fires the once-per-session review.
            ``"no_tool_call"`` (default) reviews the executor's first turn
            without tool calls — right for function-calling agent harnesses.
            ``"pattern"`` reviews the first turn whose text matches
            ``gate_trigger_pattern`` — for text-protocol harnesses (e.g.
            Terminal-Bench's terminus), where every turn lacks tool calls and
            completion is declared with a textual marker instead.
        gate_trigger_pattern: Regex searched against the executor turn's text
            when ``gate_trigger`` is ``"pattern"``
            (e.g. ``task_complete["\\s>:]*true`` for terminus).
        max_reviews: Budget of advisor reviews per budget scope: the caller's
            ``proxy_x_session_id`` header when present (one evaluation/task
            on benchmark harnesses, sub-agents included), else one scope for
            the whole backend instance. The default (1) preserves the
            original once-per-task gate; higher values re-review later
            trigger turns (e.g. a re-declared completion after a REDO),
            making the gate a sequential best-of-(N+1) with the advisor as
            judge. Failed advisor consults do not consume the budget.
        gate_stall_turns: When > 0, additionally trigger a review (once per
            session, consuming review budget) at the first request whose
            conversation already carries at least this many assistant turns —
            a mid-task checkpoint for executors that grind without ever
            declaring completion. 0 disables.
        gate_min_tool_results: For the ``no_tool_call`` trigger: only review
            a no-tool-call turn when the conversation carries at least this
            many tool results — skips reviewing early commentary turns on
            chatty harnesses. 0 reviews as before.

        seed_plan_advice: Consult the advisor once at the start of each
            session — before the executor's first turn — and inject its
            upfront plan into the session's first user message
            (``seed_advice_prefix`` + advice). The advice is cached per
            session (keyed by the conversation's stable prefix) and
            re-injected identically on every turn, so it stays visible for
            the whole session while the upstream cache prefix stays stable.
            Proxy-triggered, so it fires even on executors/harnesses that
            never call tools. The seed consult uses ``advisor_system_prompt``.
            Fail-open: a failed seed consult leaves the session unseeded.
        seed_advice_prefix: Prepended to the seeded advice when it is
            injected into the first user message.
        advisor_system_prompt: System prompt for the seed consult's advisor
            call; tells the advisor to plan, not act.

        advisor_max_tokens: Cap on the advisor's output per call.
        advisor_temperature: Sampling temperature for the advisor call. ``None``
            (default) omits the field — required for Anthropic targets that
            reject ``temperature``.
        transcript_max_chars: Cap on the serialized transcript handed to the
            advisor, so a long agent conversation can't blow its context.
            The default (200k chars ≈ 50k tokens) fits comfortably in a
            frontier advisor's window; the middle of an over-cap conversation
            is dropped (task head + recent tail survive).
        fail_open: When ``True`` (default), an advisor-call failure degrades
            gracefully — the turn passes through as APPROVE. When ``False``,
            the failure surfaces as 5xx.
        enable_stats: Record executor success/error + latency into the shared
            accumulator and stamp ``ctx.selected_model``.
        preset: Optional name of the preset that produced this config.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    executor: LlmTarget
    advisor: LlmTarget

    # review gate
    reviewer_system_prompt: str = REVIEWER_SYSTEM_PROMPT
    redo_feedback_prefix: str = REDO_FEEDBACK_PREFIX
    gate_trigger: Literal["no_tool_call", "pattern"] = "no_tool_call"
    gate_trigger_pattern: str = ""
    max_reviews: int = Field(default=1, ge=1)
    gate_stall_turns: int = Field(default=0, ge=0)
    gate_min_tool_results: int = Field(default=0, ge=0)

    # seed advice
    seed_plan_advice: bool = False
    seed_advice_prefix: str = SEED_ADVICE_PREFIX
    advisor_system_prompt: str = ADVISOR_SYSTEM_PROMPT

    # shared
    advisor_max_tokens: int = Field(default=2048, ge=1)
    advisor_temperature: float | None = None
    transcript_max_chars: int = Field(default=200_000, ge=256)
    fail_open: bool = True
    enable_stats: bool = True
    preset: str | None = None

    @field_validator("executor", "advisor", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("executor", "advisor")
    @classmethod
    def _target_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        if not tier.model:
            raise ValueError("target.model must be a non-empty string")
        return tier

    @field_validator("executor", "advisor")
    @classmethod
    def _target_format_supported(cls, tier: LlmTarget, info: ValidationInfo) -> LlmTarget:
        if tier.format == BackendFormat.RESPONSES:
            raise ValueError(
                f"{info.field_name}.format 'responses' is not supported by the advisor "
                "backend (the loop is Chat-shaped); use 'openai' or 'anthropic'"
            )
        return tier

    @field_validator("gate_trigger_pattern")
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        if value:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"gate_trigger_pattern is not a valid regex: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _pattern_trigger_requires_pattern(self) -> Self:
        if self.gate_trigger == "pattern" and not self.gate_trigger_pattern:
            raise ValueError(
                "gate_trigger 'pattern' requires a non-empty gate_trigger_pattern"
            )
        return self

__all__ = ["AdvisorConfig"]
