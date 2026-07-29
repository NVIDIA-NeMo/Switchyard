# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config model for the advisor profile.

An advisor chain pairs an **executor** (the base model under test) with a
stronger **advisor**. ``strategy`` selects how the advisor participates:

* ``"tool_call"`` (default, the shipping strategy) — the executor is offered a
  real, parameterless ``advisor`` tool. When it calls, the backend consults the
  advisor model on the full transcript and feeds the guidance back as the tool
  result, looping until the executor stops asking. The proxy-side re-creation
  of Anthropic's ``advisor_20260301`` server tool for gateways that cannot run
  it server-side. See ``switchyard/lib/backends/advisor_tool_call_backend.py``.

* ``"review_gate"`` — no advisor tool is injected. The executor works the task
  with its own tools; when it first produces a no-tool-call turn — a plan, or
  a claim of "done" — the backend consults the advisor once to APPROVE or send
  it back (REDO) with an optimized plan. See
  ``switchyard/lib/backends/advisor_loop_backend.py``.

Both tiers are ordinary targets; each tier's ``format`` selects its wire
independently and tiers mix freely under either strategy. ``anthropic``
targets are served native Anthropic-Messages with the body passed through
verbatim (the client's prompt caching survives); ``openai`` targets (Qwen,
DeepSeek, vLLM/NIM, OpenAI) are served OpenAI Chat Completions, likewise
verbatim. ``responses`` targets are rejected (the advisor loop is Chat-shaped).
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

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget, coerce_llm_target
from switchyard.lib.profiles.advisor_prompts import (
    ADVISOR_LENGTH_LINE,
    ADVISOR_SYSTEM_PROMPT,
    ADVISOR_TOOL_DESCRIPTION,
    EXECUTOR_STEERING,
    REDO_FEEDBACK_PREFIX,
    REVIEWER_SYSTEM_PROMPT,
    SEED_ADVICE_PREFIX,
)


class AdvisorConfig(BaseModel):
    """Configuration for the advisor profile.

    Attributes:
        executor: The base model under test. Runs the user-visible chat
            completion with the client's own tools.
        advisor: The stronger advisor model. Must be at least as capable.
        strategy: How the advisor participates. ``"tool_call"`` offers the
            executor a real ``advisor`` tool it calls mid-generation;
            ``"review_gate"`` consults the advisor once per session at the
            executor's first no-tool-call turn.

        advisor_tool_name: (tool_call) Tool name the executor calls to consult
            the advisor. Must match what the steering prompt references
            (``"advisor"``).
        max_uses: (tool_call) Per-request cap on advisor consultations. Over-cap
            calls receive a ``max_uses exceeded`` tool result without a consult
            (mirroring the native tool's ``max_uses_exceeded`` error result)
            and the executor continues. Failed (fail-open) consultations count
            toward this cap, which bounds retry storms when the advisor is
            unavailable.
        inject_steering: (tool_call) Prepend ``executor_steering`` to the
            executor's system prompt and append ``advisor_length_line`` to the
            latest user turn. The doc notes executors under-call the advisor
            without this; keep it a toggle so a benchmark can ablate steering.
        executor_steering: (tool_call) Verbatim doc steering for the executor.
        advisor_length_line: (tool_call) Verbatim doc length hint, injected into
            the latest user turn (the advisor sees it via the transcript).
        advisor_system_prompt: (tool_call) System prompt for the advisor's own
            LLM call (authored — the doc publishes none).
        advisor_tool_description: (tool_call) Description for the synthetic
            ``advisor`` tool.

        reviewer_system_prompt: (review_gate) System prompt for the advisor's
            review call; instructs the APPROVE / REDO contract.
        redo_feedback_prefix: (review_gate) Prepended to the advisor's REDO
            plan when it is injected back to the executor as a user turn.
            Tune per executor family (e.g. append "continue using tool calls
            only" for small OSS executors).
        gate_trigger: (review_gate) What fires the once-per-session review.
            ``"no_tool_call"`` (default) reviews the executor's first turn
            without tool calls — right for function-calling agent harnesses.
            ``"pattern"`` reviews the first turn whose text matches
            ``gate_trigger_pattern`` — for text-protocol harnesses (e.g.
            Terminal-Bench's terminus), where every turn lacks tool calls and
            completion is declared with a textual marker instead.
        gate_trigger_pattern: (review_gate) Regex searched against the
            executor turn's text when ``gate_trigger`` is ``"pattern"``
            (e.g. ``task_complete["\\s>:]*true`` for terminus).
        max_reviews: (review_gate) Per-session budget of advisor reviews.
            The default (1) preserves the original once-per-session gate;
            higher values re-review later trigger turns (e.g. a re-declared
            completion after a REDO), making the gate a sequential
            best-of-(N+1) with the advisor as judge.
        gate_stall_turns: (review_gate) When > 0, additionally trigger a
            review (once per session, consuming review budget) at the first
            request whose conversation already carries at least this many
            assistant turns — a mid-task checkpoint for executors that grind
            without ever declaring completion. 0 disables.
        gate_min_tool_results: (review_gate) For the ``no_tool_call``
            trigger: only review a no-tool-call turn when the conversation
            carries at least this many tool results — skips reviewing early
            commentary turns on chatty harnesses. 0 reviews as before.

        seed_plan_advice: (both strategies) Consult the advisor once at the
            start of each session — before the executor's first turn — and
            inject its upfront plan into the session's first user message
            (``seed_advice_prefix`` + advice). The advice is cached per
            session (keyed by the conversation's stable prefix) and
            re-injected identically on every turn, so it stays visible for
            the whole session while the upstream cache prefix stays stable.
            Proxy-triggered, so it fires even on executors/harnesses that
            never call tools. The seed consult uses ``advisor_system_prompt``.
            Fail-open: a failed seed consult leaves the session unseeded.
        seed_advice_prefix: Prepended to the seeded advice when it is
            injected into the first user message.

        advisor_max_tokens: Cap on the advisor's output per call (the doc's
            recommended starting point is 2048).
        advisor_temperature: Sampling temperature for the advisor call. ``None``
            (default) omits the field — required for Anthropic targets that
            reject ``temperature``.
        transcript_max_chars: Cap on the serialized transcript handed to the
            advisor, so a long agent conversation can't blow its context.
        fail_open: When ``True`` (default), an advisor-call failure degrades
            gracefully — the executor proceeds unadvised (tool_call) or the
            turn passes through as APPROVE (review_gate). When ``False``, the
            failure surfaces as 5xx.
        enable_stats: Record executor success/error + latency into the shared
            accumulator and stamp ``ctx.selected_model``.
        preset: Optional name of the preset that produced this config.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    executor: LlmTarget
    advisor: LlmTarget
    strategy: Literal["tool_call", "review_gate"] = "tool_call"

    # tool_call strategy
    advisor_tool_name: str = "advisor"
    max_uses: int = Field(default=2, ge=1)
    inject_steering: bool = True
    executor_steering: str = EXECUTOR_STEERING
    advisor_length_line: str = ADVISOR_LENGTH_LINE
    advisor_system_prompt: str = ADVISOR_SYSTEM_PROMPT
    advisor_tool_description: str = ADVISOR_TOOL_DESCRIPTION

    # review_gate strategy
    reviewer_system_prompt: str = REVIEWER_SYSTEM_PROMPT
    redo_feedback_prefix: str = REDO_FEEDBACK_PREFIX
    gate_trigger: Literal["no_tool_call", "pattern"] = "no_tool_call"
    gate_trigger_pattern: str = ""
    max_reviews: int = Field(default=1, ge=1)
    gate_stall_turns: int = Field(default=0, ge=0)
    gate_min_tool_results: int = Field(default=0, ge=0)

    # shared
    seed_plan_advice: bool = False
    seed_advice_prefix: str = SEED_ADVICE_PREFIX
    advisor_max_tokens: int = Field(default=2048, ge=1)
    advisor_temperature: float | None = None
    transcript_max_chars: int = Field(default=24_000, ge=256)
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
                "profile (the loop is Chat-shaped); use 'openai' or 'anthropic'"
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
