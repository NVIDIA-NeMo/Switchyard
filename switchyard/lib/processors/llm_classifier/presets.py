# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile bundles for the LLM-classifier routing chain.

A "profile" is the indivisible unit that pairs a classifier system prompt with
its expected JSON schema and a 2-tier routing policy. Pairing the wrong prompt
with the wrong schema would produce JSON the validator rejects and degrade
silently through fail-open — so the profile factories below ship them together.

Three profiles are provided:

- ``general_2_tier``      — mixed traffic, conservative SIMPLE→weak mapping.
- ``coding_agent_2_tier`` — Claude Code / Codex / Cursor; SIMPLE+MEDIUM→weak.
- ``openclaw_2_tier``     — OpenClaw personal-agent channels
                            (Telegram / Discord / Slack / iMessage / WhatsApp);
                            SIMPLE+MEDIUM→weak.

The tier selector itself stays profile-agnostic — it consumes only the contract
exposed on :class:`RouteDecision` (``policy_tier()``, ``confidence``,
``abstain``). Per-profile schemas exist for prompting and analytics; routing
reads the computed policy tier, not LLM-emitted hints.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from switchyard.lib.processors.llm_classifier.request_processor import (
    DEFAULT_CLASSIFIER_SYSTEM_PROMPT,
    DEFAULT_MAX_REQUEST_CHARS,
    LLMClassifierConfig,
)
from switchyard.lib.processors.llm_classifier.signals import (
    CodingAgentRouteDecision,
    OpenClawRouteDecision,
    RouteDecision,
    RouteSignals,
    RouteTier,
)
from switchyard.lib.processors.llm_classifier.tier_selector_request_processor import (
    SignalTierSelectorConfig,
)
from switchyard.lib.processors.reasoning_hint import model_accepts_reasoning_hint

CODING_AGENT_CLASSIFIER_SYSTEM_PROMPT = """\
You are a routing classifier inside a coding-agent harness (Claude Code,
Codex CLI, Cursor, or similar). You see one *turn* in a multi-turn session
where a human and an agent collaborate on a real codebase. Rate this turn —
not the overall session — and return exactly one JSON object matching this
schema:

{
  "recommended_tier": one of ["simple", "medium", "complex", "reasoning"],
  "confidence": number between 0 and 1,
  "abstain": boolean,
  "turn_type": one of ["chitchat", "planning", "exploration", "edit",
    "debug", "explanation", "clarification", "summarize"],
  "code_modification_scope": one of ["none", "single_line", "function",
    "file", "multi_file", "cross_module"],
  "tool_call_count_estimate": non-negative integer,
  "requires_codebase_context": boolean
}

# Tier rubric — RELATIVE TO THE WEAK MODEL, NOT ABSOLUTE DIFFICULTY

You are routing between two near-frontier models. The **weak tier in 2026
is itself a top-class model** (DeepSeek V4 Pro 1.6T MoE, Kimi K2.6 1T-class,
or similar). It handles routine coding, file exploration, single-file
edits, normal debugging, multi-tool sequences, and most refactors without
meaningful loss vs the strong tier. The **strong tier** (Claude Opus 4.7
on Bedrock, GPT-5.2 frontier, or similar) only earns its keep on the
long-tail of subtle cross-module reasoning that the weak tier fumbles.

**Ask yourself for every turn: would Opus 4.7 actually outperform V4 Pro
here, or would both produce equivalent output?** If both would be fine,
this is MEDIUM. Most agent-loop turns fall in this band.

- simple    — chitchat, single-line edits, simple explanations, status
              questions answerable from the latest message. Either tier
              is overkill.
- medium    — **DEFAULT for substantive coding work.** Function- or
              file-scope edits, exploration with several tool calls,
              planning a known approach, debugging localized errors,
              routine refactors, mechanical test or build fixes, running
              a build pipeline, setting up a project from a template.
              The weak tier handles all of these — picking strong here
              just burns money for no quality gain.
- complex   — Only turns where the weak tier would *meaningfully degrade*:
              cross-module refactors with subtle invariants, debugging
              across several files where the root cause requires reasoning
              the weak tier would miss (race conditions, type-system
              interactions, non-obvious data flow), architectural choices
              with non-obvious trade-offs, security-sensitive logic with
              easy-to-miss footguns.
- reasoning — Frontier-only territory. Novel algorithm synthesis,
              correctness proofs, deep multi-step derivation that *no
              one-shot model* handles well. Rare in agent traffic; if
              you're tempted to pick this, it's almost always COMPLEX.

Default to MEDIUM; reach for COMPLEX when the turn **plausibly** stresses
cross-file reasoning, subtle invariants, or correctness-sensitive logic
the weak tier could miss. You don't need certainty — a plausible failure
mode is enough to escalate. Vague unease about "this is hard" still isn't
a reason; vague unease about "this cluster of files has subtle invariants"
*is*.

``recommended_tier`` is your own best guess. The downstream router also
computes a deterministic tier from the other fields, so focus on
extracting accurate observable features; the tier vote is a sanity check.

Counting ``tool_call_count_estimate``: count tool calls **this turn
alone** will need, not anticipated future turns. A single `cat foo.py`
is 1. A read-then-edit is 2. A grep-spread + multi-file patch is 4+.
Conservative is fine — under-estimating only loses tier escalation;
over-counting wastes the strong tier on simple turns.

Worked examples (tier calibrated to weak-tier capability in 2026, none
drawn from any benchmark task set):

* "Read /etc/hosts and tell me what's in it." → SIMPLE, 1 tool,
  exploration, requires_codebase_context=false.
* "List every file in this repo that imports a deprecated module." →
  MEDIUM, 2-3 tools, exploration. Multi-tool but mechanical; weak tier
  breezes through.
* "Scaffold a Click-based CLI that wraps an existing function and adds
  --json output." → **MEDIUM** (not COMPLEX). Tool-heavy but procedural;
  the weak tier follows the recipe correctly. Escalate only if the
  wrapper has to preserve a non-obvious behavior contract.
* "This unit test is failing — figure out why and fix it." → **MEDIUM**
  (not COMPLEX) when the failure is localized to one file. Escalate to
  COMPLEX only when the root cause genuinely spans modules or hinges on
  a subtle invariant the weak tier would miss.
* "Update this single function so its error envelope matches the
  convention the rest of the package uses." → **COMPLEX**, edit,
  scope=file, requires_codebase_context=true. **Single-file edit but
  cross-codebase synthesis** — the model has to read elsewhere in the
  package to learn the convention, then apply it consistently here.
  Exactly the case where the weak tier's narrower synthesis loses
  ground; not all file-scope edits are MEDIUM.
* "Move the retry-with-jitter logic out of the worker module and into
  a shared utility used by both the worker and the ingestion pipeline."
  → COMPLEX. Cross-module refactor; multiple call sites share contracts
  that must stay consistent.
* "Derive a closed-form expression for the variance of this estimator."
  → REASONING. Multi-step formal derivation, rare in agent traffic.
* Terminal output appearing alone (e.g. "New Terminal Output: $
  ls /app") with no question → look at the *first* user message
  (the task framing) to infer turn_type; rate against the original
  task's difficulty band, not the terminal echo.

Set "abstain": true with low confidence when the turn is too ambiguous to
classify (e.g. user message is "yes" and you cannot tell what was being
agreed to). Do not emit markdown, commentary, or chain-of-thought.
"""

OPENCLAW_CLASSIFIER_SYSTEM_PROMPT = """\
You are a routing classifier inside an OpenClaw personal-assistant agent.
OpenClaw runs across messaging channels (Telegram, Discord, Slack, Signal,
iMessage, WhatsApp). The agent has long-term memory in MEMORY.md, a fixed
personality in SOUL.md, and a user profile in USER.md. You see one inbound
*message* on one channel. Rate this message — not the broader conversation —
and return exactly one JSON object matching this schema:

{
  "recommended_tier": one of ["simple", "medium", "complex", "reasoning"],
  "confidence": number between 0 and 1,
  "abstain": boolean,
  "turn_type": one of ["chitchat", "lookup", "memory_recall", "planning",
    "tool_orchestration", "action", "explanation", "clarification"],
  "tool_call_count_estimate": non-negative integer (how many tools the
    response will likely need to call),
  "memory_dependency": one of ["none", "light", "heavy"] (does the
    response need to read across MEMORY.md / long history?),
  "external_action_required": boolean (will the response send a real
    message, post to a channel, hit a third-party API, or otherwise do
    something visible to the outside world?),
  "precision_requirement": one of ["low", "medium", "high"] (how costly
    is a small mistake?),
  "ambiguity": one of ["low", "medium", "high"] (chat is full of "yes"
    / "ok" / "sure" — is this one of those?),
  "channel_kind": one of ["casual", "deliberate"] (WhatsApp group chat
    is casual; a curated Telegram bot or business channel is deliberate)
}

Tier rubric (per message):

- simple    — chitchat, casual greetings, looking up a single fact from
              memory, clarifying questions, short status replies.
- medium    — planning a short response, explanations that need 1–3 tool
              calls, light cross-referencing of memory.
- complex   — multi-tool orchestration, irreversible external actions
              (sending messages, posting, calling third-party APIs),
              heavy memory reasoning, high-precision tasks.
- reasoning — multi-step problem solving, careful derivation, anything
              where getting the right answer requires substantial thought.

``recommended_tier`` is your own best guess. The downstream router will
also compute a deterministic tier from the other fields, so focus on
extracting accurate observable features; the tier vote is a sanity check.

Set "abstain": true with low confidence when the message is too thin to
classify (e.g. "ok", "sure", "👍" with no prior context). Do not emit
markdown, commentary, or chain-of-thought.
"""


@dataclass(frozen=True)
class LLMClassifierProfile:
    """A bundle pairing a classifier prompt with its schema and routing policy.

    Treat this as the indivisible unit. Don't mix-and-match a prompt from
    one profile with the schema from another — the JSON shapes diverge by
    design.
    """

    name: str
    system_prompt: str
    signal_schema: type[RouteDecision]
    tier_mapping: Mapping[RouteTier, str]
    default_tier: str
    min_confidence: float = 0.0
    escalate_on_tool_planning: bool = False
    """Bias the profile toward ``default_tier`` (typically strong) on
    multi-step tool-driven turns. See
    :attr:`SignalTierSelectorConfig.escalate_on_tool_planning`."""

    align_with_llm_recommendation: bool = False
    """Trust the classifier's own ``recommended_tier`` when it
    confidently says COMPLEX/REASONING but feature-based scoring
    landed on SIMPLE/MEDIUM. One-way bump. See
    :attr:`SignalTierSelectorConfig.align_with_llm_recommendation`."""

    alignment_min_confidence: float = 0.85
    """Floor on ``signals.confidence`` for the alignment bump to fire.
    See :attr:`SignalTierSelectorConfig.alignment_min_confidence`.

    Default ``0.85`` (was ``0.7`` through 2026-05-13): the bump is a
    one-way weak→strong override that bypasses the entire policy_tier
    scoring scheme, so the calibration bar should be high. Empirically
    DeepSeek V4 Flash emits confidence ≥0.7 on a substantial fraction
    of TB turns where the underlying features don't actually warrant
    COMPLEX — the 0.7 floor lets the LLM be a soft-escalation backdoor.
    0.85 keeps the bump available for genuinely-clear COMPLEX reads
    while filtering out the chronic-pessimism case."""

    def make_classifier_config(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_request_chars: int = DEFAULT_MAX_REQUEST_CHARS,
        fail_open: bool = True,
        fallback_recommended_tier: RouteTier = RouteTier.MEDIUM,
        recent_turn_window: int = 0,
        extra_headers: dict[str, str] | None = None,
        disable_reasoning: bool | None = None,
        system_prompt: str | None = None,
    ) -> LLMClassifierConfig:
        """Return an :class:`LLMClassifierConfig` with this profile's prompt baked in.

        ``disable_reasoning=None`` (default) auto-detects from ``model`` via
        :func:`~switchyard.lib.processors.reasoning_hint.model_accepts_reasoning_hint`.
        Pass an explicit bool to override.

        ``system_prompt=None`` uses the profile's built-in prompt. Passing
        a string overrides only the prompt text; callers must still keep it
        aligned with this profile's ``signal_schema``.
        """
        if disable_reasoning is None:
            disable_reasoning = model_accepts_reasoning_hint(model)
        return LLMClassifierConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_request_chars=max_request_chars,
            fail_open=fail_open,
            fallback_recommended_tier=fallback_recommended_tier,
            system_prompt=self.system_prompt if system_prompt is None else system_prompt,
            recent_turn_window=recent_turn_window,
            extra_headers=extra_headers,
            disable_reasoning=disable_reasoning,
        )

    def make_tier_selector_config(
        self,
        *,
        tier_mapping: Mapping[RouteTier, str] | None = None,
        default_tier: str | None = None,
        min_confidence: float | None = None,
        escalate_on_tool_planning: bool | None = None,
        align_with_llm_recommendation: bool | None = None,
        alignment_min_confidence: float | None = None,
    ) -> SignalTierSelectorConfig:
        """Return a :class:`SignalTierSelectorConfig` with this profile's mapping."""
        return SignalTierSelectorConfig(
            tier_mapping=self.tier_mapping if tier_mapping is None else tier_mapping,
            default_tier=self.default_tier if default_tier is None else default_tier,
            min_confidence=self.min_confidence if min_confidence is None else min_confidence,
            escalate_on_tool_planning=(
                self.escalate_on_tool_planning
                if escalate_on_tool_planning is None
                else escalate_on_tool_planning
            ),
            align_with_llm_recommendation=(
                self.align_with_llm_recommendation
                if align_with_llm_recommendation is None
                else align_with_llm_recommendation
            ),
            alignment_min_confidence=(
                self.alignment_min_confidence
                if alignment_min_confidence is None
                else alignment_min_confidence
            ),
        )


class LLMClassifierPresets:
    """Factory for pre-built classifier profiles."""

    @staticmethod
    def general_2_tier(*, weak: str, strong: str) -> LLMClassifierProfile:
        """General-purpose mixed traffic, conservative SIMPLE→weak mapping.

        Default-tier on abstain / low-confidence is ``strong`` (fail safer,
        not cheaper).
        """
        return LLMClassifierProfile(
            name="general_2_tier",
            system_prompt=DEFAULT_CLASSIFIER_SYSTEM_PROMPT,
            signal_schema=RouteSignals,
            tier_mapping={
                RouteTier.SIMPLE: weak,
                RouteTier.MEDIUM: strong,
                RouteTier.COMPLEX: strong,
                RouteTier.REASONING: strong,
            },
            default_tier=strong,
        )

    @staticmethod
    def coding_agent_2_tier(*, weak: str, strong: str) -> LLMClassifierProfile:
        """Claude Code / Codex / Cursor traffic; SIMPLE+MEDIUM→weak.

        Routine turns inside an agentic session (single-line edits, file
        exploration, localized debugging) route to the weak tier; multi-file
        refactors and deep reasoning route to strong.
        """
        return LLMClassifierProfile(
            name="coding_agent_2_tier",
            system_prompt=CODING_AGENT_CLASSIFIER_SYSTEM_PROMPT,
            signal_schema=CodingAgentRouteDecision,
            tier_mapping={
                RouteTier.SIMPLE: weak,
                RouteTier.MEDIUM: weak,
                RouteTier.COMPLEX: strong,
                RouteTier.REASONING: strong,
            },
            default_tier=strong,
            # Coding-agent turns with 2+ tool calls escalate to strong.
            # SIMPLE+MEDIUM scope w/ multi-step tool sequences are
            # exactly where weak models fumble; the override pushes
            # those to strong without changing the base mapping.
            escalate_on_tool_planning=True,
            # Trust the LLM's gestalt read when it confidently flags
            # COMPLEX/REASONING but feature scoring undercounted. The
            # classifier sees session-level context (multi-turn
            # debugging, cross-file refactoring) that discrete schema
            # fields can't fully capture.
            align_with_llm_recommendation=True,
        )

    @staticmethod
    def openclaw_2_tier(*, weak: str, strong: str) -> LLMClassifierProfile:
        """OpenClaw personal-assistant traffic; SIMPLE+MEDIUM→weak.

        Casual chitchat, lookups, light memory recall route to the weak
        tier; tool orchestration, irreversible external actions, and
        heavy memory reasoning route to strong.
        """
        return LLMClassifierProfile(
            name="openclaw_2_tier",
            system_prompt=OPENCLAW_CLASSIFIER_SYSTEM_PROMPT,
            signal_schema=OpenClawRouteDecision,
            tier_mapping={
                RouteTier.SIMPLE: weak,
                RouteTier.MEDIUM: weak,
                RouteTier.COMPLEX: strong,
                RouteTier.REASONING: strong,
            },
            default_tier=strong,
            # Personal-assistant turns chain tools (messaging + memory
            # + actions); escalate multi-step tool sequences to strong
            # since irreversible external actions are unforgiving.
            escalate_on_tool_planning=True,
            # Same rationale as coding_agent — assistant turns carry
            # channel/memory context the schema doesn't fully express,
            # so a confident LLM COMPLEX/REASONING read deserves trust.
            align_with_llm_recommendation=True,
        )


PROFILE_FACTORIES: Mapping[str, Callable[..., LLMClassifierProfile]] = {
    "general": LLMClassifierPresets.general_2_tier,
    "coding_agent": LLMClassifierPresets.coding_agent_2_tier,
    "openclaw": LLMClassifierPresets.openclaw_2_tier,
}


def profile_default_prompt(profile_name: str) -> str:
    """Return the built-in classifier prompt for ``profile_name``."""
    try:
        factory = PROFILE_FACTORIES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {profile_name!r}; expected one of "
            f"{sorted(PROFILE_FACTORIES)}"
        ) from exc
    return factory(weak="weak", strong="strong").system_prompt


def resolve_classifier_prompt(profile_name: str, prompt_override: str | None) -> str:
    """Return the effective classifier prompt for a profile + optional override."""
    if prompt_override is not None:
        return prompt_override
    return profile_default_prompt(profile_name)


def classifier_prompt_sha256(prompt: str) -> str:
    """Return a stable SHA-256 hex digest for a classifier prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


__all__ = [
    "CODING_AGENT_CLASSIFIER_SYSTEM_PROMPT",
    "LLMClassifierPresets",
    "LLMClassifierProfile",
    "OPENCLAW_CLASSIFIER_SYSTEM_PROMPT",
    "PROFILE_FACTORIES",
    "classifier_prompt_sha256",
    "profile_default_prompt",
    "resolve_classifier_prompt",
]
