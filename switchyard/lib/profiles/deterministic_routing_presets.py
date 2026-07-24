# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named :class:`DeterministicRoutingConfig` presets keyed by shipping bundle.

Each preset bundles a validated strong/weak/classifier trio plus the
matching profile.  Callers vary only the per-deployment knobs (``api_key``,
``base_url``).

The shipping default :meth:`DeterministicRoutingPresets.coding_agent_default`
is the trio validated end-to-end on the 100-task
``openthoughts-tblite@2.0`` Terminal-Bench sweep at 2026-05 — 88.0%
classifier solve vs 84.9% force-strong at ~−25% cost vs all-strong. Use
this for any coding-agent launcher (Claude Code, Codex, Cursor) unless you
have a strong reason to deviate.

Example::

    from switchyard import (
        DeterministicRoutingPresets,
        DeterministicRoutingProfileConfig,
        ProfileSwitchyard,
    )

    config = DeterministicRoutingPresets.coding_agent_default(
        api_key=nvidia_api_key,
    )
    switchyard = ProfileSwitchyard(
        DeterministicRoutingProfileConfig.from_config(config).build()
    )
"""

from __future__ import annotations

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles.deterministic_routing_config import (
    DeterministicRoutingConfig,
)

# Shipping presets route through OpenRouter's OpenAI-compatible endpoint
# by default; callers override with ``base_url=`` for another gateway.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter model ids for the zero-flag coding-agent launcher default.
_MODEL_OPUS_4_7 = "anthropic/claude-opus-4.7"
_MODEL_KIMI_K2_6 = "moonshotai/kimi-k2.6"
_MODEL_GEMINI_3_5_FLASH = "google/gemini-3.5-flash"


class DeterministicRoutingPresets:
    """Builder of pre-built :class:`DeterministicRoutingConfig` bundles."""

    @staticmethod
    def coding_agent_default(
        *,
        api_key: str,
        base_url: str = _OPENROUTER_BASE_URL,
        timeout_secs: float | None = 600.0,
    ) -> DeterministicRoutingConfig:
        """The coding-agent-launcher default trio.

        Strong: Claude Opus 4.7.
        Weak: Kimi K2.6.
        Classifier: Gemini 3.5 Flash.
        Profile: ``coding_agent`` (SIMPLE+MEDIUM→weak, COMPLEX/REASONING→strong,
        with tool-planning escalation and high-confidence LLM alignment).

        Use for any coding-agent launcher (Claude Code, Codex, Cursor).
        """
        return DeterministicRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_7,
                # AUTO probes for Anthropic Messages; keeps cache_control on Claude.
                format=BackendFormat.AUTO,
                api_key=api_key,
                base_url=base_url,
                timeout_secs=timeout_secs,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_KIMI_K2_6,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
                timeout_secs=timeout_secs,
            ),
            classifier=LlmTarget(
                id="classifier",
                model=_MODEL_GEMINI_3_5_FLASH,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
                timeout_secs=30.0,
            ),
            profile_name="coding_agent",
            classifier_min_confidence=0.0,
            classifier_fail_open=True,
            classifier_recent_turn_window=4,
            classifier_timeout_s=30.0,
            fallback_target_on_evict="strong",
            preset="coding_agent_default",
        )


__all__ = ["DeterministicRoutingPresets"]
