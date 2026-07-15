# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named :class:`AdvisorConfig` presets keyed by shipping bundle.

The shipping default :meth:`AdvisorPresets.opus47_exec_opus48_advisor` pairs an
Opus 4.7 executor with an Opus 4.8 advisor, both served native Anthropic from
NVIDIA Inference Hub. This is the proxy-side re-creation of Anthropic's
server-side ``advisor_20260301`` beta for a gateway that cannot run it.
``strategy`` selects how the advisor participates: ``"tool_call"`` (default)
offers the executor a real, parameterless ``advisor`` tool it calls
mid-generation; ``"review_gate"`` consults the advisor once per session at the
executor's first no-tool-call turn to APPROVE or send it back (REDO).

Example::

    from switchyard import AdvisorPresets, AdvisorProfileConfig, ProfileSwitchyard

    config = AdvisorPresets.opus47_exec_opus48_advisor(api_key=nvidia_api_key)
    switchyard = ProfileSwitchyard(
        AdvisorProfileConfig.from_config(config).build().with_runtime_components()
    )
"""

from __future__ import annotations

from typing import Literal

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles.advisor_config import AdvisorConfig

# All shipping presets route through NVIDIA Inference Hub's Anthropic Messages
# endpoint by default; callers override with ``base_url=`` for a different gateway.
_INFERENCE_HUB_BASE_URL = "https://inference-api.nvidia.com/v1"

# Inference Hub model ids. Both tiers (Opus 4.7 executor, Opus 4.8 advisor) are
# served native Anthropic-Messages at ``/v1/messages`` — no OpenAI translation,
# so prompt caching survives. Both are overridable via the preset's
# ``executor_model`` / ``advisor_model``.
_MODEL_OPUS_4_7_EXECUTOR = "aws/anthropic/bedrock-claude-opus-4-7"
_MODEL_OPUS_4_8_ADVISOR = "aws/anthropic/bedrock-claude-opus-4-8"


class AdvisorPresets:
    """Factory of pre-built :class:`AdvisorConfig` bundles."""

    @staticmethod
    def opus47_exec_opus48_advisor(
        *,
        api_key: str,
        base_url: str = _INFERENCE_HUB_BASE_URL,
        timeout_secs: float | None = 600.0,
        executor_model: str = _MODEL_OPUS_4_7_EXECUTOR,
        advisor_model: str = _MODEL_OPUS_4_8_ADVISOR,
        strategy: Literal["tool_call", "review_gate"] = "tool_call",
    ) -> AdvisorConfig:
        """Opus 4.7 executor + Opus 4.8 advisor on NVIDIA Inference Hub.

        Args:
            api_key: Inference Hub API key, used for both tiers (one tenancy).
            base_url: OpenAI-compatible gateway base URL.
            timeout_secs: Per-call timeout for both tiers. Generous by default
                because the advisor consult adds an extra round-trip inside a
                single client request.
            executor_model: Override the executor model id if your tenancy
                serves Opus 4.7 under a different string.
            advisor_model: Override the advisor model id likewise.
            strategy: Advisor strategy — ``"tool_call"`` (default) or
                ``"review_gate"``.
        """
        return AdvisorConfig(
            executor=LlmTarget(
                # Native Anthropic Messages (``/v1/messages``): the request passes
                # through verbatim so the client's cache_control breakpoints reach
                # the upstream and prompt caching is honored. Inference Hub wants
                # Bearer auth (not Anthropic's x-api-key), so suppress x-api-key
                # (api_key="") and carry the key in an Authorization header.
                id="executor",
                model=executor_model,
                format=BackendFormat.ANTHROPIC,
                api_key="",
                base_url=base_url,
                timeout_secs=timeout_secs,
                extra_headers={"Authorization": f"Bearer {api_key}"},
            ),
            advisor=LlmTarget(
                # Anthropic Messages format: consulted via ``/v1/messages`` (the
                # advisor caller sends Bearer auth directly from ``api_key``).
                id="advisor",
                model=advisor_model,
                format=BackendFormat.ANTHROPIC,
                api_key=api_key,
                base_url=base_url,
                timeout_secs=timeout_secs,
            ),
            strategy=strategy,
            preset="opus47_exec_opus48_advisor",
        )


__all__ = ["AdvisorPresets"]
