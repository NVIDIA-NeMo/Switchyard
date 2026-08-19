# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind Stage routing to the candidates supplied by LiteLLM."""

from __future__ import annotations

from litellm.types.router import RoutingContext

from switchyard.libsy import algorithms

from .lite_llm_request_rewriter import LiteLLMRequestRewriter
from .switchyard_routing_plugin import SwitchyardRoutingPlugin


class StageRoutingPlugin(LiteLLMRequestRewriter):
    """Route between LiteLLM's first capable and second efficient candidate."""

    def __init__(
        self,
        *,
        picker: str,
        confidence_threshold: float,
        recent_window: int | None = None,
        escalation_note: str | None = None,
        deescalation_note: str | None = None,
        only_on_wrong_signal_escalation: bool = True,
        capable_system_prompt: str | None = None,
        efficient_system_prompt: str | None = None,
    ) -> None:
        super().__init__()
        self._picker = picker
        self._confidence_threshold = confidence_threshold
        self._recent_window = recent_window
        self._escalation_note = escalation_note
        self._deescalation_note = deescalation_note
        self._only_on_wrong_signal_escalation = only_on_wrong_signal_escalation
        self._capable_system_prompt = capable_system_prompt
        self._efficient_system_prompt = efficient_system_prompt

    async def run(self, context: RoutingContext) -> RoutingContext:
        """Build Stage routing from the current LiteLLM candidate order."""
        candidates = list(dict.fromkeys(context.candidate_models))
        if len(candidates) != 2:
            raise ValueError(
                "Stage routing requires exactly two unique LiteLLM candidates "
                "ordered as capable then efficient"
            )
        plugin = SwitchyardRoutingPlugin(
            algorithms.stage_router(
                candidates[0],
                candidates[1],
                picker=self._picker,
                confidence_threshold=self._confidence_threshold,
                recent_window=self._recent_window,
                escalation_note=self._escalation_note,
                deescalation_note=self._deescalation_note,
                only_on_wrong_signal_escalation=self._only_on_wrong_signal_escalation,
                capable_system_prompt=self._capable_system_prompt,
                efficient_system_prompt=self._efficient_system_prompt,
            )
        )
        return await plugin.run(context)


__all__ = ["StageRoutingPlugin"]
