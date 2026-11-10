# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind capability-classifier routing to the candidates supplied by LiteLLM."""

from __future__ import annotations

from litellm.types.router import RoutingContext

from switchyard.libsy import Algorithm, LlmClassifierConfig, TaskClassifierConfig, algorithms

from .judge import JudgeClient, is_judge_call
from .lite_llm_request_rewriter import LiteLLMRequestRewriter
from .switchyard_routing_plugin import SwitchyardRoutingPlugin

_RESPONSE_FORMAT_TYPES = frozenset({"json_schema", "json_object"})


class ClassifierRoutingPlugin(LiteLLMRequestRewriter):
    """Route between LiteLLM's first capable and second efficient candidate by a judge's verdict.

    The judge is a separate LiteLLM deployment, named by ``judge_model`` and
    served by ``judge``; it must not be one of the routing candidates. As with
    Stage, LiteLLM preserves ``model_list`` declaration order, so declare the
    capable model first and the efficient model second.
    """

    def __init__(
        self,
        *,
        judge: JudgeClient,
        judge_model: str,
        base_threshold: float,
        threshold_step: float = 0.0,
        prompt: str | None = None,
        max_output_tokens: int = 4096,
        response_format_type: str = "json_schema",
    ) -> None:
        """Validate the judge settings and keep them for each routing call."""
        super().__init__()
        if not judge_model:
            raise ValueError("Classifier routing requires a judge_model")
        if response_format_type not in _RESPONSE_FORMAT_TYPES:
            raise ValueError("response_format_type must be 'json_schema' or 'json_object'")
        self._judge = judge
        self._judge_model = judge_model
        self._base_threshold = base_threshold
        self._threshold_step = threshold_step
        self._prompt = prompt
        self._max_output_tokens = max_output_tokens
        self._response_format_type = response_format_type

    def _algorithm(self) -> Algorithm:
        """Build the capability classifier that judges one request."""
        config = TaskClassifierConfig(
            self._base_threshold,
            threshold_step=self._threshold_step,
            max_output_tokens=self._max_output_tokens,
            response_format_type=self._response_format_type,
            **({"prompt": self._prompt} if self._prompt is not None else {}),
        )
        return algorithms.llm_classifier(LlmClassifierConfig.capability(config=config))

    async def run(self, context: RoutingContext) -> RoutingContext:
        """Judge the request once, then keep the candidate Switchyard selected."""
        if is_judge_call():
            return context
        candidates = list(dict.fromkeys(context.candidate_models))
        if len(candidates) != 2:
            raise ValueError(
                "Classifier routing requires exactly two unique LiteLLM candidates "
                "ordered as capable then efficient"
            )
        if self._judge_model in candidates:
            raise ValueError(
                f"Classifier routing judge_model {self._judge_model!r} must not be a routing candidate"
            )
        plugin = SwitchyardRoutingPlugin(
            self._algorithm(),
            models={
                "any": candidates,
                "capable": [candidates[0]],
                "efficient": [candidates[1]],
                "judge": [self._judge_model],
            },
            judge=self._judge,
        )
        return await plugin.run(context)


__all__ = ["ClassifierRoutingPlugin"]
