# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned plan-execute construction."""

from __future__ import annotations

from typing import Any, Self

from switchyard.lib import metrics, spans
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.plan_execute_config import PlanExecuteConfig
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.proxy_context import CTX_ROUTER_NAME, ProxyContext
from switchyard_rust.components import set_stats_route_label
from switchyard_rust.core import ChatRequest

_EXECUTOR_STATS_TARGET = "executor"


class _ExecutorStatsTargetProcessor:
    """Stamp executor target metadata so stats rollups stay route-local."""

    def __init__(self, target_id: str) -> None:
        self._target_id = target_id

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        """Record the executor as the selected stats target before planning."""
        ctx.selected_target = self._target_id
        set_stats_route_label(ctx, _EXECUTOR_STATS_TARGET)
        # Plan-execute always serves the executor tier; the planner's own LLM call
        # is timed under its own span so it stays out of the in-process overhead.
        ctx.metadata[CTX_ROUTER_NAME] = "plan_execute"
        with spans.route_decision_span(
            router="plan_execute",
            tier=_EXECUTOR_STATS_TARGET,
            selected_target=self._target_id,
        ):
            pass
        metrics.record_routing_decision(
            router="plan_execute", source="planner", tier=_EXECUTOR_STATS_TARGET
        )
        return request


@profile_config("plan_execute")
class PlanExecuteProfileConfig:
    """Profile config wrapper for strong-planner / weak-executor profiles."""

    config: PlanExecuteConfig

    @classmethod
    def from_config(cls, config: PlanExecuteConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the plan-execute profile runtime."""
        from switchyard.lib.backends.multi_llm_backend import build_native_backend
        from switchyard.lib.processors.plan_execute import (
            PlanningConfig,
            PlanningRequestProcessor,
            is_anthropic_model,
        )
        from switchyard.lib.processors.reasoning_effort_normalizer import (
            ReasoningEffortNormalizer,
        )

        request_processors: list[Any] = [ReasoningEffortNormalizer()]

        config = self.config
        planner_endpoint = config.planner.endpoint
        is_anthropic_planner = is_anthropic_model(config.planner.model)
        if config.enable_stats:
            request_processors.append(_ExecutorStatsTargetProcessor(config.executor.id))
        request_processors.append(
            PlanningRequestProcessor(
                PlanningConfig(
                    model=config.planner.model,
                    api_key=planner_endpoint.api_key,
                    base_url=planner_endpoint.base_url,
                    timeout_s=planner_endpoint.timeout_secs,
                    disable_reasoning=config.disable_reasoning,
                    cadence_n=config.cadence_n,
                    fail_open=config.fail_open,
                    temperature=None if is_anthropic_planner else 0.0,
                ),
            )
        )

        backend = build_native_backend(config.executor)

        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


__all__ = ["PlanExecuteProfileConfig"]
