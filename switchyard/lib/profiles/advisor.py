# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned advisor construction.

Chain shape::

    [ReasoningEffortNormalizer] → AdvisorToolCallBackend | AdvisorLoopBackend

``config.strategy`` selects the backend: ``"tool_call"`` (default) offers the
executor a proxy-intercepted ``advisor`` tool
(:class:`~switchyard.lib.backends.advisor_tool_call_backend.AdvisorToolCallBackend`);
``"review_gate"`` consults the advisor once per session at the executor's first
no-tool-call turn
(:class:`~switchyard.lib.backends.advisor_loop_backend.AdvisorLoopBackend`).
Both delegate the executor call to a native Anthropic backend (caching intact),
stamp ``ctx.selected_model``, and record the advisor's usage themselves; they
cannot be wrapped by ``StatsLlmBackend``, so the runtime attaches the shared
accumulator through the ``_stats`` compatibility hook — mirroring
``LatencyServiceLLMBackend``. Serving-level stats processors arrive via
``with_runtime_components``, like every other profile.
"""

from __future__ import annotations

from typing import Any, Self

from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config


@profile_config("advisor")
class AdvisorProfileConfig:
    """Profile config wrapper for executor + stronger-advisor chains."""

    config: AdvisorConfig

    @classmethod
    def from_config(cls, config: AdvisorConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the advisor profile runtime for the configured strategy."""
        from switchyard.lib.processors.reasoning_effort_normalizer import (
            ReasoningEffortNormalizer,
        )

        config = self.config
        backend: Any
        if config.strategy == "review_gate":
            from switchyard.lib.backends.advisor_loop_backend import AdvisorLoopBackend

            backend = AdvisorLoopBackend(config)
        else:
            from switchyard.lib.backends.advisor_tool_call_backend import (
                AdvisorToolCallBackend,
            )

            backend = AdvisorToolCallBackend(config)

        # Normalize unsupported ``reasoning_effort`` values (Claude Code's
        # ``/effort xhigh`` in particular) before they reach the executor.
        # Same placement as the plan-execute profile.
        return ComponentChainProfile(
            request_processors=[ReasoningEffortNormalizer()],
            backend=backend,
        )


__all__ = ["AdvisorProfileConfig"]
