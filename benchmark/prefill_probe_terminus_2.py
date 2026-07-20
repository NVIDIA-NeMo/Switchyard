# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor Terminus 2 adapter for explicit prefill-probe input."""

from __future__ import annotations

import copy

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PREFILL_PROBE_INPUT_FIELD = "_switchyard_prefill_probe_input"


class PrefillProbeTerminus2(Terminus2):
    """Attach the exact task instruction to every prefill-probe LLM call."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run stock Terminus 2 with a persistent profile-local probe input."""
        call_kwargs = dict(self._llm_call_kwargs)
        configured_extra_body = call_kwargs.get("extra_body")
        if configured_extra_body is None:
            extra_body: dict[str, object] = {}
        elif isinstance(configured_extra_body, dict):
            extra_body = copy.deepcopy(configured_extra_body)
        else:
            raise TypeError("llm_call_kwargs.extra_body must be a dictionary")

        extra_body[PREFILL_PROBE_INPUT_FIELD] = instruction
        call_kwargs["extra_body"] = extra_body
        self._llm_call_kwargs = call_kwargs
        await super().run(instruction, environment, context)
